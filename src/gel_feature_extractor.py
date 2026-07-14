from __future__ import annotations
import argparse
import math
import os
import re
from pathlib import Path
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy.signal import find_peaks, peak_widths

def natural_key(s: str):
    parts = re.split('(\\d+)', str(s))
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_image_rgb(image_path: str) -> np.ndarray:
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f'Could not read image: {image_path}')
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def preprocess_image(img_rgb: np.ndarray):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    B_chan = img_rgb[:, :, 2].astype(np.float32)
    inv = 1.0 - gray / 255.0
    bg = cv2.GaussianBlur((inv * 255).astype(np.uint8), (0, 0), 35).astype(np.float32) / 255.0
    I_corr = np.clip(inv - bg, 0, 1)
    return (gray, B_chan, I_corr)

def lane_energy(I_corr: np.ndarray, y_frac=(0.05, 0.55), sig_y=12, sig_x=8):
    H, W = I_corr.shape
    y0, y1 = (int(y_frac[0] * H), int(y_frac[1] * H))
    Ic = I_corr[y0:y1, :]
    hp = Ic - ndi.gaussian_filter(Ic, sigma=(sig_y, 0))
    E = np.mean(np.abs(hp), axis=0)
    E = ndi.gaussian_filter1d(E, sigma=sig_x)
    E = (E - E.min()) / (E.max() - E.min() + 1e-09)
    return E

def estimate_lane_intervals(I_corr: np.ndarray, lane_n=15, y_frac=(0.05, 0.55), prom=0.02, active_thr=0.12, alpha=0.2):
    E = lane_energy(I_corr, y_frac=y_frac)
    H, W = I_corr.shape
    idx = np.where(E > active_thr)[0]
    if idx.size > 0:
        splits = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[idx[0], idx[splits + 1]]
        ends = np.r_[idx[splits], idx[-1]]
        j = int(np.argmax(ends - starts))
        a, b = (int(starts[j]), int(ends[j]))
    else:
        a, b = (0, W - 1)
    Eseg = E[a:b + 1]
    est_spacing = max(10, Eseg.size / lane_n)
    peaks, _ = find_peaks(Eseg, distance=int(est_spacing * 0.55), prominence=prom)
    if len(peaks) >= lane_n:
        order = np.argsort(Eseg[peaks])[::-1]
        chosen = []
        for p in peaks[order]:
            if all((abs(p - c) > est_spacing * 0.45 for c in chosen)):
                chosen.append(int(p))
            if len(chosen) == lane_n:
                break
        chosen = np.array(sorted(chosen), dtype=int)
    else:
        centers0 = np.linspace(int(0.05 * Eseg.size), int(0.95 * Eseg.size), lane_n).astype(int)
        win = int(est_spacing * 0.6)
        chosen = []
        for c0 in centers0:
            l = max(0, c0 - win)
            r = min(Eseg.size - 1, c0 + win)
            chosen.append(l + int(np.argmax(Eseg[l:r + 1])))
        chosen = np.array(sorted(chosen), dtype=int)
    centers = chosen + a
    bounds = []
    for i in range(lane_n - 1):
        l, r = (centers[i], centers[i + 1])
        local = E[l:r + 1]
        bnd = l + int(np.argmin(local))
        bounds.append(bnd)
    spacing = int(np.median(np.diff(centers)))
    left_edge = max(0, int(round(centers[0] - spacing * 0.55)))
    right_edge = min(W, int(round(centers[-1] + spacing * 0.55)))
    lanes = []
    prev = left_edge
    for bnd in bounds:
        lanes.append((prev, bnd))
        prev = bnd
    lanes.append((prev, right_edge))
    lanes_n = []
    for i, (L, R) in enumerate(lanes):
        c = int(np.clip(centers[i], L, R - 1))
        e_val = min(E[L], E[R - 1])
        T = e_val + alpha * (E[c] - e_val)
        ll = c
        while ll > L and E[ll] > T:
            ll -= 1
        rr = c
        while rr < R - 1 and E[rr] > T:
            rr += 1
        minw = max(8, int(0.35 * spacing))
        if rr - ll < minw:
            ll = max(L, c - int(0.45 * spacing))
            rr = min(R, c + int(0.45 * spacing))
        lanes_n.append((ll, rr))
    return {'E': E, 'active_span': (a, b), 'centers': centers, 'bounds': bounds, 'lanes': lanes_n}

def otsu_threshold(values, nbins=128):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 50:
        return float(np.mean(v)) if v.size else 0.0
    v = np.clip(v, 0, 1)
    hist, bins = np.histogram(v, bins=nbins, range=(0, 1))
    prob = hist / (hist.sum() + 1e-12)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * ((bins[:-1] + bins[1:]) / 2))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    k = int(np.argmax(sigma_b2))
    return float((bins[k] + bins[k + 1]) / 2)

def detect_bands_and_mask(I_corr: np.ndarray, lanes):
    H, W = I_corr.shape
    y_top = int(0.16 * H)
    y_bot = int(0.92 * H)
    accepted_bands = [[] for _ in range(len(lanes))]
    full_mask = np.zeros((H, W), dtype=bool)
    for lane_idx, (L, R) in enumerate(lanes):
        lane = I_corr[:, L:R]
        if lane.size < 100:
            continue
        p = np.mean(lane, axis=1)
        p_s = ndi.gaussian_filter1d(p, sigma=2)
        baseline = ndi.median_filter(p_s, size=151)
        p2 = np.clip(p_s - baseline, 0, None)
        scale = np.percentile(p2[y_top:y_bot], 95) + 1e-09
        pn = np.clip(p2 / scale, 0, 3)
        peaks, _ = find_peaks(pn[y_top:y_bot], distance=16, prominence=0.12, height=0.12)
        peaks = peaks + y_top
        if len(peaks) == 0:
            continue
        widths = peak_widths(pn, peaks, rel_height=0.5)
        for li, ri in zip(widths[2], widths[3]):
            s = int(max(y_top, math.floor(li) - 3))
            e = int(min(y_bot, math.ceil(ri) + 3))
            if e - s < 5:
                continue
            win = I_corr[s:e, L:R]
            flat = win.ravel()
            k_hi = max(10, int(0.1 * flat.size))
            k_lo = max(10, int(0.2 * flat.size))
            hi = float(np.mean(np.partition(flat, -k_hi)[-k_hi:]))
            lo = float(np.mean(np.partition(flat, k_lo)[:k_lo]))
            if hi - lo < 0.02 or hi < 0.03:
                continue
            thr = max(otsu_threshold(flat), float(np.percentile(flat, 80)) * 0.6)
            m = win > thr
            m = ndi.binary_opening(m, np.ones((2, 2)))
            m = ndi.binary_closing(m, np.ones((3, 3)))
            lbl, n = ndi.label(m)
            if n == 0:
                continue
            sizes = ndi.sum(m, lbl, index=range(1, n + 1))
            k = int(np.argmax(sizes)) + 1
            m_big = lbl == k
            cols = np.where(np.any(m_big, axis=0))[0]
            span = int(cols.max() - cols.min() + 1) if cols.size else 0
            if sizes[k - 1] < max(30, 0.012 * win.size) or span < 0.35 * (R - L):
                continue
            accepted_bands[lane_idx].append((s, e))
            full_mask[s:e, L:R] |= m_big
    full_mask = ndi.binary_dilation(full_mask, np.ones((3, 3)))
    return (accepted_bands, full_mask)

def compute_numbered_values(gray: np.ndarray, B_chan: np.ndarray, mask: np.ndarray, lanes, image_id: str) -> pd.DataFrame:
    labels, n = ndi.label(mask)
    rows = []
    all_mask = mask.copy()
    comps = []
    for comp_id in range(1, n + 1):
        comp = labels == comp_id
        area = int(comp.sum())
        if area < 20:
            continue
        ys, xs = np.where(comp)
        y_center = float(np.mean(ys))
        x_center = float(np.mean(xs))
        x_left, x_right = (int(xs.min()), int(xs.max()))
        y_top, y_bottom = (int(ys.min()), int(ys.max()))
        lane_guess = None
        for i, (L, R) in enumerate(lanes, start=1):
            if L <= x_center <= R:
                lane_guess = i
                break
        if lane_guess is None:
            centers = [0.5 * (L + R) for L, R in lanes]
            lane_guess = int(np.argmin([abs(c - x_center) for c in centers])) + 1
        comps.append({'raw_comp_id': comp_id, 'mask': comp, 'area_px': area, 'x_center': x_center, 'y_center': y_center, 'x_left': x_left, 'x_right': x_right, 'y_top': y_top, 'y_bottom': y_bottom, 'lane_guess': lane_guess})
    if len(comps) == 0:
        return pd.DataFrame(columns=['image_id', 'component_id', 'component_label', 'lane_guess', 'band_guess', 'x_center', 'y_center', 'x_left', 'x_right', 'y_top', 'y_bottom', 'area_px', 'bg_gray', 'bg_blue', 'mean_gray', 'mean_blue', 'IOD_gray', 'IOD_blue', 'IOD_log', 'IOD_gray_per_area', 'IOD_blue_per_area'])
    comps_df = pd.DataFrame([{k: v for k, v in c.items() if k != 'mask'} for c in comps])
    comps_df = comps_df.sort_values(['lane_guess', 'y_center']).reset_index(drop=True)
    comps_df['band_guess'] = comps_df.groupby('lane_guess').cumcount() + 1
    comps_df['component_id'] = np.arange(1, len(comps_df) + 1)
    rawid_to_sorted = {int(row['raw_comp_id']): int(row['component_id']) for _, row in comps_df.iterrows()}
    out_rows = []
    for c in comps:
        raw_comp_id = c['raw_comp_id']
        comp_id = rawid_to_sorted[raw_comp_id]
        row_meta = comps_df[comps_df['raw_comp_id'] == raw_comp_id].iloc[0]
        comp = c['mask']
        dil = ndi.binary_dilation(comp, iterations=8)
        ring = dil & ~all_mask
        lane_idx = int(row_meta['lane_guess']) - 1
        L, R = lanes[lane_idx]
        lane_region = np.zeros_like(ring, dtype=bool)
        lane_region[:, L:R] = True
        ring = ring & lane_region
        if ring.sum() < 20:
            ring = lane_region & ~all_mask
        if ring.sum() < 20:
            bg_gray = float(np.median(gray))
            bg_blue = float(np.median(B_chan))
        else:
            bg_gray = float(np.median(gray[ring]))
            bg_blue = float(np.median(B_chan[ring]))
        band_gray = gray[comp].astype(np.float32)
        band_blue = B_chan[comp].astype(np.float32)
        mean_gray = float(np.mean(band_gray))
        mean_blue = float(np.mean(band_blue))
        iod_gray = float(np.sum(np.clip(bg_gray - band_gray, 0, None)))
        iod_blue = float(np.sum(np.clip(bg_blue - band_blue, 0, None)))
        iod_log = float(np.sum(np.log((bg_gray + 1.0) / (band_gray + 1.0))))
        area_px = int(row_meta['area_px'])
        iod_gray_per_area = iod_gray / (area_px + 1e-09)
        iod_blue_per_area = iod_blue / (area_px + 1e-09)
        comp_label = f"L{int(row_meta['lane_guess'])}B{int(row_meta['band_guess'])}"
        out_rows.append({'image_id': image_id, 'component_id': comp_id, 'component_label': comp_label, 'lane_guess': int(row_meta['lane_guess']), 'band_guess': int(row_meta['band_guess']), 'x_center': float(row_meta['x_center']), 'y_center': float(row_meta['y_center']), 'x_left': int(row_meta['x_left']), 'x_right': int(row_meta['x_right']), 'y_top': int(row_meta['y_top']), 'y_bottom': int(row_meta['y_bottom']), 'area_px': area_px, 'bg_gray': bg_gray, 'bg_blue': bg_blue, 'mean_gray': mean_gray, 'mean_blue': mean_blue, 'IOD_gray': iod_gray, 'IOD_blue': iod_blue, 'IOD_log': iod_log, 'IOD_gray_per_area': iod_gray_per_area, 'IOD_blue_per_area': iod_blue_per_area})
    out_df = pd.DataFrame(out_rows).sort_values(['lane_guess', 'band_guess']).reset_index(drop=True)
    return out_df

def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 1.0
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return float(max(1.4826 * mad, 1.0))

def binary_hysteresis_from_signal(signal: np.ndarray, sigma: float, high_k: float=1.5, low_k: float=0.75) -> np.ndarray:
    high = signal >= high_k * sigma
    low = signal >= low_k * sigma
    if not high.any():
        return low
    return ndi.binary_propagation(high, mask=low)

def compute_improved_features(image_id: str, img_rgb: np.ndarray, mask: np.ndarray, numbered_df: pd.DataFrame, ring_iter: int=8) -> pd.DataFrame:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    red = img_rgb[:, :, 0].astype(np.float32)
    green = img_rgb[:, :, 1].astype(np.float32)
    blue = img_rgb[:, :, 2].astype(np.float32)
    labels, n = ndi.label(mask)
    objs = ndi.find_objects(labels)
    bbox_to_lab = {}
    for lab, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        ys, xs = sl
        bbox_to_lab[xs.start, xs.stop - 1, ys.start, ys.stop - 1] = lab
    rows = []
    for _, row in numbered_df.iterrows():
        bbox = (int(row.x_left), int(row.x_right), int(row.y_top), int(row.y_bottom))
        lab = bbox_to_lab.get(bbox, None)
        if lab is None:
            continue
        comp = labels == lab
        x1, x2, y1, y2 = bbox
        pad = max(ring_iter + 4, 10)
        wx1, wy1 = (max(0, x1 - pad), max(0, y1 - pad))
        wx2, wy2 = (min(mask.shape[1] - 1, x2 + pad), min(mask.shape[0] - 1, y2 + pad))
        comp_local = comp[wy1:wy2 + 1, wx1:wx2 + 1]
        all_local = mask[wy1:wy2 + 1, wx1:wx2 + 1]
        gray_local = gray[wy1:wy2 + 1, wx1:wx2 + 1]
        red_local = red[wy1:wy2 + 1, wx1:wx2 + 1]
        green_local = green[wy1:wy2 + 1, wx1:wx2 + 1]
        blue_local = blue[wy1:wy2 + 1, wx1:wx2 + 1]
        dil = ndi.binary_dilation(comp_local, iterations=ring_iter)
        ring = dil & ~all_local
        if ring.sum() < 20:
            dil = ndi.binary_dilation(comp_local, iterations=ring_iter * 2)
            ring = dil & ~all_local
        if ring.sum() < 20:
            ring = ~all_local
        if ring.sum() < 20:
            continue
        bg_gray = float(np.median(gray_local[ring]))
        bg_red = float(np.median(red_local[ring]))
        bg_green = float(np.median(green_local[ring]))
        bg_blue = float(np.median(blue_local[ring]))
        sigma_gray = robust_sigma(gray_local[ring])
        sigma_blue = robust_sigma(blue_local[ring])
        sig_gray = np.clip(bg_gray - gray_local, 0, None)
        sig_blue = np.clip(bg_blue - blue_local, 0, None)
        iod_gray_soft = float(np.sum(np.clip(sig_gray[comp_local] - 1.0 * sigma_gray, 0, None)))
        iod_blue_soft = float(np.sum(np.clip(sig_blue[comp_local] - 1.0 * sigma_blue, 0, None)))
        sel_gray = comp_local & binary_hysteresis_from_signal(sig_gray, sigma_gray, 1.5, 0.75)
        sel_blue = comp_local & binary_hysteresis_from_signal(sig_blue, sigma_blue, 1.5, 0.75)
        if not sel_gray.any():
            sel_gray = comp_local & (sig_gray >= 1.0 * sigma_gray)
        if not sel_blue.any():
            sel_blue = comp_local & (sig_blue >= 1.0 * sigma_blue)
        if not sel_gray.any():
            sel_gray = comp_local
        if not sel_blue.any():
            sel_blue = comp_local
        iod_gray_hyst = float(np.sum(sig_gray[sel_gray]))
        iod_blue_hyst = float(np.sum(sig_blue[sel_blue]))
        od_r = np.log((bg_red + 1.0) / (red_local + 1.0))
        od_g = np.log((bg_green + 1.0) / (green_local + 1.0))
        od_b = np.log((bg_blue + 1.0) / (blue_local + 1.0))
        od_neutral = np.minimum(np.minimum(od_r, od_g), od_b)
        od_yellow = np.clip(od_b - 0.5 * (od_r + od_g), 0, None)
        sigma_neutral = robust_sigma(od_neutral[ring])
        sigma_yellow = robust_sigma(od_yellow[ring])
        iod_neutral = float(np.sum(np.clip(od_neutral[comp_local] - 0.5 * sigma_neutral, 0, None)))
        iod_yellow = float(np.sum(np.clip(od_yellow[comp_local] - 0.5 * sigma_yellow, 0, None)))
        area_px = int(row.area_px)
        area_gray_hyst_px = int(sel_gray.sum())
        area_blue_hyst_px = int(sel_blue.sum())
        iod_hybrid = float(iod_gray_hyst + 0.5 * iod_yellow)
        yellow_ratio = float(iod_yellow / max(iod_neutral, 0.001))
        d = row.to_dict()
        d.update({'bg_gray_recalc': bg_gray, 'bg_blue_recalc': bg_blue, 'sigma_gray': sigma_gray, 'sigma_blue': sigma_blue, 'IOD_gray_soft': iod_gray_soft, 'IOD_blue_soft': iod_blue_soft, 'IOD_gray_hyst': iod_gray_hyst, 'IOD_blue_hyst': iod_blue_hyst, 'area_gray_hyst_px': area_gray_hyst_px, 'area_blue_hyst_px': area_blue_hyst_px, 'mask_area_ratio_gray_hyst': float(area_gray_hyst_px / max(area_px, 1)), 'mask_area_ratio_blue_hyst': float(area_blue_hyst_px / max(area_px, 1)), 'IOD_gray_hyst_per_area': float(iod_gray_hyst / max(area_gray_hyst_px, 1)), 'IOD_blue_hyst_per_area': float(iod_blue_hyst / max(area_blue_hyst_px, 1)), 'IOD_neutral': iod_neutral, 'IOD_yellow': iod_yellow, 'IOD_hybrid': iod_hybrid, 'yellow_ratio': yellow_ratio})
        rows.append(d)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(['lane_guess', 'band_guess']).reset_index(drop=True)
    return out

def save_lane_overlay(img_rgb, lanes, save_path, title='Detected lane intervals'):
    H, W, _ = img_rgb.shape
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img_rgb)
    ax.axis('off')
    for i, (L, R) in enumerate(lanes, start=1):
        xc = 0.5 * (L + R)
        ax.plot([L, L], [0, H], linewidth=1, alpha=0.85, color='cyan')
        ax.plot([R, R], [0, H], linewidth=1, alpha=0.85, color='cyan')
        ax.text(xc, 14, str(i), ha='center', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def save_mask_overlay(img_rgb, mask, save_path, title='mask overlay'):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img_rgb)
    ax.imshow(mask, alpha=0.35, cmap='Reds')
    ax.axis('off')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def save_pretty_numbered(img_rgb, comp_df, save_path, title='pretty numbered'):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img_rgb)
    ax.axis('off')
    for _, row in comp_df.iterrows():
        rect = patches.Rectangle((row['x_left'], row['y_top']), row['x_right'] - row['x_left'], row['y_bottom'] - row['y_top'], fill=False, linewidth=0.9, edgecolor='cyan', alpha=0.85)
        ax.add_patch(rect)
        ax.text(row['x_left'], max(0, row['y_top'] - 3), str(int(row['component_id'])), color='red', fontsize=7, fontweight='bold', ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.2))
        ax.text(row['x_left'], row['y_bottom'] + 3, str(row['component_label']), color='blue', fontsize=6, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', pad=0.2))
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def process_one_image(image_path: str, output_root: str):
    img_name = Path(image_path).stem
    image_id = img_name
    out_dir = os.path.join(output_root, img_name)
    ensure_dir(out_dir)
    img_rgb = read_image_rgb(image_path)
    gray, B_chan, I_corr = preprocess_image(img_rgb)
    lane_info = estimate_lane_intervals(I_corr, lane_n=15, alpha=0.2)
    lanes = lane_info['lanes']
    accepted_bands, full_mask = detect_bands_and_mask(I_corr, lanes)
    numbered_df = compute_numbered_values(gray, B_chan, full_mask, lanes, image_id)
    features_df = compute_improved_features(image_id, img_rgb, full_mask, numbered_df)
    band_rows = []
    for lane_idx, segs in enumerate(accepted_bands, start=1):
        L, R = lanes[lane_idx - 1]
        for s, e in segs:
            band_rows.append({'image_id': image_id, 'lane_id': lane_idx, 'x_left': int(L), 'x_right': int(R), 'y_top': int(s), 'y_bottom': int(e)})
    bands_df = pd.DataFrame(band_rows)
    lane_overlay_path = os.path.join(out_dir, f'{img_name}_lane_overlay.jpg')
    mask_path = os.path.join(out_dir, f'{img_name}_mask.png')
    mask_overlay_path = os.path.join(out_dir, f'{img_name}_mask_overlay.jpg')
    pretty_path = os.path.join(out_dir, f'{img_name}_pretty_numbered.png')
    bands_csv_path = os.path.join(out_dir, f'{img_name}_bands.csv')
    values_csv_path = os.path.join(out_dir, f'{img_name}_numbered_values.csv')
    features_csv_path = os.path.join(out_dir, f'{img_name}_features.csv')
    save_lane_overlay(img_rgb, lanes, lane_overlay_path)
    cv2.imwrite(mask_path, full_mask.astype(np.uint8) * 255)
    save_mask_overlay(img_rgb, full_mask, mask_overlay_path)
    save_pretty_numbered(img_rgb, numbered_df, pretty_path)
    bands_df.to_csv(bands_csv_path, index=False, encoding='utf-8-sig')
    numbered_df.to_csv(values_csv_path, index=False, encoding='utf-8-sig')
    features_df.to_csv(features_csv_path, index=False, encoding='utf-8-sig')
    summary = {'image_id': image_id, 'n_components': int(len(numbered_df)), 'mask_pixels': int(full_mask.sum()), 'lane_overlay': lane_overlay_path, 'mask_path': mask_path, 'mask_overlay': mask_overlay_path, 'pretty_numbered': pretty_path, 'bands_csv': bands_csv_path, 'numbered_values_csv': values_csv_path, 'features_csv': features_csv_path}
    return (numbered_df, features_df, summary)

def main(input_dir: str, output_dir: str):
    ensure_dir(output_dir)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']:
        image_files.extend(Path(input_dir).glob(ext))
    image_files = sorted([str(p) for p in image_files], key=natural_key)
    if len(image_files) == 0:
        raise FileNotFoundError(f'No images were found in the input directory: {input_dir}')
    all_numbered, all_features, summaries = ([], [], [])
    for image_path in image_files:
        try:
            print(f'Processing: {Path(image_path).stem}')
            numbered_df, features_df, summary = process_one_image(image_path, output_dir)
            all_numbered.append(numbered_df)
            all_features.append(features_df)
            summaries.append(summary)
        except Exception as e:
            print(f'[FAILED] {image_path} -> {e}')
    numbered_master = pd.concat(all_numbered, ignore_index=True) if all_numbered else pd.DataFrame()
    features_master = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    numbered_master.to_csv(os.path.join(output_dir, 'component_measurements_master.csv'), index=False, encoding='utf-8-sig')
    features_master.to_csv(os.path.join(output_dir, 'master_features.csv'), index=False, encoding='utf-8-sig')
    summary_df.to_csv(os.path.join(output_dir, 'image_summary.csv'), index=False, encoding='utf-8-sig')
    print('\nProcessing complete.')
    print('component measurements:', os.path.join(output_dir, 'component_measurements_master.csv'))
    print('feature master:', os.path.join(output_dir, 'master_features.csv'))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing gel images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
