from __future__ import annotations
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import importlib.util
import joblib

def import_module_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_images(folder: str) -> List[str]:
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    out = []
    for name in os.listdir(folder):
        if Path(name).suffix.lower() in exts:
            out.append(os.path.join(folder, name))
    out.sort()
    return out

def draw_box(img, box, color=(0, 255, 0), label=None, thickness=2):
    x1, y1, x2, y2 = map(int, [box['x1'], box['y1'], box['x2'], box['y2']])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label is not None:
        cv2.putText(img, str(label), (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def safe_median(vals: List[float], default: float) -> float:
    return float(np.median(vals)) if vals else float(default)

def extract_box_candidates(result) -> List[Dict]:
    out = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = box
        out.append({'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2), 'w': float(x2 - x1), 'h': float(y2 - y1), 'cx': float((x1 + x2) / 2), 'cy': float((y1 + y2) / 2), 'conf': float(conf), 'synthetic': False})
    return out

def coarse_filter_lanes(cands: List[Dict], img_h: int, img_w: int) -> List[Dict]:
    if not cands:
        return []
    med_w = safe_median([c['w'] for c in cands], max(10, img_w * 0.03))
    med_h = safe_median([c['h'] for c in cands], max(50, img_h * 0.6))
    out = []
    for c in cands:
        if c['h'] < max(0.35 * img_h, 0.45 * med_h):
            continue
        if c['w'] < max(0.008 * img_w, 0.35 * med_w):
            continue
        if c['w'] > 0.3 * img_w:
            continue
        out.append(c)
    return out

def dedup_lanes_by_center(cands: List[Dict]) -> List[Dict]:
    if len(cands) <= 1:
        return cands
    cands = sorted(cands, key=lambda z: z['cx'])
    med_w = safe_median([c['w'] for c in cands], 20.0)
    thr = max(6.0, 0.6 * med_w)
    groups, current = ([], [cands[0]])
    for c in cands[1:]:
        if abs(c['cx'] - current[-1]['cx']) <= thr:
            current.append(c)
        else:
            groups.append(current)
            current = [c]
    groups.append(current)
    keep = []
    med_h = safe_median([c['h'] for c in cands], 100.0)
    for g in groups:

        def score(x):
            wp = abs(x['w'] - med_w) / max(med_w, 1.0)
            hp = abs(x['h'] - med_h) / max(med_h, 1.0)
            return x['conf'] - 0.15 * wp - 0.1 * hp
        keep.append(max(g, key=score))
    return sorted(keep, key=lambda z: z['cx'])

def reduce_lanes_to_target(cands: List[Dict], target: int) -> List[Dict]:
    if len(cands) <= target:
        return sorted(cands, key=lambda z: z['cx'])
    cands = sorted(cands, key=lambda z: z['cx'])
    left, right = (cands[0]['cx'], cands[-1]['cx'])
    targets = np.linspace(left, right, target)
    chosen, used = ([], set())
    for t in targets:
        best_i, best_score = (None, 1e+18)
        for i, c in enumerate(cands):
            if i in used:
                continue
            score = abs(c['cx'] - t) - 10.0 * c['conf']
            if score < best_score:
                best_i, best_score = (i, score)
        used.add(best_i)
        chosen.append(cands[best_i])
    return sorted(chosen, key=lambda z: z['cx'])

def expand_lanes_to_target(cands: List[Dict], target: int, img_h: int) -> List[Dict]:
    cands = sorted(cands, key=lambda z: z['cx'])
    if not cands:
        return []
    if len(cands) >= target:
        return cands
    med_w = safe_median([c['w'] for c in cands], 20.0)
    med_y1 = safe_median([c['y1'] for c in cands], 0.2 * img_h)
    med_y2 = safe_median([c['y2'] for c in cands], 0.9 * img_h)
    left, right = (cands[0]['cx'], cands[-1]['cx'])
    targets = np.linspace(left, right, target)
    assigned = []
    for t in targets:
        best = min(cands, key=lambda z: abs(z['cx'] - t))
        if abs(best['cx'] - t) <= 0.6 * med_w:
            assigned.append(best.copy())
        else:
            assigned.append({'x1': float(t - med_w / 2), 'y1': float(med_y1), 'x2': float(t + med_w / 2), 'y2': float(med_y2), 'w': float(med_w), 'h': float(med_y2 - med_y1), 'cx': float(t), 'cy': float((med_y1 + med_y2) / 2), 'conf': 0.01, 'synthetic': True})
    out = []
    for c in sorted(assigned, key=lambda z: z['cx']):
        if out and abs(c['cx'] - out[-1]['cx']) < 1e-06:
            continue
        out.append(c)
    while len(out) < target:
        idx = len(out)
        t = np.linspace(left, right, target)[idx]
        out.append({'x1': float(t - med_w / 2), 'y1': float(med_y1), 'x2': float(t + med_w / 2), 'y2': float(med_y2), 'w': float(med_w), 'h': float(med_y2 - med_y1), 'cx': float(t), 'cy': float((med_y1 + med_y2) / 2), 'conf': 0.01, 'synthetic': True})
    return sorted(out[:target], key=lambda z: z['cx'])

def fixed_15_lanes(result, img_h: int, img_w: int, target: int=15) -> List[Dict]:
    cands = extract_box_candidates(result)
    cands = coarse_filter_lanes(cands, img_h, img_w)
    cands = dedup_lanes_by_center(cands)
    if len(cands) > target:
        cands = reduce_lanes_to_target(cands, target)
    if len(cands) < target:
        cands = expand_lanes_to_target(cands, target, img_h)
    cands = sorted(cands, key=lambda z: z['cx'])
    for i, c in enumerate(cands, 1):
        c['lane_id'] = i
    return cands

def lanes_to_intervals(lanes: List[Dict], img_w: int) -> List[Tuple[int, int]]:
    lanes = sorted(lanes, key=lambda z: z['cx'])
    centers = [c['cx'] for c in lanes]
    mids = [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)]
    intervals = []
    for i, c in enumerate(lanes):
        if i == 0:
            L = max(0, int(round(c['x1'])))
        else:
            L = max(0, int(round(mids[i - 1])))
        if i == len(lanes) - 1:
            R = min(img_w, int(round(c['x2'])))
        else:
            R = min(img_w, int(round(mids[i])))
        if R <= L:
            R = min(img_w, L + 1)
        intervals.append((L, R))
    return intervals

def coarse_filter_bands(bands: List[Dict], img_h: int, img_w: int) -> List[Dict]:
    out = []
    for b in bands:
        if b['h'] < 4 or b['w'] < 4:
            continue
        if b['h'] > 0.35 * img_h:
            continue
        if b['w'] > 0.2 * img_w:
            continue
        out.append(b)
    return out

def assign_band_to_lane(band: Dict, lanes: List[Dict], lane_intervals: List[Tuple[int, int]]) -> Dict:
    band_cx = band['cx']
    inside = []
    for lane, (L, R) in zip(lanes, lane_intervals):
        if L <= band_cx <= R:
            inside.append((lane, L, R))
    if inside:
        lane = min(inside, key=lambda t: abs(t[0]['cx'] - band_cx))[0]
        method = 'inside'
    else:
        lane = min(lanes, key=lambda z: abs(z['cx'] - band_cx))
        method = 'nearest'
    out = band.copy()
    out['lane_id'] = int(lane['lane_id'])
    out['assign_method'] = method
    return out

def dedup_bands_within_lane(bands: List[Dict]) -> List[Dict]:
    groups = defaultdict(list)
    for b in bands:
        groups[int(b['lane_id'])].append(b)
    out = []
    for lane_id, items in groups.items():
        items = sorted(items, key=lambda z: z['cy'])
        merged = []
        for b in items:
            if not merged:
                merged.append(b)
                continue
            prev = merged[-1]
            ythr = 0.35 * max(prev['h'], b['h']) + 4
            xthr = 0.5 * max(prev['w'], b['w']) + 4
            if abs(b['cy'] - prev['cy']) <= ythr and abs(b['cx'] - prev['cx']) <= xthr:
                merged[-1] = prev if prev['conf'] >= b['conf'] else b
            else:
                merged.append(b)
        merged = sorted(merged, key=lambda z: z['cy'])
        for i, b in enumerate(merged, 1):
            b['band_id'] = i
            out.append(b)
    return out

def refine_band_box_to_mask(I_corr: np.ndarray, band: Dict, lane_interval: Tuple[int, int], otsu_threshold_func) -> np.ndarray:
    H, W = I_corr.shape
    x1 = max(0, int(np.floor(band['x1'])))
    x2 = min(W, int(np.ceil(band['x2'])))
    y1 = max(0, int(np.floor(band['y1'])))
    y2 = min(H, int(np.ceil(band['y2'])))
    L, R = lane_interval
    x1 = max(x1, L)
    x2 = min(x2, R)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((H, W), dtype=bool)
    win = I_corr[y1:y2, x1:x2]
    flat = win.ravel()
    if flat.size == 0:
        return np.zeros((H, W), dtype=bool)
    thr = max(otsu_threshold_func(flat), float(np.percentile(flat, 80)) * 0.6)
    m = win > thr
    k2 = np.ones((2, 2), dtype=bool)
    k3 = np.ones((3, 3), dtype=bool)
    import scipy.ndimage as ndi
    m = ndi.binary_opening(m, structure=k2)
    m = ndi.binary_closing(m, structure=k3)
    lbl, n = ndi.label(m)
    if n == 0:
        out = np.zeros((H, W), dtype=bool)
        out[y1:y2, x1:x2] = True
        return out
    sizes = ndi.sum(m, lbl, index=range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    m_big = lbl == k
    out = np.zeros((H, W), dtype=bool)
    out[y1:y2, x1:x2] = m_big
    return out

def save_overlay(img_rgb, lanes: List[Dict], lane_intervals: List[Tuple[int, int]], bands: List[Dict], save_path: str):
    vis = img_rgb.copy()
    for lane, (L, R) in zip(lanes, lane_intervals):
        box = {'x1': L, 'y1': 0, 'x2': R, 'y2': img_rgb.shape[0] - 1}
        draw_box(vis, box, color=(0, 255, 0), label=f"L{lane['lane_id']}", thickness=2)
    for b in bands:
        draw_box(vis, b, color=(0, 0, 255), label=f"{b['lane_id']}-{b.get('band_id', '?')}", thickness=2)
    cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lane_model', required=True)
    ap.add_argument('--band_model', required=True)
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--feature_extractor_script', required=True)
    ap.add_argument('--predict_script', default='')
    ap.add_argument('--low_model', default='')
    ap.add_argument('--high_model', default='')
    ap.add_argument('--imgsz', type=int, default=1024)
    ap.add_argument('--lane_conf', type=float, default=0.05)
    ap.add_argument('--band_conf', type=float, default=0.1)
    args = ap.parse_args()
    ensure_dir(args.output_dir)
    ov_dir = os.path.join(args.output_dir, 'overlay')
    ensure_dir(ov_dir)
    feature_extractor = import_module_from_path('gel_feature_extractor', args.feature_extractor_script)
    predictor = None
    low_model = high_model = None
    if args.predict_script and args.low_model and args.high_model:
        predictor = import_module_from_path('quantitative_prediction', args.predict_script)
        low_model = joblib.load(args.low_model)
        high_model = joblib.load(args.high_model)
    lane_model = YOLO(args.lane_model)
    band_model = YOLO(args.band_model)
    lane_rows, band_rows, feat_frames, pred_frames, review_rows = ([], [], [], [], [])
    for image_path in list_images(args.input_dir):
        img_name = Path(image_path).name
        image_id = Path(image_path).stem
        print('processing', img_name)
        img_rgb = feature_extractor.read_image_rgb(image_path)
        H, W, _ = img_rgb.shape
        gray, B_chan, I_corr = feature_extractor.preprocess_image(img_rgb)
        lane_res = lane_model.predict(source=image_path, imgsz=args.imgsz, conf=args.lane_conf, iou=0.5, max_det=50, verbose=False, save=False)[0]
        lanes = fixed_15_lanes(lane_res, H, W, target=15)
        lane_intervals = lanes_to_intervals(lanes, W)
        for lane, (L, R) in zip(lanes, lane_intervals):
            lane_rows.append({'image': img_name, 'image_id': image_id, 'lane_id': lane['lane_id'], 'x1': L, 'y1': 0, 'x2': R, 'y2': H - 1, 'cx': lane['cx'], 'w': R - L, 'h': H, 'conf': lane['conf'], 'synthetic': int(lane.get('synthetic', False))})
        band_res = band_model.predict(source=image_path, imgsz=args.imgsz, conf=args.band_conf, iou=0.5, max_det=100, verbose=False, save=False)[0]
        bands = extract_box_candidates(band_res)
        bands = coarse_filter_bands(bands, H, W)
        assigned = [assign_band_to_lane(b, lanes, lane_intervals) for b in bands]
        assigned = dedup_bands_within_lane(assigned)
        for b in assigned:
            band_rows.append({'image': img_name, 'image_id': image_id, 'lane_id': b['lane_id'], 'band_id': b['band_id'], 'x1': b['x1'], 'y1': b['y1'], 'x2': b['x2'], 'y2': b['y2'], 'cx': b['cx'], 'cy': b['cy'], 'w': b['w'], 'h': b['h'], 'conf': b['conf'], 'assign_method': b['assign_method']})
        full_mask = np.zeros((H, W), dtype=bool)
        for b in assigned:
            lane_interval = lane_intervals[b['lane_id'] - 1]
            m = refine_band_box_to_mask(I_corr, b, lane_interval, feature_extractor.otsu_threshold)
            full_mask |= m
        component_df = feature_extractor.compute_numbered_values(gray, B_chan, full_mask, lane_intervals, image_id)
        features_df = feature_extractor.compute_improved_features(image_id, img_rgb, full_mask, component_df)
        if len(features_df):
            feat_frames.append(features_df)
            if predictor is not None:
                pred_df = predictor.predict_df(features_df.copy(), low_model, high_model)
                pred_frames.append(pred_df)
        else:
            review_rows.append({'image': img_name, 'reason': 'no_features_after_mask'})
        save_overlay(img_rgb, lanes, lane_intervals, assigned, os.path.join(ov_dir, img_name))
    pd.DataFrame(lane_rows).to_csv(os.path.join(args.output_dir, 'lane_summary.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(band_rows).to_csv(os.path.join(args.output_dir, 'band_summary.csv'), index=False, encoding='utf-8-sig')
    feat_master = pd.concat(feat_frames, ignore_index=True) if feat_frames else pd.DataFrame()
    feat_master.to_csv(os.path.join(args.output_dir, 'feature_table.csv'), index=False, encoding='utf-8-sig')
    pred_master = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    if len(pred_master):
        pred_master.to_csv(os.path.join(args.output_dir, 'prediction_table.csv'), index=False, encoding='utf-8-sig')
    review_df = pd.DataFrame(review_rows, columns=['image', 'reason'])
    review_df.to_csv(os.path.join(args.output_dir, 'review_required.csv'), index=False, encoding='utf-8-sig')
    print('done:', args.output_dir)
if __name__ == '__main__':
    main()
