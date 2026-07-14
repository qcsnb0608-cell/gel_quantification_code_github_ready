from pathlib import Path
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np
import pandas as pd

def natural_sort_key(path):
    stem = Path(path).stem
    parts = re.split('(\\d+(?:\\.\\d+)?)', stem)
    return [float(p) if re.fullmatch('\\d+(?:\\.\\d+)?', p) else p.lower() for p in parts]

def safe_name(text):
    return re.sub('[^0-9A-Za-z_.\\-]+', '_', str(text))

def ensure_ultralytics():
    try:
        import ultralytics
        return ultralytics
    except Exception as exc:
        raise ImportError('ultralytics is required. Install dependencies with: pip install -r requirements.txt') from exc

def extract_zips(input_zips, extracted_dir):
    input_zips = Path(input_zips)
    extracted_dir = Path(extracted_dir)
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    zip_files = sorted(input_zips.glob('*.zip'), key=natural_sort_key)
    records = []
    for i, zp in enumerate(zip_files, start=1):
        out_dir = extracted_dir / f'{i:03d}_{safe_name(zp.stem)}'
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'[UNZIP] {zp.name}')
        try:
            with zipfile.ZipFile(zp, 'r') as z:
                z.extractall(out_dir)
            status, error = ('success', '')
        except Exception as e:
            status, error = ('failed', repr(e))
        records.append({'zip_name': zp.name, 'extract_dir': str(out_dir), 'status': status, 'error': error})
    return pd.DataFrame(records)

def collect_images_from_dirs(dirs):
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
    images = []
    rows = []
    for root in dirs:
        root = Path(root)
        if not root.exists():
            continue
        for p in root.rglob('*'):
            if p.is_file() and p.suffix.lower() in exts:
                images.append(p)
                try:
                    rel = str(p.relative_to(root))
                except Exception:
                    rel = str(p)
                rows.append({'source_root': str(root), 'relative_path': rel, 'original_image_name': p.name, 'original_image_path': str(p)})
    order = sorted(range(len(images)), key=lambda i: natural_sort_key(images[i]))
    images = [images[i] for i in order]
    rows = [rows[i] for i in order]
    return (images, pd.DataFrame(rows))

def make_unified_inputs(images, image_index, unified_dir):
    unified_dir = Path(unified_dir)
    if unified_dir.exists():
        shutil.rmtree(unified_dir)
    unified_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in enumerate(images, start=1):
        meta = image_index.iloc[i - 1].to_dict()
        new_name = f'{i:04d}__{safe_name(Path(p).stem)}{Path(p).suffix.lower()}'
        dst = unified_dir / new_name
        shutil.copy2(p, dst)
        rows.append({**meta, 'unified_image_name': new_name, 'unified_image_path': str(dst)})
    return pd.DataFrame(rows)

def locate_files(package_root):
    package_root = Path(package_root)
    files = {'pipeline_script': package_root / 'src' / 'yolo_gel_pipeline.py', 'feature_extractor_script': package_root / 'src' / 'gel_feature_extractor.py', 'lane_model': package_root / 'models' / 'lane_segmentation' / 'best.pt', 'band_model': package_root / 'models' / 'band_detection' / 'best.pt'}
    for key, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f'Missing required file for {key}: {p}')
    return files

def run_one(files, image_path, tmp_input, raw_output, imgsz=1024, lane_conf=0.05, band_conf=0.1, timeout=900):
    tmp_input = Path(tmp_input)
    raw_output = Path(raw_output)
    if tmp_input.exists():
        shutil.rmtree(tmp_input)
    if raw_output.exists():
        shutil.rmtree(raw_output)
    tmp_input.mkdir(parents=True, exist_ok=True)
    raw_output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, tmp_input / Path(image_path).name)
    cmd = [sys.executable, str(files['pipeline_script']), '--lane_model', str(files['lane_model']), '--band_model', str(files['band_model']), '--input_dir', str(tmp_input), '--output_dir', str(raw_output), '--feature_extractor_script', str(files['feature_extractor_script']), '--imgsz', str(imgsz), '--lane_conf', str(lane_conf), '--band_conf', str(band_conf)]
    print('\n[RUN]', Path(image_path).name)
    t0 = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', timeout=timeout)
    print(res.stdout)
    if res.returncode != 0:
        raise RuntimeError(f'Pipeline failed returncode={res.returncode}')
    return time.time() - t0

def read_feature_table(path, meta):
    df = pd.read_csv(path)
    df['original_image_name'] = meta.get('original_image_name', '')
    df['relative_path'] = meta.get('relative_path', '')
    df['unified_image_name'] = meta.get('unified_image_name', '')
    df['IOD_gray_hyst'] = pd.to_numeric(df['IOD_gray_hyst'], errors='coerce')
    df['area_gray_hyst_px'] = pd.to_numeric(df['area_gray_hyst_px'], errors='coerce')
    df['log_IOD_gray_hyst'] = np.log1p(np.clip(df['IOD_gray_hyst'], 0, None))
    df['log_area_gray_hyst_px'] = np.log1p(np.clip(df['area_gray_hyst_px'], 0, None))
    return df

def make_three(df):
    id_cols = [c for c in ['original_image_name', 'relative_path', 'unified_image_name', 'image_id', 'image_name', 'lane_guess', 'band_guess', 'component_id', 'component_label', 'lane_id', 'band_id'] if c in df.columns]
    return df[id_cols + ['IOD_gray_hyst', 'log_IOD_gray_hyst', 'log_area_gray_hyst_px']]

def copy_overlay(raw_output, overlay_dir, prefix):
    overlay_dir = Path(overlay_dir)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    src = Path(raw_output) / 'overlay'
    copied = 0
    if src.exists():
        for p in src.iterdir():
            if p.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                shutil.copy2(p, overlay_dir / f'{prefix}__{p.name}')
                copied += 1
    return copied

def run_batch(package_root='.', input_zips='data/input_zips', input_images='data/input_images', output_dir='results', imgsz=1024, lane_conf=0.05, band_conf=0.1, timeout=900, continue_on_error=True):
    package_root = Path(package_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_ultralytics()
    files = locate_files(package_root)
    extracted_dir = output_dir / '_extracted_zips'
    zip_status = extract_zips(input_zips, extracted_dir)
    zip_status.to_csv(output_dir / 'zip_extract_status.csv', index=False, encoding='utf-8-sig')
    images, index_df = collect_images_from_dirs([extracted_dir, input_images])
    if len(images) == 0:
        raise FileNotFoundError('No input images were found. Add zip files to data/input_zips or image files to data/input_images.')
    unified_dir = output_dir / '_unified_images'
    unified_index = make_unified_inputs(images, index_df, unified_dir)
    unified_index.to_csv(output_dir / 'image_index.csv', index=False, encoding='utf-8-sig')
    final = output_dir / 'final'
    raw_root = output_dir / 'raw_per_image'
    tmp_root = output_dir / '_tmp_inputs'
    overlay_dir = final / 'overlay'
    final.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    unified_images = sorted([p for p in unified_dir.iterdir() if p.is_file()], key=natural_sort_key)
    meta_map = unified_index.set_index('unified_image_name').to_dict(orient='index')
    full_list, three_list, status_rows = ([], [], [])
    for i, img in enumerate(unified_images, start=1):
        print(f'\n========== [{i}/{len(unified_images)}] {img.name} ==========')
        meta = meta_map.get(img.name, {})
        row = {'unified_image_name': img.name, 'original_image_name': meta.get('original_image_name', ''), 'relative_path': meta.get('relative_path', ''), 'status': 'pending', 'n_feature_rows': 0, 'error': ''}
        safe_stem = safe_name(img.stem)
        try:
            elapsed = run_one(files, img, tmp_input=tmp_root / safe_stem, raw_output=raw_root / safe_stem, imgsz=imgsz, lane_conf=lane_conf, band_conf=band_conf, timeout=timeout)
            df = read_feature_table(raw_root / safe_stem / 'feature_table.csv', {**meta, 'unified_image_name': img.name})
            three = make_three(df)
            df.to_csv(final / f'{safe_stem}_full_feature_table_log1p.csv', index=False, encoding='utf-8-sig')
            three.to_csv(final / f'{safe_stem}_three_features_log1p.csv', index=False, encoding='utf-8-sig')
            n_overlay = copy_overlay(raw_root / safe_stem, overlay_dir, safe_stem)
            full_list.append(df)
            three_list.append(three)
            row.update({'status': 'success', 'n_feature_rows': len(df), 'n_overlay': n_overlay, 'elapsed_sec': elapsed})
        except Exception as e:
            row.update({'status': 'failed', 'error': repr(e)})
            print('[ERROR]', repr(e))
            if not continue_on_error:
                status_rows.append(row)
                pd.DataFrame(status_rows).to_csv(final / 'run_status.csv', index=False, encoding='utf-8-sig')
                raise
        status_rows.append(row)
        pd.DataFrame(status_rows).to_csv(final / 'run_status.csv', index=False, encoding='utf-8-sig')
    full_master = pd.concat(full_list, ignore_index=True) if full_list else pd.DataFrame()
    three_master = pd.concat(three_list, ignore_index=True) if three_list else pd.DataFrame()
    full_master.to_csv(final / 'gel_full_feature_table_log1p.csv', index=False, encoding='utf-8-sig')
    three_master.to_csv(final / 'gel_three_features_log1p.csv', index=False, encoding='utf-8-sig')
    if len(full_master):
        summary = full_master.groupby(['original_image_name', 'relative_path', 'unified_image_name'], dropna=False).size().reset_index(name='n_feature_rows')
    else:
        summary = pd.DataFrame()
    summary.to_csv(final / 'summary_by_image.csv', index=False, encoding='utf-8-sig')
    report = {'n_images': len(unified_images), 'n_success': int(sum((r['status'] == 'success' for r in status_rows))), 'n_failed': int(sum((r['status'] == 'failed' for r in status_rows))), 'n_feature_rows': int(len(full_master)), 'lane_model': str(files['lane_model']), 'band_model': str(files['band_model']), 'log_definition': 'log_IOD_gray_hyst = ln(1 + IOD_gray_hyst); log_area_gray_hyst_px = ln(1 + area_gray_hyst_px)', 'outputs': {'three_features': str(final / 'gel_three_features_log1p.csv'), 'full_features': str(final / 'gel_full_feature_table_log1p.csv'), 'summary': str(final / 'summary_by_image.csv'), 'status': str(final / 'run_status.csv'), 'overlay': str(overlay_dir)}}
    (final / 'RUN_REPORT.json').write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return final

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package_root', default='.')
    ap.add_argument('--input_zips', default='data/input_zips')
    ap.add_argument('--input_images', default='data/input_images')
    ap.add_argument('--output_dir', default='results')
    ap.add_argument('--imgsz', type=int, default=1024)
    ap.add_argument('--lane_conf', type=float, default=0.05)
    ap.add_argument('--band_conf', type=float, default=0.1)
    ap.add_argument('--timeout', type=int, default=900)
    ap.add_argument('--stop_on_error', action='store_true')
    args = ap.parse_args()
    run_batch(package_root=args.package_root, input_zips=args.input_zips, input_images=args.input_images, output_dir=args.output_dir, imgsz=args.imgsz, lane_conf=args.lane_conf, band_conf=args.band_conf, timeout=args.timeout, continue_on_error=not args.stop_on_error)
if __name__ == '__main__':
    main()
