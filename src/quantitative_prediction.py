import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
LOW_FEATS = ['IOD_gray_hyst', 'IOD_blue_hyst', 'IOD_neutral', 'IOD_hybrid', 'area_gray_hyst_px', 'IOD_gray_per_area_fix', 'IOD_blue_per_area_fix', 'log_effective_signal', 'log_area']
HIGH_FEATS = ['IOD_yellow', 'IOD_hybrid', 'yellow_ratio', 'log_yellow_ratio', 'chroma_shift', 'IOD_yellow_per_area', 'area_gray_hyst_px', 'effective_signal', 'log_IOD_yellow']

def prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    area = np.maximum(out['area_gray_hyst_px'].astype(float).values, 1.0)
    Igray = out['IOD_gray_hyst'].astype(float).values
    Iblue = out['IOD_blue_hyst'].astype(float).values
    Iyel = out['IOD_yellow'].astype(float).values
    out['IOD_gray_per_area_fix'] = Igray / area
    out['IOD_blue_per_area_fix'] = Iblue / area
    out['IOD_yellow_per_area'] = Iyel / area
    out['yellow_frac'] = Iyel / (Iyel + Iblue + Igray + 1.0)
    out['log_yellow_ratio'] = np.log1p(np.clip(out['yellow_ratio'].astype(float).values, 0, None))
    out['chroma_shift'] = np.log1p(Iyel + 1.0) - np.log1p(Iblue + 1.0)
    out['effective_signal'] = (1 - out['yellow_frac']) * Iblue + 2.5 * out['yellow_frac'] * Iyel + 0.03 * area
    out['log_effective_signal'] = np.log1p(out['effective_signal'])
    out['log_IOD_yellow'] = np.log1p(Iyel)
    out['log_area'] = np.log1p(area)
    return out

def predict_df(df: pd.DataFrame, low_model, high_model) -> pd.DataFrame:
    df = prep(df)
    pred_low = low_model.predict(df[LOW_FEATS])
    pred_high = high_model.predict(df[HIGH_FEATS])
    w = 1 / (1 + np.exp(-(df['IOD_yellow'].values - 250) / 40))
    pred_log = (1 - w) * pred_low + w * pred_high
    out = df.copy()
    out['mix_weight_high'] = w
    out['pred_log10'] = pred_log
    out['pred_raw'] = 10 ** pred_log
    return out

def calibrate_predictions(pred_df: pd.DataFrame, std_map: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    use = std_map[(std_map['is_standard'].astype(int) == 1) & std_map['known_concentration'].notna()].copy()
    if use.empty:
        pred_df['pred_raw_calibrated'] = pred_df['pred_raw']
        return (pred_df, {'mode': 'none'})
    merged = pred_df.merge(use[['image_id', 'component_label', 'known_concentration']], on=['image_id', 'component_label'], how='left')
    frames = []
    info = {'mode': 'prediction_level_linear', 'images': {}}
    for image_id, sub in merged.groupby('image_id', sort=False):
        std_sub = sub[sub['known_concentration'].notna()].copy()
        out = sub.copy()
        if len(std_sub) >= 2:
            x = std_sub['pred_raw'].astype(float).values
            y = std_sub['known_concentration'].astype(float).values
            a, b = np.polyfit(x, y, 1)
            out['pred_raw_calibrated'] = a * out['pred_raw'].astype(float).values + b
            info['images'][str(image_id)] = {'a': float(a), 'b': float(b), 'n_standards': int(len(std_sub))}
        elif len(std_sub) == 1:
            ratio = float(std_sub['known_concentration'].iloc[0]) / max(float(std_sub['pred_raw'].iloc[0]), 1e-09)
            out['pred_raw_calibrated'] = out['pred_raw'].astype(float).values * ratio
            info['images'][str(image_id)] = {'ratio': ratio, 'n_standards': 1}
        else:
            out['pred_raw_calibrated'] = out['pred_raw']
            info['images'][str(image_id)] = {'n_standards': 0}
        frames.append(out)
    return (pd.concat(frames, ignore_index=True), info)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features_csv', required=True)
    ap.add_argument('--low_model', required=True)
    ap.add_argument('--high_model', required=True)
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--standard_map_csv', default='')
    ap.add_argument('--calibration_json', default='')
    args = ap.parse_args()
    df = pd.read_csv(args.features_csv)
    low_model = joblib.load(args.low_model)
    high_model = joblib.load(args.high_model)
    pred_df = predict_df(df, low_model, high_model)
    calib_info = {'mode': 'none'}
    if args.standard_map_csv:
        std_map = pd.read_csv(args.standard_map_csv)
        pred_df, calib_info = calibrate_predictions(pred_df, std_map)
    else:
        pred_df['pred_raw_calibrated'] = pred_df['pred_raw']
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
    if args.calibration_json:
        Path(args.calibration_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.calibration_json, 'w', encoding='utf-8') as f:
            json.dump(calib_info, f, ensure_ascii=True, indent=2)
if __name__ == '__main__':
    main()
