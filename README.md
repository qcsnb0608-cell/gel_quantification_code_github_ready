# Gel Image Band Quantification Workflow

This repository provides the code and trained model files used for gel-image lane detection, band detection, band-region refinement, feature extraction, and optional regression-based concentration prediction. It is intended for manuscript-associated code release and reproducible analysis.

## What this package does

The workflow processes gel images and produces quantitative band-level feature tables. It performs the following steps:

1. Collects input gel images from an image folder and, when provided, from zip archives.
2. Detects lane candidates using a trained lane-segmentation model.
3. Filters and standardizes lane candidates to a fixed 15-lane layout.
4. Detects band candidates using a trained band-detection model.
5. Assigns each detected band to the corresponding lane.
6. Refines each band box into a pixel-level signal mask using local background correction and hysteresis-based signal selection.
7. Extracts grayscale, blue-channel, neutral optical-density, and yellow optical-density features.
8. Writes compact and full feature tables, run-status files, and overlay images for visual review.
9. Optionally applies trained regression models to predict concentrations from extracted features.

## Repository structure

```text
.
├── config/
│   └── pipeline_parameters.json
├── data/
│   ├── input_images/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   ├── ...
│   │   └── 10.png
│   └── input_zips/
│       └── .gitkeep
├── models/
│   ├── band_detection/
│   │   └── best.pt
│   ├── lane_segmentation/
│   │   └── best.pt
│   └── quantitative_regression/
│       ├── high_signal_model.joblib
│       └── low_signal_model.joblib
├── src/
│   ├── gel_feature_extractor.py
│   ├── quantitative_prediction.py
│   └── yolo_gel_pipeline.py
├── MANIFEST.json
├── README.md
├── requirements.txt
├── run_batch_feature_extraction.py
└── run_windows.bat
```

## File-by-file guide

| Path | Required for feature extraction | Required for prediction | Purpose |
|---|---:|---:|---|
| `README.md` | No | No | Explains the package contents, installation, input preparation, commands, outputs, and reproducibility notes. |
| `requirements.txt` | Yes | Yes | Lists the Python packages needed to run the workflow. |
| `MANIFEST.json` | No | No | Records the released file list and file sizes. This is useful for archiving and integrity checking, but the code can run without it. |
| `config/pipeline_parameters.json` | No | No | Documents the default model paths and inference parameters used in this release. The current scripts use command-line arguments and internal defaults, so this file is primarily for transparency and reproducibility. |
| `data/input_images/1.png` to `data/input_images/10.png` | No | No | Example gel images included for demonstration and command testing. They are not required for using the workflow with new data and may be replaced by user-provided images. |
| `data/input_zips/.gitkeep` | No | No | Keeps the empty zip-input folder visible in a public repository. It can be deleted after zip archives are placed in the folder. |
| `models/lane_segmentation/best.pt` | Yes | No | Trained model file used to detect gel lanes. |
| `models/band_detection/best.pt` | Yes | No | Trained model file used to detect band candidates. |
| `models/quantitative_regression/low_signal_model.joblib` | No | Yes | Optional regression model used for lower-signal band prediction. |
| `models/quantitative_regression/high_signal_model.joblib` | No | Yes | Optional regression model used for higher-signal band prediction. |
| `src/gel_feature_extractor.py` | Yes | No | Implements image reading, background correction, lane-interval estimation, band-mask refinement, signal extraction, and overlay generation. It is imported by the detection pipeline. |
| `src/yolo_gel_pipeline.py` | Yes | No | Main single-folder detection and feature-extraction pipeline. It loads the lane and band models, performs lane and band assignment, refines masks, and writes per-image results. |
| `src/quantitative_prediction.py` | No | Yes | Applies the optional regression models to an extracted feature table and can perform standard-based calibration when a standard-map CSV is supplied. |
| `run_batch_feature_extraction.py` | Yes | No | Recommended entry-point script for users. It collects images, extracts zip archives, runs the single-folder pipeline image by image, merges outputs, and writes final batch-level tables. |
| `run_windows.bat` | No | No | Optional Windows helper that runs the default batch command. It is not needed on Linux, macOS, or command-line Python environments. |

## Minimal files needed for feature extraction

For feature extraction only, the following files and folders are sufficient:

```text
requirements.txt
run_batch_feature_extraction.py
src/gel_feature_extractor.py
src/yolo_gel_pipeline.py
models/lane_segmentation/best.pt
models/band_detection/best.pt
data/input_images/ or data/input_zips/
```

The regression models are not required unless concentration prediction is needed.

## Minimal files needed for concentration prediction

For prediction from an existing feature table, the following files are sufficient:

```text
requirements.txt
src/quantitative_prediction.py
models/quantitative_regression/low_signal_model.joblib
models/quantitative_regression/high_signal_model.joblib
results/final/gel_full_feature_table_log1p.csv
```

## Installation

Use Python 3.10 or later. From the repository root, install dependencies with:

```bash
pip install -r requirements.txt
```

The workflow requires PyTorch, torchvision, and ultralytics in the same Python environment. This release pins `ultralytics==8.4.36`, the version used to load and verify the released YOLO26 model files, and `scikit-learn==1.8.0`, the version recorded in the released regression models. GPU acceleration can be used when the installed PyTorch environment supports it, but the scripts can also run on CPU.

## Input preparation

This release includes ten example gel images in:

```text
data/input_images/1.png to data/input_images/10.png
```

To analyze another dataset, replace these files or add supported image files to:

```text
data/input_images/
```

or place zip archives containing supported image files in:

```text
data/input_zips/
```

Supported image extensions are:

```text
.png, .jpg, .jpeg, .bmp, .tif, .tiff, .webp
```

Zip archives may contain nested folders. The batch runner extracts zip archives and collects supported image files recursively.

## Running feature extraction

From the repository root, run the following command. With the released package, this command processes the included example images `1.png` to `10.png` directly.

```bash
python run_batch_feature_extraction.py --package_root . --input_images data/input_images --input_zips data/input_zips --output_dir results
```

The same command can be run with default folders on Windows by double-clicking:

```text
run_windows.bat
```

### Command-line options for `run_batch_feature_extraction.py`

| Option | Default | Description |
|---|---:|---|
| `--package_root` | `.` | Repository root containing `src/` and `models/`. |
| `--input_images` | `data/input_images` | Folder containing image files. |
| `--input_zips` | `data/input_zips` | Folder containing zip archives with image files. |
| `--output_dir` | `results` | Output folder for final tables, overlays, and temporary processing files. |
| `--imgsz` | `1024` | Inference image size passed to the detection models. |
| `--lane_conf` | `0.05` | Confidence threshold for lane detection. |
| `--band_conf` | `0.1` | Confidence threshold for band detection. |
| `--timeout` | `900` | Maximum processing time in seconds for one image. |
| `--stop_on_error` | off | Stops the batch run after the first failed image. Without this flag, failed images are recorded and the batch continues. |

Example with custom thresholds:

```bash
python run_batch_feature_extraction.py --package_root . --input_images data/input_images --output_dir results --imgsz 1024 --lane_conf 0.05 --band_conf 0.10
```

## Main feature-extraction outputs

After a successful batch run, the main outputs are written to:

```text
results/final/
├── gel_three_features_log1p.csv
├── gel_full_feature_table_log1p.csv
├── summary_by_image.csv
├── run_status.csv
├── RUN_REPORT.json
└── overlay/
```

### `gel_three_features_log1p.csv`

This compact table contains the primary feature set used for downstream quantitative analysis:

```text
source_image
image_id
lane_id
band_id
IOD_gray_hyst
log_IOD_gray_hyst
log_area_gray_hyst_px
```

### `gel_full_feature_table_log1p.csv`

This table contains the full band-level feature output, including image identifiers, lane and band identifiers, band geometry, mask-derived area, local background estimates, raw and corrected signal features, optical-density features, and log-transformed features.

### `summary_by_image.csv`

This table summarizes processing results by image, including the number of detected or extracted band records for each image.

### `run_status.csv`

This table reports whether each image was processed successfully. Failed images are recorded with an error message.

### `RUN_REPORT.json`

This file records package-level run information, including input counts, output locations, and batch-level status.

### `overlay/`

This folder contains visual review images showing detected lanes, assigned bands, and extracted regions. These images are intended for quality control and manual inspection.

## Running optional concentration prediction

After feature extraction, run:

```bash
python src/quantitative_prediction.py --features_csv results/final/gel_full_feature_table_log1p.csv --low_model models/quantitative_regression/low_signal_model.joblib --high_model models/quantitative_regression/high_signal_model.joblib --output_csv results/final/prediction_table.csv
```

### Command-line options for `src/quantitative_prediction.py`

| Option | Required | Description |
|---|---:|---|
| `--features_csv` | Yes | Full feature table generated by the extraction workflow. |
| `--low_model` | Yes | Path to the lower-signal regression model. |
| `--high_model` | Yes | Path to the higher-signal regression model. |
| `--output_csv` | Yes | Output CSV path for prediction results. |
| `--standard_map_csv` | No | Optional CSV containing standard-band metadata for calibration. |
| `--calibration_json` | No | Optional output path for fitted calibration parameters. |

If standard-based calibration is required, provide a standard-map CSV with these columns:

```text
image_id
component_label
is_standard
known_concentration
```

Example:

```bash
python src/quantitative_prediction.py --features_csv results/final/gel_full_feature_table_log1p.csv --low_model models/quantitative_regression/low_signal_model.joblib --high_model models/quantitative_regression/high_signal_model.joblib --output_csv results/final/prediction_table.csv --standard_map_csv data/standard_map.csv --calibration_json results/final/calibration.json
```

## Notes on the source-code files

### `run_batch_feature_extraction.py`

This is the main script users should run. It prepares a unified input set, handles zip extraction, processes each image through `src/yolo_gel_pipeline.py`, collects feature tables, creates compact and full batch-level outputs, copies overlay images, and records processing status.

### `src/yolo_gel_pipeline.py`

This script performs model-based image analysis for one input folder. It loads the lane and band detection models, obtains lane and band boxes, filters low-quality candidates, standardizes lane order, assigns bands to lanes, calls the mask-refinement and feature-extraction functions, and saves image-level feature tables and overlays.

### `src/gel_feature_extractor.py`

This file contains the lower-level image-processing functions. It reads gel images, corrects broad background variation, estimates lane intervals, refines band masks, computes signal features, and generates visual overlays. It is used by `src/yolo_gel_pipeline.py` and normally does not need to be called directly.

### `src/quantitative_prediction.py`

This script is independent of the detection workflow. It uses a completed feature table and the released regression models to generate concentration predictions. If a standard-map file is provided, it also produces calibrated prediction values.

## Reproducibility notes

The released model files in `models/lane_segmentation/` and `models/band_detection/` are required to reproduce the detection-based feature extraction results. They were re-serialized with Ultralytics 8.4.36 to remove optimizer state and machine-specific training-path metadata; predictions on all ten included example images were verified to be identical to the source checkpoints. The default inference parameters are listed in `config/pipeline_parameters.json` and in the command-line defaults of `run_batch_feature_extraction.py`. For manuscript-associated use, report the Python version, package versions, model files, input image preprocessing, command-line parameters, and generated output tables.

## Data note

This repository contains code, model files, and ten example gel images for demonstration. Manuscript raw images and data tables should still be deposited separately when required by the journal or public archive. To reproduce manuscript results, replace the example images with the corresponding raw images in `data/input_images/` or place zip archives in `data/input_zips/`, then run the feature-extraction command above.

## Citation note

When using this package with a manuscript, cite the associated article and describe this repository as the implementation used for gel-image lane detection, band detection, band-region refinement, feature extraction, and optional concentration prediction.
