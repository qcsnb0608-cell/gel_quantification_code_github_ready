@echo off
chcp 65001
python run_batch_feature_extraction.py ^
  --package_root "." ^
  --input_zips "data\input_zips" ^
  --input_images "data\input_images" ^
  --output_dir "results"
pause
