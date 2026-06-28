# Peak-Lord

## Overview
Peak-LoRD is a novel approach written in Python (v3.11.8) for peaks (aka features) detection and quantification in raw LC-MS data.  It adopts a YOLO object detection model to automatically recognize chromatographic peaks and estimate their boundaries for accurate quantification. Combined with a dual-student semi-supervised learning framework, Peak-LoRD effectively leverages multi-source training data to enhance the detection of diverse peaks. The current method is developed for high-resolution LC-MS data for metabolomics purposes, but it can also be applied to other detections that take peaks as the targets.

Supported formats:
- .mzML

## Supported Operating System
Peak-LoRD has been tested successfully with:
- CentOS 7

## Installation
To install and run Peak-LoRD you should do a few simple steps:
1. Clone Peak-LoRD.
```bash
git clone https://github.com/WeizhenLiuBioinform/Peak_LoRD.git
```

2. Create the conda environment from the provided environment file:

```bash
cd /data1/zhaohaowei/peak_detection/peaklord
conda env create -f environment.yml
```
If you want to change the name of environment, please edit the "name" row in the environment.yml file by yourself.

3. Activate environment:

```bash
conda activate peaklord
```

## Usage
Run the main inference script:

```bash
python ultralytics/sl_inference.py --datadir data_path --model weights_path
```
We provide the peak_lord.pt file as the weight file;


### Main parameters
- `--ppm` (int, default: `15`)
  - ppm tolerance for ROI dynamic m/z matching.
- `--roi_delta_mz` (float, default: `0.01`)
  - absolute m/z tolerance lower bound (Da).
- `--roi_required_points` (int, default: `10`)
  - minimum number of points required per ROI.
- `--roi_dropped_points` (int, default: `5`)
  - maximum allowed consecutive missing points when tracking ROIs.
- `--min_nonzero_points` (int, default: `5`)
  - minimum number of non-zero points in an EIC to keep it.
- `--min_height` (float, default: `1000`)
  - minimum peak intensity to keep an EIC.
- `--window_counts` (list of int, default: `[1,2,3,4,5,6,7,8,9,10,11,12,13,14]`)
  - numbers of windows used to split each EIC for feature extraction.
- `--window_count_thresholds` (list of float, default: `[1,2,3,4,5,6,7,8,9,10,11,12,13]`)
  - thresholds used to define the number of windows.

### IO and model parameters
- `--model` (default: `weights/best.pt` or the `PEAK_MODEL_PATH` environment variable)
  - path to the peak detection model.
- `--datadir` (default: `./data/mzml` or the `PEAK_MZML_DIR` environment variable)
  - directory containing input `.mzML` files.
- `--img_tmp_path` (default: `./output/tmp.jpg` or `PEAK_TMP_IMAGE`)
  - temporary image path used during processing.
- `--save_csv` (flag, default: `True`)
  - save detected peak results as CSV files.
- `--csv_save_dir` (default: `./output/csv` or `PEAK_CSV_DIR`)
  - directory for output CSV files.
- `--noise_thresold` (float, default: `1000`)
  - noise threshold used during peak processing.

### Image output parameters
- `--save_images` (flag, default: `False`)
  - save local EIC plots around predicted peaks.
- `--images_dir` (default: `./output/peak_images` or `PEAK_IMAGE_DIR`)
  - directory for saved peak images.
- `--images_rt_margin` (float, default: `0.1`)
  - retention-time margin (minutes) added around the detected peak window.
- `--images_window_size` (float, default: `1`)
  - EIC plotting window size in minutes.

### Smoothing and sampling
- `--smooth_mode` (bool, default: `False`)
  - whether to smooth RT–intensity pairs when plotting.
- `--down_sample` (bool, default: `False`)
  - whether to downsample RT–intensity pairs in windows.
- `--use_min_windowsize` (bool, default: `False`)
  - use the smaller of the predefined window size and the actual RT span of the EIC.
- `--min_window_size` (float, default: `0.8`)
  - minimum window size when `use_min_windowsize` is enabled.

### Debug
- `--isDebug` (bool, default: `False`)
  - enable debug mode with verbose output.
- `--debugPlotImgPath` (default: `./output/debug_plot` or `PEAK_DEBUG_PLOT_DIR`)
  - directory for debug plot images.

## Example
```bash
python ultralytics/sl_inference.py \
  --datadir ./data/mzml \
  --model weights/best.pt \
  --ppm 15 \
  --roi_delta_mz 0.01 \
  --min_height 1000 \
  --save_images
```

