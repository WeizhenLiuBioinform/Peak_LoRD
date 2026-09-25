# Peak-Lord

## Overview
Peak-LoRD is a novel approach written in Python (v3.11) for peaks (aka features) detection and quantification in raw LC-MS data. It adopts a YOLO object detection model to automatically recognize chromatographic peaks and estimate their boundaries for accurate quantification. Combined with a dual-student semi-supervised learning framework, Peak-LoRD effectively leverages multi-source training data to enhance the detection of diverse peaks. The current method is developed for high-resolution LC-MS data for metabolomics purposes, but it can also be applied to other detections that take peaks as the targets.

Supported formats:
- .mzML

## Pipeline
1. `ultralytics/sl_inference.py` — peak detection on `.mzML` files (ROI/EIC building + YOLO inference), outputs one peak CSV per sample.
2. `ultralytics/train_detect_ssl.py` — semi-supervised detection training (dual-student, semi_detect task).
3. `ultralytics/train_ssl.py` — semi-supervised segmentation training (dual-student, semi_segment task).
4. `ultralytics/align.R` — R script that imports the peak CSVs into XCMS for alignment / correspondence / fillPeaks across samples.

## Requirements
- [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- Linux / macOS / Windows
- NVIDIA GPU with CUDA driver (recommended for training/inference; CPU-only also works but is slow)

## Installation (single conda environment with Python + R)
All Python and R dependencies are installed into ONE conda environment (Python 3.11 + R from conda-forge).

```bash
# 1. Clone the repository
git clone https://github.com/hollowsamadesu/mytest_pl.git peak_lord
cd peak_lord

# 2. Create the environment (Python 3.11)
conda create -n peaklord python=3.11 -y
conda activate peaklord

# 3. (Recommended for GPU) Install a CUDA build of PyTorch BEFORE step 4.
#    `pip install -e ultralytics/` alone pulls the latest CPU-only torch from PyPI
#    (PyPI has no CUDA wheels for Windows). Pick the index for your CUDA driver at
#    https://pytorch.org/get-started/locally/ ; e.g. for CUDA 12.8+:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
#    CPU-only users can skip this step.

# 4. Install the Python package (editable)
pip install -e ultralytics/

# 5. Merge the R environment into the same conda env (r-base + optparse + xcms + MSnbase + BiocParallel)
conda env update -n peaklord -f ultralytics/r_environment.yml
# Bioconductor packages are not on conda channels for Windows (bioconda = Linux/macOS only),
# so install them from inside R via BiocManager (also works on Linux/macOS):
Rscript -e 'install.packages("BiocManager")'
Rscript -e 'BiocManager::install(c("xcms", "MSnbase", "BiocParallel"), update = FALSE, ask = FALSE)'
# (BiocManager downloads many dependencies and can take 10-30 minutes.)
```

### Verify the installation
```bash
# Python side
python -c "import torch, ultralytics; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# R side
Rscript -e 'suppressPackageStartupMessages({library(optparse); library(xcms); library(MSnbase); library(BiocParallel)}); cat("R OK\n")'
```

## Usage

### 1. Peak detection — `sl_inference.py`
```bash
cd ultralytics
python sl_inference.py \
  --model /path/to/best.pt \
  --datadir /path/to/mzml_dir \
  --img_tmp_path /tmp/tmp.jpg \
  --csv_save_dir /path/to/csv_dir
```
- `--datadir`: directory containing the input `.mzML` files (one CSV is written per file).
- `--model`: YOLO weights for peak detection (e.g. a model trained with `train_detect_ssl.py`). The pretrained Peak-LoRD model is included at `ultralytics/peak_lord.pt`.
- `--csv_save_dir`: output directory for the peak CSVs (columns: `mz, rt, mzmin, mzmax, rtmin, rtmax, into, maxo, sample, conf`).

#### Main parameters
- `--ppm` (int, default: `15`) — ppm tolerance for ROI dynamic m/z matching.
- `--roi_delta_mz` (float, default: `0.01`) — absolute m/z tolerance lower bound (Da).
- `--roi_required_points` (int, default: `10`) — minimum number of points required per ROI.
- `--roi_dropped_points` (int, default: `5`) — maximum allowed consecutive missing points when tracking ROIs.
- `--min_nonzero_points` (int, default: `5`) — minimum number of non-zero points in an EIC to keep it.
- `--min_height` (float, default: `1000`) — minimum peak intensity to keep an EIC.
- `--window_counts` (list of int, default: `[1..14]`) — numbers of windows used to split each EIC for feature extraction.
- `--window_count_thresholds` (list of float, default: `[1..13]`) — thresholds used to define the number of windows.
- `--save_images` (flag, default: `False`) — save local EIC plots around predicted peaks to `--images_dir`.
- `--images_rt_margin` (float, default: `0.1`) — RT margin (minutes) around the detected peak window.
- `--images_window_size` (float, default: `1`) — EIC plotting window size in minutes.
- `--smooth_mode` / `--down_sample` / `--use_min_windowsize` / `--min_window_size` — smoothing and sampling options for plotting.
- `--isDebug` / `--debugPlotImgPath` — debug mode with verbose output.

### 2. Semi-supervised detection training — `train_detect_ssl.py`
Edit the `overrides` dict at the top of the script (paths, `model`, `unsup_model`, `data`, `unsup_data`, `batch`, `epochs`, `device`, ...), then:
```bash
cd ultralytics
python train_detect_ssl.py
```
- `model`: detection model config (e.g. `yolov8n.yaml`) or a pretrained `.pt`.
- `unsup_model`: pretrained teacher weights; set to `None` to copy-initialize the teacher from the student model.
- `data` / `unsup_data`: labeled / unlabeled dataset YAMLs (YOLO format).
- `self_train`: `False` = standard semi-supervised (labeled + unlabeled), `True` = self-training on unlabeled data only.

### 3. Semi-supervised segmentation training — `train_ssl.py`
Same pattern as above (`task="semi_segment"`, e.g. `model="yolo11-seg.yaml"`). Edit the `overrides` dict (paths are placeholders — replace them with your own), then:
```bash
cd ultralytics
python train_ssl.py
```

**Note**: Peak_LoRD supports both detection and segmentation models for training, and the two work identically during inference.

### 4. RT alignment across samples — `align.R`
Imports the peak CSVs produced by `sl_inference.py` into XCMS and performs RT correction (obiwarp), peak grouping and fillPeaks. CSV columns required: `mz, mzmin, mzmax, rt, rtmin, rtmax, into, maxo`. `rt` must be in **seconds**.
```bash
cd ultralytics
Rscript align.R \
  --mzml_dir /path/to/mzml_dir \
  --csv_dir /path/to/csv_dir \
  --out_dir /path/to/align_output \
  --threads 4
```
Options: `--rt_in_minutes` (if your CSVs use minutes), `--retcor_method` (`obiwarp` or `peakgroups`), `--group_mzwid`, `--group_bw`, `--group_minfrac`, `--threads`, `--verbose`.

Outputs in `--out_dir`:
- `feature_table_into.csv` — feature × sample matrix,
- `feature_definitions.csv` — feature definitions (m/z, RT),
- `aligned_chromPeaks.csv` — all aligned chromPeaks.
