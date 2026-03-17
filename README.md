# DistriMuSe-UC3

> **University of Torino** — Distributed Multi-Sensor Systems for Human Safety and Health

---

## Use Case 3 · Safe Interaction with Robots

An anomaly detection pipeline based on variational autoencoder models that monitors industrial safety areas in real time. The system processes video input, trains per-area models, calibrates detection thresholds, and runs inference on live or recorded footage.

---

## Pipeline Overview

```
Raw Video → Preprocessing → Training → Threshold Calibration → Inference
```

| Stage | Script | Description |
|---|---|---|
| Preprocess | — | Crop safety areas from 2540px-wide video, resize to 128×128 |
| Train | `scripts/train.py` | Train autoencoder per safety area |
| Compute Threshold | `scripts/compute_threshold.py` | Estimate reconstruction-error thresholds on validation data |
| Calibrate Threshold | `scripts/calibrate_threshold.py` | Tune thresholds using labelled or unlabelled data |
| Inference | `scripts/inference.py` | Run anomaly detection on preprocessed frames, raw frames, video, or live stream |

---

## Safety Areas

| ID | Description |
|---|---|
| `RoboArm` | Robot arm zone |
| `ConvBelt` | Conveyor belt zone |
| `PLeft` | Personnel zone — left |
| `PRight` | Personnel zone — right |
| `ALL` | Run all areas sequentially |

---

## ⚙️ 1. Setup
### 1.1 Create environment


```bash
conda create -n dm_unito python==3.9.18
conda activate dm_unito
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

```

### 1.2 Clone GitHub Repo

```bash
git clone https://github.com/rashidrao-pk/distrimuse_unito
conda activate dm_unito
pip install -r requirments.txt
```

### 1.3 Retreive Model Checkpoints
```bash
cd distrimuse_unito/scripts
git clone https://gitlab.di.unito.it/rashid/dm_checkpoints
cd ..
```

---

## 1 · Train

Train an autoencoder model on one or all safety areas.

```bash
# Single area (default settings)
python scripts/train.py --safety_area RoboArm

# All areas sequentially
python scripts/train.py --safety_area ALL

# All areas with custom settings
python scripts/train.py --safety_area ALL --epochs 1000 --batch_size 64
```

**Full argument reference**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--safety_area` | `str` | `RoboArm` | Area to train (`RoboArm`, `ConvBelt`, `PLeft`, `PRight`, `ALL`) |
| `--epochs` | `int` | `1000` | Number of training epochs |
| `--batch_size` | `int` | `64` | Batch size (set automatically based on device if omitted) |
| `--data_split` | `int` | `80` | Train/validation split percentage (remainder used for validation) |
| `--augmentation_level` | `int` | `0` | `0` = none, `1` = custom augmentation |
| `--verbose_level` | `int` | `0` | `0` = silent, `1` = standard, `2` = detailed |
| `--save_figures` | flag | off | Save reconstruction plots, learning curves, and latent space visualisations |

```bash
# Fast mode — saves curves and checkpoints only
python scripts/train.py --safety_area RoboArm

# Full mode — also saves reconstruction figures
python scripts/train.py --safety_area RoboArm --save_figures
```

---

## 2 · Compute Threshold

Estimate per-area anomaly thresholds from reconstruction errors on the validation set.

```bash
# Default (max strategy, offset=1, sigma=1.0)
python scripts/compute_threshold.py --safety_area RoboArm

# All areas using percentile strategy
python scripts/compute_threshold.py --safety_area ALL \
    --threshold_strategy percentile \
    --threshold_percentile 95


# Faster computation with larger batches
python scripts/compute_threshold.py --safety_area ALL --batch_size 64
```

---

## 3 · Calibrate Threshold

Calibrate thresholds using either unlabelled validation data or a labelled test set.

```bash
# Validation mode — no labels required
python scripts/calibrate_threshold.py --mode val --safety_area ALL

# Test mode — labelled CSV required
python scripts/calibrate_threshold.py --mode test --safety_area ALL \
    --gt_csv scripts/data/annotations/anom_metadata_unexpected_person.csv

# Test mode — tune monitoring metric and search grid
python scripts/calibrate_threshold.py --mode test --safety_area RoboArm \
    --gt_csv scripts/data/annotations/anom_metadata.csv \
    --monitor_score recall \
    --offset_ls 1,2 \
    --sigma_ls 1.0,1.5
```

---

## 4 · Inference

Run anomaly detection from multiple input sources.

```bash
# Pre-cropped frames, evaluate against annotations
python scripts/inference.py \
    --data_source preprocessed \
    --input_dir /data/test/RoboArm \
    --safety_area RoboArm \
    --gt_csv /data/annotations.csv

# Raw video frames, all areas, save output figures
python scripts/inference.py \
    --data_source raw \
    --input_dir /data/frames \
    --safety_area ALL \
    --save_figures

# Video file, process first 500 frames
python scripts/inference.py \
    --data_source video \
    --input_video /data/test.avi \
    --max_frames 500

# Live IP camera stream, all areas
python scripts/inference.py \
    --data_source ipcam \
    --camera_url rtsp://192.168.1.10/stream \
    --safety_area ALL \
    --save_figures
```

**Data source options**

| `--data_source` | Input | Notes |
|---|---|---|
| `preprocessed` | Pre-cropped image folder | Fastest; requires prior preprocessing |
| `raw` | Raw frame folder | Crops and resizes on the fly |
| `video` | `.avi` / video file | Supports `--max_frames` limit |
| `ipcam` | RTSP stream URL | For live camera feeds |

---

## Repository Structure

```
distrimuse_unito/
├── scripts/
│   ├── train.py
│   ├── compute_threshold.py
│   ├── calibrate_threshold.py
│   ├── inference.py
│   └── results/                # Figures and threshold files
│       └── models/             # Saved model weights
│       └── threshold/          # Saved model weights
│       └── training/           # Saved Learning Curves
│   └── data/
│       └── annotations/

└── README.md
```

---

## Acknowledgements

Developed at the **University of Torino** as part of the [**_DistriMuse_**](https://distrimuse.eu/) project on distributed multi-sensor systems for human safety and health.
