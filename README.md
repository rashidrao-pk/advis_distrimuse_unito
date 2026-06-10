# ADVIS_UniGra

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?color=00E5C3&lines=RGB+Anomaly+Detection+for+Safe+Human-Robot+Interaction;Synthetic+Dataset+%7C+VAE-GAN+%7C+Industrial+Safety+Monitoring;Safety-Area+Inference+%7C+Threshold+Calibration+%7C+DistriMuSe+UC3&center=true&width=900&height=45">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Project-DistriMuSe-0A192F?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Use%20Case-UC3-00E5C3?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Model-VAE--GAN-7A3EFF?style=for-the-badge" />
</p>

> **ADVIS_UniGra** is a research repository for reconstruction-based visual anomaly detection in collaborative robotic environments. It supports safety-area training, threshold calibration, and inference using VAE-GAN models.

---

## Overview

This repository provides the anomaly detection pipeline developed for **DistriMuSe UC3: Safe Interaction with Robots**. The system monitors predefined safety areas such as the robot arm, conveyor belt, and personnel zones. A VAE-GAN model is trained on normal data, and unexpected conditions are detected using reconstruction-based anomaly scores.

```text
Raw Frames / Video
        ↓
Safety-area preprocessing
        ↓
VAE-GAN training
        ↓
Threshold calibration
        ↓
Inference and evaluation
```

---

## Key Features

- VAE-GAN based visual anomaly detection
- Per-safety-area model training
- Validation-based and labelled-test threshold calibration
- Config-driven experiments using `configs/config.yaml`
- Installable Python package named `advis`
- CLI commands for training, calibration, and inference
- Support for preprocessed frames, raw frames, videos, and camera streams
- Folder-level README files for easier navigation

---

## Safety Areas

| Area | Description |
|---|---|
| `RoboArm` | Robot arm safety region |
| `ConvBelt` | Conveyor belt safety region |
| `PLeft` | Left personnel safety region |
| `PRight` | Right personnel safety region |
| `ALL` | Run all available safety areas |

---

## Repository Structure

```text
ADVIS_UniGra/
├── advis/
│   ├── config.py
│   ├── data.py
│   ├── models.py
│   ├── scoring.py
│   ├── utils.py
│   └── cad_utils.py
├── configs/
│   └── config.yaml
├── scripts/
│   ├── train.py
│   ├── calibrate_threshold.py
│   └── inference.py
├── docs/
├── data/
├── results/
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone repository

```bash
git clone https://github.com/rashidrao-pk/advis_distrimuse_unito.git
cd advis_distrimuse_unito
```

### 2. Create environment

```bash
conda create -n advis python=3.9 -y
conda activate advis
```

### 3. Install PyTorch

For CUDA 11.8:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

For CPU only:

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

### 4. Install ADVIS package

```bash
pip install -e .
```

This installs the package name `advis` and provides these CLI commands:

```bash
advis-train
advis-calibrate
advis-infer
```

---

## Configuration

All default paths and parameters are stored in:

```text
configs/config.yaml
```

Important fields:

```yaml
data:
  dataset_version: V6
  dataset_type: fronttop

model:
  latent_dim: 64

training:
  epochs: 1000
  batch_size: 32

threshold:
  score_method: max
  offset: 1
  sigma: 1.0
  quantile: 1.0
```

You can still override most values from the command line.

---

## Training

Train one safety area:

```bash
advis-train --config configs/config.yaml --safety_area RoboArm
```

Train all safety areas:

```bash
advis-train --config configs/config.yaml --safety_area ALL
```

Override parameters:

```bash
advis-train \
  --config configs/config.yaml \
  --safety_area PLeft \
  --epochs 500 \
  --batch_size 32
```

Equivalent script command:

```bash
python scripts/train.py --config configs/config.yaml --safety_area RoboArm
```

---

## Threshold Calibration

### Validation mode

Use this when only normal validation data is available.

```bash
advis-calibrate \
  --config configs/config.yaml \
  --mode val \
  --safety_area ALL
```

### Labelled-test mode

Use this when labelled normal/anomalous test data is available.

```bash
advis-calibrate \
  --config configs/config.yaml \
  --mode test \
  --safety_area ALL \
  --gt_csv scripts/data/annotations/anom_metadata_unexpected_person.csv
```

Calibration outputs are saved under:

```text
results/thresholds/
```

---

## Inference

Run inference on preprocessed frames:

```bash
advis-infer \
  --config configs/config.yaml \
  --data_source preprocessed \
  --input_dir data/test_processed/unexpected_person \
  --safety_area RoboArm
```

Run inference on all safety areas:

```bash
advis-infer \
  --config configs/config.yaml \
  --data_source preprocessed \
  --input_dir data/test_processed/unexpected_person \
  --safety_area ALL
```

Run inference on video:

```bash
advis-infer \
  --config configs/config.yaml \
  --data_source video \
  --input_video data/videos/test_video.mp4 \
  --safety_area ALL
```

Run inference and save figures:

```bash
advis-infer \
  --config configs/config.yaml \
  --data_source preprocessed \
  --input_dir data/test_processed/unexpected_person \
  --safety_area ALL \
  --save_figures
```

---

## Pipeline Figures

### Workflow

<p align="center">
  <img src="docs/workflow.svg" width="95%">
</p>

### Preprocessing

<p align="center">
  <img src="docs/preprocessing.svg" width="95%">
</p>

### VAE-GAN Architecture

<p align="center">
  <img src="docs/VAE_GAN_Model.svg" width="95%">
</p>

### Sample Results

<p align="center">
  <img src="docs/sample_results.svg" width="95%">
</p>

---

## Supported Dataset

| Dataset | Source | Description |
|---|---|---|
| Synthetic Palletizing | University of Granada, Valeria Lab | Synthetic collaborative robotics dataset for DistriMuSe UC3 |

Dataset generation tool:

```text
https://github.com/valerialabugr/SimIndus-Dataset
```

---

## Outputs

```text
results/
├── models/
├── thresholds/
├── training/
├── inference/
└── logs/
```

Large checkpoints and generated outputs should not be committed to Git.

---

## Notes on Refactoring

The active scripts now import only from the installable `advis` package. The old notebook utility files are removed from the runtime path. Reusable code is organized into `advis.models`, `advis.cad_utils`, `advis.scoring`, `advis.data`, and `advis.config`.

---

## Citation
Coming Soon

---

## Acknowledgements

This work is developed in the context of the **DistriMuSe** project on distributed multi-sensor systems for human safety and health.

---

## Keywords

Anomaly Detection · VAE-GAN · Explainable AI · Industrial Safety · Collaborative Robotics · Computer Vision · Human-Robot Interaction
