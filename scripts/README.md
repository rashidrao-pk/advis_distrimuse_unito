# scripts/

This folder contains command-line entry scripts.

Scripts:

- `train.py` — train VAE-GAN models for one or all safety areas
- `calibrate_threshold.py` — estimate or tune anomaly thresholds
- `inference.py` — run anomaly detection on preprocessed frames, raw frames, videos, or camera streams

After installation:

```bash
pip install -e .
```

Use:

```bash
advis-train
advis-calibrate
advis-infer
```
