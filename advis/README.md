# advis/

This folder contains the installable Python package used by ADVIS_UniGra.

Main modules:

- `config.py` — loads `configs/config.yaml` and applies defaults to CLI scripts
- `data.py` — dataset and transform helpers
- `models.py` — VAE-GAN model definitions and checkpoint helpers
- `scoring.py` — anomaly map and anomaly score functions
- `utils.py` — reproducibility, device, and path utilities
- `cad_utils.py` — cleaned runtime helpers used by training, calibration, and inference

New reusable code should be added here instead of inside `scripts/`.
