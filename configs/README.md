# configs/

This folder contains configuration files for ADVIS experiments.

The main file is:

```text
config.yaml

```

It defines dataset paths, safety areas, model parameters, training settings, threshold calibration options, and output directories.

```bash
advis-train --config configs/config.yaml --safety_area RoboArm
advis-calibrate --config configs/config.yaml --mode val --safety_area ALL
advis-infer --config configs/config.yaml --data_source preprocessed --input_dir data/test --safety_area ALL
```