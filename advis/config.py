from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _set_if_present(args: Any, name: str, value: Any) -> None:
    if value is None:
        return
    if hasattr(args, name):
        setattr(args, name, value)


def apply_config_defaults(args: Any, config: dict[str, Any]) -> Any:
    """Apply config values to an argparse Namespace.

    CLI values still work normally, but this function gives the project a
    single source of defaults through configs/config.yaml.
    """
    project = config.get("project", {})
    data = config.get("data", {})
    model = config.get("model", {})
    training = config.get("training", {})
    threshold = config.get("threshold", {})
    inference = config.get("inference", {})
    paths = config.get("paths", {})

    _set_if_present(args, "dataset_version", data.get("dataset_version"))
    _set_if_present(args, "dataset_type", data.get("dataset_type"))
    _set_if_present(args, "mask_image_name", data.get("mask_image_name"))
    _set_if_present(args, "latent_dims", model.get("latent_dim"))
    _set_if_present(args, "epochs", training.get("epochs"))
    _set_if_present(args, "batch_size", training.get("batch_size"))
    _set_if_present(args, "num_workers", training.get("num_workers"))
    _set_if_present(args, "pin_memory", training.get("pin_memory"))
    _set_if_present(args, "augmentation_type", training.get("augmentation"))
    _set_if_present(args, "val_every", training.get("val_every"))
    _set_if_present(args, "val_offset", training.get("val_offset"))
    _set_if_present(args, "offset", threshold.get("offset"))
    _set_if_present(args, "sigma", threshold.get("sigma"))
    _set_if_present(args, "quantile", threshold.get("quantile"))
    _set_if_present(args, "threshold_strategy", threshold.get("strategy"))
    _set_if_present(args, "threshold_percentile", threshold.get("percentile"))
    _set_if_present(args, "threshold_n_sigma", threshold.get("n_sigma"))
    _set_if_present(args, "monitor_score", threshold.get("monitor_score"))
    _set_if_present(args, "max_frames", inference.get("max_frames"))
    _set_if_present(args, "save_path_type", paths.get("save_path_type"))

    if hasattr(args, "checkpoints") and paths.get("checkpoints_dir"):
        args.checkpoints = paths["checkpoints_dir"]
    if hasattr(args, "threshold_dir") and paths.get("thresholds_dir"):
        args.threshold_dir = paths["thresholds_dir"]
    if hasattr(args, "output_dir") and paths.get("inference_dir") and getattr(args, "output_dir", None) is None:
        args.output_dir = paths["inference_dir"]
    if hasattr(args, "gt_csv") and data.get("annotation_csv"):
        args.gt_csv = data["annotation_csv"]

    seed = project.get("seed")
    if seed is not None:
        try:
            import random
            import numpy as np
            import torch
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    for key in ["results_dir", "models_dir", "thresholds_dir", "training_dir", "inference_dir", "logs_dir"]:
        if paths.get(key):
            Path(paths[key]).mkdir(parents=True, exist_ok=True)

    return args


def resolve_safety_areas(config: dict[str, Any], safety_area: str) -> list[str]:
    available = config.get("safety_areas", {}).get("available", [])
    if safety_area == "ALL":
        return list(available)
    if available and safety_area not in available:
        raise ValueError(f"Unknown safety_area={safety_area}. Available: {available}")
    return [safety_area]
