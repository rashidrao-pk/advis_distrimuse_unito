"""Clean utility helpers used by ADVIS scripts.

This module replaces the previous large notebook utility files. It contains only
runtime utilities needed by train/calibrate/inference scripts.
"""

from __future__ import annotations

import os
import platform
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


@dataclass
class Parameters:
    dummy: bool = False
    model_depth: int = 3
    latent_dims: int = 64
    learning_rate_enc_dec: float | None = None
    learning_rate_dis: float | None = None
    target_size: tuple[int, int] = (128, 128)
    batch_size: int = 64
    input_shape: tuple[int, int, int] = (3, 128, 128)
    model_name: str = "VAE-GAN"
    train_mode: str = ""
    num_workers: int = 4
    pin_memory: bool = False
    persistent_workers: bool = False
    aug_type: str = "min"


@dataclass
class Paths:
    dummy: bool = False
    path_datasets: str = ""
    path_datasets_main: str = "data"
    dataset_version: str = "v4"
    dataset_type: str = "back_view"
    path_dataset_selected: str = ""
    train_dir: str = ""
    test_dir: str = ""
    train_classes: list[str] | str = ""
    test_classes: list[str] | str = ""
    class_names_train: list[str] | str = ""
    path_codes: str = ""
    path_codes_main: str = ""
    path_codes_cloud: str = ""
    path_codes_local: str = ""
    path_results_local: str = "results"
    path_results_cloud: str = "results"
    path_models: str = "results/models"
    path_results: str = "results/training"
    history_fname: str = ""
    log_file_full: str = "results/logs/log.txt"


def get_params_paths() -> tuple[Parameters, Paths]:
    params = Parameters()
    paths = Paths(path_codes=os.getcwd())
    return params, paths


def natural_sort_key(filename: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", filename)]


def get_paths(paths: Paths, verbose: bool = False) -> Paths:
    """Resolve project-level paths without machine-specific hardcoding."""
    cwd = Path.cwd()
    paths.path_codes = str(cwd)
    paths.path_codes_main = str(cwd / "scripts")
    paths.path_results_local = str(cwd / "results")
    paths.path_results_cloud = str(cwd / "results")
    paths.path_datasets_main = str(cwd / "data")
    paths.path_models = str(cwd / "results" / "models")
    paths.path_results = str(cwd / "results" / "training")
    if verbose:
        print(f"system -- OS({os.name}) - node({platform.node()})")
        print(f"path_codes: {paths.path_codes}")
        print(f"path_datasets_main: {paths.path_datasets_main}")
        print(f"path_results_local: {paths.path_results_local}")
    return paths


def get_parameters_by_experiment(params: Parameters, verbose: bool = False) -> Parameters:
    exp_type = getattr(params, "exp_type", "E2")
    params.learning_rate_enc_dec = 0.001
    params.learning_rate_dis = 0.0001
    params.beta_kl = 0.0001
    params.beta_gan = 0.0001 if exp_type == "E3" else 0.001
    params.reconstruction_loss_fn = "MSE"
    params.adversarial_loss_fn = "BCWithLogits"
    if verbose:
        print(f"Experiment: {exp_type}")
        print(f"LR enc/dec: {params.learning_rate_enc_dec}")
        print(f"LR dis: {params.learning_rate_dis}")
        print(f"beta_kl: {params.beta_kl}")
        print(f"beta_gan: {params.beta_gan}")
    return params


def get_dataset_version(
    paths: Paths,
    params: Parameters,
    dataset_version: str = "v4",
    dataset_type: str = "back_view",
    subgroup: str = "RoboArm",
    mask_image_name: int = 3015,
    verbose: bool = False,
):
    paths.dataset_version = dataset_version
    paths.dataset_type = dataset_type
    paths.path_datasets = os.path.join(paths.path_datasets_main, dataset_version)
    paths.path_dataset_selected = os.path.join(paths.path_datasets, dataset_type)
    paths.train_dir = os.path.join(paths.path_dataset_selected, "train")
    paths.test_dir = os.path.join(paths.path_dataset_selected, "test")
    paths.mask_dir = os.path.join(paths.path_dataset_selected, "masks")
    paths.mask_image_name = mask_image_name

    params.subgroup = subgroup
    params.subgroup_mask = "mask"
    params.aug_type = f"auto-{params.subgroup_mask}"

    paths.train_dir_processed = os.path.join(paths.path_dataset_selected, "train_processed")
    paths.test_dir_processed = os.path.join(paths.path_dataset_selected, "test_processed")
    paths.train_dir_subgroup = os.path.join(paths.train_dir, subgroup)
    paths.test_dir_subgroup = os.path.join(paths.test_dir, subgroup)
    paths.train_dir_processed_subgroup = os.path.join(paths.train_dir_processed, subgroup)
    paths.test_dir_processed_subgroup = os.path.join(paths.test_dir_processed, subgroup)

    if verbose:
        print(f"Dataset root: {paths.path_dataset_selected}")
        print(f"Train processed: {paths.train_dir_processed_subgroup}")
        print(f"Test processed: {paths.test_dir_processed_subgroup}")
    return paths, params


def get_create_results_path(
    subgroup: str,
    params: Parameters,
    args: Any,
    paths: Paths,
    dir: str = "scripts",
    save_path_type: str = "local",
    models_dir: str = "models",
    create_dirs: bool = True,
    verbose: bool = True,
):
    base_dir = paths.path_results_local if save_path_type == "local" else paths.path_results_cloud
    suffix = f"{subgroup}_{params.latent_dims}"

    paths.history_fname = f"vae_gan_train_history_{subgroup}.csv"
    paths.path_models = os.path.join(base_dir, models_dir)
    paths.path_results = os.path.join(base_dir, "training", suffix)
    paths.path_results_fix = os.path.join(base_dir, "monitor", suffix)
    paths.log_file_full = os.path.join(paths.path_results, "log_file_full.txt")

    if create_dirs:
        os.makedirs(paths.path_models, exist_ok=True)
        os.makedirs(paths.path_results, exist_ok=True)
        if getattr(args, "save_figures", False):
            os.makedirs(paths.path_results_fix, exist_ok=True)

    if verbose:
        print(f"Component: {subgroup}")
        print(f"base_dir: {base_dir}")
        print(f"path_models: {paths.path_models}")
        print(f"path_results: {paths.path_results}")
    return suffix, paths


def get_colormap(mode: int = 1):
    if mode == 1:
        return LinearSegmentedColormap.from_list(
            "colormap_anomaly_map",
            [(0.0, "white"), (0.45, "skyblue"), (0.455, "violet"), (1.0, "red")],
        )
    return LinearSegmentedColormap.from_list(
        "colormap_anomaly_map", [(0.0, "#ffffff"), (0.1, "#ff0000"), (1.0, "#03fcf8")]
    )


def get_time(suff: str = "", verbose: bool = True):
    now = datetime.now()
    if verbose:
        print(f"{suff}: {now}")
    return now


def get_header(params: Parameters, paths: Paths, verbose: bool = False):
    input_shape = f"{params.input_shape[0]}x{params.input_shape[1]}x{params.input_shape[2]}"
    stat = f"| {'dataset':^10} | {'camera':^10} | {'subgroup':^10} | {'epochs':^10} | {'latent_dims':^15} | {'input_shape':^15} | {'batch_size':^10} | {'LR_enc_dec':^15} | {'LR_dis':^15} |\n"
    dyn = f"| {paths.dataset_version:^10} | {paths.dataset_type:^10} | {params.subgroup:^10} | {params.epochs:^10} | {params.latent_dims:^15} | {input_shape:^15} | {params.batch_size:^10} | {params.learning_rate_enc_dec:^15} | {params.learning_rate_dis:^15} |\n"
    if verbose:
        print(stat.strip())
        print(dyn.strip())
    return stat, dyn


def create_log_file(params: Parameters, paths: Paths, start_time, verbose: bool = False, read_mode: str = "w") -> str:
    header, dyn = get_header(params, paths, verbose=False)
    text = "=" * 160 + "\n"
    text += "START CODE\n"
    text += f"code started:\t{start_time}\n"
    text += "-" * 160 + "\n"
    text += header
    text += dyn
    text += "-" * 160 + "\n"
    text += f"paths_results:\t{paths.path_results}\n"
    text += f"paths_models:\t{paths.path_models}\n"
    if verbose:
        print(text)
    return text


def save_log_file(log_file_full: str | Path, log_messages: str, read_mode: str = "a", verbose: bool = True) -> None:
    path = Path(log_file_full)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(read_mode, encoding="utf-8") as f:
        f.write(log_messages)
    if verbose:
        print(f"Log file saved to {path}")


class MaskedCrop:
    """Crop image using a boolean safety-area mask, then apply the mask."""

    def __init__(self, subgroup=None, mask=None, fill_color=(0, 0, 0), verbose: bool = False):
        self.subgroup = subgroup
        self.fill_color = fill_color
        self.mask = mask
        self.verbose = verbose
        if mask is None:
            raise ValueError("MaskedCrop requires a boolean mask.")
        mask = np.asarray(mask).astype(bool)
        rows, cols = np.where(mask)
        self.x1, self.x2 = np.min(cols), np.max(cols)
        self.y1, self.y2 = np.min(rows), np.max(rows)
        cropped_mask_1ch = mask[self.y1 : self.y2 + 1, self.x1 : self.x2 + 1]
        self.cropped_mask = np.stack([cropped_mask_1ch] * 3, axis=-1)

    def __call__(self, image):
        if self.subgroup is None:
            return image
        image_np = np.array(image)
        image_np = image_np[self.y1 : self.y2 + 1, self.x1 : self.x2 + 1, :]
        image_np = image_np * self.cropped_mask
        return Image.fromarray(image_np.astype("uint8"))

    def uncrop(self, image_np):
        assert image_np.shape[0] == self.cropped_mask.shape[0]
        assert image_np.shape[1] == self.cropped_mask.shape[1]
        image_np = image_np * self.cropped_mask
        no_channels = image_np.shape[2]
        out = np.zeros((self.mask.shape[0], self.mask.shape[1], no_channels), dtype=image_np.dtype)
        out[self.y1 : self.y2 + 1, self.x1 : self.x2 + 1, :] = image_np
        return out


def _to_display_image(t: torch.Tensor) -> np.ndarray:
    img = t.detach().cpu() * 0.5 + 0.5
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def plot_images(
    original,
    reconstructed,
    epoch,
    paths,
    ttl="train",
    data_type=None,
    anomaly_scores=None,
    save_path=None,
    cmap=None,
    plot_anomaly_scores: bool = False,
    plot_suptitle: bool = True,
    destroy_fig: bool = False,
    save_fig: bool = True,
    interval: int = 10,
    fontsize: int = 12,
    fontcolor: str = "black",
):
    n_images = min(3, original.shape[0])
    fig, axes = plt.subplots(1, n_images * 3, figsize=(10, 2))
    if n_images == 1:
        axes = np.array([axes]).reshape(-1)
    for i in range(n_images):
        orig_img = _to_display_image(original[i])
        recon_img = _to_display_image(reconstructed[i])
        diff_img = np.linalg.norm(orig_img - recon_img, axis=2)
        axes[i * 3].imshow(orig_img)
        axes[i * 3].set_title("x", fontsize=fontsize)
        axes[i * 3 + 1].imshow(recon_img)
        axes[i * 3 + 1].set_title("x'", fontsize=fontsize)
        axes[i * 3 + 2].imshow(diff_img, cmap=cmap or get_colormap(), vmin=0, vmax=2)
        axes[i * 3 + 2].set_title("m", fontsize=fontsize)
        if plot_anomaly_scores and anomaly_scores is not None:
            axes[i * 3 + 2].text(5, 12, f"Score: {float(anomaly_scores[i]):.3f}", color=fontcolor, fontsize=fontsize - 2)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    if plot_suptitle:
        plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout(pad=0.1)
    if save_fig and epoch % interval == 0:
        save_dir = Path(save_path or paths.path_results)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{ttl}_{epoch}.png", bbox_inches="tight", dpi=150)
    if destroy_fig:
        plt.close(fig)


def plot_images_tracking(
    original_random,
    reconstructed_random,
    original_fixed,
    reconstructed_fixed,
    original_noise,
    reconstructed_noise,
    epoch,
    path_results,
    ttl="train",
    cmap=None,
    plot_anomaly_scores: bool = False,
    destroy_fig: bool = False,
    save_fig: bool = True,
    interval: int = 10,
    fontsize: int = 10,
    fontcolor: str = "black",
):
    fig, axes = plt.subplots(1, 9, figsize=(9, 1.7))
    groups = [
        (original_random, reconstructed_random, "Random"),
        (original_fixed, reconstructed_fixed, "Fixed"),
        (original_noise, reconstructed_noise, "Noise"),
    ]
    for g, (orig, rec, label) in enumerate(groups):
        orig_img = _to_display_image(orig[0])
        rec_img = _to_display_image(rec[0])
        diff = np.linalg.norm(orig_img - rec_img, axis=2)
        j = g * 3
        axes[j].imshow(orig_img); axes[j].set_title(f"{label} x", fontsize=fontsize)
        axes[j + 1].imshow(rec_img); axes[j + 1].set_title(f"{label} x'", fontsize=fontsize)
        axes[j + 2].imshow(diff, cmap=cmap or get_colormap(), vmin=0, vmax=2); axes[j + 2].set_title(f"{label} m", fontsize=fontsize)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f"Epoch {epoch}", fontsize=fontsize)
    plt.tight_layout(pad=0.05)
    if save_fig and epoch % interval == 0:
        Path(path_results).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(path_results) / f"{ttl}_{epoch}.png", bbox_inches="tight", dpi=150)
    if destroy_fig:
        plt.close(fig)


def plot_loss_sep(loss_history, params, paths, plot_type=3, save_fig=True, destroy_fig=True, verbose_print=False, plot_long_header=True, fontsize=12):
    if not loss_history:
        return
    keys = [k for k in loss_history[0].keys() if k not in {"epoch", "elapsed_time"}]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = [row.get("epoch", i) for i, row in enumerate(loss_history)]
    for key in keys:
        vals = [row.get(key) for row in loss_history]
        if all(v is not None for v in vals):
            ax.plot(x, vals, label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=fontsize - 2)
    ax.grid(True, alpha=0.3)
    if save_fig:
        Path(paths.path_results).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(paths.path_results) / "loss_history.png", bbox_inches="tight", dpi=150)
    if destroy_fig:
        plt.close(fig)
