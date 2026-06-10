"""Core VAE-GAN model definitions and checkpoint helpers for ADVIS.

This module replaces the old notebook utility imports. Scripts should import
models only from ``advis.models``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class Encoder(nn.Module):
    """VAE encoder for 128x128 RGB inputs."""

    def __init__(self, z_size: int = 64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(512 * 8 * 8, z_size)
        self.fc_logvar = nn.Linear(512 * 8 * 8, z_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_layers(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """VAE decoder for 128x128 RGB outputs."""

    def __init__(self, z_size: int = 64):
        super().__init__()
        self.fc = nn.Linear(z_size, 512 * 8 * 8)
        self.deconv_layers = nn.Sequential(
            nn.ReLU(),
            nn.Unflatten(1, (512, 8, 8)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.deconv_layers(self.fc(z))


class Discriminator(nn.Module):
    """CNN discriminator used by the VAE-GAN training objective."""

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_layers(self.conv_layers(x))


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def get_reconstructed(Enc: Encoder, Dec: Decoder, data_: torch.Tensor, device: str | torch.device = "cuda") -> torch.Tensor:
    data_ = Variable(data_).to(device)
    mu, logvar = Enc(data_)
    z = reparameterize(mu, logvar)
    return Dec(z)


def get_anomaly_score(recon_batch: torch.Tensor, data_batch: torch.Tensor) -> torch.Tensor:
    device = data_batch.device
    recon_batch = recon_batch.to(device)
    abs_diff = torch.abs(recon_batch - data_batch)
    mean_diff = abs_diff.mean(dim=1)
    return mean_diff.max(dim=-1).values.max(dim=-1).values


def get_loss_functions(verbose: bool = True):
    reconstruction_loss_fn = nn.MSELoss()
    adversarial_loss_fn = nn.BCEWithLogitsLoss()
    if verbose:
        print("Loss functions initialized:")
        print("Reconstruction Loss Function:", reconstruction_loss_fn)
        print("Adversarial Loss Function:", adversarial_loss_fn)
    return reconstruction_loss_fn, adversarial_loss_fn


def get_optimizers(
    Enc: Encoder,
    Dec: Decoder,
    Dis: Discriminator,
    learning_rate_enc_dec: float = 0.001,
    learning_rate_dis: float = 0.0001,
    verbose: bool = True,
):
    optEncDec = optim.Adam(list(Enc.parameters()) + list(Dec.parameters()), lr=learning_rate_enc_dec)
    optDis = optim.Adam(Dis.parameters(), lr=learning_rate_dis)
    if verbose:
        print(f"Optimizers initialized: enc/dec lr={learning_rate_enc_dec}, dis lr={learning_rate_dis}")
    return optEncDec, optDis


def model_override(model_path: str | Path, suffix: str) -> None:
    model_path = Path(model_path)
    current = model_path / f"model_{suffix}.pt"
    backup = model_path / f"model_{suffix}_old.pt"
    if current.exists():
        current.rename(backup)
        print(f"Existing checkpoint renamed: {current} -> {backup}")
    else:
        print(f"No checkpoint to override: {current}")


def save_model(
    Enc: Encoder,
    Dec: Decoder,
    D: Discriminator,
    optEncDec: torch.optim.Optimizer,
    optD: torch.optim.Optimizer,
    paths: Any,
    loss_history: list[dict],
    suffix: str,
    verbose: bool = False,
) -> None:
    model_dir = Path(paths.path_models)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"model_{suffix}.pt"
    if verbose:
        print(f"Saving model at: {model_path}")
    torch.save(
        {
            "encoder_state_dict": Enc.state_dict(),
            "decoder_state_dict": Dec.state_dict(),
            "discriminator_state_dict": D.state_dict(),
            "optimizer_enc_state_dict": optEncDec.state_dict(),
            "optimizer_dec_state_dict": optD.state_dict(),
            "loss_history": pd.DataFrame(loss_history),
        },
        model_path,
    )


def load_model(
    Enc: Encoder,
    Dec: Decoder,
    D: Discriminator,
    optEncDec: torch.optim.Optimizer,
    optD: torch.optim.Optimizer,
    paths: Any,
    suffix: str,
    device: str | torch.device = "cuda",
    verbose: bool = False,
) -> list[dict]:
    model_path = Path(paths.path_models) / f"model_{suffix}.pt"
    if verbose:
        print(f"Trying model from: {model_path}")
    if not model_path.exists():
        if verbose:
            print(f"Checkpoint does not exist: {model_path}")
        return []

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    Enc.load_state_dict(checkpoint["encoder_state_dict"])
    Dec.load_state_dict(checkpoint["decoder_state_dict"])
    if "discriminator_state_dict" in checkpoint:
        D.load_state_dict(checkpoint["discriminator_state_dict"])
    if "optimizer_enc_state_dict" in checkpoint:
        optEncDec.load_state_dict(checkpoint["optimizer_enc_state_dict"])
    if "optimizer_dec_state_dict" in checkpoint:
        optD.load_state_dict(checkpoint["optimizer_dec_state_dict"])

    history = checkpoint.get("loss_history", [])
    if hasattr(history, "to_dict"):
        history = history.to_dict("records")
    if verbose:
        print(f"Model loaded with {len(history)} history rows: {model_path}")
    return history
