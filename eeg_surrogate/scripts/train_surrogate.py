"""Entry point for training the EEG surrogate brain model."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from eeg_surrogate.data.dataset import create_datasets
from eeg_surrogate.data.preprocess import preprocess_directory
from eeg_surrogate.train.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train surrogate brain model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"),
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", None))

    subjects = preprocess_directory(config["data_root"], config)
    if len(subjects) == 0:
        raise RuntimeError("No valid subjects found. Check data_root and file formats.")

    train_dataset, val_dataset = create_datasets(subjects, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and config.get("use_cuda", True) else "cpu")
    trainer = Trainer(config, train_loader, val_loader, device)
    trainer.train()


if __name__ == "__main__":
    main()
