"""Training loop for the EEG surrogate brain model."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from eeg_surrogate.models.surrogate import SurrogateModel
from eeg_surrogate.train.losses import get_loss


class Trainer:
    def __init__(
        self,
        config: Dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.model = SurrogateModel(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["input_dim"],
            model_type=config.get("model_type", "gru"),
            num_layers=config.get("num_layers", 1),
            dropout=config.get("dropout", 0.0),
            bidirectional=config.get("bidirectional", False),
        ).to(device)

        self.loss_fn = get_loss(config.get("loss_type", "mse"))
        self.optimizer = self._build_optimizer()
        self.best_val = float("inf")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = self.config.get("optimizer", "adam").lower()
        if name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.get("learning_rate", 1e-3),
                weight_decay=self.config.get("weight_decay", 0.0),
            )
        raise ValueError(f"Unsupported optimizer: {name}")

    def train(self) -> None:
        num_epochs = self.config.get("num_epochs", 1)
        log_interval = self.config.get("log_interval", 100)
        save_best_only = self.config.get("save_best_only", True)
        monitor_metric = self.config.get("monitor_metric", "val_loss")
        save_dir = Path(self.config.get("save_dir", "./checkpoints"))
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            train_loss = self._run_epoch(training=True)
            val_loss = self._run_epoch(training=False)
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

            metric_value = val_loss if monitor_metric == "val_loss" else train_loss
            if metric_value < self.best_val:
                self.best_val = metric_value
                self._save_checkpoint(save_dir / "best.pt")
            if not save_best_only:
                self._save_checkpoint(save_dir / f"epoch_{epoch:03d}.pt")

            if epoch % log_interval == 0:
                self._save_log(epoch, train_loss, val_loss)

    def _run_epoch(self, training: bool) -> float:
        loader = self.train_loader if training else self.val_loader
        if training:
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.set_grad_enabled(training):
            for batch in loader:
                x, y, _ = batch
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)
                loss = self.loss_fn(pred, y)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": self.model.state_dict(), "config": self.config}, path)

    def _save_log(self, epoch: int, train_loss: float, val_loss: float) -> None:
        log_dir = Path(self.config.get("log_dir", "./logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "training_log.json"
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        if log_path.exists():
            with log_path.open("r") as f:
                data = json.load(f)
            if isinstance(data, list):
                data.append(record)
            else:
                data = [data, record]
        else:
            data = [record]
        with log_path.open("w") as f:
            json.dump(data, f, indent=2)
