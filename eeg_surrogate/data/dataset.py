"""Dataset utilities for EEG surrogate modeling."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import SubjectState


class BrainStateDataset(Dataset):
    """Sequence dataset that mixes samples from all subjects."""

    def __init__(
        self,
        subjects: List[SubjectState],
        history_length: int,
        prediction_horizon: int,
        stride: int = 1,
        return_subject_id: bool = False,
    ) -> None:
        self.subjects = subjects
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.return_subject_id = return_subject_id

        self.sample_index: List[Tuple[int, int]] = []
        for subj_idx, subj in enumerate(self.subjects):
            max_start = subj.states.shape[0] - (history_length + prediction_horizon)
            for start in range(0, max_start + 1, stride):
                self.sample_index.append((subj_idx, start))

    def __len__(self) -> int:  # noqa: D401
        """Total number of sliced samples across all subjects."""
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[str]]:
        subj_idx, start = self.sample_index[idx]
        subj = self.subjects[subj_idx]
        target_idx = start + self.history_length - 1 + self.prediction_horizon
        history = subj.states[start : start + self.history_length]
        target = subj.states[target_idx]
        x = torch.from_numpy(history).float()
        y = torch.from_numpy(target).float()
        if self.return_subject_id:
            return x, y, subj.subject_id
        return x, y, None


def create_datasets(
    subjects: List[SubjectState],
    config: Dict,
) -> Tuple[BrainStateDataset, BrainStateDataset]:
    """Convenience helper to split subjects and construct datasets."""
    from .preprocess import train_val_split

    train_subjects, val_subjects = train_val_split(
        subjects,
        val_ratio=config.get("val_subject_ratio", 0.2),
        shuffle=config.get("shuffle_subjects", True),
        seed=config.get("seed", None),
    )

    dataset_kwargs = dict(
        history_length=config["history_length"],
        prediction_horizon=config["prediction_horizon"],
        stride=config.get("stride", 1),
        return_subject_id=True,
    )
    train_dataset = BrainStateDataset(train_subjects, **dataset_kwargs)
    val_dataset = BrainStateDataset(val_subjects, **dataset_kwargs)
    return train_dataset, val_dataset
