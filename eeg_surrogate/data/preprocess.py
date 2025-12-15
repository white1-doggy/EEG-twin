"""Preprocessing routines for building ROI-level brain state sequences."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .utils import (
    compute_envelopes,
    ensure_min_length,
    load_roi_timeseries,
    validate_timeseries_shape,
)


class SubjectState:
    """Container for a subject's processed state sequence."""

    def __init__(self, subject_id: str, states: np.ndarray):
        self.subject_id = subject_id
        self.states = states

    def __len__(self) -> int:  # noqa: D401
        """Return the number of time points in the state sequence."""
        return self.states.shape[0]


def preprocess_subject(
    file_path: Path,
    config: Dict,
) -> SubjectState | None:
    """Load, validate, and preprocess a single subject time series.

    Returns ``None`` when the sequence is shorter than ``min_sequence_length``.
    """
    timeseries = load_roi_timeseries(file_path)
    validate_timeseries_shape(timeseries, n_rois=config["n_rois"])
    if not ensure_min_length(timeseries, config.get("min_sequence_length", 0)):
        return None

    states = compute_envelopes(
        timeseries=timeseries,
        bands=config["bands"],
        raw_sampling_rate=config["raw_sampling_rate"],
        target_sampling_rate=config["target_sampling_rate"],
        use_log=config.get("use_log_envelope", True),
        use_zscore=config.get("use_zscore", True),
        resample_method=config.get("resample_method", "polyphase"),
    )
    subject_id = file_path.stem
    return SubjectState(subject_id=subject_id, states=states)


def preprocess_directory(data_root: str | Path, config: Dict) -> List[SubjectState]:
    """Preprocess all subject files in a directory.

    The function scans for ``*.npy`` and ``*.npz`` files and aggregates
    processed state sequences.
    """
    data_root = Path(data_root)
    files = sorted(list(data_root.glob("*.npy")) + list(data_root.glob("*.npz")))
    subjects: List[SubjectState] = []
    for path in files:
        processed = preprocess_subject(path, config=config)
        if processed is not None:
            subjects.append(processed)
    return subjects


def save_state_summary(subjects: List[SubjectState], output_path: str | Path) -> None:
    """Persist a summary of processed subjects and sequence lengths."""
    summary = {s.subject_id: len(s) for s in subjects}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2)


def train_val_split(
    subjects: List[SubjectState],
    val_ratio: float = 0.2,
    shuffle: bool = True,
    seed: int | None = None,
) -> Tuple[List[SubjectState], List[SubjectState]]:
    """Split subjects into train/validation sets at the subject level."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(subjects))
    if shuffle:
        rng.shuffle(indices)
    cutoff = int(len(indices) * (1 - val_ratio))
    train_idx, val_idx = indices[:cutoff], indices[cutoff:]
    train = [subjects[i] for i in train_idx]
    val = [subjects[i] for i in val_idx]
    return train, val
