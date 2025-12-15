"""Utility functions for EEG surrogate preprocessing."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import signal
from scipy.io import loadmat


def load_roi_timeseries(
    path: str | Path,
    n_rois: int,
    mat_variable: str | None = None,
) -> np.ndarray:
    """Load a ROI time series array.

    Supports ``.npy``, ``.npz`` and ``.mat`` files. ``.mat`` files can
    optionally specify a variable name via ``mat_variable``; otherwise the
    first non-private variable is used. The returned array is validated to have
    ROI dimension ``n_rois`` and will be transposed if the ROI dimension is the
    first axis.
    """
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if hasattr(data, "files") and len(data.files) > 0:
            return data[data.files[0]]
        raise ValueError(f"No arrays found in npz file: {path}")
    if path.suffix == ".mat":
        mat = loadmat(path)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        if mat_variable is not None:
            if mat_variable not in mat:
                raise KeyError(f"Variable '{mat_variable}' not found in {path}")
            arr = mat[mat_variable]
        elif keys:
            arr = mat[keys[0]]
        else:
            raise ValueError(f"No data arrays found in mat file: {path}")
        arr = np.array(arr).squeeze()
        if arr.ndim != 2:
            raise ValueError(
                f"Expected 2D array in mat file {path}, got shape {arr.shape}"
            )
        if arr.shape[1] == n_rois:
            return arr
        if arr.shape[0] == n_rois:
            return arr.T
        raise ValueError(
            f"MAT array shape {arr.shape} incompatible with n_rois={n_rois}"
        )
    return np.load(path)


def butter_bandpass_sos(low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    """Design a Butterworth bandpass filter in SOS form."""
    nyq = 0.5 * fs
    return signal.butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def apply_bandpass(x: np.ndarray, band: Tuple[float, float], fs: float, order: int = 4) -> np.ndarray:
    """Apply zero-phase bandpass filtering to the signal."""
    sos = butter_bandpass_sos(band[0], band[1], fs=fs, order=order)
    return signal.sosfiltfilt(sos, x, axis=0)


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """Compute the analytic amplitude envelope via Hilbert transform."""
    analytic = signal.hilbert(x, axis=0)
    return np.abs(analytic)


def resample_signal(x: np.ndarray, orig_fs: float, target_fs: float, method: str = "polyphase") -> np.ndarray:
    """Resample signal along time axis to target sampling rate."""
    if math.isclose(orig_fs, target_fs):
        return x
    if method == "polyphase":
        gcd = math.gcd(int(orig_fs), int(target_fs))
        up = int(target_fs / gcd)
        down = int(orig_fs / gcd)
        return signal.resample_poly(x, up=up, down=down, axis=0)
    # fallback to Fourier-based resampling
    num = int(round(x.shape[0] * target_fs / orig_fs))
    return signal.resample(x, num=num, axis=0)


def log_transform(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Log-transform envelope values for stabilizing variance."""
    return np.log(x + eps)


def zscore_subject(x: np.ndarray) -> np.ndarray:
    """Z-score each feature (ROI-band) across time."""
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True) + 1e-8
    return (x - mean) / std


def compute_envelopes(
    timeseries: np.ndarray,
    bands: Dict[str, Iterable[float]],
    raw_sampling_rate: float,
    target_sampling_rate: float,
    use_log: bool = True,
    use_zscore: bool = True,
    resample_method: str = "polyphase",
) -> np.ndarray:
    """Compute band-limited envelopes and resample to target rate.

    Args:
        timeseries: Raw ROI signals with shape ``(T, n_rois)``.
        bands: Mapping of band names to [low, high] frequency pairs.
        raw_sampling_rate: Original sampling rate of the data.
        target_sampling_rate: Desired sampling rate after envelope extraction.
        use_log: Whether to log-transform envelope values.
        use_zscore: Whether to z-score per feature after aggregation.
        resample_method: Resampling strategy ("polyphase" or "fft").

    Returns:
        Array of shape ``(T_resampled, n_rois * n_bands)`` representing
        the concatenated envelopes for each ROI and band.
    """
    envelopes: List[np.ndarray] = []
    for band_name, (low, high) in bands.items():
        filtered = apply_bandpass(timeseries, (low, high), fs=raw_sampling_rate)
        env = hilbert_envelope(filtered)
        env = resample_signal(env, orig_fs=raw_sampling_rate, target_fs=target_sampling_rate, method=resample_method)
        envelopes.append(env)

    stacked = np.concatenate(envelopes, axis=1)
    if use_log:
        stacked = log_transform(stacked)
    if use_zscore:
        stacked = zscore_subject(stacked)
    return stacked


def validate_timeseries_shape(timeseries: np.ndarray, n_rois: int) -> None:
    if timeseries.ndim != 2 or timeseries.shape[1] != n_rois:
        raise ValueError(
            f"Expected timeseries shape (T, {n_rois}), got {timeseries.shape}"
        )


def ensure_min_length(timeseries: np.ndarray, min_length: int) -> bool:
    """Return True if the sequence is long enough."""
    return timeseries.shape[0] >= min_length
