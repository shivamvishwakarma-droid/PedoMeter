"""
Offline training pipeline for the Smart AI Pedometer.

This module is responsible for:
- loading labeled sensor datasets,
- computing windowed features as defined in feature_spec.py,
- training classical ML models for:
    * activity classification (multi-class),
    * step-validity detection (binary or multi-class),
- exporting the trained models in a form suitable for on-device inference.

NOTE: This is a template pipeline. You still need to provide one or more
CSV/Parquet datasets with columns matching the expected schema, and
invoke the `train_*` functions from a small CLI or notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from .feature_spec import (
    ActivityLabel,
    get_feature_names,
    get_window_params,
)


@dataclass
class WindowedExample:
    """One window of features with its activity label and step-validity flag."""

    features: np.ndarray
    activity_label: int
    is_valid_step: int  # 1 for step-like movement, 0 for non-step (e.g., driving)


def _compute_basic_stats(values: np.ndarray) -> Tuple[float, ...]:
    """Return (mean, std, var, min, max, median) for a 1D array."""

    if values.size == 0:
        return (0.0,) * 6

    mean = float(np.mean(values))
    std = float(np.std(values))
    var = float(np.var(values))
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    median = float(np.median(values))
    return mean, std, var, vmin, vmax, median


def _compute_fft_features(values: np.ndarray, sampling_hz: float) -> Tuple[float, float, float, float]:
    """Compute simple FFT-based features: dominant frequency, its power, spectral energy, entropy."""

    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Remove DC component
    detrended = signal.detrend(values)
    freqs = np.fft.rfftfreq(detrended.size, d=1.0 / sampling_hz)
    spectrum = np.fft.rfft(detrended)
    power = np.abs(spectrum) ** 2

    # Dominant frequency (excluding 0 Hz)
    if freqs.size > 1:
        dom_idx = np.argmax(power[1:]) + 1
    else:
        dom_idx = 0
    dom_freq = float(freqs[dom_idx])
    dom_freq_power = float(power[dom_idx])

    # Spectral energy and entropy
    spectral_energy = float(np.sum(power))
    prob = power / (spectral_energy + 1e-8)
    spectral_entropy = float(stats.entropy(prob + 1e-12))
    return dom_freq, dom_freq_power, spectral_energy, spectral_entropy


def _detect_peaks(values: np.ndarray, sampling_hz: float) -> Tuple[int, float]:
    """Crude peak detection to approximate step cadence based on acceleration magnitude."""

    if values.size == 0:
        return 0, 0.0

    # Use a simple prominence-based peak detector
    peaks, _ = signal.find_peaks(values, prominence=np.std(values) * 0.5)
    peak_count = len(peaks)
    if peak_count <= 1:
        return peak_count, 0.0

    # Average interval between consecutive peaks in seconds
    intervals_samples = np.diff(peaks)
    mean_interval_samples = float(np.mean(intervals_samples))
    mean_interval_seconds = mean_interval_samples / sampling_hz
    return peak_count, mean_interval_seconds


def window_sensor_dataframe(
    df: pd.DataFrame,
    label_col: str,
    step_valid_col: str,
    timestamp_col: str = "timestamp",
) -> List[WindowedExample]:
    """Convert a long sensor dataframe into a list of windowed, featureized examples.

    Expected dataframe columns (at minimum):
        timestamp (float or int, seconds or ms),
        acc_x, acc_y, acc_z,
        gyro_x, gyro_y, gyro_z,
        label_col (int or str activity label),
        step_valid_col (0/1 for step validity).
    """

    win_secs, overlap, target_hz = get_window_params()
    win_size = int(win_secs * target_hz)
    hop_size = int(win_size * (1.0 - overlap))
    if hop_size <= 0:
        hop_size = 1

    # Sort by timestamp to ensure temporal order
    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)

    acc_xyz = df_sorted[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=float)
    gyro_xyz = df_sorted[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=float)
    labels = df_sorted[label_col].to_numpy()
    step_valid = df_sorted[step_valid_col].to_numpy()

    examples: List[WindowedExample] = []

    for start in range(0, len(df_sorted) - win_size + 1, hop_size):
        end = start + win_size

        window_acc = acc_xyz[start:end]
        window_gyro = gyro_xyz[start:end]
        window_labels = labels[start:end]
        window_step_valid = step_valid[start:end]

        # Majority label in the window
        maj_label = int(stats.mode(window_labels, keepdims=True)[0][0])
        maj_step_valid = int(round(float(np.mean(window_step_valid) >= 0.5)))

        # Build feature vector in the same order as feature_spec.get_feature_names()
        acc_x = window_acc[:, 0]
        acc_y = window_acc[:, 1]
        acc_z = window_acc[:, 2]
        acc_mag = np.linalg.norm(window_acc, axis=1)

        gyro_x = window_gyro[:, 0]
        gyro_y = window_gyro[:, 1]
        gyro_z = window_gyro[:, 2]
        gyro_mag = np.linalg.norm(window_gyro, axis=1)

        feats: List[float] = []

        # 1. Time-domain stats for acc axes + magnitude
        for arr in (acc_x, acc_y, acc_z, acc_mag):
            feats.extend(_compute_basic_stats(arr))

        # 2. Time-domain stats for gyro axes + magnitude
        for arr in (gyro_x, gyro_y, gyro_z, gyro_mag):
            feats.extend(_compute_basic_stats(arr))

        # 3. Cross-axis correlations (accelerometer)
        for a, b in ((acc_x, acc_y), (acc_x, acc_z), (acc_y, acc_z)):
            if a.size == 0:
                feats.append(0.0)
            else:
                feats.append(float(np.corrcoef(a, b)[0, 1]))

        # 4. Frequency domain features on acceleration magnitude
        feats.extend(_compute_fft_features(acc_mag, target_hz))

        # 5. Peak-based cadence proxies
        peak_count, mean_interval = _detect_peaks(acc_mag, target_hz)
        feats.append(float(peak_count))
        feats.append(float(mean_interval))

        examples.append(
            WindowedExample(
                features=np.asarray(feats, dtype=np.float32),
                activity_label=maj_label,
                is_valid_step=maj_step_valid,
            )
        )

    return examples


def build_feature_matrix(
    examples: Iterable[WindowedExample],
    target: str = "activity",
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a list of WindowedExample objects into (X, y) for sklearn."""

    feats = [ex.features for ex in examples]
    X = np.stack(feats, axis=0)

    if target == "activity":
        y = np.array([ex.activity_label for ex in examples], dtype=int)
    elif target == "step_valid":
        y = np.array([ex.is_valid_step for ex in examples], dtype=int)
    else:
        raise ValueError(f"Unknown target: {target}")

    return X, y


def train_activity_model(
    examples: List[WindowedExample],
    model_dir: Path,
) -> Path:
    """Train a multi-class activity classifier and save it to disk."""

    X, y = build_feature_matrix(examples, target="activity")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("Activity classification report:")
    print(classification_report(y_val, y_pred))

    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / "activity_model.joblib"
    joblib.dump(clf, out_path)
    return out_path


def train_step_valid_model(
    examples: List[WindowedExample],
    model_dir: Path,
) -> Path:
    """Train a binary classifier distinguishing step-like vs non-step movement."""

    X, y = build_feature_matrix(examples, target="step_valid")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("Step-validity classification report:")
    print(classification_report(y_val, y_pred))

    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / "step_valid_model.joblib"
    joblib.dump(clf, out_path)
    return out_path


def load_and_window_dataset(
    csv_paths: List[Path],
    label_col: str,
    step_valid_col: str,
) -> List[WindowedExample]:
    """Load one or more CSV files and convert them to windowed examples."""

    all_examples: List[WindowedExample] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        examples = window_sensor_dataframe(
            df,
            label_col=label_col,
            step_valid_col=step_valid_col,
        )
        all_examples.extend(examples)

    return all_examples


def train_all_models(
    data_paths: List[str],
    label_col: str,
    step_valid_col: str,
    output_dir: str = "ml_models",
) -> None:
    """End-to-end training entry point.

    Example usage:

        from pathlib import Path
        from ml.pipeline import train_all_models

        train_all_models(
            data_paths=[\"/path/to/dataset1.csv\", \"/path/to/dataset2.csv\"],
            label_col=\"activity_label\",
            step_valid_col=\"is_valid_step\",
            output_dir=\"models\",
        )
    """

    csv_paths = [Path(p) for p in data_paths]
    examples = load_and_window_dataset(csv_paths, label_col, step_valid_col)

    model_dir = Path(output_dir)
    activity_model_path = train_activity_model(examples, model_dir)
    step_valid_model_path = train_step_valid_model(examples, model_dir)

    print(f"Saved activity model to: {activity_model_path}")
    print(f"Saved step-validity model to: {step_valid_model_path}")

