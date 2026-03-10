"""
Shared sensor windowing and feature specification for the Smart AI Pedometer.

This module defines the canonical configuration that BOTH:
- the offline Python training pipeline, and
- the on-device Android/Kotlin implementation
must follow to stay in sync.

If you change any constant here, you must also update the Android side.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class ActivityLabel(int, Enum):
    """Canonical numeric IDs for activity classes.

    These IDs must match the labels used when training models and
    the mapping used on-device in the Android app.
    """

    UNKNOWN = 0
    WALKING = 1
    RUNNING = 2
    CYCLING = 3
    SITTING = 4
    STANDING = 5
    DRIVING = 6
    PHONE_SHAKING = 7


# Windowing configuration

# Target duration of each sensor window in seconds.
WINDOW_SECONDS: float = 3.0

# Fractional overlap between consecutive windows (e.g. 0.5 = 50% overlap).
WINDOW_OVERLAP: float = 0.5

# Expected sampling rate in Hz for accelerometer/gyroscope.
# The actual device rate will vary; the pipeline should be reasonably
# robust to small deviations, but this is the design target.
TARGET_SAMPLING_HZ: float = 50.0


@dataclass(frozen=True)
class FeatureDef:
    """Definition of a single scalar feature in the model input vector."""

    name: str
    description: str


def get_feature_spec() -> List[FeatureDef]:
    """Return the ordered list of features used for model training/inference.

    The order of this list defines the layout of the feature vector that is
    fed into the ML models. The Android implementation must produce features
    in this exact order.
    """

    features: List[FeatureDef] = []

    # 1. Time-domain statistics per axis and magnitude for accelerometer
    for signal in ("acc_x", "acc_y", "acc_z", "acc_mag"):
        for stat in ("mean", "std", "var", "min", "max", "median"):
            features.append(
                FeatureDef(
                    name=f"{signal}_{stat}",
                    description=f"{stat} of {signal} over the window",
                )
            )

    # 2. Time-domain statistics for gyroscope (angular velocity)
    for signal in ("gyro_x", "gyro_y", "gyro_z", "gyro_mag"):
        for stat in ("mean", "std", "var", "min", "max", "median"):
            features.append(
                FeatureDef(
                    name=f"{signal}_{stat}",
                    description=f"{stat} of {signal} over the window",
                )
            )

    # 3. Cross-axis correlations (capturing gait patterns and orientation)
    for a, b in (("acc_x", "acc_y"), ("acc_x", "acc_z"), ("acc_y", "acc_z")):
        features.append(
            FeatureDef(
                name=f"{a}_{b}_corr",
                description=f"Pearson correlation between {a} and {b}",
            )
        )

    # 4. Frequency-domain features for acceleration magnitude
    # (computed via FFT on acc_mag)
    for stat in ("dom_freq", "dom_freq_power", "spectral_energy", "spectral_entropy"):
        features.append(
            FeatureDef(
                name=f"acc_mag_{stat}",
                description=f"{stat} of acceleration magnitude spectrum",
            )
        )

    # 5. Simple step-cadence proxies (peaks in acceleration magnitude)
    features.append(
        FeatureDef(
            name="acc_mag_peak_count",
            description="Number of significant peaks in acc_mag (step-like events)",
        )
    )
    features.append(
        FeatureDef(
            name="acc_mag_peak_mean_interval",
            description="Mean time between peaks in acc_mag (inverse of cadence)",
        )
    )

    return features


def get_feature_names() -> List[str]:
    """Convenience helper used by training code to build feature matrices."""

    return [f.name for f in get_feature_spec()]


def get_window_params() -> Tuple[float, float, float]:
    """Return (window_seconds, window_overlap, target_sampling_hz)."""

    return WINDOW_SECONDS, WINDOW_OVERLAP, TARGET_SAMPLING_HZ

