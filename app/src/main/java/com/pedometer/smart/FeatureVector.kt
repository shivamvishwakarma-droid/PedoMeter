package com.pedometer.smart

/**
 * Immutable container for a single feature vector produced from a sensor window.
 *
 * The order and length of [values] must exactly match the feature list defined
 * on the Python side (see ml/feature_spec.py).
 */
data class FeatureVector(
    val values: DoubleArray,
)

