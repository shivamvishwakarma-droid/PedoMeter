package com.pedometer.smart

/**
 * Output of the ML inference layer for a single feature vector.
 */
data class PredictionResult(
    val activityType: ActivityType,
    val isValidStep: Boolean,
    val confidence: Double,
)

