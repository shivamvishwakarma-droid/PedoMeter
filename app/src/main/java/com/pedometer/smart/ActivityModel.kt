package com.pedometer.smart

/**
 * Interface for any on-device activity/step-validity model.
 *
 * A concrete implementation could wrap a TensorFlow Lite, ONNX, or
 * custom RandomForest/GradientBoosting evaluator. This keeps the
 * SmartPedometerEngine independent of the underlying ML library.
 */
fun interface ActivityModel {
    fun predict(features: FeatureVector): PredictionResult
}

/**
 * Simple rule-based fallback implementation that can be used until
 * a trained model is integrated. It uses crude thresholds on peak
 * counts and variance to produce a reasonable baseline.
 */
class HeuristicActivityModel : ActivityModel {

    override fun predict(features: FeatureVector): PredictionResult {
        val vals = features.values
        if (vals.isEmpty()) {
            return PredictionResult(ActivityType.UNKNOWN, isValidStep = false, confidence = 0.0)
        }

        // By design, the last two features are:
        //   acc_mag_peak_count, acc_mag_peak_mean_interval
        val peakCount = vals.getOrNull(vals.size - 2) ?: 0.0
        val meanInterval = vals.getOrNull(vals.size - 1) ?: 0.0

        val approxCadence = if (meanInterval > 1e-3) 60.0 / meanInterval else 0.0

        val activity: ActivityType
        val isValidStep: Boolean
        val confidence: Double

        if (peakCount < 1.0 || approxCadence < 20.0) {
            activity = ActivityType.SITTING
            isValidStep = false
            confidence = 0.6
        } else if (approxCadence < 130.0) {
            activity = ActivityType.WALKING
            isValidStep = true
            confidence = 0.7
        } else {
            activity = ActivityType.RUNNING
            isValidStep = true
            confidence = 0.7
        }

        return PredictionResult(activity, isValidStep, confidence)
    }
}

