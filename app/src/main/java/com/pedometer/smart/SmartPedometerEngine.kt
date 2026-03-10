package com.pedometer.smart

import java.util.ArrayDeque
import kotlin.math.max

/**
 * High-level engine that:
 *  - consumes sensor samples,
 *  - performs windowing and feature extraction,
 *  - calls the ActivityModel for predictions,
 *  - exposes smoothed activity and step events to callers.
 */
class SmartPedometerEngine(
    private val model: ActivityModel = HeuristicActivityModel(),
    private val windowSeconds: Double = 3.0,
    private val overlapFraction: Double = 0.5,
    private val samplingHz: Double = 50.0,
    private val smoothingWindowSize: Int = 5,
) {

    private val buffer = SensorWindowBuffer(
        windowSeconds = windowSeconds,
        overlapFraction = overlapFraction,
        targetSamplingHz = samplingHz,
    )

    private val recentPredictions: ArrayDeque<PredictionResult> = ArrayDeque()

    private var totalSteps: Long = 0L
    private var currentActivity: ActivityType = ActivityType.UNKNOWN

    /**
     * Called whenever a new fused SensorSample is available.
     *
     * Returns an optional StepUpdate which reports the latest
     * step count and current activity if anything changed.
     */
    fun onSensorSample(sample: SensorSample): StepUpdate? {
        buffer.addSample(sample)

        var stepDelta = 0L
        var activityChanged = false

        val windows = buffer.drainWindows()
        for (w in windows) {
            val feat = FeatureExtractor.extract(w, samplingHz)
            val rawPred = model.predict(feat)
            val smoothed = smoothPrediction(rawPred)

            if (shouldCountStep(smoothed)) {
                totalSteps += 1
                stepDelta += 1
            }

            val newActivity = smoothed.activityType
            if (newActivity != currentActivity) {
                currentActivity = newActivity
                activityChanged = true
            }
        }

        return if (stepDelta > 0L || activityChanged) {
            StepUpdate(
                totalSteps = totalSteps,
                activityType = currentActivity,
            )
        } else {
            null
        }
    }

    private fun smoothPrediction(pred: PredictionResult): PredictionResult {
        recentPredictions.addLast(pred)
        while (recentPredictions.size > smoothingWindowSize) {
            recentPredictions.removeFirst()
        }

        if (recentPredictions.isEmpty()) return pred

        // Majority vote on activity; average isValidStep and confidence
        val counts = mutableMapOf<ActivityType, Int>()
        var stepScoreSum = 0.0
        var confSum = 0.0

        for (p in recentPredictions) {
            counts[p.activityType] = (counts[p.activityType] ?: 0) + 1
            stepScoreSum += if (p.isValidStep) 1.0 else 0.0
            confSum += p.confidence
        }

        val majorityActivity = counts.maxByOrNull { it.value }?.key ?: pred.activityType
        val avgStepScore = stepScoreSum / recentPredictions.size.toDouble()
        val avgConfidence = confSum / recentPredictions.size.toDouble()

        val isValidStep = avgStepScore >= 0.5
        val confidence = max(0.0, minOf(1.0, avgConfidence))

        return PredictionResult(
            activityType = majorityActivity,
            isValidStep = isValidStep,
            confidence = confidence,
        )
    }

    private fun shouldCountStep(pred: PredictionResult): Boolean {
        if (!pred.isValidStep) return false
        if (pred.confidence < 0.5) return false

        return when (pred.activityType) {
            ActivityType.WALKING,
            ActivityType.RUNNING -> true
            else -> false
        }
    }
}

/**
 * Lightweight DTO returned to UI / higher layers when there is a
 * change in steps or dominant activity.
 */
data class StepUpdate(
    val totalSteps: Long,
    val activityType: ActivityType,
)

