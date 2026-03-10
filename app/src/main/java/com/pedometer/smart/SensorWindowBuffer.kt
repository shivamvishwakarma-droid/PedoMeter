package com.pedometer.smart

import kotlin.math.max

/**
 * Buffers incoming sensor samples into fixed-size, overlapping windows.
 *
 * The windowing configuration (windowSeconds, overlapFraction, targetSamplingHz)
 * must match the values used in the Python training pipeline.
 */
class SensorWindowBuffer(
    private val windowSeconds: Double,
    private val overlapFraction: Double,
    private val targetSamplingHz: Double,
) {

    private val samples: MutableList<SensorSample> = ArrayList()

    // Computed sizes in samples
    private val windowSizeSamples: Int =
        max(1, (windowSeconds * targetSamplingHz).toInt())
    private val hopSizeSamples: Int =
        max(1, (windowSizeSamples * (1.0 - overlapFraction)).toInt())

    /**
     * Add a new sample to the buffer.
     */
    fun addSample(sample: SensorSample) {
        samples.add(sample)
    }

    /**
     * Extract as many complete windows as are currently available.
     *
     * Returned windows are disjoint views (copies) of the underlying samples.
     * After windows are emitted, the consumed prefix is dropped to keep memory bounded.
     */
    fun drainWindows(): List<List<SensorSample>> {
        val result: MutableList<List<SensorSample>> = ArrayList()
        var start = 0

        while (start + windowSizeSamples <= samples.size) {
            val end = start + windowSizeSamples
            val window = samples.subList(start, end).map { it }
            result.add(window)
            start += hopSizeSamples
        }

        // Drop the consumed prefix, keep tail that may form the next window
        if (start > 0 && start < samples.size) {
            val tail = samples.subList(start, samples.size).map { it }
            samples.clear()
            samples.addAll(tail)
        } else if (start >= samples.size) {
            samples.clear()
        }

        return result
    }
}

