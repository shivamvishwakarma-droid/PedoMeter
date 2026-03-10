package com.pedometer.smart

import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Computes the same features as the Python training pipeline for a single window.
 *
 * This implementation is pure Kotlin and does not depend on any third-party
 * numeric libraries, so it can run on-device in a normal Android module.
 */
object FeatureExtractor {

    /**
     * Compute a FeatureVector from a list of SensorSample forming a time window.
     */
    fun extract(window: List<SensorSample>, samplingHz: Double): FeatureVector {
        if (window.isEmpty()) {
            return FeatureVector(DoubleArray(0))
        }

        val n = window.size

        val accX = DoubleArray(n)
        val accY = DoubleArray(n)
        val accZ = DoubleArray(n)
        val accMag = DoubleArray(n)

        val gyroX = DoubleArray(n)
        val gyroY = DoubleArray(n)
        val gyroZ = DoubleArray(n)
        val gyroMag = DoubleArray(n)

        for (i in 0 until n) {
            val s = window[i]
            accX[i] = s.accX
            accY[i] = s.accY
            accZ[i] = s.accZ
            accMag[i] = magnitude3(s.accX, s.accY, s.accZ)

            gyroX[i] = s.gyroX
            gyroY[i] = s.gyroY
            gyroZ[i] = s.gyroZ
            gyroMag[i] = magnitude3(s.gyroX, s.gyroY, s.gyZ)
        }

        val features = ArrayList<Double>()

        // 1. Time-domain stats for acc axes + magnitude
        listOf(accX, accY, accZ, accMag).forEach { arr ->
            features.addAll(basicStats(arr))
        }

        // 2. Time-domain stats for gyro axes + magnitude
        listOf(gyroX, gyroY, gyroZ, gyroMag).forEach { arr ->
            features.addAll(basicStats(arr))
        }

        // 3. Cross-axis correlations (accelerometer)
        features.add(correlation(accX, accY))
        features.add(correlation(accX, accZ))
        features.add(correlation(accY, accZ))

        // 4. Frequency-domain features on acceleration magnitude
        val fftFeatures = fftFeatures(accMag, samplingHz)
        features.addAll(fftFeatures)

        // 5. Peak-based cadence proxies
        val (peakCount, meanInterval) = detectPeaks(accMag, samplingHz)
        features.add(peakCount.toDouble())
        features.add(meanInterval)

        return FeatureVector(features.toDoubleArray())
    }

    private fun magnitude3(x: Double, y: Double, z: Double): Double {
        return sqrt(x * x + y * y + z * z)
    }

    /**
     * (mean, std, var, min, max, median)
     */
    private fun basicStats(values: DoubleArray): List<Double> {
        if (values.isEmpty()) {
            return listOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        }

        var sum = 0.0
        var sumSq = 0.0
        var vMin = Double.POSITIVE_INFINITY
        var vMax = Double.NEGATIVE_INFINITY

        for (v in values) {
            sum += v
            sumSq += v * v
            if (v < vMin) vMin = v
            if (v > vMax) vMax = v
        }

        val n = values.size.toDouble()
        val mean = sum / n
        val variance = max(0.0, sumSq / n - mean * mean)
        val std = sqrt(variance)

        val sorted = values.copyOf()
        sorted.sort()
        val median = if (sorted.size % 2 == 1) {
            sorted[sorted.size / 2]
        } else {
            val mid = sorted.size / 2
            (sorted[mid - 1] + sorted[mid]) / 2.0
        }

        return listOf(mean, std, variance, vMin, vMax, median)
    }

    /**
     * Pearson correlation between two equal-length arrays.
     */
    private fun correlation(a: DoubleArray, b: DoubleArray): Double {
        if (a.isEmpty() || b.isEmpty() || a.size != b.size) return 0.0

        val n = a.size
        var sumA = 0.0
        var sumB = 0.0
        var sumASq = 0.0
        var sumBSq = 0.0
        var sumAB = 0.0

        for (i in 0 until n) {
            val va = a[i]
            val vb = b[i]
            sumA += va
            sumB += vb
            sumASq += va * va
            sumBSq += vb * vb
            sumAB += va * vb
        }

        val meanA = sumA / n
        val meanB = sumB / n

        val cov = sumAB / n - meanA * meanB
        val varA = sumASq / n - meanA * meanA
        val varB = sumBSq / n - meanB * meanB

        val denom = sqrt(max(1e-12, varA) * max(1e-12, varB))
        return cov / denom
    }

    /**
     * FFT-based features approximating the Python implementation:
     *  (domFreq, domFreqPower, spectralEnergy, spectralEntropy).
     *
     * We use a simple radix-2 Cooley–Tukey FFT with zero-padding to the next
     * power of two.
     */
    private fun fftFeatures(values: DoubleArray, samplingHz: Double): List<Double> {
        if (values.isEmpty()) {
            return listOf(0.0, 0.0, 0.0, 0.0)
        }

        // Detrend by removing mean
        val mean = values.average()
        val detrended = DoubleArray(values.size) { i -> values[i] - mean }

        // Zero-pad to next power of two
        var n = 1
        while (n < detrended.size) {
            n = n shl 1
        }
        val re = DoubleArray(n)
        val im = DoubleArray(n)
        for (i in detrended.indices) {
            re[i] = detrended[i]
        }

        fftInPlace(re, im)

        // Only use non-negative frequencies (first n/2+1 bins)
        val half = n / 2
        val power = DoubleArray(half + 1)
        for (k in 0..half) {
            val r = re[k]
            val ii = im[k]
            power[k] = r * r + ii * ii
        }

        // Dominant frequency (excluding DC if possible)
        var domIdx = 0
        if (power.size > 1) {
            var maxVal = power[1]
            domIdx = 1
            for (k in 2 until power.size) {
                if (power[k] > maxVal) {
                    maxVal = power[k]
                    domIdx = k
                }
            }
        }

        val domFreq = domIdx * samplingHz / n.toDouble()
        val domFreqPower = power[domIdx]

        // Spectral energy and entropy
        var spectralEnergy = 0.0
        for (v in power) spectralEnergy += v
        val prob = DoubleArray(power.size)
        val eps = 1e-8
        val denom = spectralEnergy + eps
        for (i in power.indices) {
            prob[i] = power[i] / denom
        }

        var spectralEntropy = 0.0
        for (p in prob) {
            if (p > 0.0) {
                spectralEntropy -= p * ln(p)
            }
        }

        return listOf(domFreq, domFreqPower, spectralEnergy, spectralEntropy)
    }

    /**
     * In-place radix-2 Cooley–Tukey FFT.
     *
     * re and im are the real and imaginary parts of the input; on return they
     * contain the frequency-domain representation.
     */
    private fun fftInPlace(re: DoubleArray, im: DoubleArray) {
        val n = re.size
        if (n == 0) return

        // Bit-reversal permutation
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                val tmpRe = re[i]
                val tmpIm = im[i]
                re[i] = re[j]
                im[i] = im[j]
                re[j] = tmpRe
                im[j] = tmpIm
            }
        }

        // Cooley–Tukey
        var len = 2
        while (len <= n) {
            val ang = -2.0 * Math.PI / len
            val wLenRe = kotlin.math.cos(ang)
            val wLenIm = kotlin.math.sin(ang)
            var i = 0
            while (i < n) {
                var wRe = 1.0
                var wIm = 0.0
                for (k in 0 until len / 2) {
                    val uRe = re[i + k]
                    val uIm = im[i + k]
                    val vRe = re[i + k + len / 2]
                    val vIm = im[i + k + len / 2]

                    val tRe = vRe * wRe - vIm * wIm
                    val tIm = vRe * wIm + vIm * wRe

                    re[i + k] = uRe + tRe
                    im[i + k] = uIm + tIm
                    re[i + k + len / 2] = uRe - tRe
                    im[i + k + len / 2] = uIm - tIm

                    val nextWRe = wRe * wLenRe - wIm * wLenIm
                    val nextWIm = wRe * wLenIm + wIm * wLenRe
                    wRe = nextWRe
                    wIm = nextWIm
                }
                i += len
            }
            len = len shl 1
        }
    }

    /**
     * Crude peak detector for acceleration magnitude to approximate step cadence.
     */
    private fun detectPeaks(values: DoubleArray, samplingHz: Double): Pair<Int, Double> {
        if (values.isEmpty()) return 0 to 0.0

        val mean = values.average()
        val std = sqrt(values.map { (it - mean).pow(2.0) }.average())
        val threshold = mean + 0.5 * std

        val peaks = ArrayList<Int>()
        for (i in 1 until values.size - 1) {
            val v = values[i]
            if (v > threshold && v > values[i - 1] && v >= values[i + 1]) {
                peaks.add(i)
            }
        }

        if (peaks.size <= 1) return peaks.size to 0.0

        var sumIntervals = 0.0
        for (i in 1 until peaks.size) {
            sumIntervals += (peaks[i] - peaks[i - 1]).toDouble()
        }
        val meanIntervalSamples = sumIntervals / (peaks.size - 1).toDouble()
        val meanIntervalSeconds = meanIntervalSamples / samplingHz
        return peaks.size to meanIntervalSeconds
    }
}

