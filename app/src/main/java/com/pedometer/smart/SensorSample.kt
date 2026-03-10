package com.pedometer.smart

/**
 * Single fused sensor reading used by the smart pedometer pipeline.
 *
 * This mirrors the schema expected by the Python training pipeline:
 *  - timestamp: seconds since epoch (or any monotonic reference)
 *  - accX/Y/Z: linear acceleration in m/s^2
 *  - gyroX/Y/Z: angular velocity in rad/s
 */
data class SensorSample(
    val timestampSeconds: Double,
    val accX: Double,
    val accY: Double,
    val accZ: Double,
    val gyroX: Double,
    val gyroY: Double,
    val gyroZ: Double,
)

