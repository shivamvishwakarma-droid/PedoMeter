package com.pedometer.smart

import com.pedometer.smart.data.ActivityLogEntity
import com.pedometer.smart.data.DailySummaryEntity
import com.pedometer.smart.data.SmartPedometerDao
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.launch
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId

/**
 * Aggregates SmartPedometerEngine outputs into daily summaries
 * and exposes reactive streams for the UI.
 */
class AnalyticsRepository(
    private val dao: SmartPedometerDao,
    private val coroutineScope: CoroutineScope,
    private val userStrideMeters: Double = 0.78, // average adult stride
    private val userWeightKg: Double = 70.0,
) {

    fun observeTodaySummary(): Flow<DailySummaryEntity?> {
        val today = LocalDate.now().toString()
        return dao.observeDailySummary(today)
    }

    fun observeRecentSummaries(limit: Int = 7): Flow<List<DailySummaryEntity>> {
        return dao.observeRecentDailySummaries(limit)
    }

    /**
     * Called whenever the engine produces a new StepUpdate.
     * This updates both the activity log and the per-day aggregates.
     */
    fun onStepUpdate(update: StepUpdate) {
        val now = Instant.now()
        val localDate = LocalDate.ofInstant(now, ZoneId.systemDefault()).toString()
        val millis = now.toEpochMilli()

        coroutineScope.launch(Dispatchers.IO) {
            dao.insertActivityLog(
                ActivityLogEntity(
                    timestampMillis = millis,
                    activityType = update.activityType,
                    stepsTotal = update.totalSteps,
                )
            )

            // For v1 we only keep simple aggregates; more detailed duration
            // tracking can be added later based on ActivityLog deltas.
            val totalSteps = update.totalSteps
            val distance = totalSteps * userStrideMeters
            val calories = estimateCalories(distanceMeters = distance)

            val summary = DailySummaryEntity(
                date = localDate,
                totalSteps = totalSteps,
                distanceMeters = distance,
                calories = calories,
                walkingMinutes = 0,
                runningMinutes = 0,
                cyclingMinutes = 0,
                sittingMinutes = 0,
                standingMinutes = 0,
                drivingMinutes = 0,
            )

            dao.upsertDailySummary(summary)
        }
    }

    private fun estimateCalories(distanceMeters: Double): Double {
        val distanceKm = distanceMeters / 1000.0
        val met = 3.5 // light-to-moderate walking
        val hours = distanceKm / 4.0 // assume 4 km/h
        return met * 3.5 * userWeightKg * hours / 200.0
    }
}

