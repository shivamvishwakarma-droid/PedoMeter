package com.pedometer.smart.data

import androidx.room.Entity
import androidx.room.PrimaryKey
import com.pedometer.smart.ActivityType

@Entity(tableName = "daily_summaries")
data class DailySummaryEntity(
    @PrimaryKey val date: String, // ISO-8601 local date, e.g. 2026-03-10
    val totalSteps: Long,
    val distanceMeters: Double,
    val calories: Double,
    val walkingMinutes: Int,
    val runningMinutes: Int,
    val cyclingMinutes: Int,
    val sittingMinutes: Int,
    val standingMinutes: Int,
    val drivingMinutes: Int,
)

