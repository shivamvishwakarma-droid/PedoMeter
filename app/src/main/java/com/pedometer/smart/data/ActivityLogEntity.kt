package com.pedometer.smart.data

import androidx.room.Entity
import androidx.room.PrimaryKey
import com.pedometer.smart.ActivityType

@Entity(tableName = "activity_logs")
data class ActivityLogEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val timestampMillis: Long,
    val activityType: ActivityType,
    val stepsTotal: Long,
)

