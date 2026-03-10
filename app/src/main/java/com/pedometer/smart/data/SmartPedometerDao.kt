package com.pedometer.smart.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface SmartPedometerDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsertDailySummary(entity: DailySummaryEntity)

    @Insert
    suspend fun insertActivityLog(entity: ActivityLogEntity)

    @Query("SELECT * FROM daily_summaries WHERE date = :date LIMIT 1")
    fun observeDailySummary(date: String): Flow<DailySummaryEntity?>

    @Query("SELECT * FROM daily_summaries ORDER BY date DESC LIMIT :limit")
    fun observeRecentDailySummaries(limit: Int): Flow<List<DailySummaryEntity>>
}

