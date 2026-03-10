package com.pedometer.smart.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverters

@Database(
    entities = [
        DailySummaryEntity::class,
        ActivityLogEntity::class,
    ],
    version = 1,
)
@TypeConverters(TypeConvertersActivityType::class)
abstract class SmartPedometerDatabase : RoomDatabase() {

    abstract fun dao(): SmartPedometerDao

    companion object {
        @Volatile
        private var INSTANCE: SmartPedometerDatabase? = null

        fun get(context: Context): SmartPedometerDatabase {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    SmartPedometerDatabase::class.java,
                    "smart_pedometer.db",
                ).build().also { INSTANCE = it }
            }
        }
    }
}

