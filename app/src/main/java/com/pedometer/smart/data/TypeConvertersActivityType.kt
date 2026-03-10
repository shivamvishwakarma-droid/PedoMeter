package com.pedometer.smart.data

import androidx.room.TypeConverter
import com.pedometer.smart.ActivityType

class TypeConvertersActivityType {

    @TypeConverter
    fun fromActivityType(type: ActivityType?): Int? = type?.id

    @TypeConverter
    fun toActivityType(id: Int?): ActivityType? =
        id?.let { value -> ActivityType.values().firstOrNull { it.id == value } }
}

