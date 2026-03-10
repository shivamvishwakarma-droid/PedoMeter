package com.pedometer.smart

import com.pedometer.smart.data.DailySummaryEntity

/**
 * Simple rule-based recommendation generator based on recent activity.
 */
object RecommendationEngine {

    fun buildRecommendations(today: DailySummaryEntity?, dailyGoalSteps: Long): String {
        if (today == null) {
            return "Start walking with your phone in your pocket to see smart insights here."
        }

        val steps = today.totalSteps
        val distanceKm = today.distanceMeters / 1000.0

        val lines = mutableListOf<String>()

        if (steps < dailyGoalSteps) {
            val remaining = dailyGoalSteps - steps
            lines += "You are $remaining steps away from your daily goal of $dailyGoalSteps."
            lines += "Try a short 10–15 minute walk to close the gap."
        } else {
            lines += "Great job! You reached your daily goal of $dailyGoalSteps steps."
        }

        if (distanceKm < 3.0) {
            lines += "Aim for at least 3 km of movement today for better health."
        }

        return lines.joinToString(separator = "\n")
    }
}

