package com.pedometer.smart

/**
 * Activity categories recognized by the smart pedometer.
 *
 * IDs should stay in sync with ActivityLabel in the Python code.
 */
enum class ActivityType(val id: Int) {
    UNKNOWN(0),
    WALKING(1),
    RUNNING(2),
    CYCLING(3),
    SITTING(4),
    STANDING(5),
    DRIVING(6),
    PHONE_SHAKING(7),
}

