package com.pedometer.smart

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.pedometer.smart.data.SmartPedometerDatabase
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var accSensor: Sensor? = null
    private var gyroSensor: Sensor? = null

    private val engine = SmartPedometerEngine()

    private val appScope: CoroutineScope =
        CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    private val analyticsRepository by lazy {
        val dao = SmartPedometerDatabase.get(this).dao()
        AnalyticsRepository(dao, appScope)
    }

    private lateinit var textSteps: TextView
    private lateinit var textDistance: TextView
    private lateinit var textCalories: TextView
    private lateinit var textActivity: TextView
    private lateinit var textRecommendations: TextView

    private var lastGyroX: Double = 0.0
    private var lastGyroY: Double = 0.0
    private var lastGyroZ: Double = 0.0

    private val dailyGoalSteps: Long = 8000

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        textSteps = findViewById(R.id.textSteps)
        textDistance = findViewById(R.id.textDistance)
        textCalories = findViewById(R.id.textCalories)
        textActivity = findViewById(R.id.textActivity)
        textRecommendations = findViewById(R.id.textRecommendations)

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        accSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        observeSummaries()
    }

    override fun onResume() {
        super.onResume()
        accSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    override fun onPause() {
        super.onPause()
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return

        val timestampSeconds = event.timestamp / 1_000_000_000.0

        when (event.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> {
                lastGyroX = event.values[0].toDouble()
                lastGyroY = event.values[1].toDouble()
                lastGyroZ = event.values[2].toDouble()
            }
            Sensor.TYPE_ACCELEROMETER -> {
                val sample = SensorSample(
                    timestampSeconds = timestampSeconds,
                    accX = event.values[0].toDouble(),
                    accY = event.values[1].toDouble(),
                    accZ = event.values[2].toDouble(),
                    gyroX = lastGyroX,
                    gyroY = lastGyroY,
                    gyroZ = lastGyroZ,
                )

                val update = engine.onSensorSample(sample)
                if (update != null) {
                    // Update analytics and UI
                    analyticsRepository.onStepUpdate(update)
                    updateStepUi(update)
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // no-op
    }

    private fun observeSummaries() {
        lifecycleScope.launch {
            analyticsRepository.observeTodaySummary().collectLatest { summary ->
                if (summary != null) {
                    val km = summary.distanceMeters / 1000.0
                    textSteps.text = "Steps: ${summary.totalSteps}"
                    textDistance.text = String.format("Distance: %.2f km", km)
                    textCalories.text = String.format("Calories: %.0f kcal", summary.calories)

                    val recs = RecommendationEngine.buildRecommendations(summary, dailyGoalSteps)
                    textRecommendations.text = recs
                }
            }
        }
    }

    private fun updateStepUi(update: StepUpdate) {
        textActivity.text = "Current activity: ${update.activityType.name.lowercase().replaceFirstChar { it.uppercase() }}"
    }
}

