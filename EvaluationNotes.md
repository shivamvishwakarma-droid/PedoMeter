## Smart AI Pedometer – Evaluation Checklist

Use this checklist when testing the app on a real device.

- **Scenarios to record**
  - Normal walking (at least 5–10 minutes)
  - Brisk walking / light running
  - Cycling
  - Sitting/standing while using the phone (typing, scrolling)
  - Sitting in a car / bus (driving / passenger)
  - Intentional phone shaking

- **What to compare**
  - Count your steps manually over short intervals and compare to:
    - The app’s step count (dashboard)
    - Any baseline pedometer app you trust (optional)
  - Note false positives:
    - Steps counted while driving or sitting
    - Steps counted during phone shaking but not walking
  - Watch the detected activity label in real time.

- **Metrics to track**
  - Step detection accuracy = 1 – |app_steps – true_steps| / true_steps
  - Activity classification accuracy = correct_activity_seconds / total_seconds
  - False positive rate during non-walking activities (driving, sitting)
  - Battery impact over at least 1–2 hours of continuous use.

- **Tuning suggestions**
  - If steps are under-counted:
    - Reduce the peak detection threshold in `FeatureExtractor.detectPeaks`.
  - If steps are over-counted (especially while sitting/driving):
    - Increase `SmartPedometerEngine` confidence threshold.
    - Retrain models with more negative examples for those activities.
  - If activity labels flicker:
    - Increase `smoothingWindowSize` in `SmartPedometerEngine`.

