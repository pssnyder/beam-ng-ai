# Telemetry Logging System

**Date**: January 3, 2026
**Purpose**: Detailed step-by-step data collection for threshold analysis

---

## Overview

Added comprehensive telemetry logging to capture every training step's data, enabling analysis of crash/stuck detection thresholds and vehicle behavior patterns.

## File Structure

```
beam-ng-ai/
├── training_logs/           # CSV and text logs
│   ├── training_session_YYYYMMDD_HHMMSS.csv    # Episode summaries
│   ├── telemetry_YYYYMMDD_HHMMSS.csv           # Step-by-step telemetry (NEW)
│   └── training_log_YYYYMMDD_HHMMSS.txt        # Text logs
├── models/                  # Model checkpoints (moved from root)
│   ├── highway_best.pth                        # Best distance model
│   ├── highway_best_reward.pth                 # Best reward model
│   ├── highway_checkpoint_ep10.pth             # Periodic checkpoints
│   ├── highway_final.pth                       # Final model after training
│   └── highway_interrupted.pth                 # Saved on Ctrl+C
```

## Telemetry CSV Columns

**File**: `training_logs/telemetry_YYYYMMDD_HHMMSS.csv`

| Column | Description | Usage |
|--------|-------------|-------|
| **Episode** | Episode number | Group steps by episode |
| **Step** | Step within episode | Track progression |
| **Timestamp** | Exact timestamp (ms precision) | Timeline analysis |
| **Speed_ms** | Vehicle speed (m/s) | Speed patterns |
| **Throttle** | Throttle input [0-1] | AI control behavior |
| **Steering** | Steering input [-1, 1] | AI control behavior |
| **Brake** | Brake input [0-1] | AI control behavior |
| **Damage** | Current damage value | Crash severity |
| **Damage_Increase** | Δ damage this step | Crash detection threshold |
| **Total_Damage** | Cumulative damage | Wreck detection |
| **Gx, Gy, Gz** | G-forces (x, y, z) | Orientation/flip detection |
| **Distance_From_Origin** | Total distance traveled | Progress tracking |
| **Distance_From_Checkpoint** | Distance since last checkpoint | Episode progress |
| **Position_X/Y/Z** | World coordinates | Position tracking |
| **Position_Delta** | Movement this step (m) | Stuck detection threshold |
| **Stationary_Timer** | Time not moving (s) | Stuck timeout analysis |
| **Crash_Detected** | Boolean crash flag | Crash events |
| **Stationary_Timeout** | Boolean timeout flag | Stuck events |
| **Flipped** | Boolean flip flag | Orientation events |
| **Reward** | Reward this step | Reward engineering |
| **Event_Type** | CRASH/STUCK/SPEEDING | Event categorization |

## Usage Examples

### Analyze Crash Thresholds

```python
import pandas as pd

# Load telemetry
df = pd.read_csv('training_logs/telemetry_20260103_195000.csv')

# Find all crashes
crashes = df[df['Crash_Detected'] == True]

# Analyze damage increase distribution at crash
print("Damage increase stats at crash:")
print(crashes['Damage_Increase'].describe())

# Check if threshold (0.15) is appropriate
print(f"Min damage at crash: {crashes['Damage_Increase'].min()}")
print(f"Max damage at crash: {crashes['Damage_Increase'].max()}")
print(f"Median damage at crash: {crashes['Damage_Increase'].median()}")

# Find false negatives (high damage but no crash detected)
high_damage_no_crash = df[(df['Damage_Increase'] > 0.1) & (df['Crash_Detected'] == False)]
print(f"Potential missed crashes: {len(high_damage_no_crash)}")
```

### Analyze Stuck Detection

```python
# Load telemetry
df = pd.read_csv('training_logs/telemetry_20260103_195000.csv')

# Find stuck events
stuck = df[df['Stationary_Timeout'] == True]

# Analyze position_delta distribution when stuck
print("Position delta stats when stuck:")
print(stuck['Position_Delta'].describe())

# Check stationary timer values
print(f"Average time to stuck detection: {stuck['Stationary_Timer'].mean():.1f}s")
print(f"Min stationary timer at stuck: {stuck['Stationary_Timer'].min():.1f}s")

# Find vehicle spinning wheels (high speed, low movement)
wheel_spin = df[(df['Speed_ms'] > 5) & (df['Position_Delta'] < 0.1)]
print(f"Wheel-spinning instances: {len(wheel_spin)}")
```

### Analyze G-Force Patterns

```python
# Check if flip detection is working
df = pd.read_csv('training_logs/telemetry_20260103_195000.csv')

# Distribution of Gz during normal driving
normal = df[df['Crash_Detected'] == False]
print("Gz during normal driving:")
print(normal['Gz'].describe())

# Gz at crashes
crashes = df[df['Crash_Detected'] == True]
print("Gz at crashes:")
print(crashes['Gz'].describe())

# Check current flip threshold (Gz > 0.7)
flipped = df[(df['Gz'] > 0.7)]
print(f"Times Gz > 0.7 (flip threshold): {len(flipped)}")
```

## Current Thresholds (as of Jan 3, 2026)

Based on initial testing, thresholds are set conservatively:

| Detection | Threshold | Rationale |
|-----------|-----------|-----------|
| **Crash (damage increase)** | > 0.15 | Moderate damage - catches most crashes |
| **Wrecked (total damage)** | > 0.5 | High damage - vehicle too damaged to continue |
| **Minor damage** | 0.05 - 0.15 | Small penalty, no reset |
| **Stuck (position delta)** | < 0.2m in 0.3s | Vehicle barely moving |
| **Stuck timeout** | 6.0 seconds | Maximum time stuck before recovery |
| **Flip detection (Gz)** | > 0.7 | Nearly upside down |
| **Speed limit** | 20 m/s (~45 mph) | Highway safe speed |

## Tuning Process

1. **Run training for 10-20 episodes** with telemetry logging enabled
2. **Analyze telemetry CSV** using pandas/Excel
3. **Identify false positives/negatives**:
   - False positives: Unnecessary resets (damages AI learning)
   - False negatives: Crashes/stuck not detected (wastes training time)
4. **Adjust thresholds** in `phase4c_neural_highway_training.py`
5. **Re-test** and compare telemetry

## Changes Made

### 1. Model Organization
- **Before**: Models saved to `models/` in root directory
- **After**: Models saved to `models/` directory created by TrainingMetrics class
- Cleaner separation of logs vs models

### 2. Telemetry Logging
- Added `log_telemetry()` method to TrainingMetrics
- Captures every step's state, actions, and events
- CSV format for easy analysis in Excel/pandas
- Event tagging (CRASH, STUCK, SPEEDING) for filtering

### 3. Crash Detection Improvements
- Added total damage check (> 0.5 triggers recovery)
- Lowered damage increase threshold from 0.4 to 0.15
- Reduced stuck timeout from 12s to 6s
- Now logs damage_increase value for threshold analysis

---

**Recommendation**: After first 20-episode training run, analyze `telemetry_*.csv` to validate thresholds before longer training sessions.
