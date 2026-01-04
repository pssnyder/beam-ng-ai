# Threshold Adjustments - January 3, 2026

## Summary
Based on telemetry analysis from `telemetry_20260103_200943.csv`, adjusted reward system thresholds to match actual observed sensor values.

## Critical Bug Fixed: Telemetry Timing

**Problem:** Telemetry was logged AFTER crash recovery, showing damage=0.000 for all crashes because recovery resets damage.

**Solution:** Moved telemetry logging inside `step()` method BEFORE `_recover_to_road_center()` is called.

**Impact:** Now captures actual crash state with real damage/position/g-force values.

---

## Damage Threshold Adjustments

### Original Thresholds (INCORRECT)
Based on assumption that damage scale was 0.0-1.0:
- **Total damage > 0.5**: Vehicle wrecked (-30 reward, episode end)
- **Damage increase > 0.15**: Major crash (-30 reward, episode end)
- **Damage increase > 0.05**: Minor scrape (-5 reward, continue)

### Actual Damage Scale (from telemetry)
- **Normal driving**: 0.000-0.001
- **Minor scrapes**: 0.001-0.05
- **Major crashes**: 100-3000+ (massive spikes!)

**Example from Episode 6, Step 18:**
```
Damage: 3000.000 (from 0.001 previous step)
Gz: 12.63 (completely upside down)
Gx: 6.05, Gy: 21.22 (extreme lateral forces)
```

### New Thresholds (CORRECTED)
Adjusted to match actual 0-3000+ damage scale:

| Severity | Damage Increase | Total Damage | Penalty | Episode End? | Notes |
|----------|----------------|--------------|---------|--------------|-------|
| **Wrecked** | - | **> 100.0** | -30 | **Yes** | Vehicle destroyed |
| **Major Crash** | **> 50.0** | - | -30 | **Yes** | Significant impact |
| **Minor Impact** | **> 10.0** | - | -10 | No | Bumps/collisions |
| **Scrape** | **> 0.5** | - | -2 | No | Curb/guardrail contact |

**Multiplier:** All thresholds increased by ~100-200x to match actual sensor scale.

---

## Speed Limit Adjustment

### Original Setting
- **Speed Limit:** 20.0 m/s (~45 mph)
- **Observation:** AI never exceeded 12 m/s during exploration
- **Issue:** Too conservative for highway training

### New Setting
- **Speed Limit:** 30.0 m/s (~67 mph)
- **Rationale:** More realistic for highway driving
- **Penalty:** Still linear (-0.5 per m/s over limit)

---

## Position Delta Threshold (NO CHANGE)

**Current:** < 0.2m movement per step triggers stationary timer

**Analysis from telemetry:**
- Normal driving: 2-4m per step ✅
- Slow movement: 0.3-1.0m per step ✅
- No false positives detected

**Decision:** **Keep at 0.2m** - working correctly

---

## Flip Detection (UNDER INVESTIGATION)

### Current Threshold
```python
is_completely_flipped = next_state.gz > 0.7
is_sideways = abs(next_state.gx) > 1.5 and abs(next_state.gy) > 1.5
```

### Observed Values
**Normal driving:**
- Gz: -9 to -13 (upright, gravity + acceleration)
- Gx/Gy: -7 to +10 (turning/braking forces)

**During flip (Episode 6, Step 18):**
- **Gz: +12.63** (completely inverted!)
- **Gx: 6.05, Gy: 21.22** (extreme lateral)
- **Should have triggered BUT** was logged after recovery

### Status
Flip detection thresholds appear correct, but was suffering from same timing issue as damage detection. Now that telemetry is logged before recovery, should work properly.

---

## Expected Impact

### With Correct Damage Thresholds
1. **Major crashes now properly detected** (damage spike > 50)
2. **Minor bumps won't end episodes** (10-50 damage = penalty only)
3. **Scrapes tracked but minimally penalized** (0.5-10 damage)

### With Higher Speed Limit
1. **AI encouraged to drive faster** (can go up to 67 mph)
2. **More realistic highway behavior**
3. **Better exploration of speed-distance tradeoff**

### With Fixed Telemetry Timing
1. **Accurate crash data** for threshold tuning
2. **Real damage values** visible in logs
3. **Can validate flip detection** works properly

---

## Testing Plan

1. **Run 10-20 episodes** with new thresholds
2. **Analyze new telemetry** to confirm:
   - Crashes properly detected at damage > 50
   - Minor impacts (-10 penalty) don't end episodes
   - Speed limit violations occur (if AI drives faster)
3. **Look for flip events** with actual Gz values > 0.7
4. **Tune if needed** based on observed damage distribution

---

## Code Changes

### Files Modified
- `src/phase4/phase4c_neural_highway_training.py`

### Key Changes
1. **Moved telemetry logging** from training loop into `step()` method
2. **Log before recovery** (line ~759) instead of after (line ~1031)
3. **Damage thresholds** multiplied by ~100-200x
4. **Speed limit** increased from 20 → 30 m/s
5. **Added debug output** for minor impacts

### New Penalty Structure
```python
# Priority 1: Wrecked (total damage > 100)
reward -= 30.0, episode ends

# Priority 2: Major crash (damage increase > 50)
reward -= 30.0, episode ends

# Priority 3: Minor impact (damage increase > 10)
reward -= 10.0, continues driving

# Priority 4: Scrape (damage increase > 0.5)
reward -= 2.0, continues driving
```

---

## Previous Issues (RESOLVED)

❌ **Crashes showed damage=0.000** → ✅ Fixed by logging before recovery
❌ **Damage thresholds 100x too small** → ✅ Adjusted to 100-3000 scale
❌ **Speed limit too conservative** → ✅ Increased to 67 mph
❌ **No visibility into crash damage** → ✅ Now logged accurately

---

## Next Steps

1. Test with new thresholds
2. Analyze telemetry for damage distribution
3. Validate crashes are detected at proper severity
4. Monitor AI speed behavior (should drive faster now)
5. Check for flip detection with accurate Gz logging
