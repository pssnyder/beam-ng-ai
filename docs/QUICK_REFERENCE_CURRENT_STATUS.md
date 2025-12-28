# BeamNG AI - Quick Reference & Current Status

**Last Updated**: December 27, 2025

---

## 🎯 Current Training Status

**Phase**: 4C - Neural Highway Training  
**Algorithm**: SAC (Soft Actor-Critic)  
**Device**: CUDA (RTX 4070 Ti)  

### What's Working ✅
- BeamNG persistent connection (no full game reload between episodes)
- Vehicle movement and control
- Basic crash detection (damage + orientation)  
- Stationary timeout recovery
- Random exploration → Neural network transition
- Road-center crash recovery

### Current Issues ❌
- AI crashes into first obstacle (no forward vision)
- Sometimes gets stuck in damaged state in grass
- No road awareness (drives blind)
- Reward function encourages reckless speed

---

## 📊 Model Architecture Summary

### Inputs (27 values)
```
Position & Motion (6):     x, y, z, vx, vy, vz
Distance Tracking (3):     total_dist, checkpoint_dist, delta_dist
Vehicle Dynamics (15):     speed, throttle, steering, brake,
                          rpm, gear, wheelspeed,
                          gx, gy, gz (G-forces),
                          damage, abs, esc, tcs, fuel
Episode Metadata (3):      episode_time, crash_count, stationary_time
```

### Outputs (3 values)
```
throttle: [0.0 to 1.0]
steering: [-1.0 to 1.0]  
brake: [0.0 to 1.0]
```

### Neural Networks
```
Actor:   27 → 256 → 256 → 6 (mean + std for 3 actions)
Critic1: 30 → 256 → 256 → 1
Critic2: 30 → 256 → 256 → 1
Total Parameters: ~268,000
```

---

## ⚙️ Training Hyperparameters

```python
replay_start_size = 3000      # Random exploration steps
batch_size = 64               # Training batch size
learning_rate = 0.0003        # Neural network learning rate
gamma = 0.99                  # Future reward discount
tau = 0.005                   # Target network update rate
buffer_capacity = 100,000     # Experience replay size
```

### Control Loop
```
Frequency: 3.3 Hz (0.3s per action)
Episode Length: 200 steps max (~60 seconds)
Reset Time: ~1 second (scenario restart) or ~0.6s (soft reset - not yet active)
```

---

## 🎁 Reward Function

```python
# Primary goal
reward += distance_traveled * 0.5

# Speed bonus (when making progress)
if moving_forward and speed > 1.0:
    reward += speed * 0.1

# Crash penalties
if damage_increased > 0.05:
    reward -= 50.0
if flipped_or_tilted:
    reward -= 30.0

# Stationary penalties
if speed < 1.0:
    reward -= 0.5 per step
    if stationary > 2 seconds:
        reward -= 20.0 + trigger recovery
```

---

## 🔧 Recent Fixes (Dec 27, 2025)

### 1. Parking Brake Auto-Release
**Problem**: Car wouldn't move  
**Solution**: Added `parkingbrake=0` to all control calls

### 2. Balanced Random Exploration
**Problem**: AI only learned "full throttle into wall"  
**Old**: throttle [0.7-1.0], steering [±0.4], brake [0-0.02]  
**New**: throttle [0.3-1.0], steering [±0.8], brake [0-0.3]  
**Impact**: AI now explores braking and turning

### 3. Sensitive Stuck Detection
**Problem**: Car sat in grass for minutes  
**Old**: Timeout if speed < 0.5 for 3.0s  
**New**: Timeout if speed < 1.0 for 2.0s  
**Impact**: Faster recovery from stuck states

### 4. Crash Recovery
**Problem**: Crashed cars stayed crashed  
**Solution**: Teleport to crash location upright instead of full recovery  
**Impact**: Continues from crash spot, not sent back to origin

### 5. Increased Exploration
**Problem**: Not enough diverse experiences  
**Old**: 1000 random steps  
**New**: 3000 random steps  
**Impact**: Better initial data before neural training starts

---

## 🚨 Known Limitations

### What the AI CAN'T See
- ❌ Road boundaries or lane markers
- ❌ Upcoming obstacles or curves
- ❌ Distance to road edges
- ❌ Heading relative to road direction
- ❌ Surface type (pavement vs grass)

**Result**: Driving completely blind using only dashboard instruments

### What the AI CAN'T Do
- ❌ Manual gear shifting (automatic transmission)
- ❌ Parking brake control (managed automatically)
- ❌ Reverse (action space is forward-only)
- ❌ Reset own damage (requires BeamNG API call)

---

## 📈 Next Improvements Needed

### Critical (Needed Now)
1. **Add road sensors** - raycasts to detect edges
2. **Heading alignment** - angle to road direction
3. **Improve reward** - penalize leaving pavement
4. **Damage reset** - allow training to continue with broken car

### Important (Phase 5)
1. **Curvature detection** - know when turns are coming
2. **Surface detection** - different friction on grass vs pavement
3. **Better exploration** - curriculum learning (start slow)
4. **Checkpoint system** - save best models

### Nice-to-Have (Later Phases)
1. **Vision input** - camera feed for road detection
2. **Multi-map training** - generalize to different roads
3. **Traffic avoidance** - train with other vehicles
4. **God mode experiments** - physics manipulation for testing

---

## 🛠️ Common Operations

### Start Training
```powershell
cd "S:\Programming\Gaming Projects\beam-ng-ai"
python src/phase4/phase4c_neural_highway_training.py
```

### Required BeamNG Setup
1. Launch BeamNG.drive
2. Load any scenario (will be replaced)
3. Wait at main menu
4. Run training script (connects automatically)

### Monitoring Training
Watch for:
- `Explore Step X` - Random exploration phase
- `Step X: Reward=` - Neural network training phase
- `💥 CRASH` - Collision detected
- `⏱️ STATIONARY TIMEOUT` - Stuck detection
- `✓ Recovered to road center` - Recovery confirmation

### Stopping Training
- `Ctrl+C` in terminal
- BeamNG stays open (no need to restart for next run)

---

## 📚 Documentation Files

- `NEURAL_NETWORK_ARCHITECTURE_EXPLAINED.md` - Full educational guide
- `PHASE4C_HIGHWAY_DISTANCE_TRAINING.md` - Technical architecture
- `PHASE4C_QUICK_START.md` - Setup instructions
- `PROJECT_STATUS_REPORT.md` - Overall project status
- `.github/copilot-instructions.md` - AI assistant context

---

## 🐛 Debugging Tips

### AI Not Moving
- Check: Is parking brake released? (should see "🚗 Parking brake released" message)
- Check: Is throttle being applied? (look for `Action=[T:0.XX ...]` in logs)
- Check: Is car on solid ground? (not fallen through map)

### AI Crashes Immediately
- Normal during random exploration (first 3000 steps)
- Check reward logs - should see negative rewards after crashes
- If continues after 3000 steps → reward function issue

### AI Stuck
- Should trigger timeout after 2 seconds at speed < 1.0
- If not triggering → check stationary_timer in logs
- Manual fix: Ctrl+C and restart (BeamNG stays open)

### Training Not Improving
- Check if buffer size growing (`len(replay_buffer)`)
- Check if losses decreasing (`Actor Loss=` in logs)
- Check if episode rewards increasing over time
- If no improvement after 50 episodes → reward function design issue

---

## 🎓 Learning Resources

**RL Algorithms**:
- [Spinning Up in Deep RL](https://spinningup.openai.com/) - Best RL tutorial
- [SAC Paper](https://arxiv.org/abs/1801.01290) - Original algorithm

**BeamNG**:
- [BeamNGpy Docs](https://beamngpy.readthedocs.io/)
- [Examples](https://github.com/BeamNG/BeamNGpy/tree/master/examples)

**Project Docs**:
- Start with `README.md` for project vision
- Read `NEURAL_NETWORK_ARCHITECTURE_EXPLAINED.md` for deep dive
- Check `PROJECT_STATUS_REPORT.md` for current state

---

**Questions?** Check the full architecture guide or ask about specific components!
