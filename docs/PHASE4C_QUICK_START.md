# Phase 4C Quick Start Guide

## Neural Highway Training with Persistent BeamNG Instance

### Prerequisites

1. **BeamNG.drive** installed and running
2. **Python 3.13** with packages:
   ```bash
   pip install beamngpy numpy torch
   ```

### Launch Sequence

#### Step 1: Start BeamNG.drive
1. Launch BeamNG.drive normally
2. Keep it running (don't load any scenario)
3. The game should be at the main menu

#### Step 2: Run Training Script
```bash
# Using Python 3.13
"/c/Users/patss/AppData/Local/Programs/Python/Python313/python.exe" src/phase4/phase4c_neural_highway_training.py

# Or if you've added an alias
py313 src/phase4/phase4c_neural_highway_training.py
```

### What Happens

1. **Connection** (3 seconds)
   - Script connects to running BeamNG instance
   - No new launch needed!

2. **Scenario Setup** (10 seconds, one time)
   - Loads automation_test_track highway
   - Spawns AI vehicle with sensors
   - Physics stabilization

3. **Training Loop** (continuous)
   - Episode 1: Random exploration (building replay buffer)
   - Episode 10+: Neural network starts learning
   - Episode 20+: You should see improvement
   - Episode 50+: Noticeable highway driving behavior

### Training Output

```
=== Episode 15/100 ===
  Step 10: Reward=12.50, Actor Loss=0.0234, Distance=25.3m
  Step 20: Reward=8.30, Actor Loss=0.0198, Distance=18.7m
  CRASH #1 at 152.3m - checkpoint updated
  Step 30: Reward=-50.00, Actor Loss=0.0215, Distance=-2.1m

Episode 15 Complete:
  Time: 45.2s
  Steps: 35
  Total Reward: 127.85
  Crashes: 1
  Max Distance: 152.3m
  Buffer Size: 1250
```

### Key Features

#### Fast Episode Reset
- Uses `scenario.restart()` instead of full reload
- 2 seconds instead of 30 seconds
- Keeps BeamNG running throughout

#### Distance-Based Rewards
- **+0.5 points** per meter traveled
- **+0.1 * speed** bonus when making progress
- **-50 points** on crash (checkpoint moves to crash location)
- **-20 points** on stationary timeout

#### Checkpoint System
```
Start: Checkpoint at origin (0m)
  ↓
Travel 100m → Checkpoint still at origin
  ↓
💥 CRASH at 100m
  ↓
Checkpoint moves to 100m mark
  ↓
Travel 50m from crash → Total: 150m from origin, 50m from checkpoint
  ↓
Rewards based on distance from current checkpoint (50m)
```

### Controls

- **Ctrl+C** - Stop training (saves model as `highway_model_interrupted.pth`)
- BeamNG stays running - you can inspect the scene
- Script reconnects on next run (no restart needed)

### Model Checkpoints

Saved automatically every 10 episodes:
- `highway_model_ep10.pth`
- `highway_model_ep20.pth`
- etc.

### Troubleshooting

#### "Could not connect to BeamNG"
- Make sure BeamNG.drive is running
- Default port: 64256
- Check BeamNG settings → Options → Other → Tech Port

#### "ModuleNotFoundError: No module named 'torch'"
```bash
# Install PyTorch with CUDA (if you have NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Or CPU-only version
pip install torch
```

#### Vehicle falls through map
- Map coordinates may need adjustment
- Script will auto-fallback to `west_coast_usa`
- Edit spawn coordinates in `setup_scenario()`

#### Training too slow
- Reduce `batch_size` from 64 to 32
- Increase control interval from 0.5s to 1.0s
- Use CPU if GPU causes issues: Set `CUDA_VISIBLE_DEVICES=""`

### Performance Tips

1. **GPU Acceleration**: Ensure CUDA-enabled PyTorch for RTX 4070 Ti
2. **Replay Buffer**: Start size of 1000 samples (about 10-15 episodes)
3. **Training Frequency**: Updates every 2 steps (configurable)
4. **Episode Length**: Max 200 steps to prevent endless episodes

### Expected Progress

| Episodes | Behavior | Max Distance |
|----------|----------|--------------|
| 1-20 | Random exploration, frequent crashes | 50-150m |
| 20-50 | Basic forward motion, learning steering | 150-300m |
| 50-100 | Road following emerging, fewer crashes | 300-500m |
| 100+ | Confident highway driving | 500m+ |

### Next Steps

After training:

1. **Test Policy**
   ```python
   agent.load('highway_model_ep100.pth')
   action = agent.get_action(state, deterministic=True)
   ```

2. **Add Camera Input** (Phase 4D)
   - Vision-based lane detection
   - Combine with telemetry

3. **Tune Rewards** (Phase 5)
   - Lane keeping bonus
   - Smooth driving rewards
   - Speed optimization

4. **Deploy to Competition** (Phase 7)
   - Race against built-in AI
   - Lap time optimization
