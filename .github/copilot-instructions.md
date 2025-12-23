# BeamNG AI Driver - Copilot Instructions

## Project Overview

**"Data-Driven Driver"** - A reinforcement learning project using BeamNG.drive simulator to train an autonomous driving AI. The philosophy is **Maximum Data Visibility**: consume every piece of telemetry from the official BeamNGpy API rather than screen scraping or game modification.

**Current Status**: Phase 4C (Neural Highway Training with Persistent Instance)
- Phases 1-3: Complete (Connection → Control → Telemetry)
- Phase 4A: Complete (Exploratory RL Environment)
- Phase 4B: Complete (SAC neural networks with live brain dashboard)
- Phase 4C: Active (Highway distance training with persistent BeamNG)

## Repository Structure

```
beam-ng-ai/
├── src/               # Phase implementation files (organized by phase)
│   ├── phase1/        # BeamNG connection foundation
│   ├── phase2/        # Vehicle control and basic sensors
│   ├── phase3/        # Maximum telemetry integration
│   └── phase4/        # RL environment and neural networks
├── docs/              # Project documentation and phase reports
├── tests/             # Connection and API validation tests
├── scripts/           # Debug utilities and helper scripts
├── legacy/            # Deprecated experimental code (Lua API)
├── .github/           # GitHub configuration and AI instructions
├── README.md          # Project vision and roadmap
└── requirements.txt   # Python dependencies
```

## Critical Architecture Patterns

### BeamNG Integration - The Foundation Pattern

Every script follows this strict initialization sequence to avoid crashes:

```python
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging

# 1. Always set logging first
set_up_simple_logging()

# 2. BeamNG connection (hardcoded path is intentional)
bng_home = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
bng = BeamNGpy('localhost', 25252, home=bng_home)

# 3. Launch and wait
bng.open(launch=True)

# 4. Scenario setup BEFORE vehicle attachment
scenario = Scenario('west_coast_usa', 'scenario_name', description='...')
vehicle = Vehicle('vehicle_id', model='etk800', license='...')

# 5. Sensors attached BEFORE adding vehicle to scenario
vehicle.sensors.attach('electrics', Electrics())
vehicle.sensors.attach('damage', Damage())

# 6. Spawn at proven coordinates (vehicle won't fall through map)
scenario.add_vehicle(vehicle, 
    pos=(-717.121, 101, 118.675),
    rot_quat=(0, 0, 0.3826834, 0.9238795))

# 7. Build → Set physics → Load → Start
scenario.make(bng)
bng.settings.set_deterministic(60)  # or 120 for high-frequency
bng.scenario.load(scenario)
bng.scenario.start()

# 8. ALWAYS wait for physics stabilization
time.sleep(3-5)
```

**Why this order matters**: Attaching sensors after adding vehicle to scenario causes BeamNG crashes. The spawn coordinates are from validated BeamNGpy examples - other positions cause vehicles to fall through terrain.

### The 142-Channel Telemetry System

Phase 3 established a comprehensive input pipeline - **always use this full state vector**:

```python
# Vehicle State (7 channels)
position: (x, y, z)
velocity: (vx, vy, vz)  
speed: float

# Control Inputs (3 channels)
throttle, steering, brake

# Electrics Telemetry (126 channels from vehicle.sensors['electrics'])
# Key channels: throttle, brake, steering, wheelspeed, rpm, gear, fuel, 
#               gx/gy/gz (G-forces), abs_active, esc_active, tcs_active

# Damage & Physics (6 channels)
damage_level, damage zones, G-forces
```

See [src/phase3/phase3_maximum_telemetry_fixed.py](src/phase3/phase3_maximum_telemetry_fixed.py) for the `TelemetrySnapshot` dataclass pattern - reuse this structure for consistency.

### Auto-Recovery System (Phase 4A Pattern)

All RL environments must implement this collision recovery to prevent training stalls:

```python
def check_recovery_needed(vehicle) -> bool:
    """Check if vehicle needs recovery (stuck/crashed)"""
    vehicle.sensors.poll()
    
    # Collision detection
    damage = vehicle.sensors['damage']
    if damage['damage'] > self.last_damage + 50:  # Damage threshold
        return True
    
    # Stationary timeout (5 seconds)
    if vehicle.state['vel'].length() < 0.1:
        self.stationary_time += dt
        if self.stationary_time > 5.0:
            return True
    else:
        self.stationary_time = 0
    
    return False

# Recovery execution
vehicle.recover()  # BeamNG's built-in recovery - don't reimplement
```

Pattern established in [src/phase4/phase4a_exploratory_environment.py](src/phase4/phase4a_exploratory_environment.py) - proven to work with -50 reward penalty per recovery.

### Neural Network Architecture (Phase 4B)

Use **SAC (Soft Actor-Critic)** for continuous control. Standard architecture:

```python
# Actor: state_dim (142) → 256 → 256 → action_dim*2 (mean + log_std)
# Critic: (state_dim + action_dim) → 256 → 256 → 1 (Q-value)

# Device configuration for RTX 4070 Ti
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Action space: [throttle, steering, brake] all continuous [-1, 1]
# Clip to valid ranges before sending to BeamNG
```

See [src/phase4/phase4b_integrated_neural_training.py](src/phase4/phase4b_integrated_neural_training.py) for complete SAC implementation with layer hooks for visualization.

### Persistent BeamNG Connection (Phase 4C Pattern)

**Always connect to running instance, never launch new one** - dramatically faster iteration:

```python
# Connect to already-running BeamNG
bng = BeamNGpy('localhost', 64256)
bng.open(launch=False)  # False = connect to existing

# Fast episode reset (2 seconds vs 30 seconds)
bng.scenario.restart()  # Don't reload, just restart
time.sleep(2)

# Vehicle is automatically repositioned
vehicle.sensors.poll()
# Ready for next episode!
```

**Why this matters**: Full BeamNG launch takes 30+ seconds. Scenario restart takes 2 seconds. Over 100 training episodes, this saves 45+ minutes. See [src/phase4/phase4c_neural_highway_training.py](src/phase4/phase4c_neural_highway_training.py) for complete implementation.

## Development Workflows

### Running Experiments

**Phase progression files** (e.g., `src/phase1/phase1_mvp_working.py`, `src/phase2/phase2_fixed.py`) are the source of truth. Each phase has:
- `phase*_working.py` / `phase*_fixed.py` - Known good implementations
- `phase*_diagnostic.py` - Debugging tools if BeamNG acts up  
- Markdown docs in `docs/` - `PHASE*_*.md` files document achievements and next steps

**To test changes**: Copy the latest working phase file within its phase directory, increment the version (e.g., `src/phase4/phase4b_v2.py`), make changes. Never modify working files directly.

### BeamNG Troubleshooting

**Common issue**: BeamNG freezes at menu instead of loading scenario
- **Cause**: Timing race condition during startup
- **Solution**: See [src/phase2/phase2_diagnostic.py](src/phase2/phase2_diagnostic.py) - adds extended startup waits and manual scenario trigger

**Vehicle falls through map**:
- Only use validated spawn coordinates: `(-717.121, 101, 118.675)` on `west_coast_usa`
- Other maps require coordinate validation through manual testing

### Dependencies

From [requirements.txt](requirements.txt) - minimal by design:
```
beamngpy          # Core API (v1.34.1 validated)
torch             # Neural networks (CUDA for RTX 4070 Ti)
numpy, matplotlib # Data processing and visualization
```

Install PyTorch with CUDA 12.1+ for GPU training: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

## Project-Specific Conventions

### File Naming & Organization
- `src/phase*/` - Phase implementation scripts organized by phase number
  - `phase*_working.py` / `phase*_fixed.py` - Stable, working implementations
  - `phase*_diagnostic.py` - Debugging and troubleshooting tools
- `docs/` - All project documentation
  - `PHASE*_*.md` - Phase reports and planning docs (caps for visibility)
  - `PROJECT_STATUS_REPORT.md` - Overall project status
- `tests/` - Connection and API validation scripts (`test_*.py`, `check_*.py`)
- `scripts/` - Debug and utility scripts (`debug_*.py`, `launch_*.py`)
- `legacy/` - Deprecated experimental code (Lua API attempts)

### Phase System
The 8-phase roadmap in [README.md](README.md) is the project plan. Each phase builds on the previous:
1. Connection → 2. Control → 3. Telemetry → 4. RL Environment → 5. Feature Engineering → 6. Training → 7. Competition → 8. Superhuman

**Never skip phases** - each validates foundations for the next.

### Code Style
- Type hints and dataclasses for telemetry structures
- Emoji in print statements for visual scanning during runs (e.g., `🚀`, `✅`, `❌`)
- Extensive error reporting with troubleshooting hints
- Time delays explicit (`time.sleep(5)`) with comments explaining physics stabilization

### Control Ranges (BeamNGpy API)
```python
# Correct ranges for vehicle.control()
throttle: 0.0 to 1.0   # NOT -1 to 1
steering: -1.0 to 1.0  # Left negative, right positive  
brake: 0.0 to 1.0      # NOT -1 to 1
```

## Key Integration Points

### BeamNGpy API Quirks
- `vehicle.sensors.poll()` required before reading sensor data
- `vehicle.state` is a dict, not an object - use `vehicle.state['pos']` not `vehicle.state.pos`
- Control updates at ~2Hz are smooth, faster causes jitter
- Physics determinism: `bng.settings.set_deterministic(60)` for 60Hz or 120Hz max

### Live Brain Visualization (Phase 4B)
Multi-threaded architecture: training thread + Tkinter UI thread
- `queue.Queue` for thread-safe brain state communication
- Forward hooks registered on `nn.Linear` layers to capture activations
- See [src/phase4/phase4b_live_brain_dashboard.py](src/phase4/phase4b_live_brain_dashboard.py) for visualization patterns

### Reward Engineering Philosophy
Start minimal (Phase 4A: distance + collision penalty), progressively add complexity:
- Phase 4A: `distance * 0.1 - 50 * collisions`
- Phase 4B: Add speed optimization, smoothness rewards
- Phase 5: Feature engineering (TTC, path deviation, friction estimation)

**Never add rewards without validation** - test each component independently.

## What NOT to Do

- Don't use BeamNG's Lua API (see `legacy/beamng_api_server.lua`) - it's experimental, Python API is stable
- Don't spawn vehicles without proven coordinates - map collision is unreliable
- Don't poll sensors faster than 20Hz - diminishing returns vs CPU load
- Don't use screen capture for observations - Phase 3 provides 142 structured channels
- Don't modify working phase files - copy and version instead

## Getting Oriented

**New to the project?** Read in order:
1. [README.md](README.md) - Vision, philosophy, 8-phase roadmap
2. [docs/PROJECT_STATUS_REPORT.md](docs/PROJECT_STATUS_REPORT.md) - Current state, achievements
3. [src/phase1/phase1_mvp_working.py](src/phase1/phase1_mvp_working.py) - Foundation pattern
4. [src/phase3/phase3_maximum_telemetry_fixed.py](src/phase3/phase3_maximum_telemetry_fixed.py) - Telemetry architecture
5. [docs/PHASE4A_SUCCESS_REPORT.md](docs/PHASE4A_SUCCESS_REPORT.md) - RL environment validation

**Working on neural networks?** Start with [src/phase4/phase4b_integrated_neural_training.py](src/phase4/phase4b_integrated_neural_training.py) - complete SAC implementation with visualization hooks.
