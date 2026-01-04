# West Coast USA - Waypoint Coordinates for Phase 5

**Map**: west_coast_usa  
**Spawn Point**: (-717.121, 101.0, 118.675)  
**Coordinate System**: Cartesian (X, Y, Z) in meters

---

## Route Design Philosophy

The west_coast_usa highway provides a natural progression of difficulty:

1. **Urban Section** (0-500m): Tight maneuvering, obstacles, low speed
2. **Transition Zone** (500-1500m): Opening curves, moderate speed
3. **Open Highway** (1500m+): Smooth flowing, high speed optimal

Waypoints are placed to guide the AI through this natural curriculum.

---

## Waypoint Definitions

### Route 1: Urban Basics (3 Waypoints)
**Purpose**: Learn basic waypoint navigation in constrained environment  
**Distance**: ~500m  
**Difficulty**: ⭐⭐☆☆☆

```python
ROUTE_1_URBAN = [
    {'id': 'spawn', 'pos': (-717.121, 101.0, 118.675), 'name': 'Start'},
    {'id': 'wp1', 'pos': (-730.0, 85.0, 118.5), 'name': 'Urban Straight'},
    {'id': 'wp2', 'pos': (-745.0, 70.0, 118.5), 'name': 'Urban Exit'},
    {'id': 'wp3', 'pos': (-760.0, 55.0, 118.5), 'name': 'Transition Start'}
]
```

**Characteristics**:
- Straight initial section (~25m)
- Gentle curve (~20m)
- Building obstacles on sides
- Target speed: 10-15 m/s

---

### Route 2: Transition Curves (5 Waypoints)
**Purpose**: Navigate through opening curves with speed variation  
**Distance**: ~1000m  
**Difficulty**: ⭐⭐⭐☆☆

```python
ROUTE_2_TRANSITION = [
    {'id': 'spawn', 'pos': (-717.121, 101.0, 118.675), 'name': 'Start'},
    {'id': 'wp1', 'pos': (-745.0, 70.0, 118.5), 'name': 'Urban Exit'},
    {'id': 'wp2', 'pos': (-780.0, 45.0, 118.3), 'name': 'First Curve'},
    {'id': 'wp3', 'pos': (-820.0, 30.0, 118.0), 'name': 'Curve Apex'},
    {'id': 'wp4', 'pos': (-860.0, 25.0, 117.8), 'name': 'Straightaway'},
    {'id': 'wp5', 'pos': (-900.0, 20.0, 117.5), 'name': 'Highway Entry'}
]
```

**Characteristics**:
- Mix of straight and curved sections
- Increasing speed zones
- Grace runoffs to grass (off-road detection)
- Target speed: 15-20 m/s

---

### Route 3: Highway Loop (8 Waypoints)
**Purpose**: Complete highway circuit with speed optimization  
**Distance**: ~2000m  
**Difficulty**: ⭐⭐⭐⭐☆

```python
ROUTE_3_HIGHWAY = [
    {'id': 'spawn', 'pos': (-717.121, 101.0, 118.675), 'name': 'Start'},
    {'id': 'wp1', 'pos': (-760.0, 55.0, 118.5), 'name': 'Transition'},
    {'id': 'wp2', 'pos': (-820.0, 30.0, 118.0), 'name': 'Curve Entry'},
    {'id': 'wp3', 'pos': (-880.0, 20.0, 117.5), 'name': 'Highway'},
    {'id': 'wp4', 'pos': (-950.0, 15.0, 117.0), 'name': 'Straight Section'},
    {'id': 'wp5', 'pos': (-1020.0, 10.0, 116.5), 'name': 'Long Straight'},
    {'id': 'wp6', 'pos': (-1100.0, 5.0, 116.0), 'name': 'Far Point'},
    {'id': 'wp7', 'pos': (-1150.0, 0.0, 115.5), 'name': 'Return Curve'},
    {'id': 'wp8', 'pos': (-1200.0, -10.0, 115.0), 'name': 'Finish'}
]
```

**Characteristics**:
- Long straight sections (ideal for high speed)
- Smooth flowing curves
- Multiple speed zones
- Target speed: 20-30 m/s on straights, 10-15 m/s in curves

---

### Route 4: Complex Navigation (12 Waypoints)
**Purpose**: Advanced pathfinding with tight tolerances  
**Distance**: ~3000m  
**Difficulty**: ⭐⭐⭐⭐⭐

```python
ROUTE_4_COMPLEX = [
    # To be defined after Route 3 success
    # Will include: tight curves, elevation changes, multiple surface types
]
```

**Status**: Reserved for Phase 5D extended training

---

## Waypoint Placement Guidelines

### 1. Spacing Considerations
- **Minimum spacing**: 20m (prevents waypoint skipping)
- **Maximum spacing**: 200m (prevents AI getting lost)
- **Curve waypoints**: Place at apex and exit
- **Straight waypoints**: Every 100-150m

### 2. Height Coordinates (Z-axis)
- West coast highway is relatively flat
- Z variation: 115.0 - 118.7m
- Use terrain height at waypoint location
- Small variations are normal (±0.5m)

### 3. Coordinate Validation
```python
def validate_waypoint(wp):
    """Check if waypoint is reachable and safe"""
    # Distance from spawn should be reasonable
    distance = np.linalg.norm(np.array(wp['pos']) - SPAWN_POS)
    assert 10 < distance < 5000, "Waypoint too close/far"
    
    # Z coordinate should be near ground level
    assert 110 < wp['pos'][2] < 125, "Waypoint elevation invalid"
    
    return True
```

### 4. Tolerance Zones
Each waypoint has a "reached" radius:
- **Urban waypoints**: 10m radius (tight)
- **Highway waypoints**: 20m radius (looser)
- **Curve apex**: 15m radius (moderate)

---

## Route Testing Procedure

Before using in training:

1. **Manual Drive-Through**
   - Drive route manually in BeamNG
   - Verify all waypoints are reachable
   - Check for obstacles/terrain issues
   - Note approximate time to complete

2. **Coordinate Verification**
   - Spawn vehicle at each waypoint
   - Verify coordinates are correct
   - Check terrain height
   - Ensure no map geometry issues

3. **Distance Calculation**
   ```python
   def calculate_route_length(route):
       total = 0
       for i in range(len(route) - 1):
           p1 = np.array(route[i]['pos'])
           p2 = np.array(route[i+1]['pos'])
           total += np.linalg.norm(p2 - p1)
       return total
   ```

4. **Path Visualization**
   - Plot waypoints on 2D map
   - Verify path makes sense
   - Check for sharp angles (>90°)
   - Ensure smooth flow

---

## Dynamic Waypoint Generation

For advanced training (Phase 5D+):

```python
def generate_random_route(start_pos, num_waypoints=5, avg_spacing=100):
    """Generate procedural route for varied training"""
    route = [{'id': 'spawn', 'pos': start_pos, 'name': 'Start'}]
    
    current_pos = np.array(start_pos)
    current_heading = np.random.uniform(0, 2*np.pi)
    
    for i in range(num_waypoints):
        # Add some curvature
        heading_change = np.random.uniform(-np.pi/6, np.pi/6)
        current_heading += heading_change
        
        # Random spacing variation
        distance = np.random.uniform(avg_spacing*0.8, avg_spacing*1.2)
        
        # Calculate next position
        next_pos = current_pos + distance * np.array([
            np.cos(current_heading),
            np.sin(current_heading),
            np.random.uniform(-0.5, 0.5)  # Small height variation
        ])
        
        route.append({
            'id': f'wp{i+1}',
            'pos': tuple(next_pos),
            'name': f'Auto Waypoint {i+1}'
        })
        
        current_pos = next_pos
    
    return route
```

---

## Coordinate System Notes

### BeamNG Coordinate System
- **X**: East-West (negative = west)
- **Y**: North-South (negative = south)
- **Z**: Elevation (up is positive)
- **Origin**: Map-specific, varies by map

### Heading/Bearing
- **0° / 0 rad**: North (+Y direction)
- **90° / π/2 rad**: East (+X direction)
- **180° / π rad**: South (-Y direction)
- **270° / 3π/2 rad**: West (-X direction)

### Distance Calculations
```python
def distance_2d(pos1, pos2):
    """Horizontal distance (ignore Z)"""
    return np.sqrt((pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2)

def distance_3d(pos1, pos2):
    """True 3D distance"""
    return np.linalg.norm(np.array(pos2) - np.array(pos1))

def bearing_to(pos_from, pos_to):
    """Calculate bearing in radians (0 = north, clockwise)"""
    dx = pos_to[0] - pos_from[0]
    dy = pos_to[1] - pos_from[1]
    return np.arctan2(dx, dy)  # Note: atan2(x, y) for north=0
```

---

## Waypoint Tolerance Tuning

### Phase 5A (Learning)
- Large tolerance: 30m radius
- Forgiving heading error: ±45°
- Goal: Learn basic navigation

### Phase 5B (Refining)
- Medium tolerance: 20m radius
- Moderate heading error: ±30°
- Goal: Improve accuracy

### Phase 5C (Mastery)
- Tight tolerance: 10m radius
- Strict heading error: ±15°
- Goal: Precise navigation

### Phase 5D (Racing)
- Racing line: 5m radius
- Optimal heading: ±10°
- Goal: Fastest lap times

---

## Example Route Usage

```python
from phase5_waypoints import ROUTE_1_URBAN, ROUTE_2_TRANSITION

# Load route
current_route = ROUTE_1_URBAN
current_waypoint_idx = 1  # Start at first waypoint after spawn

# During episode
vehicle_pos = vehicle.state['pos']
target_waypoint = current_route[current_waypoint_idx]

# Calculate navigation
distance = np.linalg.norm(np.array(target_waypoint['pos']) - vehicle_pos)
bearing = bearing_to(vehicle_pos, target_waypoint['pos'])

# Check if reached
if distance < WAYPOINT_TOLERANCE:
    print(f"Reached {target_waypoint['name']}!")
    current_waypoint_idx += 1
    
    if current_waypoint_idx >= len(current_route):
        print("Route complete!")
        episode_done = True
```

---

## Future Enhancements (Phase 6+)

- **Elevation-aware routes**: Use Z-coordinate for hill climb challenges
- **Multi-path routes**: Branching waypoints with choices
- **Time trial mode**: Fastest completion time
- **Rally mode**: Off-road waypoints
- **Traffic integration**: Waypoints with AI traffic avoidance

---

**Status**: INITIAL DRAFT - Coordinates to be validated with manual testing  
**Next Action**: Test Route 1 coordinates in BeamNG, adjust as needed
