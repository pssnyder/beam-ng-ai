# Phase 4C: Episode-Based Training Design

**Date**: December 27, 2025  
**Status**: Proposed Improvement

---

## New Training Paradigm

### Fixed-Time Episodes

**Current Problem**:
- Car crashes → recovers to bad spot → crashes again loop
- Inconsistent episode lengths
- Hard to compare performance across episodes

**New Approach**:
- Each episode = exactly 60 seconds
- Always reset to origin after episode
- Score based on total distance achieved

---

## Episode Structure

```
Episode Timeline (60 seconds):
  0s: Spawn at origin, fresh car (damage=0)
  0-60s: AI drives, accumulating distance
  60s: Episode END
  
  Scoring:
    + Distance traveled (primary metric)
    + Bonus if still moving at end (+10 if speed > 5 m/s)
    - Penalty for damage at end (-damage * 20)
    - Small penalties during: stationary, minor scrapes
  
  Reset: Teleport back to origin, repeat
```

---

## Reward Function Changes

### During Episode (per step)
```python
reward = 0

# Distance (primary)
reward += distance_this_step * 1.0  # 1 point per meter

# Speed efficiency  
if speed > 2.0 and speed <= 20.0:
    reward += speed * 0.05  # Small bonus for moving well
elif speed > 20.0:
    reward -= (speed - 20.0) * 0.2  # Penalize excessive speed

# Minor damage
if damage_increase > 0.05:
    reward -= damage_increase * 10.0  # Penalize but don't reset

# Stationary
if not_moving:
    reward -= 0.1  # Small continuous penalty
```

### At Episode End (60s)
```python
# Final bonuses/penalties
if still_moving (speed > 5.0):
    reward += 10.0  # Bonus for momentum

if damage > 0.1:
    reward -= damage * 20.0  # Penalty for ending damaged
```

---

## Benefits

**1. Consistent Episodes**
- Every episode = 60 seconds
- Fair comparison between episodes
- Clear metric: distance per minute

**2. No Recovery Loops**
- No mid-episode recovery to bad spots
- If AI crashes, episode continues (with damage penalty)
- Learn to drive carefully to maximize distance

**3. Better Learning Signal**
- "How far can I get in 60 seconds?"
- Encourages efficient, safe driving
- Damage matters for final score

**4. Future: Multiple Spawn Points**
```python
spawn_points = [
    ("straight", pos1, rot1),    # Easy: long straight
    ("gentle_curve", pos2, rot2), # Medium: slight turn
    ("sharp_turn", pos3, rot3),   # Hard: tight curve
]

# Randomly select spawn for each episode
spawn = random.choice(spawn_points)
```

---

## Implementation Changes

### Environment
```python
def step(action):
    # Execute action
    next_state = get_state()
    reward = calculate_reward()
    
    # Check episode time
    episode_time = time.time() - episode_start
    done = (episode_time >= 60.0)
    
    if done:
        # Final scoring
        if speed > 5.0:
            reward += 10.0
        if damage > 0.1:
            reward -= damage * 20.0
    
    return next_state, reward, done

def reset_episode():
    # Always teleport to spawn
    vehicle.teleport(spawn_pos, spawn_rot, reset=True)
    episode_start = time.time()
    return initial_state
```

### Training Loop
```python
for episode in range(num_episodes):
    state = env.reset_episode()  # Back to spawn
    episode_distance = 0
    
    while True:  # 60 second loop
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        
        episode_distance += info['distance_progress']
        replay_buffer.push(state, action, reward, next_state, done)
        
        if done:  # 60 seconds elapsed
            print(f"Episode {episode}: {episode_distance:.1f}m")
            break
```

---

## Expected Behavior

### Random Exploration Phase
```
Episode 1: 15.3m (crashed at 20s, sat still rest of time)
Episode 2: 45.2m (drove slow but steady)
Episode 3: 23.1m (went too fast, crashed at 30s)
...
```

### Neural Training Phase
```
Episode 50: 125.4m (learning to stay on road)
Episode 100: 287.6m (smooth driving, few crashes)
Episode 200: 512.3m (near-optimal for straight road)
```

---

## Metrics to Track

Per Episode:
- Total distance
- Final speed
- Final damage
- Total reward
- Number of stationary moments
- Average speed

Across Training:
- Best distance
- Average distance (last 10 episodes)
- Improvement rate
- Crash frequency

---

## Future Extensions

### 1. Curriculum Learning
```python
# Start easy, progress to hard
if avg_distance > 300:
    spawn = "gentle_curve"
if avg_distance > 500:
    spawn = "sharp_turn"
```

### 2. Different Maps
```python
maps = ["west_coast_usa", "italy", "jungle_rock_island"]
# Train on variety for generalization
```

### 3. Variable Episode Length
```python
# Once mastered 60s, increase challenge
episode_duration = min(60 + (episode // 100) * 10, 300)  # Max 5 minutes
```

---

## Migration Plan

1. ✅ Remove crash recovery mid-episode
2. ✅ Implement 60s timer
3. ✅ Add end-of-episode scoring
4. ✅ Simplify reset to always use spawn
5. 🔄 Test with current setup
6. 🔄 Add spawn point variety
7. 🔄 Implement curriculum learning

---

**Next Step**: Implement in phase4c_neural_highway_training.py
