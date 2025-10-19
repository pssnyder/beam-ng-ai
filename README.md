# **Project: Black Box Driver (BeamNG.drive AI Automation)**

## **🎯 Project Motto**

**Solve. The. Problem.**

## **💡 Project Vision & Philosophy**

This project aims to develop a robust, high-performance Reinforcement Learning (RL) environment using the BeamNG.drive simulator. The primary technical challenge is designing a system that treats the simulator as a **true "black box,"** restricting the AI's data stream (vision and telemetry) to mimic the information constraints of a real-world sensor suite.

We will intentionally move away from "white-box" solutions (like direct memory reading) to focus on pure perception, control, and decision-making logic.

### **Core Technical Pillars:**

1. **Low Coupling:** The AI's decision-making logic must be entirely decoupled from the BeamNG engine's internal workings. Inputs are generic (throttle, steer), and outputs are generic (camera feed, simple ground-truth telemetry).  
2. **High Cohesion:** Each subsystem (Vision Processor, Telemetry Handler, Action Injector) will focus on a single, clear task.  
3. **Core Tool:** The official, open-source **BeamNGpy** Python API will be the sole client interface, leveraging its structured control features.

## **🛣 Project Roadmap: Progressive Phases**

The project is structured into 8 progressive phases, moving from basic setup (MVP) to advanced superhuman exploits (North Star).

### **Phase 1: Basic Environment Setup (MVP)**

Idea: Establish a stable, authenticated Python-to-BeamNG connection.  
Goal: Successfully launch the simulator, spawn a vehicle, and exchange a single data packet.  
Benefit: Confirms environmental readiness and establishes the foundation for the control loop.  
Tasks:

* Install BeamNGpy (pip install beamngpy).  
* Verify BeamNG.tech/drive version compatibility.  
* Write an initial Python script to instantiate BeamNGpy and connect to the local port.  
* Create the simplest possible scenario (one car, one flat map).  
* Add "Clip Cursor" logic (via ctypes/pywin32 for Windows) to the main script to ensure input safety outside the simulation process.  
  Priority: Urgent

### **Phase 2: Core Game Control and Visibility Layers (Defining the Black Box)**

Idea: Define the AI's sensory and action boundaries.  
Goal: Create the initial observation (state) and action (input) functions and run a manual control loop.  
Benefit: Solidifies the "black box" constraints by dictating exactly what the AI can see and do.  
Tasks:

* **Vision Layer (Observation):** Implement a function to capture the front-facing vehicle camera feed via BeamNGpy's sensor module, reducing the resolution to the minimum required (e.g., 84x84 grayscale) to force the AI to rely on simple visual cues.  
* **Control Layer (Action):** Map discrete actions (e.g., \[STEER\_LEFT, STEER\_RIGHT, THROTTLE\_ON, BRAKE\_ON\]) to vehicle inputs via BeamNGpy commands.  
* First Test: Write a script where you manually input commands (e.g., press 'S' to throttle, 'A' to steer left) and verify the low-resolution image stream.  
  Priority: Urgent

### **Phase 3: Telemetry and Feedback Systems (Ground Truth)**

Idea: Incorporate minimal, necessary structured data that would be available from a real-world car (e.g., an OBD-II port).  
Goal: Extract only the essential numerical data points required for the reward function.  
Benefit: Provides highly accurate, low-latency performance metrics without forcing the AI to visually scrape the HUD.  
Tasks:

* Identify minimum required telemetry fields: speed, engine\_rpm, current\_damage\_factor, wheel\_slip.  
* Implement a separate function in Python to poll and parse this telemetry data via BeamNGpy's vehicle state access.  
* Create a rudimentary Reward Function MVP: Reward \= distance\_traveled \- time\_elapsed \- 10 \* damage\_factor.  
  Priority: High

### **Phase 4: Directed Driving Simulation (Directional and Coordinal Controls)**

Idea: Teach the AI fundamental stability and path-following.  
Goal: Develop an initial Imitation Learning script that allows the AI to follow a simple, predefined road segment without crashing.  
Benefit: Achieves the first level of "autonomous" behavior and provides a baseline performance metric.  
Tasks:

* Define a simple scenario with clear waypoints on a closed loop track.  
* Implement a **Proportional-Derivative (PD)** controller logic in Python as a non-AI baseline to prove the control system works.  
* Begin collecting Training Data: Log the camera feed, current telemetry, and corresponding correct human inputs while you drive the simple route.  
  Priority: High

### **Phase 5: Environmental Factors and Feedback System Deep Dive Enhancements**

Idea: Introduce physics and sensory challenges to stress-test the AI's resilience.  
Goal: Expand the observation space to include external factors and enhance the AI's ability to "feel" the environment.  
Benefit: Makes the RL problem more realistic and transferable to complex tasks.  
Tasks:

* **Advanced Telemetry:** Add ground-truth sensor data for **G-forces** and **IMU/Angular Velocity** to represent wheel-spin and skidding (essential for racing/stunts).  
* **Environmental Inputs:** Incorporate variables like **weather condition** (rain/dry) and **time of day** (light/dark) into the AI's state vector, forcing it to generalize its driving logic.  
* Collision Feedback: Fine-tune the damage logging for granular, immediate negative reinforcement (penalizing near misses before they become full crashes).  
  Priority: High

### **Phase 6: Training and Challenges (The RL Loop)**

Idea: Implement the full Reinforcement Learning agent.  
Goal: Train a CNN-based deep RL agent (e.g., using Stable Baselines3) to autonomously drive the scenario without human input.  
Benefit: Moves the project from automation to self-learning AI research.  
Tasks:

* Integrate the Python environment with an RL framework (e.g., **SB3**).  
* Refine the **Reward Function** to incentivize smoothness and speed over simple survival.  
* Execute the first round of training runs (e.g., 1 million steps) and monitor learning curves.  
* Challenge MVP: Set a consistent lap time goal on the training track.  
  Priority: High

### **Phase 7: Racing, Performance Tuning, and Competition**

Idea: Test the AI against other entities and optimize for absolute speed and efficiency.  
Goal: Beat the fastest lap time set by a human driver or the in-game AI on a complex track.  
Benefit: Validates the efficiency and control quality of the trained agent.  
Tasks:

* Introduce the in-game AI (controlled via BeamNGpy) as a competitor.  
* Train the agent on an entirely new, higher-complexity track (e.g., a twisty mountain road).  
* Implement Hyperparameter Tuning to find the optimal balance between speed and control precision.  
  Priority: Low

### **Phase 8: Stunts and Super-Human Exploits**

Idea: Leverage the AI's precise control to execute physics-defying or extremely difficult tasks.  
Goal: Complete challenges that are impossible or inconsistent for human drivers.  
Benefit: Demonstrates the limits of the simulation and the superiority of machine precision.  
Tasks:

* **Stunt Challenge 1:** Perfect, high-speed ramp jumps with minimal rotation.  
* **Exploit Challenge:** Identify a scenario where the AI can use a physics quirk (e.g., soft-body deformation) to gain an advantage (e.g., fitting through a gap a human couldn't).  
* Documentation: Record videos of the "super-human" feats for future articles\!  
  Priority: Low

## **🛠 Next Step: Execute Phase 1**

Your next immediate action is to set up the Phase 1 MVP.

**Initial Task:**

1. Verify BeamNG.drive/BeamNG.tech installation.  
2. Install **pip install beamngpy**.  
3. Write the initial connection and teardown script, ensuring your Windows input restriction (ClipCursor) is ready for testing in Phase 2\.

Remember to save and back up your work\! 💾