# **Project: Data-Driven Driver (BeamNG.drive AI Automation)**

## **🎯 Project Motto**

**Solve. The. Problem.**

## **💡 Project Vision & Philosophy**

This project aims to develop a robust, high-performance Reinforcement Learning (RL) environment using the BeamNG.drive simulator. The primary technical challenge is designing a system that treats the simulator as a **Maximum Data Visibility Environment**—meaning we will consume and exploit all available data provided by the official API.

We will intentionally embrace the rich, structured data streams of the simulator to build the most informed and intelligent AI agent possible. Unlike previous projects where we had the ability to inject custom data points or modify the game engine directly, this project constrains us to work exclusively with BeamNG's native telemetry streams, sensor outputs, and control interfaces. If data is needed beyond the API, it must be derived or proxied using the existing telemetry, not via external game manipulation.

### **Core Technical Pillars:**

1. **Full Data Visibility:** Leverage every piece of telemetry and sensor data the BeamNGpy API offers (position, rotation, G-forces, full physics state).  
2. **Feature Engineering Focus:** The intelligence challenge shifts from perception (reading the screen) to data refinement (calculating derived metrics like projected impact or ideal racing line deviation).  
3. **Core Tool:** The official, open-source **BeamNGpy** Python API will be the sole client interface, leveraging its structured control features.

## **🛣 Project Roadmap: Progressive Phases**

The project is structured into 8 progressive phases, moving from basic setup to superhuman exploits.

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

### **Phase 2: Core Game Control and Dynamic Sensor Implementation**

Idea: Define the AI's sensory and action boundaries using the highest fidelity data available.  
Goal: Implement full-resolution visual and sensor data streams and continuous vehicle control.  
Benefit: Provides the AI with the complete, high-fidelity observation state necessary for sophisticated decision-making.  
Tasks:

* **Vision Layer (Observation):** Implement a function to capture the front-facing vehicle camera feed via BeamNGpy's sensor module, using full color and high resolution (e.g., 640x480 or 1280x720) to maximize visual information.  
* **Dynamic Sensor Suite:** Instantiate and read simulated Lidar, GPS, and Advanced IMU sensor data from the vehicle via BeamNGpy to access raw ground truth.  
* **Control Layer (Action):** Map continuous actions (Steering: -1.0 to 1.0, Throttle: 0.0 to 1.0, Brake: 0.0 to 1.0) to vehicle inputs via BeamNGpy's control commands.  
  Priority: Urgent

### **Phase 3: Comprehensive Telemetry and State Capture**

Idea: Incorporate every available piece of structured physics and vehicle state data.  
Goal: Poll the complete vehicle state (position, rotation, velocity, G-forces, etc.) at the physics tick rate (up to 2000Hz).  
Benefit: Provides the precise, low-latency ground truth required for accurate reward function calculation and debugging.  
Tasks:

* Identify and poll ALL available telemetry from the Vehicle.sensors.state, GForces, and Electrics modules, including: pos, dir (rotation/quaternion), linear_velocity, angular_velocity, and wheel_data.  
* Implement a data buffer system in Python to handle the high update rate (up to 2000Hz physics ticks) for feature calculation.  
* Create a rudimentary Reward Function MVP: Reward = f(Position_Vector, Path_Deviation_Calculated) - time_elapsed - f(Damage_Rate).  
  Priority: High

### **Phase 4: Directed Driving Simulation (Directional and Coordinal Controls)**

Idea: Teach the AI fundamental stability and path-following.  
Goal: Develop an initial Imitation Learning script that allows the AI to follow a simple, predefined road segment without crashing, using the full state vector.  
Benefit: Achieves the first level of "autonomous" behavior and provides a baseline performance metric using the maximum data set.  
Tasks:

* Define a simple scenario with clear waypoints on a closed loop track.  
* Implement a **Proportional-Derivative (PD)** controller logic in Python as a non-AI baseline to prove the control system works with the continuous, full-telemetry input stream.  
* Begin collecting Training Data: Log the camera feed, full telemetry state vector, and corresponding correct human control inputs while driving the simple route.  
  Priority: High

### **Phase 5: Feature Engineering and Derived Metrics**

Idea: Transform the raw ground-truth data into highly predictive, calculated features that drive intelligence.  
Goal: Create derived metrics for the AI's observation space that represent concepts not directly available (e.g., time-to-impact, ideal steering angle).  
Benefit: Makes the RL problem solvable by translating massive amounts of raw data into a small, meaningful set of features.  
Tasks:

* **Collision Prediction Proxy:** Calculate the Time-to-Collision (TTC) feature using the car's current velocity and LiDAR/Mesh sensor distance readings.  
* **Path Deviation Feature:** Calculate the deviation angle between the car's current heading (dir from state) and the vector to the nearest unvisited waypoint.  
* **Wheel Control Feature:** Calculate a filtered, smoothed steering angle rate of change to penalize abrupt, unnatural inputs.  
* **Environmental Inputs:** Calculate and integrate derived variables like road friction estimate based on wheel slip and G-forces.  
  Priority: High

### **Phase 6: Training and Challenges (The RL Loop)**

Idea: Implement the full Reinforcement Learning agent.  
Goal: Train a CNN-based deep RL agent (e.g., using Stable Baselines3) to autonomously drive the scenario without human input, leveraging the rich feature set.  
Benefit: Moves the project from automation to self-learning AI research.  
Tasks:

* Integrate the Python environment with an RL framework (e.g., **SB3**).  
* Refine the **Reward Function** to heavily incentivize smoothness and speed, while using the Phase 5 derived features to enforce soft constraints (e.g., high penalty for high TTC or high path deviation angle).  
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
* Documentation: Record videos of the "super-human" feats for future articles!  
  Priority: Low

## **🛠 Next Step: Execute Phase 1**

Your next immediate action is to set up the Phase 1 MVP.

**Initial Task:**

1. Verify BeamNG.drive/BeamNG.tech installation.  
2. Install **pip install beamngpy**.  
3. Write the initial connection and teardown script, ensuring your Windows input restriction (ClipCursor) is ready for testing in Phase 2.

Remember to save and back up your work! 💾