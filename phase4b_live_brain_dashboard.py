#!/usr/bin/env python3
"""
Phase 4B: Live Neural Network Visualization Dashboard
BeamNG AI Driver - Real-Time Brain Activity Monitoring

This creates a live dashboard to visualize AI neural network activity
while the AI learns to drive in BeamNG.
"""

import time
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
import json

# BeamNG imports
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Electrics, Damage, GForces

# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    print("🧠 PyTorch available - full neural network integration enabled")
except ImportError:
    TORCH_AVAILABLE = False
    print("📊 Running in visualization mode - install PyTorch for full neural network training")

@dataclass
class BrainActivityData:
    """Data structure for neural network brain activity"""
    timestamp: float
    input_channels: np.ndarray
    layer_activities: Dict[str, float]  # Layer name -> activity level
    output_actions: np.ndarray  # [throttle, steering, brake]
    reward: float
    episode: int
    step: int
    distance: float
    speed: float

class LiveBrainDashboard:
    """Real-time neural network visualization dashboard"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧠 BeamNG AI Brain Monitor - Live Neural Activity")
        self.root.geometry("1200x800")
        
        # Thread-safe data queues
        self.brain_data_queue = queue.Queue(maxsize=50)
        self.activity_history = deque(maxlen=500)  # Store last 500 data points
        
        # Visualization state
        self.is_running = False
        self.update_counter = 0
        
        # Setup the dashboard UI
        self.setup_dashboard()
        
        # Start update timer
        self.start_live_updates()
    
    def setup_dashboard(self):
        """Create the main dashboard interface"""
        # Create main notebook with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Live Brain Activity
        self.setup_brain_activity_tab()
        
        # Tab 2: Training Progress
        self.setup_training_progress_tab()
        
        # Tab 3: Input Channel Analysis
        self.setup_input_analysis_tab()
        
        # Control panel at bottom
        self.setup_control_panel()
    
    def setup_brain_activity_tab(self):
        """Setup the main brain activity visualization tab"""
        self.brain_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.brain_tab, text="🧠 Live Brain Activity")
        
        # Create matplotlib figure for brain visualization
        self.brain_fig, self.brain_axes = plt.subplots(2, 2, figsize=(12, 8))
        self.brain_fig.suptitle("🧠 AI Brain Activity - Real Time", fontsize=14, fontweight='bold')
        
        # Configure subplots
        self.ax_network = self.brain_axes[0, 0]
        self.ax_layers = self.brain_axes[0, 1] 
        self.ax_actions = self.brain_axes[1, 0]
        self.ax_rewards = self.brain_axes[1, 1]
        
        # Setup each subplot
        self.ax_network.set_title("Network Architecture")
        self.ax_layers.set_title("Layer Activity Levels")
        self.ax_actions.set_title("Action Outputs")
        self.ax_rewards.set_title("Reward Over Time")
        
        # Embed matplotlib in tkinter
        self.brain_canvas = FigureCanvasTkAgg(self.brain_fig, self.brain_tab)
        self.brain_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_training_progress_tab(self):
        """Setup training progress monitoring tab"""
        self.progress_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.progress_tab, text="📈 Training Progress")
        
        # Create training metrics display
        self.progress_text = tk.Text(self.progress_tab, font=('Courier', 11), bg='black', fg='green')
        self.progress_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Insert initial text
        self.progress_text.insert(tk.END, "🚀 BeamNG AI Training Monitor\n")
        self.progress_text.insert(tk.END, "=" * 50 + "\n")
        self.progress_text.insert(tk.END, "Waiting for neural network data...\n\n")
    
    def setup_input_analysis_tab(self):
        """Setup input channel utilization analysis tab"""
        self.input_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.input_tab, text="📡 Input Channels")
        
        # Create input channel visualization
        self.input_fig, (self.ax_input_usage, self.ax_input_importance) = plt.subplots(1, 2, figsize=(12, 6))
        self.input_fig.suptitle("📡 Input Channel Analysis", fontsize=14)
        
        self.ax_input_usage.set_title("Channel Utilization")
        self.ax_input_importance.set_title("Channel Importance")
        
        # Embed in tkinter
        self.input_canvas = FigureCanvasTkAgg(self.input_fig, self.input_tab)
        self.input_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_control_panel(self):
        """Setup control panel for dashboard"""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Status: Waiting for AI connection...", 
                                     font=('Arial', 10, 'bold'))
        self.status_label.pack(side=tk.LEFT)
        
        # Control buttons
        self.start_button = ttk.Button(control_frame, text="▶️ Start Monitoring", 
                                      command=self.start_monitoring)
        self.start_button.pack(side=tk.RIGHT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="⏹️ Stop Monitoring", 
                                     command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_button.pack(side=tk.RIGHT, padx=5)
    
    def start_live_updates(self):
        """Start the live update timer"""
        self.root.after(200, self.update_dashboard)  # Update every 200ms (5 Hz)
    
    def update_dashboard(self):
        """Update dashboard with latest data"""
        try:
            # Process any new data from queue
            new_data_count = 0
            while not self.brain_data_queue.empty():
                try:
                    brain_data = self.brain_data_queue.get_nowait()
                    self.activity_history.append(brain_data)
                    new_data_count += 1
                except queue.Empty:
                    break
            
            # Update visualizations if we have data
            if self.activity_history and new_data_count > 0:
                self.update_brain_visualizations()
                self.update_training_progress()
                self.update_input_analysis()
                self.update_counter += 1
            
            # Update status
            if self.activity_history:
                latest = self.activity_history[-1]
                self.status_label.config(text=f"Status: Active - Episode {latest.episode}, Step {latest.step}")
            
        except Exception as e:
            print(f"⚠️  Dashboard update error: {e}")
        
        # Schedule next update
        self.root.after(200, self.update_dashboard)
    
    def update_brain_visualizations(self):
        """Update the brain activity visualizations"""
        if not self.activity_history:
            return
        
        latest_data = self.activity_history[-1]
        
        # Update network architecture visualization
        self.visualize_network_architecture(latest_data)
        
        # Update layer activity levels
        self.visualize_layer_activities(latest_data)
        
        # Update action outputs over time
        self.visualize_action_outputs()
        
        # Update reward progression
        self.visualize_reward_progression()
        
        # Refresh the canvas
        self.brain_canvas.draw_idle()
    
    def visualize_network_architecture(self, data: BrainActivityData):
        """Visualize the neural network architecture with live activity"""
        self.ax_network.clear()
        self.ax_network.set_title("Network Architecture - Live Activity")
        
        # Define network layers for visualization
        layers = [
            {"name": "Input\n54ch", "pos": (1, 3), "size": 1000},
            {"name": "Hidden1\n256", "pos": (3, 3), "size": 1500},
            {"name": "Hidden2\n128", "pos": (5, 3), "size": 1200},
            {"name": "Output\n3act", "pos": (7, 3), "size": 800}
        ]
        
        # Draw layers with activity-based coloring
        for i, layer in enumerate(layers):
            # Calculate activity level (simulate if no real data)
            if layer["name"].startswith("Input"):
                activity = np.mean(np.abs(data.input_channels)) if len(data.input_channels) > 0 else 0.5
            elif layer["name"].startswith("Output"):
                activity = np.mean(np.abs(data.output_actions)) if len(data.output_actions) > 0 else 0.3
            else:
                # Get from layer activities or simulate
                layer_key = f"hidden_{i}"
                activity = data.layer_activities.get(layer_key, np.random.uniform(0.2, 0.8))
            
            # Color intensity based on activity
            color_intensity = min(max(activity, 0), 1)
            color = (1.0, 1.0 - color_intensity, 1.0 - color_intensity)  # Red gradient
            
            # Draw layer node
            self.ax_network.scatter(layer["pos"][0], layer["pos"][1], s=layer["size"], 
                                  c=[color], alpha=0.7, edgecolors='black', linewidth=2)
            
            # Add layer label
            self.ax_network.text(layer["pos"][0], layer["pos"][1], layer["name"], 
                               ha='center', va='center', fontweight='bold', fontsize=8)
        
        # Draw connections between layers
        for i in range(len(layers) - 1):
            start_pos = layers[i]["pos"]
            end_pos = layers[i + 1]["pos"]
            self.ax_network.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                               'k-', alpha=0.3, linewidth=1)
        
        self.ax_network.set_xlim(0, 8)
        self.ax_network.set_ylim(1, 5)
        self.ax_network.set_aspect('equal')
        self.ax_network.axis('off')
    
    def visualize_layer_activities(self, data: BrainActivityData):
        """Visualize layer activity levels as bar chart"""
        self.ax_layers.clear()
        
        # Extract layer activities
        layer_names = list(data.layer_activities.keys())
        activities = list(data.layer_activities.values())
        
        if not layer_names:
            # Show simulation data if no real layer data
            layer_names = ['Input', 'Hidden1', 'Hidden2', 'Output']
            activities = [np.random.uniform(0.2, 0.9) for _ in layer_names]
        
        # Create bar chart
        bars = self.ax_layers.bar(layer_names, activities, color=['lightblue', 'lightgreen', 'yellow', 'orange'])
        
        # Color bars based on activity level
        for bar, activity in zip(bars, activities):
            if activity > 0.7:
                bar.set_color('red')
            elif activity > 0.4:
                bar.set_color('yellow')
            else:
                bar.set_color('lightblue')
        
        self.ax_layers.set_ylabel("Activity Level")
        self.ax_layers.set_ylim(0, 1)
        self.ax_layers.set_title("Layer Activity Levels")
        self.ax_layers.tick_params(axis='x', rotation=45)
    
    def visualize_action_outputs(self):
        """Visualize action outputs over time"""
        self.ax_actions.clear()
        
        if len(self.activity_history) < 2:
            return
        
        # Extract last 50 data points
        recent_data = list(self.activity_history)[-50:]
        
        steps = [data.step for data in recent_data]
        throttles = [data.output_actions[0] if len(data.output_actions) > 0 else 0 for data in recent_data]
        steerings = [data.output_actions[1] if len(data.output_actions) > 1 else 0 for data in recent_data]
        brakes = [data.output_actions[2] if len(data.output_actions) > 2 else 0 for data in recent_data]
        
        # Plot action lines
        self.ax_actions.plot(steps, throttles, 'g-', label='Throttle', linewidth=2)
        self.ax_actions.plot(steps, steerings, 'b-', label='Steering', linewidth=2)
        self.ax_actions.plot(steps, brakes, 'r-', label='Brake', linewidth=2)
        
        self.ax_actions.set_ylabel("Action Value")
        self.ax_actions.set_xlabel("Step")
        self.ax_actions.legend()
        self.ax_actions.grid(True, alpha=0.3)
        self.ax_actions.set_title("Action Outputs Over Time")
    
    def visualize_reward_progression(self):
        """Visualize reward progression over time"""
        self.ax_rewards.clear()
        
        if len(self.activity_history) < 2:
            return
        
        # Extract reward history
        recent_data = list(self.activity_history)[-100:]
        steps = [data.step for data in recent_data]
        rewards = [data.reward for data in recent_data]
        
        # Plot reward line
        self.ax_rewards.plot(steps, rewards, 'purple', linewidth=2)
        self.ax_rewards.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        self.ax_rewards.set_ylabel("Reward")
        self.ax_rewards.set_xlabel("Step")
        self.ax_rewards.grid(True, alpha=0.3)
        self.ax_rewards.set_title("Reward Progression")
    
    def update_training_progress(self):
        """Update the training progress display"""
        if not self.activity_history:
            return
            
        latest = self.activity_history[-1]
        
        # Create progress report
        progress_text = f"""
🧠 AI BRAIN ACTIVITY REPORT - Live Update #{self.update_counter}
{'=' * 60}

Episode: {latest.episode}
Step: {latest.step}
Distance: {latest.distance:.1f}m
Speed: {latest.speed:.1f} m/s
Current Reward: {latest.reward:.2f}

🎮 Action Outputs:
  Throttle: {latest.output_actions[0]:.3f} {'█' * int(latest.output_actions[0] * 10)}
  Steering: {latest.output_actions[1]:+.3f} {'█' * int(abs(latest.output_actions[1]) * 10)}
  Brake:    {latest.output_actions[2]:.3f} {'█' * int(latest.output_actions[2] * 10)}

🧠 Neural Activity:
"""
        
        for layer_name, activity in latest.layer_activities.items():
            activity_bar = '█' * int(activity * 20)
            progress_text += f"  {layer_name}: {activity:.3f} {activity_bar}\n"
        
        # Add performance statistics
        if len(self.activity_history) >= 10:
            recent_rewards = [d.reward for d in list(self.activity_history)[-10:]]
            avg_reward = np.mean(recent_rewards)
            progress_text += f"\n📊 Recent Performance (last 10 steps):\n"
            progress_text += f"  Average Reward: {avg_reward:.2f}\n"
        
        progress_text += f"\n⏰ Last Update: {time.strftime('%H:%M:%S')}\n"
        
        # Update the text display
        self.progress_text.delete(1.0, tk.END)
        self.progress_text.insert(tk.END, progress_text)
        self.progress_text.see(tk.END)
    
    def update_input_analysis(self):
        """Update input channel analysis"""
        if not self.activity_history:
            return
        
        latest = self.activity_history[-1]
        
        # Clear and update input usage plot
        self.ax_input_usage.clear()
        
        if len(latest.input_channels) > 0:
            # Show input channel utilization
            channel_indices = range(min(20, len(latest.input_channels)))  # Show first 20 channels
            channel_values = latest.input_channels[:20]
            
            bars = self.ax_input_usage.bar(channel_indices, np.abs(channel_values))
            
            # Color code by magnitude
            for i, (bar, value) in enumerate(zip(bars, channel_values)):
                if abs(value) > 0.7:
                    bar.set_color('red')
                elif abs(value) > 0.3:
                    bar.set_color('orange')
                else:
                    bar.set_color('lightblue')
            
            self.ax_input_usage.set_xlabel("Input Channel")
            self.ax_input_usage.set_ylabel("Absolute Value")
            self.ax_input_usage.set_title("Input Channel Utilization")
        
        # Update input importance (simulated for now)
        self.ax_input_importance.clear()
        categories = ['Vehicle State', 'Controls', 'Engine', 'Physics', 'Sensors']
        importance = np.random.uniform(0.3, 0.9, len(categories))  # Simulated importance
        
        self.ax_input_importance.pie(importance, labels=categories, autopct='%1.1f%%')
        self.ax_input_importance.set_title("Input Category Importance")
        
        # Refresh canvas
        self.input_canvas.draw_idle()
    
    def add_brain_data(self, brain_data: BrainActivityData):
        """Thread-safe method to add new brain activity data"""
        try:
            self.brain_data_queue.put_nowait(brain_data)
        except queue.Full:
            # Remove oldest and add new
            try:
                self.brain_data_queue.get_nowait()
                self.brain_data_queue.put_nowait(brain_data)
            except queue.Empty:
                pass
    
    def start_monitoring(self):
        """Start the monitoring"""
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        print("▶️ Brain monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring"""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        print("⏹️ Brain monitoring stopped")
    
    def run(self):
        """Run the dashboard"""
        print("🖥️  Starting live brain visualization dashboard...")
        self.root.mainloop()

def simulate_brain_activity(dashboard: LiveBrainDashboard):
    """Simulate brain activity data for demonstration"""
    episode = 1
    step = 0
    
    while True:
        if not dashboard.is_running:
            time.sleep(0.5)
            continue
        
        # Simulate brain activity data
        brain_data = BrainActivityData(
            timestamp=time.time(),
            input_channels=np.random.uniform(-1, 1, 54),  # 54 input channels
            layer_activities={
                'input': np.random.uniform(0.3, 0.8),
                'hidden1': np.random.uniform(0.2, 0.9),
                'hidden2': np.random.uniform(0.1, 0.7),
                'output': np.random.uniform(0.2, 0.6)
            },
            output_actions=np.random.uniform(0, 1, 3),  # [throttle, steering, brake]
            reward=np.random.uniform(-2, 5),
            episode=episode,
            step=step,
            distance=step * 0.5,  # Simulate distance progression
            speed=np.random.uniform(0, 15)
        )
        
        # Add to dashboard
        dashboard.add_brain_data(brain_data)
        
        step += 1
        if step > 1000:  # Reset episode
            episode += 1
            step = 0
        
        time.sleep(0.1)  # 10 Hz simulation

def main():
    """Phase 4B Main: Live Neural Network Visualization"""
    print("🚀 Phase 4B: Live Neural Network Visualization Dashboard")
    print("=" * 60)
    print("Features:")
    print("• Real-time neural network activity monitoring")
    print("• Live brain visualization with layer activities")
    print("• Action output tracking and reward progression")
    print("• Input channel utilization analysis")
    print()
    
    # Create the dashboard
    dashboard = LiveBrainDashboard()
    
    # Start simulation in separate thread
    sim_thread = threading.Thread(target=simulate_brain_activity, args=(dashboard,), daemon=True)
    sim_thread.start()
    
    print("🖥️  Dashboard ready! Click 'Start Monitoring' to begin")
    
    # Run the dashboard
    dashboard.run()
    
    print("🔒 Dashboard closed")

if __name__ == "__main__":
    main()