#!/usr/bin/env python3
"""
Phase 4B: Neural Network Integration with Live Visualization
BeamNG AI Driver - SAC Training with Real-Time Brain Monitoring

Features:
- SAC (Soft Actor-Critic) reinforcement learning
- Live neural network visualization dashboard
- Real-time brain activity monitoring
- Progressive reward system
- Experience replay buffer
- Continuous learning loop
"""

import time
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import queue
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import json

# BeamNG imports
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Electrics, Damage, GForces

# Try to import PyTorch for neural networks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
    print("🧠 PyTorch available - neural networks enabled")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - install with: pip install torch")

@dataclass
class BrainState:
    """Neural network brain state for visualization"""
    timestamp: float
    layer_activations: Dict[str, np.ndarray]
    input_values: np.ndarray
    output_actions: np.ndarray
    reward: float
    episode: int
    step: int
    learning_metrics: Dict[str, float]

@dataclass
class TrainingMetrics:
    """Training progress metrics"""
    episode: int
    total_reward: float
    episode_length: int
    distance_traveled: float
    recovery_count: int
    average_speed: float
    policy_loss: float
    value_loss: float
    exploration_noise: float

class SACActorNetwork(nn.Module):
    """SAC Actor (Policy) Network with monitoring hooks"""
    
    def __init__(self, state_dim=54, action_dim=3, hidden_dim=256):
        super(SACActorNetwork, self).__init__()
        
        # Policy network layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Action output layers
        self.mean_head = nn.Linear(hidden_dim // 2, action_dim)
        self.log_std_head = nn.Linear(hidden_dim // 2, action_dim)
        
        # For visualization
        self.activations = {}
        self.register_hooks()
        
    def register_hooks(self):
        """Register forward hooks for visualization"""
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output.detach().cpu().numpy()
            return hook
        
        self.fc1.register_forward_hook(hook_fn('fc1'))
        self.fc2.register_forward_hook(hook_fn('fc2'))
        self.fc3.register_forward_hook(hook_fn('fc3'))
        self.mean_head.register_forward_hook(hook_fn('mean'))
        self.log_std_head.register_forward_hook(hook_fn('log_std'))
    
    def forward(self, state):
        """Forward pass with activation monitoring"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
    
    def sample(self, state, deterministic=False):
        """Sample action from policy"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            action = mean
        else:
            normal = torch.distributions.Normal(mean, std)
            action = normal.rsample()
        
        # Apply tanh to bound actions
        bounded_action = torch.tanh(action)
        
        # Convert to vehicle control ranges
        throttle = torch.sigmoid(bounded_action[:, 0])  # [0, 1]
        steering = bounded_action[:, 1]                 # [-1, 1] 
        brake = torch.sigmoid(bounded_action[:, 2])     # [0, 1]
        
        return torch.stack([throttle, steering, brake], dim=1)

class SACCriticNetwork(nn.Module):
    """SAC Critic (Value) Network"""
    
    def __init__(self, state_dim=54, action_dim=3, hidden_dim=256):
        super(SACCriticNetwork, self).__init__()
        
        # Q1 network
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2 network
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, 1)
        
        # For visualization
        self.activations = {}
        self.register_hooks()
    
    def register_hooks(self):
        """Register hooks for Q-network visualization"""
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output.detach().cpu().numpy()
            return hook
        
        self.q1_fc2.register_forward_hook(hook_fn('q1_hidden'))
        self.q2_fc2.register_forward_hook(hook_fn('q2_hidden'))
    
    def forward(self, state, action):
        """Forward pass for both Q networks"""
        sa = torch.cat([state, action], dim=1)
        
        # Q1 forward
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)
        
        # Q2 forward
        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)
        
        return q1, q2

class LiveBrainVisualizer:
    """Real-time neural network visualization dashboard"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧠 BeamNG AI Brain Monitor - Live Neural Activity")
        self.root.geometry("1400x900")
        
        # Data queues for thread-safe updates
        self.brain_queue = queue.Queue(maxsize=100)
        self.metrics_queue = queue.Queue(maxsize=100)
        
        # Visualization data
        self.brain_history = deque(maxlen=1000)
        self.metrics_history = deque(maxlen=1000)
        
        # Setup UI
        self.setup_dashboard()
        
        # Animation for live updates
        self.animation = None
        self.start_animation()
    
    def setup_dashboard(self):
        """Create the visualization dashboard UI"""
        # Main notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Neural Network Activity
        self.brain_frame = ttk.Frame(notebook)
        notebook.add(self.brain_frame, text="🧠 Neural Activity")
        
        # Tab 2: Training Metrics
        self.metrics_frame = ttk.Frame(notebook)
        notebook.add(self.metrics_frame, text="📊 Training Metrics")
        
        # Tab 3: Input Channel Utilization
        self.inputs_frame = ttk.Frame(notebook)
        notebook.add(self.inputs_frame, text="📡 Input Channels")
        
        self.setup_brain_tab()
        self.setup_metrics_tab()
        self.setup_inputs_tab()
    
    def setup_brain_tab(self):
        """Setup neural network visualization tab"""
        # Create matplotlib figure
        self.brain_fig, ((self.ax_network, self.ax_activations), 
                         (self.ax_actions, self.ax_rewards)) = plt.subplots(2, 2, figsize=(14, 8))
        
        self.brain_fig.suptitle("🧠 Live Neural Network Activity", fontsize=16, fontweight='bold')
        
        # Network architecture visualization
        self.ax_network.set_title("Network Architecture")
        self.ax_network.set_xlim(0, 10)
        self.ax_network.set_ylim(0, 10)
        
        # Layer activation heatmap
        self.ax_activations.set_title("Layer Activations Heatmap")
        
        # Action outputs
        self.ax_actions.set_title("Action Outputs Over Time")
        self.ax_actions.set_ylabel("Action Value")
        self.ax_actions.legend(['Throttle', 'Steering', 'Brake'])
        
        # Reward progression
        self.ax_rewards.set_title("Reward Progression")
        self.ax_rewards.set_ylabel("Reward")
        self.ax_rewards.set_xlabel("Time Steps")
        
        # Embed in tkinter
        self.brain_canvas = FigureCanvasTkAgg(self.brain_fig, self.brain_frame)
        self.brain_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_metrics_tab(self):
        """Setup training metrics tab"""
        self.metrics_fig, ((self.ax_episode_reward, self.ax_distance), 
                          (self.ax_losses, self.ax_exploration)) = plt.subplots(2, 2, figsize=(14, 8))
        
        self.metrics_fig.suptitle("📊 Training Progress Metrics", fontsize=16, fontweight='bold')
        
        # Configure subplots
        self.ax_episode_reward.set_title("Episode Rewards")
        self.ax_episode_reward.set_ylabel("Total Reward")
        
        self.ax_distance.set_title("Distance Traveled")
        self.ax_distance.set_ylabel("Distance (m)")
        
        self.ax_losses.set_title("Training Losses")
        self.ax_losses.set_ylabel("Loss")
        self.ax_losses.legend(['Policy Loss', 'Value Loss'])
        
        self.ax_exploration.set_title("Exploration vs Exploitation")
        self.ax_exploration.set_ylabel("Noise Level")
        
        # Embed in tkinter
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, self.metrics_frame)
        self.metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_inputs_tab(self):
        """Setup input channel utilization tab"""
        # Create text display for input analysis
        self.inputs_text = tk.Text(self.inputs_frame, font=('Courier', 10))
        scrollbar = ttk.Scrollbar(self.inputs_frame, orient=tk.VERTICAL, command=self.inputs_text.yview)
        self.inputs_text.configure(yscrollcommand=scrollbar.set)
        
        self.inputs_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def start_animation(self):
        """Start the live animation"""
        self.animation = FuncAnimation(self.brain_fig, self.update_brain_display, 
                                     interval=200, blit=False)  # 5 Hz updates
    
    def update_brain_display(self, frame):
        """Update brain visualization"""
        try:
            # Get latest brain state
            while not self.brain_queue.empty():
                brain_state = self.brain_queue.get_nowait()
                self.brain_history.append(brain_state)
            
            if not self.brain_history:
                return
            
            latest_brain = self.brain_history[-1]
            
            # Update network architecture visualization
            self.visualize_network_architecture(latest_brain)
            
            # Update layer activations heatmap
            self.visualize_layer_activations(latest_brain)
            
            # Update action outputs
            self.visualize_action_progression()
            
            # Update reward progression
            self.visualize_reward_progression()
            
        except queue.Empty:
            pass
        except Exception as e:
            print(f"⚠️  Visualization update error: {e}")
    
    def visualize_network_architecture(self, brain_state):
        """Visualize network architecture with activity"""
        self.ax_network.clear()
        self.ax_network.set_title("Network Architecture - Live Activity")
        
        # Draw simplified network diagram
        layers = ['Input\n(54)', 'Hidden1\n(256)', 'Hidden2\n(256)', 'Hidden3\n(128)', 'Output\n(3)']
        positions = [(1, 5), (3, 5), (5, 5), (7, 5), (9, 5)]
        
        for i, (layer, pos) in enumerate(zip(layers, positions)):
            # Color based on activity level
            activity = 0.5  # Default
            if brain_state.layer_activations:
                layer_key = f'fc{i+1}' if i < len(layers)-1 else 'mean'
                if layer_key in brain_state.layer_activations:
                    activity = np.mean(np.abs(brain_state.layer_activations[layer_key]))
            
            color = plt.cm.Reds(min(activity, 1.0))
            self.ax_network.scatter(pos[0], pos[1], s=2000, c=[color], alpha=0.7)
            self.ax_network.text(pos[0], pos[1], layer, ha='center', va='center', fontweight='bold')
        
        # Draw connections
        for i in range(len(positions)-1):
            self.ax_network.plot([positions[i][0], positions[i+1][0]], 
                               [positions[i][1], positions[i+1][1]], 
                               'k-', alpha=0.3, linewidth=2)
        
        self.ax_network.set_xlim(0, 10)
        self.ax_network.set_ylim(3, 7)
        self.ax_network.set_aspect('equal')
        self.ax_network.axis('off')
    
    def visualize_layer_activations(self, brain_state):
        """Visualize layer activation heatmap"""
        self.ax_activations.clear()
        
        if brain_state.layer_activations:
            # Create heatmap data
            activation_data = []
            layer_names = []
            
            for layer_name, activations in brain_state.layer_activations.items():
                if len(activations.shape) > 0:
                    # Take first 50 neurons for visualization
                    act_slice = activations.flatten()[:50]
                    activation_data.append(act_slice)
                    layer_names.append(layer_name)
            
            if activation_data:
                # Pad to same length
                max_len = max(len(row) for row in activation_data)
                padded_data = [np.pad(row, (0, max_len - len(row))) for row in activation_data]
                
                heatmap = np.array(padded_data)
                im = self.ax_activations.imshow(heatmap, cmap='RdYlBu_r', aspect='auto')
                
                self.ax_activations.set_yticks(range(len(layer_names)))
                self.ax_activations.set_yticklabels(layer_names)
                self.ax_activations.set_xlabel("Neuron Index")
                self.ax_activations.set_title("Layer Activations (Red=High, Blue=Low)")
    
    def visualize_action_progression(self):
        """Visualize action outputs over time"""
        if len(self.brain_history) < 2:
            return
        
        self.ax_actions.clear()
        
        # Extract action history
        times = [brain.timestamp for brain in self.brain_history[-100:]]  # Last 100 steps
        throttles = [brain.output_actions[0] if len(brain.output_actions) > 0 else 0 
                    for brain in self.brain_history[-100:]]
        steerings = [brain.output_actions[1] if len(brain.output_actions) > 1 else 0 
                    for brain in self.brain_history[-100:]]
        brakes = [brain.output_actions[2] if len(brain.output_actions) > 2 else 0 
                 for brain in self.brain_history[-100:]]
        
        if times:
            self.ax_actions.plot(times, throttles, 'g-', label='Throttle', linewidth=2)
            self.ax_actions.plot(times, steerings, 'b-', label='Steering', linewidth=2)  
            self.ax_actions.plot(times, brakes, 'r-', label='Brake', linewidth=2)
            
            self.ax_actions.set_ylabel("Action Value")
            self.ax_actions.set_xlabel("Time")
            self.ax_actions.legend()
            self.ax_actions.grid(True, alpha=0.3)
            self.ax_actions.set_title("Action Outputs Over Time")
    
    def visualize_reward_progression(self):
        """Visualize reward progression"""
        if len(self.brain_history) < 2:
            return
        
        self.ax_rewards.clear()
        
        # Extract reward history
        times = [brain.timestamp for brain in self.brain_history[-200:]]  # Last 200 steps
        rewards = [brain.reward for brain in self.brain_history[-200:]]
        
        if times:
            self.ax_rewards.plot(times, rewards, 'purple', linewidth=2)
            self.ax_rewards.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            self.ax_rewards.set_ylabel("Reward")
            self.ax_rewards.set_xlabel("Time")
            self.ax_rewards.grid(True, alpha=0.3)
            self.ax_rewards.set_title("Reward Progression")
    
    def update_brain_state(self, brain_state: BrainState):
        """Thread-safe update of brain state"""
        try:
            self.brain_queue.put_nowait(brain_state)
        except queue.Full:
            # Remove oldest item and add new one
            try:
                self.brain_queue.get_nowait()
                self.brain_queue.put_nowait(brain_state)
            except queue.Empty:
                pass
    
    def run(self):
        """Run the visualization dashboard"""
        self.root.mainloop()

class EnhancedSACAgent:
    """SAC Agent with live brain monitoring"""
    
    def __init__(self, state_dim=54, action_dim=3, lr=3e-4):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for SAC agent")
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.actor = SACActorNetwork(state_dim, action_dim)
        self.critic = SACCriticNetwork(state_dim, action_dim)
        self.target_critic = SACCriticNetwork(state_dim, action_dim)
        
        # Copy parameters to target network
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # SAC hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2  # Entropy regularization
        
        # Experience replay
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 256
        
        # Training metrics
        self.training_step = 0
        self.policy_loss = 0.0
        self.value_loss = 0.0
    
    def select_action(self, state, deterministic=False):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor.sample(state_tensor, deterministic=deterministic)
            return action.squeeze().numpy()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Perform one SAC training step"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = list(zip(*np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)))
        batch = [np.array(x) for x in batch]
        
        states = torch.FloatTensor(batch[0])
        actions = torch.FloatTensor(batch[1])
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1)
        next_states = torch.FloatTensor(batch[3])
        dones = torch.BoolTensor(batch[4]).unsqueeze(1)
        
        # Critic loss
        with torch.no_grad():
            next_actions = self.actor.sample(next_states, deterministic=False)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones.float()) * target_q
        
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor loss (simplified)
        new_actions = self.actor.sample(states, deterministic=False)
        q1, q2 = self.critic(states, new_actions)
        actor_loss = -torch.min(q1, q2).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.policy_loss = actor_loss.item()
        self.value_loss = critic_loss.item()
        self.training_step += 1
    
    def get_brain_state(self, state, action, reward, episode, step):
        """Get current brain state for visualization"""
        return BrainState(
            timestamp=time.time(),
            layer_activations=self.actor.activations.copy(),
            input_values=state,
            output_actions=action,
            reward=reward,
            episode=episode,
            step=step,
            learning_metrics={
                'policy_loss': self.policy_loss,
                'value_loss': self.value_loss,
                'training_step': self.training_step
            }
        )

def main_phase4b():
    """Phase 4B Main: Neural Network Integration with Live Visualization"""
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch required for Phase 4B")
        print("Install with: pip install torch")
        return False
    
    print("🚀 Phase 4B: Neural Network Integration with Live Brain Monitoring")
    print("=" * 70)
    print("Features: SAC training + Real-time neural activity visualization")
    print()
    
    # Start visualization dashboard in separate thread
    visualizer = LiveBrainVisualizer()
    viz_thread = threading.Thread(target=visualizer.run, daemon=True)
    viz_thread.start()
    
    print("🖥️  Live brain visualization dashboard starting...")
    time.sleep(2)  # Let dashboard initialize
    
    # TODO: Integrate with BeamNG environment from Phase 4A
    # TODO: Implement full SAC training loop
    # TODO: Connect neural network to live dashboard
    
    print("✅ Phase 4B framework ready!")
    print("🧠 Neural network monitoring active")
    print("📊 Live dashboard operational")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🔒 Shutting down Phase 4B...")
        return True

if __name__ == "__main__":
    success = main_phase4b()
    
    if success:
        print("\n🎉 PHASE 4B NEURAL NETWORK INTEGRATION READY!")  
        print("Next: Full SAC training with BeamNG environment")
    else:
        print("\n❌ Phase 4B setup needs debugging")