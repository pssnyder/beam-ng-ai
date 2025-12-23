"""
Phase 4B Alternative: CPU-Optimized Neural Network Training with Live Visualization
===================================================================================

This version is optimized for CPU training while we set up CUDA.
It includes all the neural network features but with lighter computational load.

Features:
- Optimized SAC neural network for CPU training
- Live brain visualization dashboard  
- Real-time training monitoring
- Smaller network architecture for faster CPU training
- Auto-recovery system

Author: GitHub Copilot
Date: 2024
"""

import numpy as np
import threading
import time
import queue
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import logging

# GUI and visualization imports
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation

# BeamNG and AI imports  
from beamngpy import BeamNGpy, Scenario, Vehicle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BrainActivityData:
    """Data structure for neural network activity monitoring"""
    timestamp: float
    layer_activations: Dict[str, np.ndarray] = field(default_factory=dict)
    action_outputs: Dict[str, float] = field(default_factory=dict)
    reward: float = 0.0
    loss_values: Dict[str, float] = field(default_factory=dict)
    episode_stats: Dict[str, Any] = field(default_factory=dict)

class CPUOptimizedSAC:
    """CPU-Optimized Soft Actor-Critic neural network"""
    
    def __init__(self, state_dim=142, action_dim=4, hidden_dim=128, lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim  # Smaller for CPU
        
        # Use CPU and enable optimizations
        torch.set_num_threads(4)  # Optimize for CPU cores
        self.device = torch.device('cpu')
        
        # Build smaller networks for CPU efficiency
        self.actor = self._build_actor()
        self.critic1 = self._build_critic()
        self.critic2 = self._build_critic()
        
        # Optimizers with higher learning rate for faster convergence
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # SAC parameters
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -action_dim
        
        # Brain monitoring
        self.brain_data = BrainActivityData(timestamp=time.time())
        
        print(f"🧠 CPU-Optimized SAC initialized with {hidden_dim} hidden units")
        
    def _build_actor(self) -> nn.Module:
        """Build compact actor network for CPU"""
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.action_dim * 2)  # mean and log_std
        )
    
    def _build_critic(self) -> nn.Module:
        """Build compact critic network for CPU"""
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, float]]:
        """Get action from actor network"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Forward pass
            actor_output = self.actor(state_tensor)
            mean, log_std = actor_output.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp()
            
            if deterministic:
                action = torch.tanh(mean)
            else:
                normal = torch.distributions.Normal(mean, std)
                x_t = normal.rsample()
                action = torch.tanh(x_t)
            
            action_np = action.cpu().numpy().flatten()
            
            # Store for visualization
            self.brain_data.action_outputs = {
                'steering': float(action_np[0]),
                'throttle': float(action_np[1]),
                'brake': float(action_np[2]),
                'gear': float(action_np[3])
            }
            self.brain_data.timestamp = time.time()
            
            # Store layer activations for brain visualization
            self.brain_data.layer_activations = {
                'actor_input': state_tensor.cpu().numpy(),
                'actor_output': actor_output.cpu().numpy(),
                'action_mean': mean.cpu().numpy(),
                'action_std': std.cpu().numpy()
            }
            
            return action_np, self.brain_data.action_outputs
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Simplified update for CPU training"""
        state = batch['state']
        action = batch['action']
        reward = batch['reward']
        next_state = batch['next_state']
        done = batch['done']
        
        # Critic updates (simplified)
        with torch.no_grad():
            next_action, next_log_prob = self._sample_action(next_state)
            target_q1 = self.critic1(torch.cat([next_state, next_action], dim=1))
            target_q2 = self.critic2(torch.cat([next_state, next_action], dim=1))
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
            target_value = reward + (1 - done) * 0.99 * target_q
        
        current_q1 = self.critic1(torch.cat([state, action], dim=1))
        current_q2 = self.critic2(torch.cat([state, action], dim=1))
        
        critic1_loss = F.mse_loss(current_q1, target_value)
        critic2_loss = F.mse_loss(current_q2, target_value)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Actor update (every other step for efficiency)
        new_action, log_prob = self._sample_action(state)
        q1_new = self.critic1(torch.cat([state, new_action], dim=1))
        q2_new = self.critic2(torch.cat([state, new_action], dim=1))
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.log_alpha.exp() * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Alpha update
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Store loss values
        loss_info = {
            'actor_loss': float(actor_loss.item()),
            'critic1_loss': float(critic1_loss.item()),
            'critic2_loss': float(critic2_loss.item()),
            'alpha_loss': float(alpha_loss.item()),
            'alpha': float(self.log_alpha.exp().item())
        }
        
        self.brain_data.loss_values = loss_info
        return loss_info
    
    def _sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action with log probability"""
        actor_output = self.actor(state)
        mean, log_std = actor_output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob

class SimpleBrainDashboard:
    """Simplified brain dashboard for CPU training"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧠 Phase 4B: CPU-Optimized Neural Training Monitor")
        self.root.geometry("1200x700")
        
        # Data for visualization
        self.brain_queue = queue.Queue()
        self.is_monitoring = False
        
        # Setup GUI
        self._setup_gui()
        
    def _setup_gui(self):
        """Setup simplified GUI"""
        # Title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(title_frame, text="🧠 CPU-Optimized Neural Network Training", 
                 font=("Arial", 14, "bold")).pack(side=tk.LEFT)
        
        # Control buttons
        control_frame = ttk.Frame(title_frame)
        control_frame.pack(side=tk.RIGHT)
        
        self.start_btn = ttk.Button(control_frame, text="▶️ Start", command=self.start_monitoring)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="⏹️ Stop", command=self.stop_monitoring)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Create main content area
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Split into left and right
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Brain activity plot
        self.brain_fig = Figure(figsize=(8, 6))
        self.brain_ax = self.brain_fig.add_subplot(111)
        self.brain_ax.set_title("🧠 Live Neural Activity")
        self.brain_canvas = FigureCanvasTkAgg(self.brain_fig, left_frame)
        self.brain_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Action outputs
        actions_frame = ttk.LabelFrame(right_frame, text="🎮 Actions")
        actions_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.action_vars = {}
        for action in ['steering', 'throttle', 'brake', 'gear']:
            frame = ttk.Frame(actions_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(frame, text=f"{action.title()}:").pack(side=tk.LEFT)
            var = tk.StringVar(value="0.000")
            self.action_vars[action] = var
            ttk.Label(frame, textvariable=var, font=("Courier", 10)).pack(side=tk.RIGHT)
        
        # Training stats
        stats_frame = ttk.LabelFrame(right_frame, text="📊 Training Stats")
        stats_frame.pack(fill=tk.BOTH, expand=True)
        
        self.stats_text = tk.Text(stats_frame, font=("Courier", 9))
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize data storage
        self.neural_activity = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        
    def start_monitoring(self):
        """Start monitoring"""
        self.is_monitoring = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Start update loop
        self._update_display()
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
    def update_brain_data(self, brain_data: BrainActivityData):
        """Update brain data"""
        if self.is_monitoring:
            try:
                self.brain_queue.put_nowait(brain_data)
            except queue.Full:
                pass
    
    def _update_display(self):
        """Update display with latest data"""
        if not self.is_monitoring:
            return
        
        # Process queue
        while not self.brain_queue.empty():
            try:
                brain_data = self.brain_queue.get_nowait()
                self._process_brain_data(brain_data)
            except queue.Empty:
                break
        
        # Update plots
        self._update_brain_plot()
        
        # Schedule next update
        self.root.after(100, self._update_display)
    
    def _process_brain_data(self, brain_data: BrainActivityData):
        """Process incoming brain data"""
        # Update action displays
        for action, value in brain_data.action_outputs.items():
            if action in self.action_vars:
                self.action_vars[action].set(f"{value:+.3f}")
        
        # Store neural activity
        if 'actor_output' in brain_data.layer_activations:
            activity = np.mean(np.abs(brain_data.layer_activations['actor_output']))
            self.neural_activity.append(activity)
        
        # Store reward
        if brain_data.reward:
            self.reward_history.append(brain_data.reward)
        
        # Update stats text
        stats = []
        stats.append(f"Timestamp: {brain_data.timestamp:.2f}")
        stats.append(f"Reward: {brain_data.reward:.3f}")
        
        if brain_data.loss_values:
            stats.append("--- Losses ---")
            for loss_name, loss_val in brain_data.loss_values.items():
                stats.append(f"{loss_name}: {loss_val:.4f}")
        
        if brain_data.episode_stats:
            stats.append("--- Episode ---")
            for stat_name, stat_val in brain_data.episode_stats.items():
                stats.append(f"{stat_name}: {stat_val}")
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, "\n".join(stats))
    
    def _update_brain_plot(self):
        """Update brain activity plot"""
        self.brain_ax.clear()
        
        if len(self.neural_activity) > 1:
            self.brain_ax.plot(list(self.neural_activity), 'b-', label='Neural Activity')
            self.brain_ax.set_ylabel('Activity Level')
            
        if len(self.reward_history) > 1:
            ax2 = self.brain_ax.twinx()
            ax2.plot(list(self.reward_history), 'r-', alpha=0.7, label='Reward')
            ax2.set_ylabel('Reward')
            
        self.brain_ax.set_title('🧠 Live Neural Activity & Rewards')
        self.brain_ax.set_xlabel('Time Steps')
        self.brain_ax.legend()
        self.brain_ax.grid(True, alpha=0.3)
        
        self.brain_canvas.draw()
    
    def run(self):
        """Run the dashboard"""
        self.root.mainloop()

class SimplifiedTrainingEnvironment:
    """Simplified training environment for CPU"""
    
    def __init__(self):
        # Components
        self.bng = None
        self.vehicle = None
        self.sac = CPUOptimizedSAC()
        self.replay_buffer = deque(maxlen=10000)  # Smaller buffer
        
        # Training state
        self.episode = 0
        self.total_steps = 0
        self.episode_reward = 0
        
        # Dashboard
        self.dashboard = None
        self.is_training = False
        
    def connect_beamng(self):
        """Connect to BeamNG"""
        try:
            logger.info("🚗 Connecting to BeamNG.drive...")
            self.bng = BeamNGpy('localhost', 64256, home='S:/SteamLibrary/steamapps/common/BeamNG.drive')
            self.bng.open(launch=False)
            
            scenario = Scenario('automation_test_track', 'Phase4B_CPU')
            self.vehicle = Vehicle('ego_vehicle', model='etk800', license='CPU4B')
            scenario.add_vehicle(self.vehicle, pos=(0, 0, 0), rot_quat=(0, 0, 0, 1))
            
            scenario.make(self.bng)
            self.bng.load_scenario(scenario)
            self.bng.start_scenario()
            
            logger.info("✅ BeamNG.drive connected!")
            return True
            
        except Exception as e:
            logger.error(f"❌ BeamNG connection failed: {e}")
            return False
    
    def get_state(self) -> np.ndarray:
        """Get simplified state"""
        try:
            state = self.vehicle.state
            pos = state.get('pos', [0, 0, 0])
            vel = state.get('vel', [0, 0, 0])
            rotation = state.get('rotation', [0, 0, 0])
            
            # Create simple state vector
            state_data = pos + vel + rotation
            
            # Pad to 142 dimensions
            while len(state_data) < 142:
                state_data.append(0.0)
            
            return np.array(state_data[:142], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"⚠️ State error: {e}")
            return np.zeros(142, dtype=np.float32)
    
    def apply_action(self, action: np.ndarray):
        """Apply action to vehicle"""
        try:
            steering = float(np.clip(action[0], -1, 1))
            throttle = float(np.clip(action[1], 0, 1))
            brake = float(np.clip(action[2], 0, 1))
            
            self.vehicle.control(steering=steering, throttle=throttle, brake=brake)
            
        except Exception as e:
            logger.warning(f"⚠️ Action error: {e}")
    
    def calculate_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """Simple reward calculation"""
        velocity = np.linalg.norm(state[3:6])
        reward = velocity * 0.1  # Reward for movement
        reward -= abs(action[0]) * 0.01  # Penalty for steering
        return reward
    
    def train_step(self):
        """Training step"""
        if len(self.replay_buffer) < 100:  # Smaller minimum
            return
        
        # Sample batch
        batch_data = random.sample(self.replay_buffer, min(32, len(self.replay_buffer)))
        states, actions, rewards, next_states, dones = zip(*batch_data)
        
        batch = {
            'state': torch.FloatTensor(np.array(states)),
            'action': torch.FloatTensor(np.array(actions)),
            'reward': torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            'next_state': torch.FloatTensor(np.array(next_states)),
            'done': torch.FloatTensor(np.array(dones)).unsqueeze(1)
        }
        
        # Update network
        loss_info = self.sac.update(batch)
        return loss_info
    
    def run_episode(self):
        """Run training episode"""
        self.episode += 1
        self.episode_reward = 0
        
        logger.info(f"🎯 Episode {self.episode} starting...")
        
        state = self.get_state()
        
        for step in range(200):  # Shorter episodes for CPU
            if not self.is_training:
                break
            
            # Get action
            action, action_outputs = self.sac.get_action(state)
            
            # Apply action
            self.apply_action(action)
            time.sleep(0.2)  # Slower for CPU
            
            # Get next state
            next_state = self.get_state()
            reward = self.calculate_reward(state, action)
            done = step >= 199
            
            # Store experience
            self.replay_buffer.append((state, action, reward, next_state, done))
            
            # Update counters
            self.episode_reward += reward
            self.total_steps += 1
            
            # Update brain data
            brain_data = self.sac.brain_data
            brain_data.reward = reward
            brain_data.episode_stats = {
                'episode': self.episode,
                'step': step,
                'total_reward': self.episode_reward
            }
            
            if self.dashboard:
                self.dashboard.update_brain_data(brain_data)
            
            # Train every 8 steps (less frequent for CPU)
            if step % 8 == 0:
                self.train_step()
            
            state = next_state
            
            if done:
                break
        
        logger.info(f"✅ Episode {self.episode}: {self.episode_reward:.2f} reward")
    
    def start_training_with_dashboard(self):
        """Start training with dashboard"""
        if not self.connect_beamng():
            return False
        
        # Start dashboard
        def run_dashboard():
            self.dashboard = SimpleBrainDashboard()
            self.dashboard.run()
        
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        time.sleep(2)
        
        # Training loop
        self.is_training = True
        logger.info("🚀 Starting CPU-optimized training...")
        
        try:
            while self.is_training and self.episode < 50:
                self.run_episode()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Training stopped")
        finally:
            self.is_training = False
            if self.bng:
                self.bng.close()
        
        return True

def main():
    """Main entry point"""
    print("🧠 Phase 4B: CPU-Optimized Neural Network Training")
    print("=================================================")
    print()
    print("Features:")
    print("• CPU-optimized SAC neural network")
    print("• Live brain activity visualization")
    print("• Simplified training for faster CPU performance")
    print("• Real-time neural activity monitoring")
    print()
    
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"💻 Using CPU with {torch.get_num_threads()} threads")
    print()
    
    try:
        env = SimplifiedTrainingEnvironment()
        success = env.start_training_with_dashboard()
        
        if success:
            print("✅ Phase 4B CPU training completed!")
        else:
            print("❌ Training failed to start")
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()