"""
Phase 4B: Integrated Neural Network Training with Live Brain Visualization
=========================================================================

This is the complete Phase 4B implementation that integrates:
- SAC (Soft Actor-Critic) neural network training 
- Live brain visualization dashboard
- Exploratory reinforcement learning environment
- Real-time neural activity monitoring

Features:
- Live neural network activity visualization during training
- Real-time action output and reward tracking
- Experience replay buffer monitoring
- Auto-recovery system for continuous learning
- Multi-threaded training with live dashboard updates

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
from contextlib import contextmanager

# GUI and visualization imports
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
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
    network_weights: Dict[str, np.ndarray] = field(default_factory=dict)
    gradient_norms: Dict[str, float] = field(default_factory=dict)

class SACNetwork:
    """Soft Actor-Critic neural network implementation with brain monitoring"""
    
    def __init__(self, state_dim=142, action_dim=4, hidden_dim=256, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Device configuration - optimize for RTX 4070 Ti
        self.device = self._get_optimal_device()
        print(f"🔥 Using device: {self.device}")
        
        # Initialize networks
        self.actor = self._build_actor().to(self.device)
        self.critic1 = self._build_critic().to(self.device)
        self.critic2 = self._build_critic().to(self.device)
        self.target_critic1 = self._build_critic().to(self.device)
        self.target_critic2 = self._build_critic().to(self.device)
        
        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # SAC specific parameters
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -action_dim
        
        # Brain monitoring
        self.brain_data = BrainActivityData(timestamp=time.time())
        self.layer_hooks = {}
        self._register_hooks()
        
    def _get_optimal_device(self) -> torch.device:
        """Get optimal device for training (GPU if available, CPU otherwise)"""
        if torch.cuda.is_available():
            # Use RTX 4070 Ti if available
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🎮 Detected GPU: {gpu_name}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return torch.device('cuda:0')
        else:
            print("💻 CUDA not available, using CPU (consider installing CUDA-enabled PyTorch)")
            return torch.device('cpu')
        
    def _build_actor(self) -> nn.Module:
        """Build actor network with monitored layers"""
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim * 2)  # mean and log_std
        )
    
    def _build_critic(self) -> nn.Module:
        """Build critic network with monitored layers"""
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
    
    def _register_hooks(self):
        """Register forward hooks for brain activity monitoring"""
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.brain_data.layer_activations[name] = output.detach().cpu().numpy()
            return hook
        
        # Register hooks for actor network
        for i, layer in enumerate(self.actor):
            if isinstance(layer, (nn.Linear, nn.ReLU)):
                hook = make_hook(f"actor_layer_{i}")
                layer.register_forward_hook(hook)
        
        # Register hooks for critic networks
        for i, layer in enumerate(self.critic1):
            if isinstance(layer, (nn.Linear, nn.ReLU)):
                hook = make_hook(f"critic1_layer_{i}")
                layer.register_forward_hook(hook)
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, float]]:
        """Get action from actor network with brain monitoring"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Forward pass through actor
        actor_output = self.actor(state_tensor)
        mean, log_std = actor_output.chunk(2, dim=-1)
        
        # Clamp log_std for stability
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        if deterministic:
            action = torch.tanh(mean)
        else:
            # Sample action using reparameterization trick
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
        
        # Convert to numpy and store brain data
        action_np = action.detach().cpu().numpy().flatten()
        
        # Store action outputs for visualization
        self.brain_data.action_outputs = {
            'steering': float(action_np[0]),
            'throttle': float(action_np[1]),
            'brake': float(action_np[2]),
            'gear': float(action_np[3])
        }
        self.brain_data.timestamp = time.time()
        
        return action_np, self.brain_data.action_outputs
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update networks and return loss information for visualization"""
        state = batch['state']
        action = batch['action']
        reward = batch['reward']
        next_state = batch['next_state']
        done = batch['done']
        
        # Update critics
        with torch.no_grad():
            next_action, next_log_prob = self._sample_action(next_state)
            target_q1 = self.target_critic1(torch.cat([next_state, next_action], dim=1))
            target_q2 = self.target_critic2(torch.cat([next_state, next_action], dim=1))
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
            target_value = reward + (1 - done) * 0.99 * target_q
        
        current_q1 = self.critic1(torch.cat([state, action], dim=1))
        current_q2 = self.critic2(torch.cat([state, action], dim=1))
        
        critic1_loss = F.mse_loss(current_q1, target_value)
        critic2_loss = F.mse_loss(current_q2, target_value)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        new_action, log_prob = self._sample_action(state)
        q1_new = self.critic1(torch.cat([state, new_action], dim=1))
        q2_new = self.critic2(torch.cat([state, new_action], dim=1))
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.log_alpha.exp() * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha (temperature parameter)
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update target networks
        self._soft_update(self.target_critic1, self.critic1, 0.005)
        self._soft_update(self.target_critic2, self.critic2, 0.005)
        
        # Store loss values for visualization
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
    
    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class ExperienceReplay:
    """Experience replay buffer with monitoring capabilities"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch from buffer"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return {
            'state': torch.FloatTensor(state),
            'action': torch.FloatTensor(action),
            'reward': torch.FloatTensor(reward).unsqueeze(1),
            'next_state': torch.FloatTensor(next_state),
            'done': torch.FloatTensor(done).unsqueeze(1)
        }
    
    def __len__(self):
        return len(self.buffer)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics for monitoring"""
        if len(self.buffer) == 0:
            return {'size': 0, 'capacity': self.capacity, 'utilization': 0.0}
        
        rewards = [exp[2] for exp in self.buffer]
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'avg_reward': np.mean(rewards),
            'reward_std': np.std(rewards)
        }

class LiveBrainDashboard:
    """Live neural network visualization dashboard"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧠 Phase 4B: Live Neural Network Training Monitor")
        self.root.geometry("1400x900")
        
        # Data queues for thread-safe communication
        self.brain_queue = queue.Queue()
        self.training_queue = queue.Queue()
        
        # Monitoring state
        self.is_monitoring = False
        self.update_interval = 100  # ms
        
        # Setup GUI
        self._setup_gui()
        
        # Animation for live updates
        self.ani = None
    
    def _setup_gui(self):
        """Setup the main GUI interface"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title and controls
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(title_frame, text="🧠 Live Neural Network Training Monitor", 
                               font=("Arial", 16, "bold"))
        title_label.pack(side=tk.LEFT)
        
        # Control buttons
        control_frame = ttk.Frame(title_frame)
        control_frame.pack(side=tk.RIGHT)
        
        self.start_button = ttk.Button(control_frame, text="▶️ Start Training", 
                                      command=self.start_monitoring)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="⏹️ Stop Training", 
                                     command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Setup tabs
        self._setup_network_tab()
        self._setup_training_tab()
        self._setup_performance_tab()
        self._setup_experience_tab()
    
    def _setup_network_tab(self):
        """Setup neural network visualization tab"""
        network_frame = ttk.Frame(self.notebook)
        self.notebook.add(network_frame, text="🧠 Neural Network")
        
        # Split into two columns
        left_frame = ttk.Frame(network_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(network_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Network architecture visualization
        self.network_fig = Figure(figsize=(8, 6), facecolor='black')
        self.network_ax = self.network_fig.add_subplot(111)
        self.network_ax.set_facecolor('black')
        self.network_canvas = FigureCanvasTkAgg(self.network_fig, left_frame)
        self.network_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Action outputs
        actions_frame = ttk.LabelFrame(right_frame, text="🎮 Action Outputs")
        actions_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.action_vars = {}
        for action in ['steering', 'throttle', 'brake', 'gear']:
            frame = ttk.Frame(actions_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(frame, text=f"{action.title()}:").pack(side=tk.LEFT)
            var = tk.StringVar(value="0.000")
            self.action_vars[action] = var
            ttk.Label(frame, textvariable=var, font=("Courier", 10)).pack(side=tk.RIGHT)
        
        # Network statistics
        stats_frame = ttk.LabelFrame(right_frame, text="📊 Network Stats")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=8, font=("Courier", 9))
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def _setup_training_tab(self):
        """Setup training progress tab"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="📈 Training Progress")
        
        # Training plots
        self.training_fig = Figure(figsize=(12, 8))
        
        # Subplot layout: 2x2
        self.reward_ax = self.training_fig.add_subplot(221)
        self.loss_ax = self.training_fig.add_subplot(222)
        self.episode_ax = self.training_fig.add_subplot(223)
        self.alpha_ax = self.training_fig.add_subplot(224)
        
        self.training_fig.tight_layout()
        self.training_canvas = FigureCanvasTkAgg(self.training_fig, training_frame)
        self.training_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize data storage
        self.reward_history = deque(maxlen=1000)
        self.loss_history = {'actor': deque(maxlen=1000), 'critic1': deque(maxlen=1000), 
                            'critic2': deque(maxlen=1000)}
        self.episode_history = deque(maxlen=100)
        self.alpha_history = deque(maxlen=1000)
    
    def _setup_performance_tab(self):
        """Setup performance monitoring tab"""
        perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(perf_frame, text="⚡ Performance")
        
        # Performance metrics
        metrics_frame = ttk.LabelFrame(perf_frame, text="📊 Live Metrics")
        metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.perf_vars = {}
        metrics = ['Episode', 'Total Steps', 'Episode Reward', 'Distance Traveled', 'Recovery Count']
        
        for i, metric in enumerate(metrics):
            frame = ttk.Frame(metrics_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(frame, text=f"{metric}:", width=20).pack(side=tk.LEFT)
            var = tk.StringVar(value="0")
            self.perf_vars[metric] = var
            ttk.Label(frame, textvariable=var, font=("Courier", 12, "bold")).pack(side=tk.RIGHT)
        
        # Performance plots
        self.perf_fig = Figure(figsize=(12, 6))
        self.distance_ax = self.perf_fig.add_subplot(121)
        self.speed_ax = self.perf_fig.add_subplot(122)
        self.perf_fig.tight_layout()
        
        self.perf_canvas = FigureCanvasTkAgg(self.perf_fig, perf_frame)
        self.perf_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def _setup_experience_tab(self):
        """Setup experience replay monitoring tab"""
        exp_frame = ttk.Frame(self.notebook)
        self.notebook.add(exp_frame, text="💾 Experience Replay")
        
        # Buffer status
        buffer_frame = ttk.LabelFrame(exp_frame, text="🗃️ Buffer Status")
        buffer_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.buffer_vars = {}
        buffer_metrics = ['Buffer Size', 'Capacity', 'Utilization', 'Avg Reward', 'Reward Std']
        
        for metric in buffer_metrics:
            frame = ttk.Frame(buffer_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            ttk.Label(frame, text=f"{metric}:", width=15).pack(side=tk.LEFT)
            var = tk.StringVar(value="0")
            self.buffer_vars[metric] = var
            ttk.Label(frame, textvariable=var, font=("Courier", 10)).pack(side=tk.RIGHT)
        
        # Experience visualization
        self.exp_fig = Figure(figsize=(12, 6))
        self.buffer_ax = self.exp_fig.add_subplot(121)
        self.reward_dist_ax = self.exp_fig.add_subplot(122)
        self.exp_fig.tight_layout()
        
        self.exp_canvas = FigureCanvasTkAgg(self.exp_fig, exp_frame)
        self.exp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def start_monitoring(self):
        """Start the monitoring process"""
        self.is_monitoring = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Start animation for live updates
        if self.ani is None:
            self.ani = animation.FuncAnimation(self.training_fig, self._update_plots, 
                                             interval=self.update_interval, blit=False)
        
        logger.info("🧠 Live brain monitoring started!")
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.is_monitoring = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        if self.ani:
            self.ani.event_source.stop()
            self.ani = None
        
        logger.info("🧠 Live brain monitoring stopped!")
    
    def update_brain_data(self, brain_data: BrainActivityData):
        """Update brain activity data (thread-safe)"""
        if self.is_monitoring:
            try:
                self.brain_queue.put_nowait(brain_data)
            except queue.Full:
                pass  # Drop data if queue is full
    
    def update_training_data(self, training_data: Dict[str, Any]):
        """Update training progress data (thread-safe)"""
        if self.is_monitoring:
            try:
                self.training_queue.put_nowait(training_data)
            except queue.Full:
                pass
    
    def _update_plots(self, frame):
        """Update all plots with latest data"""
        if not self.is_monitoring:
            return
        
        # Process queued data
        while not self.brain_queue.empty():
            try:
                brain_data = self.brain_queue.get_nowait()
                self._process_brain_data(brain_data)
            except queue.Empty:
                break
        
        while not self.training_queue.empty():
            try:
                training_data = self.training_queue.get_nowait()
                self._process_training_data(training_data)
            except queue.Empty:
                break
        
        # Redraw canvases
        self.training_canvas.draw_idle()
        self.perf_canvas.draw_idle()
        self.exp_canvas.draw_idle()
    
    def _process_brain_data(self, brain_data: BrainActivityData):
        """Process and visualize brain activity data"""
        # Update action outputs
        for action, value in brain_data.action_outputs.items():
            if action in self.action_vars:
                self.action_vars[action].set(f"{value:+.3f}")
        
        # Update reward history
        if hasattr(brain_data, 'reward'):
            self.reward_history.append(brain_data.reward)
        
        # Update loss history
        for loss_type, value in brain_data.loss_values.items():
            if loss_type in self.loss_history:
                self.loss_history[loss_type].append(value)
            elif loss_type == 'alpha':
                self.alpha_history.append(value)
    
    def _process_training_data(self, training_data: Dict[str, Any]):
        """Process and visualize training progress data"""
        # Update performance metrics
        for metric, value in training_data.items():
            if metric in self.perf_vars:
                if isinstance(value, float):
                    self.perf_vars[metric].set(f"{value:.2f}")
                else:
                    self.perf_vars[metric].set(str(value))
    
    def run(self):
        """Run the dashboard"""
        logger.info("🚀 Phase 4B: Live Neural Network Training Dashboard Starting...")
        logger.info("🖥️  Dashboard ready! Click 'Start Training' to begin")
        self.root.mainloop()

class IntegratedTrainingEnvironment:
    """Complete training environment with live visualization"""
    
    def __init__(self):
        # BeamNG connection
        self.bng = None
        self.vehicle = None
        
        # Neural network
        self.sac = SACNetwork()
        self.replay_buffer = ExperienceReplay()
        
        # Training state
        self.episode = 0
        self.total_steps = 0
        self.episode_reward = 0
        self.episode_steps = 0
        self.distance_traveled = 0
        self.recovery_count = 0
        
        # Live dashboard
        self.dashboard = None
        self.training_thread = None
        self.is_training = False
        
        # Auto-recovery system
        self.stuck_threshold = 2.0  # seconds
        self.last_position = None
        self.stuck_timer = 0
        
    def connect_beamng(self):
        """Connect to BeamNG.drive"""
        try:
            logger.info("🚗 Connecting to BeamNG.drive...")
            self.bng = BeamNGpy('localhost', 64256, home='S:/SteamLibrary/steamapps/common/BeamNG.drive')
            self.bng.open(launch=False)
            
            # Create scenario
            scenario = Scenario('automation_test_track', 'Phase4B_Training')
            self.vehicle = Vehicle('ego_vehicle', model='etk800', license='PHASE4B')
            scenario.add_vehicle(self.vehicle, pos=(0, 0, 0), rot_quat=(0, 0, 0, 1))
            
            scenario.make(self.bng)
            self.bng.load_scenario(scenario)
            self.bng.start_scenario()
            
            logger.info("✅ BeamNG.drive connected successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to BeamNG.drive: {e}")
            return False
    
    def get_state(self) -> np.ndarray:
        """Get current state from vehicle"""
        try:
            # Get sensor data
            sensors = self.vehicle.sensors
            state_data = []
            
            # Basic vehicle state (using polling for now)
            state = self.vehicle.state
            
            # Position and orientation
            pos = state.get('pos', [0, 0, 0])
            state_data.extend(pos)
            
            # Velocity
            vel = state.get('vel', [0, 0, 0])
            state_data.extend(vel)
            
            # Rotation
            rotation = state.get('rotation', [0, 0, 0])
            state_data.extend(rotation)
            
            # Angular velocity
            angular_vel = state.get('angular_vel', [0, 0, 0])
            state_data.extend(angular_vel)
            
            # Pad to 142 dimensions with additional sensor data
            while len(state_data) < 142:
                state_data.append(0.0)
            
            return np.array(state_data[:142], dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"⚠️ Error getting state: {e}")
            return np.zeros(142, dtype=np.float32)
    
    def apply_action(self, action: np.ndarray):
        """Apply action to vehicle"""
        try:
            # Convert network outputs to vehicle controls
            steering = float(np.clip(action[0], -1, 1))
            throttle = float(np.clip(action[1], 0, 1))
            brake = float(np.clip(action[2], 0, 1))
            
            # Apply controls
            self.vehicle.control(steering=steering, throttle=throttle, brake=brake)
            
        except Exception as e:
            logger.warning(f"⚠️ Error applying action: {e}")
    
    def calculate_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """Calculate reward for current state-action pair"""
        reward = 0.0
        
        # Extract velocity from state
        velocity = np.linalg.norm(state[3:6])  # Velocity magnitude
        
        # Reward for movement
        reward += velocity * 0.1
        
        # Small penalty for large steering to encourage smooth driving
        steering = abs(action[0])
        reward -= steering * 0.01
        
        # Penalty for excessive braking
        brake = action[2]
        reward -= brake * 0.02
        
        return reward
    
    def check_auto_recovery(self, current_pos: np.ndarray) -> bool:
        """Check if auto-recovery is needed"""
        if self.last_position is None:
            self.last_position = current_pos
            return False
        
        # Calculate distance moved
        distance_moved = np.linalg.norm(current_pos - self.last_position)
        
        if distance_moved < 0.1:  # Very slow movement
            self.stuck_timer += 0.1  # Assume 100ms steps
            if self.stuck_timer >= self.stuck_threshold:
                logger.info("🔄 Auto-recovery triggered - vehicle appears stuck")
                self.stuck_timer = 0
                self.recovery_count += 1
                return True
        else:
            self.stuck_timer = 0
        
        self.last_position = current_pos.copy()
        return False
    
    def perform_recovery(self):
        """Perform auto-recovery"""
        try:
            # Simple recovery: reset vehicle position with small random offset
            reset_pos = (random.uniform(-10, 10), random.uniform(-10, 10), 1)
            reset_rot = (0, 0, random.uniform(0, 360), 1)
            
            self.vehicle.set_position(reset_pos)
            self.vehicle.set_rotation(reset_rot)
            
            time.sleep(0.5)  # Allow physics to settle
            
        except Exception as e:
            logger.warning(f"⚠️ Recovery failed: {e}")
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < 1000:  # Minimum buffer size
            return
        
        # Sample batch and move to device
        batch = self.replay_buffer.sample(64)
        for key in batch:
            batch[key] = batch[key].to(self.sac.device)
        
        # Update network
        loss_info = self.sac.update(batch)
        
        # Update dashboard with training progress
        training_data = {
            'Episode': self.episode,
            'Total Steps': self.total_steps,
            'Episode Reward': self.episode_reward,
            'Distance Traveled': self.distance_traveled,
            'Recovery Count': self.recovery_count,
            'Buffer Size': len(self.replay_buffer),
            **loss_info
        }
        
        if self.dashboard:
            self.dashboard.update_training_data(training_data)
    
    def run_episode(self):
        """Run one training episode"""
        self.episode += 1
        self.episode_reward = 0
        self.episode_steps = 0
        episode_distance = 0
        
        # Reset environment
        try:
            # Simple reset - move to random position
            reset_pos = (random.uniform(-20, 20), random.uniform(-20, 20), 1)
            self.vehicle.set_position(reset_pos)
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"⚠️ Episode reset failed: {e}")
        
        state = self.get_state()
        last_pos = state[:3]
        
        logger.info(f"🎯 Starting Episode {self.episode}")
        
        for step in range(1000):  # Max steps per episode
            if not self.is_training:
                break
            
            # Get action from neural network
            action, action_outputs = self.sac.get_action(state)
            
            # Apply action
            self.apply_action(action)
            time.sleep(0.1)  # Control frequency
            
            # Get next state
            next_state = self.get_state()
            current_pos = next_state[:3]
            
            # Calculate reward
            reward = self.calculate_reward(state, action)
            
            # Update distance traveled
            step_distance = np.linalg.norm(current_pos - last_pos)
            episode_distance += step_distance
            self.distance_traveled += step_distance
            last_pos = current_pos
            
            # Check for auto-recovery
            done = False
            if self.check_auto_recovery(current_pos):
                reward -= 5.0  # Penalty for getting stuck
                self.perform_recovery()
                done = True
            
            # Store experience
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update counters
            self.episode_reward += reward
            self.episode_steps += 1
            self.total_steps += 1
            
            # Update brain data for visualization
            brain_data = self.sac.brain_data
            brain_data.reward = reward
            brain_data.episode_stats = {
                'episode': self.episode,
                'step': step,
                'episode_reward': self.episode_reward,
                'distance': episode_distance
            }
            
            if self.dashboard:
                self.dashboard.update_brain_data(brain_data)
            
            # Train neural network
            if step % 4 == 0:  # Train every 4 steps
                self.train_step()
            
            # Episode termination conditions
            if done or step >= 999:
                break
            
            state = next_state
        
        logger.info(f"✅ Episode {self.episode} completed: {self.episode_reward:.2f} reward, {episode_distance:.1f}m distance")
    
    def start_training_with_dashboard(self):
        """Start training with live dashboard"""
        # Connect to BeamNG
        if not self.connect_beamng():
            return False
        
        # Create and start dashboard in separate thread
        def run_dashboard():
            self.dashboard = LiveBrainDashboard()
            self.dashboard.run()
        
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        
        # Wait for dashboard to initialize
        time.sleep(2)
        
        # Start training loop
        self.is_training = True
        logger.info("🚀 Starting integrated neural network training...")
        
        try:
            while self.is_training and self.episode < 100:  # Train for 100 episodes
                self.run_episode()
                time.sleep(1)  # Brief pause between episodes
                
        except KeyboardInterrupt:
            logger.info("🛑 Training interrupted by user")
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
        finally:
            self.is_training = False
            if self.bng:
                self.bng.close()
            logger.info("🏁 Training session completed")
        
        return True

def main():
    """Main entry point for Phase 4B"""
    print("🧠 Phase 4B: Integrated Neural Network Training with Live Visualization")
    print("======================================================================")
    print()
    print("Features:")
    print("• SAC (Soft Actor-Critic) neural network training")
    print("• Live brain activity visualization")
    print("• Real-time training progress monitoring")
    print("• Auto-recovery system for continuous learning")
    print("• Experience replay buffer analysis")
    print()
    
    # Check PyTorch and device availability
    print(f"🔥 PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🎮 CUDA available - {gpu_count} GPU(s) detected:")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("� CUDA not available - training will use CPU")
        print("   Consider installing CUDA-enabled PyTorch for GPU acceleration")
    
    print()
    
    try:
        # Create training environment
        env = IntegratedTrainingEnvironment()
        
        # Start training with live dashboard
        success = env.start_training_with_dashboard()
        
        if success:
            print("✅ Phase 4B completed successfully!")
        else:
            print("❌ Phase 4B failed to start")
            
    except Exception as e:
        logger.error(f"❌ Phase 4B error: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()