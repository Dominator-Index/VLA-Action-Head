import torch
import torch.nn as nn
import torch.nn.functional as F
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from .shared_components import MLPResNet, SinusoidalPositionalEncoding, BaseActionHead

class ShortcutActionHead(nn.Module):
    """
    Shortcut Model-based action head that implements self-consistency training and variable step sampling.
    
    Based on "Shortcut Models for Fast and Robust Image Generation" which introduces:
    1. Self-consistency loss for improved training stability
    2. Variable step size sampling during inference
    3. Adaptive shortcut connections to reduce sampling steps
    
    This implementation adapts the shortcut model framework for continuous action prediction
    in vision-language-action (VLA) models.
    """
    
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_flow_steps=20,
        # Shortcut model specific parameters
        self_consistency_ratio=0.3,  # Fraction of batch used for self-consistency loss
        cfg_dropout_prob=0.1,        # Classifier-free guidance dropout probability
        min_dt=0.01,                 # Minimum step size for variable sampling
        max_dt=0.2,                  # Maximum step size for variable sampling
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_flow_steps = num_flow_steps
        self.self_consistency_ratio = self_consistency_ratio
        self.cfg_dropout_prob = cfg_dropout_prob
        self.min_dt = min_dt
        self.max_dt = max_dt
        
        # Time encoder for timestep embedding
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)
        
        # dt encoder for step size embedding (key feature of shortcut models)
        self.dt_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)
        
        # Main velocity prediction network
        # Input: [action_hidden_states + time_emb + dt_emb + current_position]
        self.velocity_net = MLPResNet(
            num_blocks=3,
            input_dim=input_dim * action_dim + 2 * hidden_dim + action_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )
        
        # Classifier-free guidance: condition dropout
        self.condition_dropout = nn.Dropout(cfg_dropout_prob)
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize parameters for the convex flow model."""
        # Initialize ICNN parameters
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize flow parameters with small values for stability
        if hasattr(self.convex_flow, 'w0'):
            nn.init.constant_(self.convex_flow.w0, -1.0)  # softplus(-1) ≈ 0.31
        if hasattr(self.convex_flow, 'w1'):
            nn.init.constant_(self.convex_flow.w1, 0.0)  # 使用固定值，因为bias_w1已经在flow初始化时设置
    
    def forward(self, actions_hidden_states, t, dt, x_t, apply_cfg_dropout=True):
        """
        Forward pass of the shortcut velocity prediction network.
        
        Args:
            actions_hidden_states: (batch, NUM_ACTIONS_CHUNK * action_dim, input_dim)
            t: (batch,) current time values [0, 1]
            dt: (batch,) step size values (key input for shortcut models)
            x_t: (batch, NUM_ACTIONS_CHUNK, action_dim) current position
            apply_cfg_dropout: Whether to apply classifier-free guidance dropout
        
        Returns:
            velocity: (batch, NUM_ACTIONS_CHUNK, action_dim) predicted velocity field
        """
        batch_size = actions_hidden_states.shape[0]
        
        # Reshape action hidden states
        x = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        
        # Apply classifier-free guidance dropout to conditioning
        if apply_cfg_dropout and self.training:
            x = self.condition_dropout(x)
        
        # Encode timestep and step size
        t_emb = self.time_encoder(t).unsqueeze(1)  # (batch, 1, hidden_dim)
        dt_emb = self.dt_encoder(dt).unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Broadcast to all action chunks
        t_emb = t_emb.expand(-1, NUM_ACTIONS_CHUNK, -1)
        dt_emb = dt_emb.expand(-1, NUM_ACTIONS_CHUNK, -1)
        
        # Concatenate all features
        combined_input = torch.cat([x, t_emb, dt_emb, x_t], dim=-1)
        
        # Predict velocity field
        velocity = self.velocity_net(combined_input)
        
        return velocity
    
    def compute_shortcut_loss(self, x0, x1, actions_hidden_states):
        """
        Compute shortcut model loss with self-consistency training.
        
        This implements the core shortcut training algorithm:
        1. Sample random timesteps t and step sizes dt
        2. Compute flow matching loss for most samples
        3. Compute self-consistency loss for a subset of samples
        4. Combine both losses for improved stability
        
        Args:
            x0: (batch, NUM_ACTIONS_CHUNK, action_dim) source points (noise)
            x1: (batch, NUM_ACTIONS_CHUNK, action_dim) target points (ground truth actions)
            actions_hidden_states: (batch, NUM_ACTIONS_CHUNK * action_dim, input_dim)
        
        Returns:
            loss_dict: Dictionary containing total loss and component losses
        """
        batch_size = x1.shape[0]
        device = x1.device
        model_dtype = next(self.parameters()).dtype
        
        # Ensure data type consistency
        x0 = x0.to(dtype=model_dtype)
        x1 = x1.to(dtype=model_dtype)
        actions_hidden_states = actions_hidden_states.to(dtype=model_dtype)
        
        # Sample random timesteps t ∈ [0, 1]
        t = torch.rand(batch_size, device=device, dtype=model_dtype)
        
        # Sample random step sizes dt with constraints
        # Ensure t + dt ≤ 1.0 for valid flow paths
        max_allowed_dt = 1.0 - t
        dt = torch.rand(batch_size, device=device, dtype=model_dtype)
        dt = torch.clamp(dt * self.max_dt, min=self.min_dt)
        dt = torch.minimum(dt, max_allowed_dt)
        
        # Determine number of samples for self-consistency
        num_self_consistency = int(batch_size * self.self_consistency_ratio)
        
        # Interpolate between source and target
        t_expanded = t.view(-1, 1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        
        # True velocity field (flow matching target)
        v_true = x1 - x0
        
        # Compute self-consistency targets for subset of batch
        if num_self_consistency > 0:
            x_t_sc = x_t[:num_self_consistency]
            t_sc = t[:num_self_consistency]
            dt_sc = dt[:num_self_consistency]
            dt_half = dt_sc * 0.5
            actions_sc = actions_hidden_states[:num_self_consistency]
            
            # Two-step prediction for self-consistency (Eq. 5 in paper)
            with torch.no_grad():
                # First half-step
                v1 = self.forward(actions_sc, t_sc, dt_half, x_t_sc, apply_cfg_dropout=False)
                
                # Second half-step
                x_mid = x_t_sc + dt_half.view(-1, 1, 1) * v1
                t_mid = t_sc + dt_half
                v2 = self.forward(actions_sc, t_mid, dt_half, x_mid, apply_cfg_dropout=False)
                
                # Average for self-consistency target
                v_target_sc = (v1 + v2) / 2.0
        
        # Set dt=0 for regular flow matching samples (no shortcut)
        dt_for_fm = dt.clone()
        dt_for_fm[num_self_consistency:] = 0.0
        
        # Forward pass with classifier-free guidance dropout
        v_pred = self.forward(actions_hidden_states, t, dt_for_fm, x_t, apply_cfg_dropout=True)
        
        # Flow matching loss (standard samples)
        if num_self_consistency < batch_size:
            loss_fm = F.mse_loss(
                v_pred[num_self_consistency:], 
                v_true[num_self_consistency:]
            )
        else:
            loss_fm = torch.tensor(0.0, device=device)
        
        # Self-consistency loss (shortcut samples)
        if num_self_consistency > 0:
            loss_sc = F.mse_loss(
                v_pred[:num_self_consistency], 
                v_target_sc
            )
        else:
            loss_sc = torch.tensor(0.0, device=device)
        
        # Combined loss
        total_loss = loss_fm + loss_sc
        
        return {
            'loss': total_loss,
            'loss_fm': loss_fm.detach(),
            'loss_sc': loss_sc.detach()
        }
    
    def _generate_adaptive_dt_list(self, num_steps):
        """
        Generate adaptive step size list for inference.
        
        Uses larger steps at the beginning and smaller steps towards the end
        for better quality-speed tradeoff.
        
        Args:
            num_steps: Number of sampling steps
            
        Returns:
            dt_list: List of step sizes that sum to approximately 1.0
        """
        if num_steps == 1:
            return [1.0]
        
        # Exponential decay: larger steps initially, smaller steps later
        dt_list = []
        remaining_time = 1.0
        
        for i in range(num_steps):
            if i == num_steps - 1:
                # Last step: use remaining time
                dt_list.append(remaining_time)
            else:
                # Adaptive step size: starts large, gets smaller
                progress = i / (num_steps - 1)
                # Exponential decay factor
                decay_factor = 0.5 + 0.5 * (1 - progress) ** 2
                dt = (remaining_time / (num_steps - i)) * decay_factor
                dt = max(dt, self.min_dt)  # Ensure minimum step size
                dt = min(dt, remaining_time - self.min_dt * (num_steps - i - 1))  # Ensure feasibility
                
                dt_list.append(dt)
                remaining_time -= dt
        
        # Normalize to ensure sum equals 1.0
        total = sum(dt_list)
        dt_list = [dt / total for dt in dt_list]
        
        return dt_list
    
    @torch.no_grad()
    def predict_action(self, actions_hidden_states, num_steps=None, dt_list=None, disable_shortcut=False):
        """
        Generate actions using the shortcut model with variable step sampling.
        
        This implements the inference procedure from shortcut models:
        1. Start from noise
        2. Use adaptive or fixed step sizes
        3. Option to disable shortcuts for comparison
        
        Args:
            actions_hidden_states: (batch, NUM_ACTIONS_CHUNK * action_dim, input_dim)
            num_steps: Number of sampling steps (if dt_list not provided)
            dt_list: Custom list of step sizes (overrides num_steps)
            disable_shortcut: If True, use dt=0 (standard flow matching)
        
        Returns:
            actions: (batch, NUM_ACTIONS_CHUNK, action_dim)
        """
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        
        # Determine step size schedule
        if dt_list is not None:
            # Use provided step sizes
            step_sizes = dt_list
        elif num_steps is not None:
            # Generate adaptive step sizes
            step_sizes = self._generate_adaptive_dt_list(num_steps)
        else:
            # Default to fixed steps
            step_sizes = self._generate_adaptive_dt_list(self.num_flow_steps)
        
        # Ensure step sizes sum to approximately 1.0
        total_time = sum(step_sizes)
        if abs(total_time - 1.0) > 1e-6:
            step_sizes = [dt / total_time for dt in step_sizes]
        
        # Initialize with noise
        x = torch.randn(
            (batch_size, NUM_ACTIONS_CHUNK, self.action_dim),
            device=device,
            dtype=actions_hidden_states.dtype
        )
        
        # Integration loop
        t_current = torch.zeros(batch_size, device=device)
        
        for dt_val in step_sizes:
            dt = torch.full((batch_size,), dt_val, device=device)
            
            # Option to disable shortcuts (use dt=0 for standard flow matching)
            if disable_shortcut:
                dt_input = torch.zeros_like(dt)
            else:
                dt_input = dt
            
            # Predict velocity
            velocity = self.forward(actions_hidden_states, t_current, dt_input, x, apply_cfg_dropout=False)
            
            # Euler integration step
            x = x + velocity * dt.view(-1, 1, 1)
            
            # Update time
            t_current = t_current + dt
        
        return x
    
    @torch.no_grad()
    def sample_actions_with_cfg(self, actions_hidden_states, cfg_scale=1.5, num_steps=None):
        """
        Sample actions with classifier-free guidance for improved quality.
        
        Args:
            actions_hidden_states: (batch, NUM_ACTIONS_CHUNK * action_dim, input_dim)
            cfg_scale: Scale factor for classifier-free guidance (>1.0 for stronger conditioning)
            num_steps: Number of sampling steps
        
        Returns:
            actions: (batch, NUM_ACTIONS_CHUNK, action_dim)
        """
        if cfg_scale == 1.0:
            # No CFG, use standard sampling
            return self.predict_action(actions_hidden_states, num_steps=num_steps)
        
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        
        # Duplicate batch for conditional and unconditional predictions
        x = torch.randn(
            (batch_size, NUM_ACTIONS_CHUNK, self.action_dim),
            device=device,
            dtype=actions_hidden_states.dtype
        )
        
        # Duplicate conditioning
        cond_doubled = actions_hidden_states.repeat(2, 1, 1)
        x_doubled = x.repeat(2, 1, 1)
        
        # Generate step sizes
        step_sizes = self._generate_adaptive_dt_list(num_steps or self.num_flow_steps)
        
        t_current = torch.zeros(batch_size * 2, device=device)
        
        for dt_val in step_sizes:
            dt = torch.full((batch_size * 2,), dt_val, device=device)
            
            # Predict velocity with and without conditioning
            velocity_doubled = self.forward(cond_doubled, t_current, dt, x_doubled, apply_cfg_dropout=False)
            
            # Split conditional and unconditional predictions
            v_cond, v_uncond = velocity_doubled.chunk(2, dim=0)
            
            # Apply classifier-free guidance
            velocity = v_uncond + cfg_scale * (v_cond - v_uncond)
            
            # Update only the original batch
            x = x + velocity * dt_val
            
            # Update doubled state for next iteration
            x_doubled = x.repeat(2, 1, 1)
            t_current = t_current + dt
        
        return x
    
    def get_sampling_schedule(self, num_steps, schedule_type="adaptive"):
        """
        Get different sampling schedules for comparison.
        
        Args:
            num_steps: Number of sampling steps
            schedule_type: "uniform", "adaptive", or "exponential"
        
        Returns:
            dt_list: List of step sizes
        """
        if schedule_type == "uniform":
            return [1.0 / num_steps] * num_steps
        elif schedule_type == "adaptive":
            return self._generate_adaptive_dt_list(num_steps)
        elif schedule_type == "exponential":
            # Exponential decay
            dt_list = []
            for i in range(num_steps):
                dt = (2.0 / num_steps) * (0.5 ** i)
                dt_list.append(dt)
            # Normalize
            total = sum(dt_list)
            return [dt / total for dt in dt_list]
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        