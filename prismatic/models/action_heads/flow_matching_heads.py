"""
Flow Matching-based Action Heads

This module contains action heads that use flow matching approaches:
- FlowMatchingActionHead: Standard flow matching
- OTFlowMatchingActionHead: Optimal transport flow matching
- COTFlowMatchingActionHead: Conditional optimal transport flow matching
- MeanFlowActionHead: Mean-reverting flow matching with two timesteps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from .shared_components import MLPResNet, SinusoidalPositionalEncoding, BaseActionHead


class FlowMatchingActionHead(nn.Module):
    """
    MLP-based action head for flow matching, consistent with L1RegressionActionHead/VAEActionHead/DiffusionActionHead.
    支持多步采样推理，每步注入时间编码。
    """
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_flow_steps=20,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_flow_steps = num_flow_steps
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)
        # 输入: [batch, NUM_ACTIONS_CHUNK, action_dim * input_dim + hidden_dim]
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * action_dim + hidden_dim + action_dim,  # 加上x_t的维度
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(self, actions_hidden_states, t_emb, x_t):
        """
        actions_hidden_states: (batch, NUM_ACTIONS_CHUNK * action_dim, input_dim)
        t_emb: (batch, 1, hidden_dim)  # 每个样本的时间步编码
        x_t: (batch, NUM_ACTIONS_CHUNK, action_dim)  # 当前位置
        return: (batch, NUM_ACTIONS_CHUNK, action_dim)
        """
        batch_size = actions_hidden_states.shape[0]
    
        # (batch, NUM_ACTIONS_CHUNK, action_dim * input_dim)
        x = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        
        # t_emb: (batch, 1, hidden_dim) -> broadcast到NUM_ACTIONS_CHUNK
        t_emb = t_emb.expand(-1, NUM_ACTIONS_CHUNK, -1).to(dtype=torch.bfloat16)
        # 拼接特征、时间编码和当前位置
        x = torch.cat([x, t_emb, x_t], dim=-1)  
        vector_field = self.model(x)  # (batch, NUM_ACTIONS_CHUNK, action_dim)
        return vector_field
    
    def compute_flow_matching_loss(self, x0, x1, t, actions_hidden_states):
        """
        x0: (batch, NUM_ACTIONS_CHUNK, action_dim)  # 源点（如高斯噪声）
        x1: (batch, NUM_ACTIONS_CHUNK, action_dim)  # 目标（真实动作）
        t: (batch,) or (batch, 1)                   # 路径采样时间
        actions_hidden_states: (batch, NUM_ACTIONS_CHUNK * action_dim, input_dim)
        """
        # 获取模型参数的数据类型
        model_dtype = next(self.parameters()).dtype
        x0 = x0.to(dtype=model_dtype)
        x1 = x1.to(dtype=model_dtype)
        # 确保输入数据类型与模型一致
        actions_hidden_states = actions_hidden_states.to(dtype=model_dtype)
        t = t.view(-1, 1, 1).to(dtype=model_dtype)  # (batch, 1, 1)
        x_t = (1 - t) * x0 + t * x1  # (batch, NUM_ACTIONS_CHUNK, action_dim)
        u_t = x1 - x0  # (batch, NUM_ACTIONS_CHUNK, action_dim)
        # 时间编码
        t_scalar = t.view(-1).to(dtype=torch.bfloat16)  # (batch,)
        t_emb = self.time_encoder(t_scalar)  # (batch, hidden_dim)
        t_emb = t_emb.unsqueeze(1)  # (batch, 1, hidden_dim)
        # 将x_t作为输入传递给模型
        flow_pred = self.forward(actions_hidden_states, t_emb, x_t)  # (batch, NUM_ACTIONS_CHUNK, action_dim)
        loss = torch.nn.functional.mse_loss(flow_pred, u_t)
        return loss
    
    @torch.no_grad()
    def predict_action(self, actions_hidden_states):
        """
        使用多步采样生成actions，参考finetune_swanlab.py中的run_flow_matching_sampling
        """
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        
        # 初始点x0 ~ N(0, I)
        x = torch.randn(
            size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
            device=device,
            dtype=torch.bfloat16,
        )
        num_steps = self.num_flow_steps

        t_vals = torch.linspace(0, 1, num_steps + 1, device=device).to(dtype=torch.bfloat16)
        dt = 1.0 / num_steps
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(num_steps):
                t_scalar = t_vals[i].expand(batch_size).to(dtype=torch.bfloat16)
                t_emb = self.time_encoder(t_scalar).unsqueeze(1)  # (batch, 1, hidden_dim)
                # 将当前位置 x 作为输入传递给模型
                v = self.forward(actions_hidden_states, t_emb, x)  # 正确传递x参数
                x = x + v * dt
        return x.reshape(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM)


class OTFlowMatchingActionHead(FlowMatchingActionHead):
    """
    Optimal Transport Flow Matching Action Head, which uses optimal transport to sample paths between source and target actions.
    """
    
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_flow_steps=20,
        ot_plan_sampler=None,  # Optional OTPlanSampler instance
    ):
        super().__init__(input_dim, hidden_dim, action_dim, num_flow_steps)
        self.ot_plan_sampler = ot_plan_sampler if ot_plan_sampler is not None else OTPlanSampler(method='exact')
        
    def compute_flow_matching_loss(self, x0, x1, t, actions_hidden_states):
        """
        Use optimal transport to find correspondences between source and target samples before computing loss.
        
        Args:
            x0: (batch, NUM_ACTIONS_CHUNK, action_dim)  # Source points (random noise)
            x1: (batch, NUM_ACTIONS_CHUNK, action_dim)  # Target points (ground truth actions)
            t: (batch,) or (batch, 1)                  # Path sampling time
            actions_hidden_states: (batch, NUM_ACTIONS_CHUNK * action_dim, input_dim)
        """
        model_dtype = next(self.parameters()).dtype
        # Get OT matched pairs
        x0_ot, x1_ot = self.ot_plan_sampler.sample_plan(x0, x1)
        x0_ot = x0_ot.to(dtype=model_dtype)
        x1_ot = x1_ot.to(dtype=model_dtype)
        
        # Now proceed with standard flow matching using OT matched pairs
        t = t.view(-1, 1, 1).to(dtype=model_dtype)  # (batch, 1, 1)
        x_t = (1 - t) * x0_ot + t * x1_ot  # (batch, NUM_ACTIONS_CHUNK, action_dim)
        u_t = x1_ot - x0_ot  # (batch, NUM_ACTIONS_CHUNK, action_dim)
        
        # Time encoding
        t_scalar = t.view(-1).to(dtype=model_dtype)  # (batch,)
        t_emb = self.time_encoder(t_scalar)  # (batch, hidden_dim)
        t_emb = t_emb.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Predict flow vector field
        flow_pred = self.forward(actions_hidden_states, t_emb, x_t)  # (batch, NUM_ACTIONS_CHUNK, action_dim)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(flow_pred, u_t)
        return loss


class COTFlowMatchingActionHead(OTFlowMatchingActionHead):
    """
    Conditional Optimal Transport Flow Matching Action Head.
    Uses a conditional OT plan that depends on task conditions.
    """    
    
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_flow_steps=20,
        condition_coordinates=None,
        eps=0.1,
    ):
        super().__init__(input_dim, hidden_dim, action_dim, num_flow_steps)
        
        # Default to using first half of dimensions as condition coordinates if not specified
        if condition_coordinates is None:
            condition_coordinates = list(range(action_dim // 2))
        
        # Create a batch COT plan sampler
        from cot import BatchCOTPlanSampler
        self.ot_plan_sampler = BatchCOTPlanSampler(
            condition_coordinates=condition_coordinates,
            eps=eps,
            method="exact"
        )
        
    def compute_flow_matching_loss(self, x0, x1, t, actions_hidden_states):
        """
        Use conditional optimal transport to find correspondences between source
        and target samples before computing loss.
        
        The difference from regular OT is that COT considers conditional information
        when computing the transport plan.
        """
        # Get COT matched pairs
        x0_cot, x1_cot = self.ot_plan_sampler.sample_plan(x0, x1)
        
        # Now proceed with standard flow matching using COT matched pairs
        t = t.view(-1, 1, 1)  # (batch, 1, 1)
        x_t = (1 - t) * x0_cot + t * x1_cot  # (batch, NUM_ACTIONS_CHUNK, action_dim)
        u_t = x1_cot - x0_cot  # (batch, NUM_ACTIONS_CHUNK, action_dim)
        
        # Time encoding
        t_scalar = t.view(-1)  # (batch,)
        t_emb = self.time_encoder(t_scalar)  # (batch, hidden_dim)
        t_emb = t_emb.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Predict flow vector field - 正确传递 x_t 参数
        flow_pred = self.forward(actions_hidden_states, t_emb, x_t)  # (batch, NUM_ACTIONS_CHUNK, action_dim)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(flow_pred, u_t)
        return loss


class MeanFlowActionHead(BaseActionHead):
    """
    MeanFlow-based action head that uses mean-reverting flow matching with two timesteps.
    
    Based on "MeanFlow: Mean-Reverting Normalizing Flow for Anomaly Detection" which uses
    a two-timestep formulation and adaptive weighting for improved stability.
    
    Key differences from standard flow matching:
    1. Uses two timesteps (t, r) where h = t - r represents time difference
    2. Includes temporal derivative term in the loss function
    3. Uses adaptive weighting for training stability
    4. Includes EMA for better inference
    """
    
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_flow_steps=20,
        # MeanFlow specific parameters
        norm_eps=1e-3,
        norm_p=-0.5,
        ema_decay=0.999,
        # Time sampling parameters (from MeanFlow paper)
        P_mean_t=-1.2,
        P_std_t=1.2,
        P_mean_r=-1.2,
        P_std_r=1.2,
        ratio=0.5,  # probability of making t != r
        tr_sampler="v0",  # time sampling strategy
    ):
        super().__init__(input_dim, hidden_dim, action_dim)
        self.num_flow_steps = num_flow_steps
        self.norm_eps = norm_eps
        self.norm_p = norm_p
        
        # Time sampling parameters
        self.P_mean_t = P_mean_t
        self.P_std_t = P_std_t
        self.P_mean_r = P_mean_r
        self.P_std_r = P_std_r
        self.ratio = ratio
        self.tr_sampler = tr_sampler
        
        # Time encoder for both timesteps t and h=t-r
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)
        
        # Main network that takes both timesteps as input
        # Input: [action_hidden_states + time_t_emb + time_h_emb + current_position]
        self.model = MLPResNet(
            num_blocks=3,
            input_dim=input_dim * action_dim + 2 * hidden_dim + action_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )
        
        # EMA for stable inference
        self.ema_decay = ema_decay
        self.register_buffer("num_updates", torch.tensor(0))
        
        # Initialize EMA model
        self.model_ema = None
        self._init_ema()
    
    def _init_ema(self):
        """Initialize EMA model with same architecture"""
        self.model_ema = MLPResNet(
            num_blocks=3,
            input_dim=self.model.fc1.in_features,
            hidden_dim=self.model.fc1.out_features,
            output_dim=self.model.fc2.out_features,
        )
        # Copy initial weights
        with torch.no_grad():
            for ema_param, param in zip(self.model_ema.parameters(), self.model.parameters()):
                ema_param.copy_(param)
    
    def update_ema(self, period=16):
        """Update EMA weights with periodic updates for efficiency"""
        self.num_updates += 1
        
        if self.num_updates % period == 0:
            decay_effective = self.ema_decay ** period
            with torch.no_grad():
                for ema_param, param in zip(self.model_ema.parameters(), self.model.parameters()):
                    # Use double precision to avoid numerical issues
                    delta = param.data.double() - ema_param.data.double()
                    ema_param_new = ema_param.data.double() + (1 - decay_effective) * delta
                    ema_param.data.copy_(ema_param_new.float())
    
    def logit_normal_timestep_sample(self, P_mean, P_std, num_samples, device):
        """Sample timesteps using logit-normal distribution"""
        rnd_normal = torch.randn((num_samples,), device=device)
        time = torch.sigmoid(rnd_normal * P_std + P_mean)
        time = torch.clip(time, min=0.0, max=1.0)
        return time
    
    def sample_two_timesteps_v0(self, num_samples, device):
        """
        Sample two timesteps (t, r) using v0 strategy from MeanFlow paper.
        Ensures t >= r with post-processing.
        """
        # Step 1: sample two independent timesteps
        t = self.logit_normal_timestep_sample(self.P_mean_t, self.P_std_t, num_samples, device)
        r = self.logit_normal_timestep_sample(self.P_mean_r, self.P_std_r, num_samples, device)
        
        # Step 2: ensure t >= r
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        
        # Step 3: make t and r different with probability (1 - ratio)
        prob = torch.rand(num_samples, device=device)
        mask = prob < (1 - self.ratio)
        r = torch.where(mask, t, r)
        
        return t, r
    
    def sample_two_timesteps_v1(self, num_samples, device):
        """
        Sample two timesteps (t, r) using v1 strategy from MeanFlow paper.
        Different post-processing order.
        """
        # Step 1: sample two independent timesteps
        t = self.logit_normal_timestep_sample(self.P_mean_t, self.P_std_t, num_samples, device)
        r = self.logit_normal_timestep_sample(self.P_mean_r, self.P_std_r, num_samples, device)
        
        # Step 2: make t and r different with probability (1 - ratio)
        prob = torch.rand(num_samples, device=device)
        mask = prob < (1 - self.ratio)
        r = torch.where(mask, t, r)
        
        # Step 3: ensure t >= r
        r = torch.minimum(t, r)
        
        return t, r
    
    def sample_two_timesteps(self, num_samples, device):
        """Sample two timesteps using specified strategy"""
        if self.tr_sampler == "v0":
            return self.sample_two_timesteps_v0(num_samples, device)
        elif self.tr_sampler == "v1":
            return self.sample_two_timesteps_v1(num_samples, device)
        else:
            raise ValueError(f"Unknown time sampler: {self.tr_sampler}")
    
    def forward(self, actions_hidden_states, t, h, x_t):
        """
        Forward pass of the mean flow network.
        
        Args:
            actions_hidden_states: (batch, NUM_ACTIONS_CHUNK * action_dim, input_dim)
            t: (batch,) timestep values
            h: (batch,) time differences (t - r)
            x_t: (batch, NUM_ACTIONS_CHUNK, action_dim) current position
        
        Returns:
            u_pred: (batch, NUM_ACTIONS_CHUNK, action_dim) predicted vector field
        """
        batch_size = actions_hidden_states.shape[0]
        
        # Reshape action hidden states
        x = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        
        # Encode both timesteps
        t_emb = self.time_encoder(t).unsqueeze(1)  # (batch, 1, hidden_dim)
        h_emb = self.time_encoder(h).unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Broadcast to all action chunks
        t_emb = t_emb.expand(-1, NUM_ACTIONS_CHUNK, -1)
        h_emb = h_emb.expand(-1, NUM_ACTIONS_CHUNK, -1)
        
        # Concatenate all features
        combined_input = torch.cat([x, t_emb, h_emb, x_t], dim=-1)
        
        # Predict vector field
        u_pred = self.model(combined_input)
        
        return u_pred
    
    def compute_meanflow_loss(self, x0, x1, actions_hidden_states):
        """
        Compute MeanFlow loss with two-timestep formulation and adaptive weighting.
        
        This implements the core MeanFlow algorithm:
        1. Sample noise e and two timesteps (t, r)
        2. Compute interpolated position z = (1-t)*x1 + t*e
        3. Compute target vector field v = e - x1
        4. Use JVP to compute temporal derivative du/dt
        5. Compute adaptive weighted loss
        
        Args:
            x0: (batch, NUM_ACTIONS_CHUNK, action_dim) source points (not used, for compatibility)
            x1: (batch, NUM_ACTIONS_CHUNK, action_dim) target points (ground truth actions)
            actions_hidden_states: (batch, NUM_ACTIONS_CHUNK * action_dim, input_dim)
        
        Returns:
            loss: scalar loss value
        """
        batch_size = x1.shape[0]
        device = x1.device
        model_dtype = next(self.parameters()).dtype
        
        # 确保数据类型一致
        x1 = x1.to(dtype=model_dtype)
        actions_hidden_states = actions_hidden_states.to(dtype=model_dtype)
        
        # 采样噪声和双时间步
        e = torch.randn_like(x1)
        t, r = self.sample_two_timesteps(batch_size, device)
        
        # 确保时间步需要梯度
        t = t.to(dtype=model_dtype).requires_grad_(True)
        r = r.to(dtype=model_dtype).requires_grad_(True)
        
        # 扩展维度
        t_expanded = t.view(-1, 1, 1)
        h = t - r
        
        # 插值位置
        z = (1 - t_expanded) * x1 + t_expanded * e
        v = e - x1
        
        # 网络函数 - 修正版
        def u_func(z_input, t_input, r_input, actions_hidden_input):
            h_input = t_input - r_input
            return self.forward(actions_hidden_input, t_input, h_input, z_input)
        
        # 正确的切向量
        tangent_z = torch.zeros_like(z)  # 对z不求导
        tangent_t = torch.ones_like(t)   # dt/dt = 1
        tangent_r = torch.zeros_like(r)  # dr/dt = 0
        tangent_actions = torch.zeros_like(actions_hidden_states)  # 对actions_hidden_states不求导
        
        with torch.amp.autocast("cuda", enabled=False):
            # 计算JVP
            u_pred, dudt = torch.func.jvp(
                u_func,
                (z, t, r, actions_hidden_states),
                (tangent_z, tangent_t, tangent_r, tangent_actions)
            )
            
            # MeanFlow目标
            h_expanded = h.view(-1, 1, 1)
            u_target = (v - h_expanded * dudt).detach()
            
            # 计算损失
            loss_per_sample = ((u_pred - u_target) ** 2).sum(dim=(1, 2))
            
            # 自适应权重
            adaptive_weight = (loss_per_sample.detach() + self.norm_eps) ** self.norm_p
            weighted_loss = loss_per_sample / adaptive_weight
            
            loss = weighted_loss.mean()
        
        return loss
    
    @torch.no_grad()
    def predict_action(self, actions_hidden_states):
        """
        Generate actions using the EMA model with single-step sampling.
        
        This implements the inference procedure from MeanFlow:
        1. Start from noise e
        2. Set t=1, r=0 (so h=1)
        3. Predict vector field u
        4. Compute clean actions as z_0 = z_1 - u
        
        Args:
            actions_hidden_states: (batch, NUM_ACTIONS_CHUNK * action_dim, input_dim)
        
        Returns:
            actions: (batch, NUM_ACTIONS_CHUNK, action_dim)
        """
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        
        # Sample initial noise
        e = torch.randn(
            (batch_size, NUM_ACTIONS_CHUNK, self.action_dim),
            device=device,
            dtype=actions_hidden_states.dtype
        )
        
        # Use single timestep for fast inference (as in original MeanFlow)
        t = torch.ones(batch_size, device=device)  # t = 1 (start from noise)
        r = torch.zeros(batch_size, device=device)  # r = 0
        h = t - r  # h = 1
        
        # Use EMA model for inference
        model_to_use = self.model_ema if self.model_ema is not None else self.model
        
        # Predict vector field and generate clean actions
        x = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        t_emb = self.time_encoder(t).unsqueeze(1).expand(-1, NUM_ACTIONS_CHUNK, -1)
        h_emb = self.time_encoder(h).unsqueeze(1).expand(-1, NUM_ACTIONS_CHUNK, -1)
        
        combined_input = torch.cat([x, t_emb, h_emb, e], dim=-1)
        u = model_to_use(combined_input)
        
        # Generate clean actions: z_0 = z_1 - u (reverse the flow)
        actions = e - u
        
        return actions
    
    @torch.no_grad()  
    def sample_actions_multi_step(self, actions_hidden_states, num_steps=None):
        """
        Multi-step sampling for higher quality (optional, for comparison).
        
        This uses Euler integration to solve the ODE from t=1 to t=0.
        
        Args:
            actions_hidden_states: (batch, NUM_ACTIONS_CHUNK * action_dim, input_dim)
            num_steps: number of integration steps (default: self.num_flow_steps)
        
        Returns:
            actions: (batch, NUM_ACTIONS_CHUNK, action_dim)
        """
        if num_steps is None:
            num_steps = self.num_flow_steps
            
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        
        # Start from noise
        z = torch.randn(
            (batch_size, NUM_ACTIONS_CHUNK, self.action_dim),
            device=device,
            dtype=actions_hidden_states.dtype
        )
        
        # Integration from t=1 to t=0 with step size dt
        dt = 1.0 / num_steps
        
        model_to_use = self.model_ema if self.model_ema is not None else self.model
        
        for i in range(num_steps):
            t_val = 1.0 - i * dt  # From 1 to 0
            t = torch.full((batch_size,), t_val, device=device)
            h = torch.full((batch_size,), dt, device=device)  # Small time difference
            
            # Predict vector field
            x = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
            t_emb = self.time_encoder(t).unsqueeze(1).expand(-1, NUM_ACTIONS_CHUNK, -1)
            h_emb = self.time_encoder(h).unsqueeze(1).expand(-1, NUM_ACTIONS_CHUNK, -1)
            
            combined_input = torch.cat([x, t_emb, h_emb, z], dim=-1)
            u = model_to_use(combined_input)
            
            # Euler integration step
            z = z - dt * u
        
        return z


__all__ = ['FlowMatchingActionHead', 'OTFlowMatchingActionHead', 'COTFlowMatchingActionHead', 'MeanFlowActionHead']