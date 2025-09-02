"""
Other Specialized Action Heads

This module contains specialized action heads that don't fit into the main categories:
- ConvexFlowActionHead: Convex potential flow-based action head
- ShortcutActionHead: Shortcut models for fast generation
- NormalizingFlowActionHead: Standard normalizing flow-based action head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from .shared_components import MLPResNet, SinusoidalPositionalEncoding, BaseActionHead, PosLinear, ActNorm, ActNormNoLogdet, SequentialFlow

class ConvexFlowActionHead(nn.Module):
    """
    Convex Potential Flow-based action head inspired by CP-Flow.
    
    This implementation uses Input-Convex Neural Networks (ICNNs) to learn 
    convex potential functions for normalizing flow-based action generation.
    """
    
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_flow_steps=20,
        # Convex flow specific parameters
        icnn_hidden_dim=64,
        icnn_num_layers=3,
        unbiased_logdet=False,
        no_bruteforce=True,
        m1=10,
        m2=None,
        rtol=0.0,
        atol=1e-3,
        bias_w1=0.0,
        trainable_w0=True,
        # ICNN activation parameters
        softplus_type='gaussian_softplus',
        zero_softplus=True,
        symm_act_first=False,
        # 新增：是否使用条件化
        use_conditioning=True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_flow_steps = num_flow_steps
        self.icnn_hidden_dim = icnn_hidden_dim
        self.icnn_num_layers = icnn_num_layers
        self.no_bruteforce = no_bruteforce
        self.use_conditioning = use_conditioning
        
        if m2 is None:
            m2 = action_dim
        self.m1, self.m2 = m1, m2
        
        # 根据是否使用条件化决定输入维度
        if use_conditioning:
            # Input projection: map from VLA hidden states to conditioning space
            self.input_projection = MLPResNet(
                num_blocks=2,
                input_dim=input_dim * action_dim,
                hidden_dim=hidden_dim,
                output_dim=icnn_hidden_dim,  # 作为ICNN的条件输入
            )
            
            # 使用条件化的ICNN (需要实现PICNN或类似结构)
            self.icnn = ConditionalICNN(
                dim=action_dim,
                dimh=icnn_hidden_dim,
                dimc=icnn_hidden_dim,  # 条件维度
                num_hidden_layers=icnn_num_layers,
                softplus_type=softplus_type,
                zero_softplus=zero_softplus,
                symm_act_first=symm_act_first,
            )
        else:
            # 无条件版本，仅用于对比
            self.input_projection = None
            self.icnn = ICNN(
                dim=action_dim,
                dimh=icnn_hidden_dim,
                num_hidden_layers=icnn_num_layers,
                softplus_type=softplus_type,
                zero_softplus=zero_softplus,
                symm_act_first=symm_act_first,
            )
        
        # Convex flow transformation
        self.convex_flow = ConditionalDeepConvexFlow(
            icnn=self.icnn,
            dim=action_dim,
            unbiased=unbiased_logdet,
            no_bruteforce=no_bruteforce,
            m1=m1,
            m2=m2,
            rtol=rtol,
            atol=atol,
            bias_w1=bias_w1,
            trainable_w0=trainable_w0,
            use_conditioning=use_conditioning,
        )
        
        # ActNorm for data-dependent initialization
        self.actnorm = ActNorm(action_dim)
        
        # Sequential flow combining ActNorm and ConvexFlow
        self.flow = ConditionalSequentialFlow([self.actnorm, self.convex_flow])
        
        # Initialize parameters
        self._initialize_parameters()
    
    def forward(self, actions_hidden_states, compute_log_prob=True):
        """
        Forward transformation: base distribution -> action distribution
        """
        batch_size = actions_hidden_states.shape[0]
        
        # 准备条件信息
        if self.use_conditioning:
            conditioning = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
            conditioning = self.input_projection(conditioning)  # (batch, NUM_ACTIONS_CHUNK, icnn_hidden_dim)
        else:
            conditioning = None
        
        # Sample from base distribution (standard Gaussian)
        z = torch.randn(
            (batch_size, NUM_ACTIONS_CHUNK, self.action_dim),
            device=actions_hidden_states.device,
            dtype=actions_hidden_states.dtype
        )
        
        # Apply convex flow transformations for each action chunk
        actions = []
        log_probs = []
        
        for i in range(NUM_ACTIONS_CHUNK):
            z_chunk = z[:, i, :]  # (batch, action_dim)
            cond_chunk = conditioning[:, i, :] if conditioning is not None else None
            
            # Apply convex flow transformation
            if compute_log_prob:
                # Compute log probability using flow
                log_prob_chunk = self.flow.logp(z_chunk, context=cond_chunk)
                log_probs.append(log_prob_chunk)
                
                # Transform to get actions
                actions_chunk, _ = self.flow.forward_transform(z_chunk, context=cond_chunk)
            else:
                # Just transform without computing log probability
                actions_chunk, _ = self.flow.forward_transform(z_chunk, context=cond_chunk)
            
            actions.append(actions_chunk)
        
        # Combine results
        actions = torch.stack(actions, dim=1)  # (batch, NUM_ACTIONS_CHUNK, action_dim)
        
        if compute_log_prob:
            log_prob = torch.stack(log_probs, dim=1).sum(dim=1)  # Sum over chunks
            return actions, log_prob
        else:
            return actions
    
    def inverse(self, actions, actions_hidden_states):
        """
        Inverse transformation: action distribution -> base distribution
        
        Args:
            actions: (batch, NUM_ACTIONS_CHUNK, action_dim)
            actions_hidden_states: (batch, NUM_ACTIONS_CHUNK * action_dim, input_dim)
        
        Returns:
            z: (batch, NUM_ACTIONS_CHUNK, action_dim) latent variables
            log_det_jacobian: (batch,) log determinant of Jacobian
        """
        batch_size = actions.shape[0]
        
        # 准备条件信息
        if self.use_conditioning:
            conditioning = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
            conditioning = self.input_projection(conditioning)
        else:
            conditioning = None
        
        # Apply inverse transformation for each action chunk
        z_list = []
        log_dets = []
        
        for i in range(NUM_ACTIONS_CHUNK):
            actions_chunk = actions[:, i, :]  # (batch, action_dim)
            cond_chunk = conditioning[:, i, :] if conditioning is not None else None
            
            # Apply inverse convex flow transformation with context
            z_chunk = self.flow.reverse(actions_chunk, context=cond_chunk)
            z_list.append(z_chunk)
            
            # Compute log determinant with context
            _, log_det = self.flow.forward_transform(z_chunk, context=cond_chunk)
            log_dets.append(log_det)
        
        z = torch.stack(z_list, dim=1)  # (batch, NUM_ACTIONS_CHUNK, action_dim)
        log_det_jacobian = torch.stack(log_dets, dim=1).sum(dim=1)  # Sum over chunks
        
        return z, log_det_jacobian
    
    def compute_convex_flow_loss(self, actions_hidden_states, ground_truth_actions):
        """
        Compute negative log-likelihood loss for convex flow training.
        """
        batch_size = ground_truth_actions.shape[0]
        
        # 准备条件信息
        if self.use_conditioning:
            conditioning = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
            conditioning = self.input_projection(conditioning)
        else:
            conditioning = None
        
        # Compute log probability for each action chunk
        log_probs = []
        
        for i in range(NUM_ACTIONS_CHUNK):
            actions_chunk = ground_truth_actions[:, i, :]  # (batch, action_dim)
            cond_chunk = conditioning[:, i, :] if conditioning is not None else None
            
            # Compute log probability using convex flow
            log_prob_chunk = self.flow.logp(actions_chunk, context=cond_chunk)
            log_probs.append(log_prob_chunk)
        
        # Total log probability
        total_log_prob = torch.stack(log_probs, dim=1).sum(dim=1)  # Sum over chunks
        
        # Return negative log-likelihood
        loss = -total_log_prob.mean()
        
        return loss
    
    @torch.no_grad()
    def predict_action(self, actions_hidden_states):
        """
        Predict actions using the convex flow model.
        
        Args:
            actions_hidden_states: (batch, NUM_ACTIONS_CHUNK * action_dim, input_dim)
        
        Returns:
            actions: (batch, NUM_ACTIONS_CHUNK, action_dim)
        """
        actions, _ = self.forward(actions_hidden_states, compute_log_prob=False)
        return actions
    
    @torch.no_grad()
    def sample_actions(self, actions_hidden_states, num_samples=1):
        """
        Sample multiple action sequences for a given observation.
        
        Args:
            actions_hidden_states: (batch, NUM_ACTIONS_CHUNK * action_dim, input_dim)
            num_samples: Number of action samples to generate
        
        Returns:
            actions: (batch, num_samples, NUM_ACTIONS_CHUNK, action_dim)
        """
        batch_size = actions_hidden_states.shape[0]
        
        # Expand conditioning for multiple samples
        conditioning_expanded = actions_hidden_states.unsqueeze(1).expand(-1, num_samples, -1, -1)
        conditioning_flat = conditioning_expanded.reshape(batch_size * num_samples, -1, actions_hidden_states.shape[-1])
        
        # Generate samples
        actions_flat, _ = self.forward(conditioning_flat, compute_log_prob=False)
        
        # Reshape back to (batch, num_samples, NUM_ACTIONS_CHUNK, action_dim)
        actions = actions_flat.reshape(batch_size, num_samples, NUM_ACTIONS_CHUNK, self.action_dim)
        
        return actions
    
    def get_potential(self, x):
        """
        Get the convex potential value for given input.
        
        Args:
            x: (batch, action_dim) input tensor
        
        Returns:
            potential: (batch, 1) potential values
        """
        return self.convex_flow.get_potential(x)
    
    def visualize_potential(self, actions_hidden_states, bounds=(-3, 3), resolution=50):
        """
        Visualize the learned convex potential (for 2D actions only).
        
        Args:
            actions_hidden_states: (1, NUM_ACTIONS_CHUNK * action_dim, input_dim)
            bounds: (min, max) bounds for visualization
            resolution: Grid resolution
        
        Returns:
            potential_grid: (resolution, resolution) potential values
        """
        if self.action_dim != 2:
            raise ValueError("Potential visualization only supported for 2D actions")
        
        # Create grid
        x1 = torch.linspace(bounds[0], bounds[1], resolution)
        x2 = torch.linspace(bounds[0], bounds[1], resolution)
        X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
        grid_points = torch.stack([X1.flatten(), X2.flatten()], dim=1)
        
        # Move to same device as model
        device = next(self.parameters()).device
        grid_points = grid_points.to(device)
        
        # Compute potential values
        with torch.no_grad():
            # Use first action chunk for visualization
            potential_values = []
            batch_size = 1000  # Process in batches to avoid memory issues
            
            for i in range(0, len(grid_points), batch_size):
                batch_points = grid_points[i:i+batch_size]
                batch_potential = self.convex_flow.get_potential(batch_points)
                potential_values.append(batch_potential)
            
            potential_values = torch.cat(potential_values, dim=0)
        
        # Reshape to grid
        potential_grid = potential_values.reshape(resolution, resolution)
        
        return potential_grid.cpu().numpy()

# 需要添加的条件化组件

class ConditionalICNN(nn.Module):
    """条件化的Input-Convex Neural Network"""
    
    def __init__(self, dim=2, dimh=16, dimc=16, num_hidden_layers=2, 
                 softplus_type='gaussian_softplus', zero_softplus=False, symm_act_first=False):
        super().__init__()
        self.dim = dim
        self.dimh = dimh
        self.dimc = dimc
        self.num_hidden_layers = num_hidden_layers
        self.symm_act_first = symm_act_first
        
        # Activation function
        self.act = self._get_softplus(softplus_type, zero_softplus)
        
        # First layer (can be negative) - 包含条件输入
        self.Wz0 = nn.Linear(dim, dimh)
        self.Wc0 = nn.Linear(dimc, dimh)  # 条件输入
        
        # Hidden layers (positive weights for convexity)
        self.Wzs = nn.ModuleList([
            PosLinear(dimh, dimh) for _ in range(num_hidden_layers - 1)
        ])
        
        # Skip connections (can be negative) - 包含条件输入
        self.Wxs = nn.ModuleList([
            nn.Linear(dim, dimh) for _ in range(num_hidden_layers - 1)
        ])
        self.Wcs = nn.ModuleList([
            nn.Linear(dimc, dimh) for _ in range(num_hidden_layers - 1)
        ])
        
        # Output layer (positive weights)
        self.Wz_out = PosLinear(dimh, 1, bias=False)
        self.Wx_out = nn.Linear(dim, 1, bias=False)
        self.Wc_out = nn.Linear(dimc, 1, bias=False)  # 条件输出
        
        # ActNorm layers for stability
        self.actnorm0 = ActNormNoLogdet(dimh)
        self.actnorms = nn.ModuleList([
            ActNormNoLogdet(dimh) for _ in range(num_hidden_layers - 1)
        ])
    
    def _get_softplus(self, softplus_type, zero_softplus):
        """Get the appropriate softplus activation function."""
        if softplus_type == 'gaussian_softplus':
            def gaussian_softplus(x):
                pi_half = torch.tensor(torch.pi / 2, device=x.device, dtype=x.dtype)
                z = torch.sqrt(pi_half)
                sqrt_2 = torch.tensor(2.0, device=x.device, dtype=x.dtype).sqrt()
                return (z * x * torch.erf(x / sqrt_2) + 
                       torch.exp(-x**2 / 2) + z * x) / (2*z)
            act = gaussian_softplus
        else:
            act = F.softplus
        
        if zero_softplus:
            def shifted_act(x):
                return act(x) - act(torch.zeros_like(x))
            return shifted_act
        return act
    
    def forward(self, x, c):
        # First layer with conditioning
        if self.symm_act_first:
            z = self.act(self.actnorm0(self.Wz0(x) + self.Wc0(c))) - 0.5 * self.actnorm0(self.Wz0(x) + self.Wc0(c))
        else:
            z = self.act(self.actnorm0(self.Wz0(x) + self.Wc0(c)))
        
        # Hidden layers with skip connections and conditioning
        for Wz, Wx, Wc, actnorm in zip(self.Wzs, self.Wxs, self.Wcs, self.actnorms):
            z = self.act(actnorm(Wz(z) + Wx(x) + Wc(c)))
        
        # Output with conditioning
        return self.Wz_out(z) + self.Wx_out(x) + self.Wc_out(c)

class ConditionalDeepConvexFlow(DeepConvexFlow):
    """条件化的Deep Convex Flow"""
    
    def __init__(self, icnn, dim, use_conditioning=True, **kwargs):
        super().__init__(icnn, dim, **kwargs)
        self.use_conditioning = use_conditioning
    
    def get_potential(self, x, context=None):
        n = x.size(0)
        if self.use_conditioning and context is not None:
            icnn_out = self.icnn(x, context)
        else:
            icnn_out = self.icnn(x)
        return F.softplus(self.w1) * icnn_out + F.softplus(self.w0) * (x.view(n, -1) ** 2).sum(1, keepdim=True) / 2
    
    def forward_transform(self, x, logdet=0, context=None):
        """Forward transformation with context support."""
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            F_val = self.get_potential(x, context)
            f = torch.autograd.grad(F_val.sum(), x, create_graph=True)[0]
        
        if self.training or self.no_bruteforce:
            # Use stochastic estimate during training
            return f, logdet + self._stochastic_logdet(x)
        else:
            # Use exact computation during evaluation
            return f, logdet + self._exact_logdet(x)
    
    def reverse(self, y, max_iter=1000, lr=1.0, tol=1e-12, context=None):
        """Inverse transformation with context support."""
        x = y.clone().detach().requires_grad_(True)
        
        def closure():
            F_val = self.get_potential(x, context)
            loss = torch.sum(F_val) - torch.sum(x * y)
            x.grad = torch.autograd.grad(loss, x)[0].detach()
            return loss
        
        optimizer = torch.optim.LBFGS([x], lr=lr, max_iter=max_iter, 
                                     tolerance_grad=tol, tolerance_change=tol)
        optimizer.step(closure)
        
        return x.detach()
        
class ConditionalSequentialFlow(SequentialFlow):
    """条件化的Sequential Flow"""
    
    def forward_transform(self, x, logdet=0, context=None):
        for flow in self.flows:
            if hasattr(flow, 'get_potential'):  # ConvexFlow
                x, logdet = flow.forward_transform(x, logdet, context=context)
            else:  # ActNorm等
                x, logdet = flow.forward_transform(x, logdet)
        return x, logdet
    
    def reverse(self, x, context=None):
        """Reverse transformation with context support."""
        for flow in reversed(self.flows):
            if hasattr(flow, 'get_potential'):  # ConvexFlow
                x = flow.reverse(x, context=context)
            else:  # ActNorm等
                x = flow.reverse(x)
        return x
    
    def logp(self, x, context=None):
        z, logdet = self.forward_transform(x, context=context)
        pi_const = torch.tensor(torch.pi, device=x.device, dtype=x.dtype)
        log_prob_z = -0.5 * (z ** 2).sum(dim=1) - 0.5 * z.size(1) * torch.log(2 * pi_const)
        return log_prob_z + logdet

# Helper classes for ICNN and Convex Flow (simplified versions)

# 在文件开头添加缺失的导入
import numpy as np

class ICNN(nn.Module):
    """
    Input-Convex Neural Network for parameterizing convex potential functions.
    """
    
    def __init__(self, dim=2, dimh=16, num_hidden_layers=2, 
                 softplus_type='gaussian_softplus', zero_softplus=False, symm_act_first=False):
        super().__init__()
        self.dim = dim
        self.dimh = dimh
        self.num_hidden_layers = num_hidden_layers
        self.symm_act_first = symm_act_first
        
        # Activation function
        self.act = self._get_softplus(softplus_type, zero_softplus)
        
        # First layer (can be negative)
        self.Wz0 = nn.Linear(dim, dimh)
        
        # Hidden layers (positive weights for convexity)
        self.Wzs = nn.ModuleList([
            PosLinear(dimh, dimh) for _ in range(num_hidden_layers - 1)
        ])
        
        # Skip connections (can be negative)
        self.Wxs = nn.ModuleList([
            nn.Linear(dim, dimh) for _ in range(num_hidden_layers - 1)
        ])
        
        # Output layer (positive weights)
        self.Wz_out = PosLinear(dimh, 1, bias=False)
        self.Wx_out = nn.Linear(dim, 1, bias=False)
        
        # ActNorm layers for stability
        self.actnorm0 = ActNormNoLogdet(dimh)
        self.actnorms = nn.ModuleList([
            ActNormNoLogdet(dimh) for _ in range(num_hidden_layers - 1)
        ])
    
    def _get_softplus(self, softplus_type, zero_softplus):
        """Get the appropriate softplus activation function."""
        if softplus_type == 'gaussian_softplus':
            def gaussian_softplus(x):
                # 修正：避免tensor类型错误
                pi_half = torch.tensor(torch.pi / 2, device=x.device, dtype=x.dtype)
                z = torch.sqrt(pi_half)
                sqrt_2 = torch.tensor(2.0, device=x.device, dtype=x.dtype).sqrt()
                return (z * x * torch.erf(x / sqrt_2) + 
                       torch.exp(-x**2 / 2) + z * x) / (2*z)
            act = gaussian_softplus
        else:
            act = F.softplus
        
        if zero_softplus:
            # Make activation zero at origin
            def shifted_act(x):
                return act(x) - act(torch.zeros_like(x))
            return shifted_act
        return act
    
    def forward(self, x):
        # First layer
        if self.symm_act_first:
            z = self.act(self.actnorm0(self.Wz0(x))) - 0.5 * self.actnorm0(self.Wz0(x))
        else:
            z = self.act(self.actnorm0(self.Wz0(x)))
        
        # Hidden layers with skip connections
        for Wz, Wx, actnorm in zip(self.Wzs, self.Wxs, self.actnorms):
            z = self.act(actnorm(Wz(z) + Wx(x)))
        
        # Output
        return self.Wz_out(z) + self.Wx_out(x)


class PosLinear(nn.Linear):
    """Linear layer with positive weights (for ICNN convexity)."""
    
    def forward(self, x):
        gain = 1 / x.size(1)
        return F.linear(x, F.softplus(self.weight), self.bias) * gain


class DeepConvexFlow(nn.Module):
    """
    Deep convex flow transformation using ICNN potential.
    """
    
    def __init__(self, icnn, dim, unbiased=False, no_bruteforce=True, 
                 m1=10, m2=None, rtol=0.0, atol=1e-3, bias_w1=0.0, trainable_w0=True):
        super().__init__()
        if m2 is None:
            m2 = dim
        
        self.icnn = icnn
        self.dim = dim
        self.no_bruteforce = no_bruteforce
        self.rtol = rtol
        self.atol = atol
        self.m1, self.m2 = m1, m2
        
        # Learnable parameters
        self.w0 = nn.Parameter(torch.log(torch.exp(torch.ones(1)) - 1), requires_grad=trainable_w0)
        self.w1 = nn.Parameter(torch.zeros(1) + bias_w1)
    
    def get_potential(self, x):
        """Compute the convex potential F(x)."""
        n = x.size(0)
        icnn_out = self.icnn(x)
        quadratic = F.softplus(self.w0) * (x.view(n, -1) ** 2).sum(1, keepdim=True) / 2
        return F.softplus(self.w1) * icnn_out + quadratic
    
    def forward_transform(self, x, logdet=0):
        """Forward transformation: x -> f(x) = ∇F(x)"""
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            F_val = self.get_potential(x)
            f = torch.autograd.grad(F_val.sum(), x, create_graph=True)[0]
        
        if self.training or self.no_bruteforce:
            # Use stochastic estimate during training
            return f, logdet + self._stochastic_logdet(x)
        else:
            # Use exact computation during evaluation
            return f, logdet + self._exact_logdet(x)
    
    def reverse(self, y, max_iter=1000, lr=1.0, tol=1e-12):
        """Inverse transformation: solve x such that f(x) = y"""
        x = y.clone().detach().requires_grad_(True)
        
        def closure():
            F_val = self.get_potential(x)
            loss = torch.sum(F_val) - torch.sum(x * y)
            x.grad = torch.autograd.grad(loss, x)[0].detach()
            return loss
        
        optimizer = torch.optim.LBFGS([x], lr=lr, max_iter=max_iter, 
                                     tolerance_grad=tol, tolerance_change=tol)
        optimizer.step(closure)
        
        return x.detach()
    
    def logp(self, x):
        """Compute log probability under the flow."""
        z, logdet = self.forward_transform(x)
        
        # 修正：确保设备和数据类型一致
        pi_const = torch.tensor(torch.pi, device=x.device, dtype=x.dtype)
        log_prob_z = -0.5 * (z ** 2).sum(dim=1) - 0.5 * self.dim * torch.log(2 * pi_const)
        
        return log_prob_z + logdet
    
    def _stochastic_logdet(self, x):
        """Stochastic estimate of log determinant."""
        # Simplified version - in practice would use more sophisticated estimators
        batch_size = x.size(0)
        return torch.zeros(batch_size, device=x.device, dtype=x.dtype)
    
    def _exact_logdet(self, x):
        """Exact computation of log determinant."""
        batch_size = x.size(0)
        
        with torch.enable_grad():
            x.requires_grad_(True)
            F_val = self.get_potential(x)
            f = torch.autograd.grad(F_val.sum(), x, create_graph=True)[0]
            
            # Compute Hessian
            H = []
            f_flat = f.reshape(batch_size, -1)
            for i in range(f_flat.shape[1]):
                retain_graph = i < (f_flat.shape[1] - 1)
                Hi = torch.autograd.grad(f_flat[:, i].sum(), x, 
                                       create_graph=False, retain_graph=retain_graph)[0]
                H.append(Hi)
            
            H = torch.stack(H, dim=1)  # (batch, dim, dim)
        
        return torch.slogdet(H).logabsdet


class ActNorm(nn.Module):
    """ActNorm layer with data-dependent initialization."""
    
    def __init__(self, num_features, logscale_factor=1., scale=1., learn_scale=True):
        super().__init__()
        self.initialized = False
        self.num_features = num_features
        self.learn_scale = learn_scale
        
        self.register_parameter('b', nn.Parameter(torch.zeros(1, num_features), requires_grad=True))
        if learn_scale:
            self.logscale_factor = logscale_factor
            self.scale = scale
            self.register_parameter('logs', nn.Parameter(torch.zeros(1, num_features), requires_grad=True))
    
    def forward_transform(self, x, logdet=0):
        if not self.initialized:
            self.initialized = True
            # Data-dependent initialization
            b = -torch.mean(x, dim=0, keepdim=True)
            self.b.data.copy_(b.data)
            
            if self.learn_scale:
                var = torch.var(x + b, dim=0, keepdim=True)
                logs = torch.log(self.scale / (torch.sqrt(var) + 1e-6)) / self.logscale_factor
                self.logs.data.copy_(logs.data)
        
        output = x + self.b
        
        if self.learn_scale:
            logs = self.logs * self.logscale_factor
            scale = torch.exp(logs)
            output = output * scale
            dlogdet = torch.sum(torch.log(scale))
            return output, logdet + dlogdet
        else:
            return output, logdet
    
    def reverse(self, y):
        assert self.initialized
        if self.learn_scale:
            logs = self.logs * self.logscale_factor
            scale = torch.exp(logs)
            x = y / scale - self.b
        else:
            x = y - self.b
        return x


class ActNormNoLogdet(ActNorm):
    """ActNorm without log determinant computation."""
    
    def forward(self, x):
        return super().forward_transform(x)[0]


class SequentialFlow(nn.Module):
    """Sequential composition of normalizing flow layers."""
    
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)
    
    def forward_transform(self, x, logdet=0):
        for flow in self.flows:
            x, logdet = flow.forward_transform(x, logdet)
        return x, logdet
    
    def reverse(self, x):
        for flow in reversed(self.flows):
            x = flow.reverse(x)
        return x
    
    def logp(self, x):
        z, logdet = self.forward_transform(x)
        # 修正：确保设备和数据类型一致
        pi_const = torch.tensor(torch.pi, device=x.device, dtype=x.dtype)
        log_prob_z = -0.5 * (z ** 2).sum(dim=1) - 0.5 * z.size(1) * torch.log(2 * pi_const)
        return log_prob_z + logdet