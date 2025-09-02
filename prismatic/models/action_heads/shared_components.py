"""
Shared Components for Action Heads

This module contains shared utilities, base classes, and common components
used across different action head implementations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sine- and cosine-based positional encoding that produces embeddings of a batch of timesteps.

    For example, at train time, the input might be a batch of 32 randomly sampled diffusion timesteps -> shape (32,)
    Then the output would be a batch of 32 timestep embeddings -> shape (32, D)

    Adapted from: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the positional encoding

    def forward(self, x):
        # x: (batch_size,)
        device = x.device
        assert self.dim % 2 == 0, f"# dimensions must be even but got {self.dim}"
        half_dim = self.dim // 2
        exponent = torch.arange(half_dim, device=device) * -math.log(10000) / (half_dim - 1)  # shape: (D/2,)
        emb = torch.exp(exponent)  # shape: (D/2,)
        emb = x[:, None] * emb[None, :]  # shape: (batch_size, 1) * (1, D/2) -> (batch_size, D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # shape: (batch_size, D)
        return emb


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    def __init__(self, num_blocks=3, input_dim=4096, hidden_dim=4096, output_dim=7):
        super().__init__()
        self.num_blocks = num_blocks
        
        # LayerNorm with elementwise_affine=False to avoid dtype issues
        self.layer_norm1 = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim, elementwise_affine=False),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim, elementwise_affine=False),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(num_blocks)
        ])
        
        self.layer_norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # 确保所有参数都是 BFloat16
        self.bfloat16()
    
    def forward(self, x):
        # 获取模型的数据类型（应该是 BFloat16）
        model_dtype = next(self.parameters()).dtype
        
        # 将输入转换为模型的数据类型
        if x.dtype != model_dtype:
            x = x.to(dtype=model_dtype)
        
        batch_size, seq_len, input_dim = x.shape
        x = x.view(batch_size * seq_len, input_dim)
        
        x = self.layer_norm1(x)
        
        x = F.relu(self.fc1(x))
        
        x = self.layer_norm2(x)
        
        for i, block in enumerate(self.residual_blocks):
            residual = x
            x = block(x)
            x = x + residual
        
        x = self.layer_norm3(x)
        
        x = self.fc2(x)
        
        x = x.view(batch_size, seq_len, -1)
        
        return x

class BaseActionHead(nn.Module):
    """Base class for all action heads providing common interface."""
    
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
    
    def predict_action(self, actions_hidden_states):
        """Predict actions - must be implemented by subclasses."""
        raise NotImplementedError
    
    def compute_loss(self, actions_hidden_states, ground_truth_actions):
        """Compute loss - must be implemented by subclasses."""
        raise NotImplementedError
    
    def _initialize_parameters(self):
        """Initialize parameters for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class PosLinear(nn.Linear):
    """Linear layer with positive weights (for ICNN convexity)."""
    
    def forward(self, x):
        gain = 1 / x.size(1)
        return F.linear(x, F.softplus(self.weight), self.bias) * gain


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


# Export shared components
__all__ = [
    'SinusoidalPositionalEncoding',
    'MLPResNetBlock', 
    'MLPResNet',
    'BaseActionHead',
    'PosLinear',
    'ActNorm',
    'ActNormNoLogdet',
    'SequentialFlow',
]