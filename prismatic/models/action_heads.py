"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX

# Add conditional flow matching 
from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
from torchcfm.optimal_transport import OTPlanSampler

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
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x

# TODO: Add action heads

class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        action = self.model(rearranged_actions_hidden_states)
        return action

class VAEActionHead(nn.Module):
    """
    Chunk-wise VAE-based action head for continuous action prediction.
    每个 chunk 独立编码/解码，结构与 L1RegressionActionHead 对齐，便于公平对比。
    """
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        latent_dim=32,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # 与 L1RegressionActionHead 保持一致：每个 chunk 一个 MLP
        self.encoder = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * action_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = MLPResNet(
            num_blocks=2,
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def encode(self, actions_hidden_states):
        # actions_hidden_states: (batch, NUM_ACTIONS_CHUNK, action_dim * input_dim)
        batch_size = actions_hidden_states.shape[0]
        chunk = actions_hidden_states.shape[1]
        x = actions_hidden_states  # (batch, chunk, action_dim * input_dim)
        h = self.encoder(x)        # (batch, chunk, hidden_dim)
        mu = self.fc_mu(h)         # (batch, chunk, latent_dim)
        logvar = self.fc_logvar(h) # (batch, chunk, latent_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # z: (batch, chunk, latent_dim)
        recon = self.decoder(z)  # (batch, chunk, action_dim)
        return recon

    def forward(self, actions_hidden_states):
        # actions_hidden_states: (batch, NUM_ACTIONS_CHUNK * action_dim, input_dim)
        batch_size = actions_hidden_states.shape[0]
        # reshape to (batch, NUM_ACTIONS_CHUNK, action_dim * input_dim)
        x = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_actions = self.decode(z)  # (batch, chunk, action_dim)
        return recon_actions, mu, logvar

    def get_vae_loss(self, actions_hidden_states, ground_truth_actions, batch_size=None, device_id=None, beta=1.0):
        # actions_hidden_states: (batch, NUM_ACTIONS_CHUNK * action_dim, input_dim)
        # ground_truth_actions: (batch, NUM_ACTIONS_CHUNK, action_dim)
        recon_actions, mu, logvar = self.forward(actions_hidden_states)
        recon_loss = torch.nn.functional.mse_loss(recon_actions, ground_truth_actions, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss
        return loss

    def predict_action(self, actions_hidden_states):
        batch_size = actions_hidden_states.shape[0]
        x = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        mu, _ = self.encode(x)
        recon_action = self.decode(mu)
        return recon_action
    
class NoisePredictionModel(nn.Module):
    """
    Diffusion noise prediction model that takes an observation embedding (which fuses the
    noisy action, diffusion timestep, and image-language observation embeddings) and
    outputs a noise prediction.
    """

    def __init__(
        self,
        transformer_hidden_dim,  # Transformer hidden embedding size
        hidden_dim,  # MLP hidden size
        action_dim=7,  # action dimensionality
    ):
        super().__init__()
        self.mlp_resnet = MLPResNet(
            num_blocks=2,
            input_dim=transformer_hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def forward(
        self,
        obs,
    ):
        # obs: observation embeddings to condition the generation on
        # - shape: (batch_size, chunk_len, rearranged_hidden_dim=action_dim*hidden_dim)
        #
        # output: predicted noise
        # - shape: (batch_size, action_dim)
        output = self.mlp_resnet(obs)
        return output


class DiffusionActionHead(nn.Module):
    """
    Simple MLP-based action head that generates continuous actions via conditional denoising diffusion process.

    Loosely inspired by: https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py
    """

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_diffusion_steps=100,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.noise_predictor = NoisePredictionModel(
            transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim
        )
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_diffusion_steps, beta_schedule="squaredcos_cap_v2")
        self.num_diffusion_steps = num_diffusion_steps
        self.time_encoder = SinusoidalPositionalEncoding(dim=hidden_dim)

    def sample_noisy_actions(self, ground_truth_actions):
        """
        Samples noise and applies noise to ground-truth actions to produce noisy actions, which are
        used as input in the noise prediction network. Returns noise, noisy actions, and the
        corresponding diffusion timestep embeddings.
        """
        # ground_truth_actions: ground-truth actions
        # - shape: (batch_size, chunk_len, action_dim)
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        # Sample random noise with shape equal to actions, used for closed-form forward diffusion.
        noise = torch.randn(size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), device=device, dtype=ground_truth_actions.dtype)  # (B, chunk_len, action_dim)
        # Sample random diffusion timesteps (one for each action in batch).
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(batch_size,), device=device
        )
        # Add noise to clean actions according to the magnitude at each diffusion timestep via
        # closed-form forward diffusion.
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)  # (B, chunk_len, action_dim)

        # Get diffusion timestep embeddings as well
        diffusion_timestep_embeddings = self.time_encoder(timesteps).to(noisy_actions.dtype).to(noisy_actions.device)  # (B, llm_dim)
        diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (B, 1, llm_dim)

        return_dict = dict(
            noise=noise,
            noisy_actions=noisy_actions,
            diffusion_timestep_embeddings=diffusion_timestep_embeddings,
        )

        return return_dict

    def predict_noise(self, actions_hidden_states):
        """
        Given a batch of last hidden Transformer layer embeddings (which fuse the vision-language observation embeddings,
        noisy action embeddings, and diffusion timestep embedding), predicts the noise applied to the actions.
        """
        # actions_hidden_states: last hidden states of Transformer corresponding to action tokens in sequence
        # - shape: (batch_size, chunk_len * action_dim, hidden_dim)
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        rearranged_actions_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)  # (batch_size, chunk_len, action_dim * hidden_dim)
        # Get diffusion model's noise prediction.
        noise_pred = self.noise_predictor(rearranged_actions_hidden_states)
        return noise_pred

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
            input_dim=input_dim * action_dim + hidden_dim,  # 拼接时间编码
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
        t_emb = t_emb.expand(-1, NUM_ACTIONS_CHUNK, -1)
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
        t = t.view(-1, 1, 1)  # (batch, 1, 1)
        x_t = (1 - t) * x0 + t * x1  # (batch, NUM_ACTIONS_CHUNK, action_dim)
        u_t = x1 - x0  # (batch, NUM_ACTIONS_CHUNK, action_dim)
        # 时间编码
        t_scalar = t.view(-1)  # (batch,)
        t_emb = self.time_encoder(t_scalar)  # (batch, hidden_dim)
        t_emb = t_emb.unsqueeze(1)  # (batch, 1, hidden_dim)
        # 将x_t作为输入传递给模型
        flow_pred = self.forward(actions_hidden_states, t_emb, x_t)  # (batch, NUM_ACTIONS_CHUNK, action_dim)
        loss = torch.nn.functional.mse_loss(flow_pred, u_t)
        return loss
    
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
        # Get OT matched pairs
        x0_ot, x1_ot = self.ot_plan_sampler.sample_plan(x0, x1)
        
        # Now proceed with standard flow matching using OT matched pairs
        t = t.view(-1, 1, 1)  # (batch, 1, 1)
        x_t = (1 - t) * x0_ot + t * x1_ot  # (batch, NUM_ACTIONS_CHUNK, action_dim)
        u_t = x1_ot - x0_ot  # (batch, NUM_ACTIONS_CHUNK, action_dim)
        
        # Time encoding
        t_scalar = t.view(-1)  # (batch,)
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


        
     
        
        
    
    
    
