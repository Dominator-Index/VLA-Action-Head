"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX


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
    Variational Autoencoder (VAE)-based action head for continuous action prediction.
    编码 action hidden states 到潜在空间，采样后解码生成动作。
    """
    
    def __init__(
        self, input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        latent_dim=32,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # 编码器：输入为 (batch, chunk_len * action_dim, input_dim)，flatten后为 (batch, chunk_len * action_dim * input_dim)
        self.encoder = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * NUM_ACTIONS_CHUNK * action_dim // ACTION_DIM,  # input_dim * chunk_len
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器：输入 latent，输出 (batch, chunk_len * action_dim)
        self.decoder = MLPResNet(
            num_blocks=2,
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=NUM_ACTIONS_CHUNK * action_dim,
        )
    
    def encode(self, actions_hidden_states):
        # actions_hidden_states: (batch, chunk_len * action_dim, input_dim)
        batch_size = actions_hidden_states.shape[0]
        x = actions_hidden_states.reshape(batch_size, -1)  # Flatten to (batch_size, chunk_len * action_dim * input_dim)
        h = self.encoder(x)  # (batch_size, hidden_dim)
        mu = self.fc_mu(h)  # (batch_size, latent_dim)
        logvar = self.fc_logvar(h)  # (batch_size, latent_dim)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # 采样 latent 向量
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=std.device)
        return mu + eps * std
    
    def decode(self, z):
        # z: (batch, latent_dim)
        recon = self.decoder(z)  # (batch_size, chunk_len * action_dim)
        recon = recon.view(-1, NUM_ACTIONS_CHUNK, self.action_dim)  # (batch, chunk_len, action_dim)
        return recon

    def forward(self, actions_hidden_states):
        """
        前向传播，返回重构的 action、均值、对数方差
        actions_hidden_states: (batch, chunk_len * action_dim, input_dim)
        """
        mu, logvar = self.encode(actions_hidden_states)  # (batch, latent_dim)
        z = self.reparameterize(mu, logvar)  # (batch, latent_dim)
        recon_actions = self.decode(z)  # (batch, chunk_len, action_dim)
        return recon_actions, mu, logvar
    
    def get_vae_loss(self, actions_hidden_states, ground_truth_actions, batch_size=None, device_id=None, beta=1.0):
        """
        计算 VAE 损失（重构损失 + KL 散度）
        actions_hidden_states: (batch, act_chunk_len, hidden_dim)
        ground_truth_actions: (batch, act_chunk_len, action_dim)
        """
        recon_actions, mu, logvar = self.forward(actions_hidden_states)  # (batch, act_chunk_len, action_dim), (batch, latent_dim), (batch, latent_dim)
        # 重构损失（L2）
        recon_loss = torch.nn.functional.mse_loss(recon_actions, ground_truth_actions, reduction='mean')  # (batch, act_chunk_len, action_dim)
        # KL 散度
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # (batch, latent_dim)
        # 总损失
        loss = recon_loss + beta * kl_loss  # beta 是 KL 散度的权重
        return loss
    
    def predict_action(self, actions_hidden_states):
        """
        推理时只用均值，不加噪声
        actions_hidden_states: (batch, chunk_len * action_dim, input_dim)
        返回: (batch, chunk_len, action_dim)
        """
        mu, _ = self.encode(actions_hidden_states)
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
