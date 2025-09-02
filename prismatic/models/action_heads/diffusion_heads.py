"""
Diffusion-based Action Heads

This module contains action heads that use diffusion models:
- DiffusionActionHead: Standard diffusion with noise prediction
- EndToEndDiffusionActionHead: End-to-end diffusion without modifying VLA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from .shared_components import MLPResNet, SinusoidalPositionalEncoding, BaseActionHead


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


class DiffusionActionHead(BaseActionHead):
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
        super().__init__(input_dim, hidden_dim, action_dim)
        self.num_diffusion_steps = num_diffusion_steps
        self.noise_predictor = NoisePredictionModel(
            transformer_hidden_dim=hidden_dim*ACTION_DIM, hidden_dim=hidden_dim, action_dim=action_dim
        )
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_diffusion_steps, beta_schedule="squaredcos_cap_v2")
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

    @torch.no_grad()
    def predict_action(self, actions_hidden_states):
        """
        Predict actions using reverse diffusion sampling.
        Note: This is an approximation since DiffusionActionHead requires VLA forward passes for accurate multi-step sampling.
        Here, we use fixed actions_hidden_states for all steps, which is not fully accurate but follows the structure of run_diffusion_sampling.
        """
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        
        # Sample random noisy action, used as the starting point for reverse diffusion
        noise = torch.randn(
            size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
            device=device,
            dtype=actions_hidden_states.dtype,
        )
        
        # Set diffusion timestep values
        self.noise_scheduler.set_timesteps(self.num_diffusion_steps)
        
        # Reverse diffusion: Iteratively denoise to generate action, conditioned on observation
        curr_noisy_actions = noise
        for t in self.noise_scheduler.timesteps:
            # Get diffusion model's noise prediction (using fixed actions_hidden_states)
            timesteps = torch.Tensor([t]).repeat(batch_size).to(device)
            diffusion_timestep_embeddings = (
                self.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
            )
            diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)
            
            # Predict noise (note: in full sampling, actions_hidden_states would be updated via VLA forward pass)
            noise_pred = self.predict_noise(actions_hidden_states)
            noise_pred = noise_pred.reshape(curr_noisy_actions.shape)
            
            # Compute the action at the previous diffusion timestep: x_t -> x_{t-1}
            curr_noisy_actions = self.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample
        
        return curr_noisy_actions

    def compute_loss(self, actions_hidden_states, ground_truth_actions):
        noisy_dict = self.sample_noisy_actions(ground_truth_actions)
        noise = noisy_dict["noise"]
        noise_pred = self.predict_noise(actions_hidden_states)
        noise_pred = noise_pred.reshape(noise.shape)
        return F.mse_loss(noise_pred, noise, reduction="mean")


class EndToEndDiffusionActionHead(BaseActionHead):
    """端到端扩散动作头：不改动VLA，直接在head内进行扩散条件噪声预测与反演。"""
    
    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=7, num_diffusion_steps=50, **kwargs):
        super().__init__(input_dim, hidden_dim, action_dim)
        self.num_diffusion_steps = num_diffusion_steps
        self.time_encoder = SinusoidalPositionalEncoding(hidden_dim)
        
        # 输入: [hidden(action_dim*input_dim) + time(hidden_dim) + noisy_action(action_dim)]
        self.denoising_mlp = MLPResNet(
            num_blocks=3,
            input_dim=input_dim * action_dim + hidden_dim + action_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )
        
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
        )
    
    def sample_noisy_actions(self, ground_truth_actions):
        # 简化版
        batch_size = ground_truth_actions.shape[0]
        device = ground_truth_actions.device
        noise = torch.randn_like(ground_truth_actions)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, 
                                 size=(batch_size,), device=device)
        noisy_actions = self.noise_scheduler.add_noise(ground_truth_actions, noise, timesteps)
        
        return {
            "noise": noise, 
            "noisy_actions": noisy_actions, 
            "timesteps": timesteps
        }
    
    def forward(self, actions_hidden_states, noisy_actions, timesteps):
        # 简化版
        batch_size = actions_hidden_states.shape[0]
        x = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        t_emb = self.time_encoder(timesteps).to(x.dtype).unsqueeze(1)
        t_emb = t_emb.expand(-1, NUM_ACTIONS_CHUNK, -1)
        
        # 拼接特征
        concat = torch.cat([x, t_emb, noisy_actions], dim=-1)
        noise_pred = self.denoising_mlp(concat)
        
        return noise_pred
        
    @torch.no_grad()
    def predict_action(self, actions_hidden_states):
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        
        # 从噪声开始
        self.noise_scheduler.set_timesteps(self.num_diffusion_steps, device=device)
        x = torch.randn(size=(batch_size, NUM_ACTIONS_CHUNK, self.action_dim), 
                       device=device, dtype=actions_hidden_states.dtype)
                       
        # 反向扩散过程
        for t in self.noise_scheduler.timesteps:
            t_scalar = torch.full((batch_size,), t, device=device, dtype=torch.long)
            noise_pred = self.forward(actions_hidden_states, x, t_scalar)
            x = self.noise_scheduler.step(noise_pred, t, x).prev_sample
            
        return x
    
    def compute_loss(self, actions_hidden_states, ground_truth_actions):
        noisy_dict = self.sample_noisy_actions(ground_truth_actions)
        noise, noisy_actions, timesteps = (
            noisy_dict["noise"],
            noisy_dict["noisy_actions"],
            noisy_dict["timesteps"],
        )
        
        noise_pred = self.forward(actions_hidden_states, noisy_actions=noisy_actions, timesteps=timesteps)
        return F.mse_loss(noise_pred, noise, reduction="mean")


__all__ = ['DiffusionActionHead', 'EndToEndDiffusionActionHead']