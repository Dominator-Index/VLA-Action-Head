"""
Regression-based Action Heads

This module contains action heads that use regression approaches:
- L1RegressionActionHead: Simple L1 regression
- VAEActionHead: Variational Autoencoder-based action prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from .shared_components import MLPResNet, BaseActionHead


class L1RegressionActionHead(BaseActionHead):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
    ):
        super().__init__(input_dim, hidden_dim, action_dim)
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim*ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim
        )
        self._initialize_parameters()
    
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
    
    def compute_loss(self, actions_hidden_states, ground_truth_actions):
        predicted_actions = self.predict_action(actions_hidden_states)
        return torch.nn.L1Loss()(ground_truth_actions, predicted_actions)


class VAEActionHead(BaseActionHead):
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
        super().__init__(input_dim, hidden_dim, action_dim)
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
        self._initialize_parameters()

    def encode(self, actions_hidden_states):
        # actions_hidden_states: (batch, NUM_ACTIONS_CHUNK, action_dim * input_dim)
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


__all__ = ['L1RegressionActionHead', 'VAEActionHead']