"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX
from transformers import LlamaForCausalLM, LlamaConfig

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

    @torch.no_grad()
    def predict_action(self, actions_hidden_states):
        """
        使用反向扩散采样生成actions，参考finetune_swanlab.py中的run_diffusion_sampling
        注意：DiffusionActionHead没有noisy_action_projector和proprio_projector参数，所以简化版
        """
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        
        # Sample random noisy action, used as the starting point for reverse diffusion
        noise = torch.randn(
            size=(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM),
            device=device,
            dtype=torch.bfloat16,
        )  # (batch, chunk_len, action_dim)

        # Set diffusion timestep values
        self.noise_scheduler.set_timesteps(self.num_diffusion_steps)

        # Reverse diffusion: Iteratively denoise to generate action, conditioned on observation
        curr_noisy_actions = noise
        for t in self.noise_scheduler.timesteps:
            # Get diffusion timestep embeddings
            timesteps = torch.Tensor([t]).repeat(batch_size).to(device)
            diffusion_timestep_embeddings = self.time_encoder(timesteps).to(curr_noisy_actions.dtype).to(curr_noisy_actions.device)
            diffusion_timestep_embeddings = diffusion_timestep_embeddings.unsqueeze(1)  # (batch, 1, hidden_dim)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Predict noise
                noise_pred = self.predict_noise(actions_hidden_states)

            # Compute the action at the previous diffusion timestep: x_t -> x_{t-1}
            curr_noisy_actions = self.noise_scheduler.step(noise_pred, t, curr_noisy_actions).prev_sample

        return curr_noisy_actions.reshape(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM)

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
        
class NormalizingFlowActionHead(nn.Module):
    """
    Normalizing Flow-based action head that uses invertible transformations to model action distributions.
    
    This implementation is inspired by Real NVP (Real-valued Non-Volume Preserving) transformations.
    It learns a bijective mapping between a simple base distribution (Gaussian) and the complex action distribution.
    
    Key features:
    1. Invertible affine coupling layers
    2. Easy-to-compute Jacobian determinants
    3. Flexible transformation capacity
    4. Stable training through proper initialization
    """
    
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_coupling_layers=8,
        coupling_hidden_dim=256,
        # Normalizing Flow specific parameters
        mask_type="alternating",  # "alternating" or "random"
        use_batch_norm=False,     # Whether to use batch normalization
        dropout_rate=0.0,         # Dropout rate for regularization
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_coupling_layers = num_coupling_layers
        self.coupling_hidden_dim = coupling_hidden_dim
        self.use_batch_norm = use_batch_norm
        
        # Input projection: map from VLA hidden states to flow input
        self.input_projection = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * action_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )
        
        # Generate masks for affine coupling layers
        self.masks = self._generate_masks(mask_type)
        
        # Create affine coupling layers
        self.coupling_layers = nn.ModuleList([
            AffineCouplingLayer(
                mask=self.masks[i],
                hidden_dim=coupling_hidden_dim,
                dropout_rate=dropout_rate
            )
            for i in range(num_coupling_layers)
        ])
        
        # Optional batch normalization layers
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(action_dim)
                for _ in range(num_coupling_layers)
            ])
        
        # Output normalization to ensure actions are in reasonable range
        self.output_scale = nn.Parameter(torch.ones(action_dim))
        self.output_bias = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _generate_masks(self, mask_type):
        """
        Generate binary masks for affine coupling layers.
        
        Args:
            mask_type: Type of masking strategy ("alternating" or "random")
        
        Returns:
            List of binary masks for each coupling layer
        """
        masks = []
        
        if mask_type == "alternating":
            # Alternating mask pattern: [1,0,1,0,...] then [0,1,0,1,...]
            for i in range(self.num_coupling_layers):
                if i % 2 == 0:
                    # Even layers: first half fixed, second half transformed
                    mask = torch.zeros(self.action_dim)
                    mask[:self.action_dim // 2] = 1.0
                else:
                    # Odd layers: second half fixed, first half transformed
                    mask = torch.zeros(self.action_dim)
                    mask[self.action_dim // 2:] = 1.0
                masks.append(mask)
        
        elif mask_type == "random":
            # Random mask pattern for each layer
            for i in range(self.num_coupling_layers):
                mask = torch.zeros(self.action_dim)
                # Randomly select half of the dimensions to keep fixed
                fixed_indices = torch.randperm(self.action_dim)[:self.action_dim // 2]
                mask[fixed_indices] = 1.0
                masks.append(mask)
        
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
        
        return masks
    
    def _initialize_parameters(self):
        """Initialize model parameters for stable training."""
        # Initialize output scale and bias
        nn.init.ones_(self.output_scale)
        nn.init.zeros_(self.output_bias)
        
        # Initialize coupling layer parameters
        for coupling_layer in self.coupling_layers:
            coupling_layer.initialize_parameters()
    
    def forward(self, actions_hidden_states, compute_log_prob=True):
        """
        Forward transformation: base distribution -> action distribution
        
        Args:
            actions_hidden_states: (batch, NUM_ACTIONS_CHUNK * action_dim, input_dim)
            compute_log_prob: Whether to compute log probability
        
        Returns:
            actions: (batch, NUM_ACTIONS_CHUNK, action_dim)
            log_prob: (batch,) log probability if compute_log_prob=True
        """
        batch_size = actions_hidden_states.shape[0]
        
        # Project VLA hidden states to conditioning vector
        conditioning = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        conditioning = self.input_projection(conditioning)  # (batch, NUM_ACTIONS_CHUNK, action_dim)
        
        # Sample from base distribution (standard Gaussian)
        z = torch.randn(
            (batch_size, NUM_ACTIONS_CHUNK, self.action_dim),
            device=actions_hidden_states.device,
            dtype=actions_hidden_states.dtype
        )
        
        # Apply normalizing flow transformations
        x = z
        log_det_jacobian = torch.zeros(batch_size, device=z.device, dtype=z.dtype)
        
        for i, coupling_layer in enumerate(self.coupling_layers):
            # Condition the transformation on VLA hidden states
            x, log_det = coupling_layer.forward(x, conditioning)
            log_det_jacobian += log_det.sum(dim=1)  # Sum over action chunks
            
            # Apply batch normalization if enabled
            if self.use_batch_norm:
                # Reshape for batch norm: (batch * NUM_ACTIONS_CHUNK, action_dim)
                x_flat = x.reshape(-1, self.action_dim)
                x_flat = self.batch_norms[i](x_flat)
                x = x_flat.reshape(batch_size, NUM_ACTIONS_CHUNK, self.action_dim)
        
        # Apply output normalization
        actions = x * self.output_scale + self.output_bias
        
        if compute_log_prob:
            # 修正：正确计算log概率，避免tensor类型问题
            pi_const = torch.tensor(torch.pi, device=z.device, dtype=z.dtype)
            log_prob_z = -0.5 * (z ** 2).sum(dim=(1, 2)) - 0.5 * NUM_ACTIONS_CHUNK * self.action_dim * torch.log(2 * pi_const)
            log_prob = log_prob_z - log_det_jacobian
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
        
        # Project VLA hidden states to conditioning vector
        conditioning = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        conditioning = self.input_projection(conditioning)
        
        # Remove output normalization
        x = (actions - self.output_bias) / self.output_scale
        
        log_det_jacobian = torch.zeros(batch_size, device=actions.device, dtype=actions.dtype)
        
        # Apply inverse transformations in reverse order
        for i in range(len(self.coupling_layers) - 1, -1, -1):
            # Apply inverse batch normalization if enabled
            if self.use_batch_norm:
                # This is approximate since batch norm is not exactly invertible
                x_flat = x.reshape(-1, self.action_dim)
                # Use running statistics for inverse
                x_flat = (x_flat - self.batch_norms[i].running_mean) / torch.sqrt(self.batch_norms[i].running_var + self.batch_norms[i].eps)
                x = x_flat.reshape(batch_size, NUM_ACTIONS_CHUNK, self.action_dim)
            
            x, log_det = self.coupling_layers[i].inverse(x, conditioning)
            log_det_jacobian += log_det.sum(dim=1)
        
        return x, log_det_jacobian
    
    def compute_nf_loss(self, actions_hidden_states, ground_truth_actions):
        """
        Compute negative log-likelihood loss for training.
        
        Args:
            actions_hidden_states: (batch, NUM_ACTIONS_CHUNK * action_dim, input_dim)
            ground_truth_actions: (batch, NUM_ACTIONS_CHUNK, action_dim)
        
        Returns:
            loss: Scalar loss value
        """
        # Transform ground truth actions to latent space
        z, log_det_jacobian = self.inverse(ground_truth_actions, actions_hidden_states)
        
        # 修正：正确计算log概率，确保device和dtype一致
        pi_const = torch.tensor(torch.pi, device=z.device, dtype=z.dtype)
        log_prob_z = -0.5 * (z ** 2).sum(dim=(1, 2)) - 0.5 * NUM_ACTIONS_CHUNK * self.action_dim * torch.log(2 * pi_const)
        
        # Compute log probability in action space
        log_prob_x = log_prob_z + log_det_jacobian
        
        # Return negative log-likelihood
        loss = -log_prob_x.mean()
        
        return loss
    
    @torch.no_grad()
    def predict_action(self, actions_hidden_states):
        """
        Predict actions using the normalizing flow model.
        
        Args:
            actions_hidden_states: (batch, NUM_ACTIONS_CHUNK * action_dim, input_dim)
        
        Returns:
            actions: (batch, NUM_ACTIONS_CHUNK, action_dim)
        """
        with torch.no_grad():
            actions, _ = self.forward(actions_hidden_states, compute_log_prob=False)
        return actions
    
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
        
        with torch.no_grad():
            actions_flat, _ = self.forward(conditioning_flat, compute_log_prob=False)
        
        # Reshape back to (batch, num_samples, NUM_ACTIONS_CHUNK, action_dim)
        actions = actions_flat.reshape(batch_size, num_samples, NUM_ACTIONS_CHUNK, self.action_dim)
        
        return actions


class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer for normalizing flows.
    
    This layer applies an affine transformation (scale and translation) to a subset of input dimensions,
    while keeping the other dimensions unchanged. The transformation parameters are computed using
    neural networks conditioned on both the unchanged dimensions and external conditioning information.
    """
    
    def __init__(self, mask, hidden_dim, dropout_rate=0.0):
        """
        Initialize affine coupling layer.
        
        Args:
            mask: Binary mask indicating which dimensions to keep fixed (1) vs transform (0)
            hidden_dim: Hidden dimension for transformation networks
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Register mask as buffer (not a parameter)
        self.register_buffer('mask', mask)
        
        # Networks for computing scale (log_scale for numerical stability)
        self.scale_net = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.hidden_dim),  # *2 for input + conditioning
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
        
        # Networks for computing translation
        self.translation_net = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.hidden_dim),  # *2 for input + conditioning
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
    
    def initialize_parameters(self):
        """Initialize parameters for stable training."""
        # Initialize final layers to output small values initially
        nn.init.zeros_(self.scale_net[-1].weight)
        nn.init.zeros_(self.scale_net[-1].bias)
        nn.init.zeros_(self.translation_net[-1].weight)
        nn.init.zeros_(self.translation_net[-1].bias)
    
    def _compute_scale_and_translation(self, x, conditioning):
        """
        Compute scale and translation parameters.
        
        Args:
            x: Input tensor (batch, NUM_ACTIONS_CHUNK, action_dim)
            conditioning: Conditioning information (batch, NUM_ACTIONS_CHUNK, action_dim)
        
        Returns:
            log_scale: Log of scale parameters
            translation: Translation parameters
        """
        # Mask input to only use unchanged dimensions
        masked_x = x * self.mask
        
        # Concatenate masked input with conditioning
        net_input = torch.cat([masked_x, conditioning], dim=-1)
        
        # Compute transformation parameters
        log_scale = self.scale_net(net_input)
        translation = self.translation_net(net_input)
        
        # Apply tanh to log_scale for stability and mask to only affect transformed dimensions
        log_scale = torch.tanh(log_scale) * (1 - self.mask)
        translation = translation * (1 - self.mask)
        
        return log_scale, translation
    
    def forward(self, x, conditioning):
        """
        Forward transformation: x -> y
        
        Args:
            x: Input tensor (batch, NUM_ACTIONS_CHUNK, action_dim)
            conditioning: Conditioning tensor (batch, NUM_ACTIONS_CHUNK, action_dim)
        
        Returns:
            y: Transformed tensor (batch, NUM_ACTIONS_CHUNK, action_dim)
            log_det: Log determinant of Jacobian (batch, NUM_ACTIONS_CHUNK)
        """
        log_scale, translation = self._compute_scale_and_translation(x, conditioning)
        
        # Apply affine transformation: y = mask * x + (1 - mask) * (x * exp(log_scale) + translation)
        y = self.mask * x + (1 - self.mask) * (x * torch.exp(log_scale) + translation)
        
        # Compute log determinant of Jacobian
        # For affine transformation, |det(J)| = exp(sum(log_scale))
        log_det = log_scale.sum(dim=-1)  # Sum over action dimensions
        
        return y, log_det
    
    def inverse(self, y, conditioning):
        """
        Inverse transformation: y -> x
        
        Args:
            y: Input tensor (batch, NUM_ACTIONS_CHUNK, action_dim)
            conditioning: Conditioning tensor (batch, NUM_ACTIONS_CHUNK, action_dim)
        
        Returns:
            x: Inverse transformed tensor (batch, NUM_ACTIONS_CHUNK, action_dim)
            log_det: Log determinant of Jacobian (batch, NUM_ACTIONS_CHUNK)
        """
        # 修正：在inverse中，需要使用y的固定部分来计算变换参数
        # 这与Real NVP 2D实现一致：s = self._compute_scale(y), t = self._compute_translation(y)
        
        # 首先从y中提取固定部分
        masked_y = y * self.mask
        net_input = torch.cat([masked_y, conditioning], dim=-1)
        
        # 使用固定部分计算变换参数
        log_scale = self.scale_net(net_input)
        translation = self.translation_net(net_input)
        
        # Apply tanh to log_scale for stability and mask to only affect transformed dimensions
        log_scale = torch.tanh(log_scale) * (1 - self.mask)
        translation = translation * (1 - self.mask)
        
        # Apply inverse affine transformation: x = mask * y + (1 - mask) * ((y - translation) * exp(-log_scale))
        x = self.mask * y + (1 - self.mask) * ((y - translation) * torch.exp(-log_scale))
        
        # Compute log determinant of inverse Jacobian
        log_det = -log_scale.sum(dim=-1)  # Negative because it's the inverse
        
        return x, log_det

class MeanFlowActionHead(nn.Module):
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
        super().__init__()
        self.action_dim = action_dim
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

class EndToEndDiffusionActionHead(nn.Module):
    """端到端扩散动作头：不改动VLA，直接在head内进行扩散条件噪声预测与反演。"""
    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=7, num_diffusion_steps=50, **kwargs):
        super().__init__()
        self.action_dim = action_dim
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
        
    
        
     
        
        
    
    
    
