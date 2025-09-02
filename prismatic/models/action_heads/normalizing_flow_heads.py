
import torch
import torch.nn as nn
import torch.nn.functional as F
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from .shared_components import MLPResNet, SinusoidalPositionalEncoding, BaseActionHead

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
