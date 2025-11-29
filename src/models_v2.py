"""
A-JEPA v2: Cognitively-Inspired World Model Architecture

Key innovations inspired by aphantasia and cognitive science:
1. Temporal Memory Core - abstract state memory (no pixels)
2. Slot/Object Factorization - multiple slots instead of one vector
3. Sparsity + Bottleneck - L1 penalty, aggressive dimensionality reduction
4. Multi-Timescale Latents - z_fast (per-step) + z_slow (global properties)
5. Uncertainty Head - explicit confidence prediction
6. Multi-Step Prediction - predict multiple future states

This represents a fundamentally different approach to world modeling:
- No reconstruction, no pixels
- Abstract relational reasoning
- Explicit uncertainty quantification
- Hierarchical temporal abstraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict


# =============================================================================
# 1. TEMPORAL MEMORY CORE
# =============================================================================

class TemporalMemoryCore(nn.Module):
    """
    GRU-based temporal memory that maintains abstract state over time.
    No pixels - only latent states flow through this module.
    
    Inspired by: working memory in aphantasia operates without imagery
    """
    
    def __init__(self, state_dim: int, memory_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.state_dim = state_dim
        self.memory_dim = memory_dim
        
        # GRU for temporal integration
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=memory_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        
        # Project memory back to state space
        self.memory_to_state = nn.Linear(memory_dim, state_dim)
        
    def forward(
        self, 
        states: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            states: (B, T, state_dim) - sequence of abstract states
            hidden: (num_layers, B, memory_dim) - optional initial hidden state
            
        Returns:
            output: (B, T, state_dim) - memory-augmented states
            hidden: (num_layers, B, memory_dim) - final hidden state
        """
        # Run through GRU
        gru_out, hidden = self.gru(states, hidden)  # (B, T, memory_dim)
        
        # Project back and add residual
        output = states + self.memory_to_state(gru_out)
        
        return output, hidden


# =============================================================================
# 2. SLOT/OBJECT FACTORIZATION
# =============================================================================

class SlotAttention(nn.Module):
    """
    Slot Attention module for object-centric representation.
    Instead of one big vector, factorize into K slots.
    
    Based on: Locatello et al., "Object-Centric Learning with Slot Attention"
    """
    
    def __init__(
        self, 
        num_slots: int = 4,
        slot_dim: int = 32,
        input_dim: int = 128,
        num_iters: int = 3,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iters = num_iters
        
        # Slot initialization (learnable)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.1)
        self.slots_sigma = nn.Parameter(torch.ones(1, 1, slot_dim) * 0.1)
        
        # Input projection
        self.project_input = nn.Linear(input_dim, slot_dim)
        
        # Attention mechanism
        self.norm_input = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        
        self.q = nn.Linear(slot_dim, slot_dim)
        self.k = nn.Linear(slot_dim, slot_dim)
        self.v = nn.Linear(slot_dim, slot_dim)
        
        # GRU for slot update
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        
        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim),
        )
        self.norm_mlp = nn.LayerNorm(slot_dim)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, N, input_dim) - N spatial/temporal positions
            
        Returns:
            slots: (B, num_slots, slot_dim) - factorized representation
        """
        B, N, _ = inputs.shape
        
        # Project inputs
        inputs = self.project_input(inputs)  # (B, N, slot_dim)
        inputs = self.norm_input(inputs)
        
        # Initialize slots with learned distribution
        slots = self.slots_mu + self.slots_sigma * torch.randn(
            B, self.num_slots, self.slot_dim, device=inputs.device
        )
        
        # Iterative attention
        for _ in range(self.num_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Attention
            q = self.q(slots)  # (B, K, D)
            k = self.k(inputs)  # (B, N, D)
            v = self.v(inputs)  # (B, N, D)
            
            # Scaled dot-product attention
            scale = self.slot_dim ** -0.5
            attn = torch.softmax(
                torch.bmm(q, k.transpose(1, 2)) * scale,
                dim=-1
            )  # (B, K, N)
            
            # Weighted sum
            updates = torch.bmm(attn, v)  # (B, K, D)
            
            # GRU update
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(B, self.num_slots, self.slot_dim)
            
            # MLP refinement with residual
            slots = slots + self.mlp(self.norm_mlp(slots))
        
        return slots


# =============================================================================
# 3. SPARSITY + BOTTLENECK CONSTRAINTS
# =============================================================================

class SparseBottleneck(nn.Module):
    """
    Aggressive dimensionality reduction with sparsity constraints.
    
    Options:
    - L1 penalty on activations (with configurable lambda)
    - Low-dimensional bottleneck (16-64 dims)
    - Optional VQ-style discrete codes
    - Optional β-VAE KL divergence
    
    FIX: Exposed sparsity_lambda parameter (default lowered from 0.01 to 0.002)
    to prevent over-regularization on small models/data.
    """
    
    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 32,
        use_vq: bool = False,
        num_codes: int = 512,
        use_vae: bool = False,
        beta: float = 4.0,
        sparsity_lambda: float = 0.002,  # SOFTENED: was 0.01
    ):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.use_vq = use_vq
        self.use_vae = use_vae
        self.beta = beta
        self.sparsity_lambda = sparsity_lambda  # Exposed for tuning
        
        # Encoder to bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, bottleneck_dim * (2 if use_vae else 1)),
        )
        
        # VQ codebook
        if use_vq:
            self.codebook = nn.Embedding(num_codes, bottleneck_dim)
            self.codebook.weight.data.uniform_(-1/num_codes, 1/num_codes)
        
        # Decoder from bottleneck
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, input_dim),
        )
        
        # Track auxiliary losses
        self.aux_loss = 0.0
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim)
            
        Returns:
            z: (B, bottleneck_dim) - compressed representation
            x_recon: (B, input_dim) - reconstructed (for residual)
        """
        h = self.encoder(x)
        
        if self.use_vae:
            # VAE: split into mean and log-variance
            mu, log_var = h.chunk(2, dim=-1)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            # KL divergence loss (β-VAE style)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            self.aux_loss = self.beta * kl_loss
            
        elif self.use_vq:
            # VQ: find nearest codebook entry
            z_e = h
            
            # Compute distances to codebook
            d = torch.sum(z_e ** 2, dim=1, keepdim=True) + \
                torch.sum(self.codebook.weight ** 2, dim=1) - \
                2 * torch.matmul(z_e, self.codebook.weight.t())
            
            # Get nearest codes
            indices = torch.argmin(d, dim=1)
            z_q = self.codebook(indices)
            
            # Straight-through estimator
            z = z_e + (z_q - z_e).detach()
            
            # Commitment loss
            commitment_loss = F.mse_loss(z_e, z_q.detach())
            codebook_loss = F.mse_loss(z_q, z_e.detach())
            self.aux_loss = commitment_loss + 0.25 * codebook_loss
            
        else:
            # Standard bottleneck with L1 sparsity (softened)
            z = h
            self.aux_loss = self.sparsity_lambda * torch.mean(torch.abs(z))
        
        # Decode
        x_recon = self.decoder(z)
        
        return z, x_recon
    
    def get_aux_loss(self) -> torch.Tensor:
        """Return auxiliary loss (KL, VQ commitment, or L1)."""
        return self.aux_loss


# =============================================================================
# 4. MULTI-TIMESCALE LATENTS
# =============================================================================

class MultiTimescaleLatent(nn.Module):
    """
    Two-level latent hierarchy:
    - z_fast: per-step encoding (positions, velocities)
    - z_slow: updated every N steps (mass, friction, environment)
    
    NEW: Temporal persistence loss encourages z_slow to be stable over time,
    capturing time-invariant properties like mass.
    
    Inspired by: different cognitive timescales, DMN, slow/fast thinking
    """
    
    def __init__(
        self,
        input_dim: int,
        fast_dim: int = 32,
        slow_dim: int = 16,
        slow_update_freq: int = 5,
        persistence_weight: float = 0.1,  # Weight for z_slow stability loss
    ):
        super().__init__()
        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        self.slow_update_freq = slow_update_freq
        self.persistence_weight = persistence_weight
        
        # Fast encoder (per-step)
        self.fast_encoder = nn.Sequential(
            nn.Linear(input_dim, fast_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(fast_dim * 2, fast_dim),
        )
        
        # Slow encoder (aggregates multiple steps)
        self.slow_encoder = nn.Sequential(
            nn.Linear(fast_dim * slow_update_freq, slow_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(slow_dim * 2, slow_dim),
        )
        
        # Slow state GRU (maintains global context)
        self.slow_gru = nn.GRUCell(slow_dim, slow_dim)
        
        # Combine fast and slow
        self.combiner = nn.Linear(fast_dim + slow_dim, fast_dim + slow_dim)
        
        # Track z_slow history for persistence loss
        self.slow_states_history = []
        self.persistence_loss = 0.0
        
    def forward(
        self, 
        x_seq: torch.Tensor,
        slow_hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_seq: (B, T, input_dim) - sequence of inputs
            slow_hidden: (B, slow_dim) - previous slow state
            
        Returns:
            z_combined: (B, T, fast_dim + slow_dim) - full latent
            z_fast: (B, T, fast_dim) - fast latent
            slow_hidden: (B, slow_dim) - updated slow state
        """
        B, T, _ = x_seq.shape
        
        # Initialize slow hidden if needed
        if slow_hidden is None:
            slow_hidden = torch.zeros(B, self.slow_dim, device=x_seq.device)
        
        # Encode fast (all timesteps)
        z_fast = self.fast_encoder(x_seq)  # (B, T, fast_dim)
        
        # Track slow states for persistence loss
        self.slow_states_history = [slow_hidden.clone()]
        
        # Update slow every N steps
        z_combined_list = []
        for t in range(T):
            if t > 0 and t % self.slow_update_freq == 0:
                # Aggregate last N fast states
                start_t = max(0, t - self.slow_update_freq)
                fast_window = z_fast[:, start_t:t, :].reshape(B, -1)
                
                # Pad if necessary
                if fast_window.shape[1] < self.fast_dim * self.slow_update_freq:
                    pad_size = self.fast_dim * self.slow_update_freq - fast_window.shape[1]
                    fast_window = F.pad(fast_window, (0, pad_size))
                
                # Update slow state
                slow_input = self.slow_encoder(fast_window)
                slow_hidden = self.slow_gru(slow_input, slow_hidden)
                
                # Track for persistence loss
                self.slow_states_history.append(slow_hidden.clone())
            
            # Combine fast and slow
            z_t = torch.cat([z_fast[:, t], slow_hidden], dim=-1)
            z_t = self.combiner(z_t)
            z_combined_list.append(z_t)
        
        z_combined = torch.stack(z_combined_list, dim=1)  # (B, T, fast+slow)
        
        # Compute persistence loss: z_slow should be stable over time
        self._compute_persistence_loss()
        
        return z_combined, z_fast, slow_hidden
    
    def _compute_persistence_loss(self):
        """
        Compute temporal persistence loss for z_slow.
        Encourages z_slow to remain stable, capturing time-invariant properties like mass.
        """
        if len(self.slow_states_history) < 2:
            self.persistence_loss = 0.0
            return
        
        # Penalize changes in z_slow between updates
        loss = 0.0
        for i in range(1, len(self.slow_states_history)):
            prev = self.slow_states_history[i-1]
            curr = self.slow_states_history[i]
            # MSE between consecutive slow states
            loss += F.mse_loss(curr, prev)
        
        self.persistence_loss = self.persistence_weight * loss / (len(self.slow_states_history) - 1)
    
    def get_persistence_loss(self) -> float:
        """Return the temporal persistence loss for z_slow."""
        return self.persistence_loss


# =============================================================================
# 5. UNCERTAINTY HEAD
# =============================================================================

class UncertaintyHead(nn.Module):
    """
    Predicts confidence / epistemic uncertainty for predictions.
    
    Outputs:
    - prediction: the actual prediction
    - log_variance: uncertainty estimate (higher = less confident)
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Mean prediction head
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        
        # Log-variance head (uncertainty)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim)
            
        Returns:
            mean: (B, output_dim) - prediction
            log_var: (B, output_dim) - uncertainty (log variance)
        """
        h = self.shared(x)
        mean = self.mean_head(h)
        log_var = self.logvar_head(h)
        
        # Clamp log_var for stability
        log_var = torch.clamp(log_var, min=-10, max=10)
        
        return mean, log_var
    
    def nll_loss(
        self, 
        pred_mean: torch.Tensor, 
        pred_logvar: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Negative log-likelihood loss with learned variance.
        Encourages model to be uncertain when wrong.
        """
        # Heteroscedastic loss
        precision = torch.exp(-pred_logvar)
        diff = (pred_mean - target) ** 2
        loss = 0.5 * (precision * diff + pred_logvar)
        return loss.mean()


# =============================================================================
# 6. MULTI-STEP PREDICTOR
# =============================================================================

class MultiStepPredictor(nn.Module):
    """
    Predicts multiple future latent states from context.
    
    Instead of just z_{t+1}, predict {z_{t+1}, z_{t+2}, ..., z_{t+k}}
    This forces learning of actual dynamics, not shortcuts.
    """
    
    def __init__(
        self, 
        latent_dim: int, 
        num_steps: int = 5,
        hidden_dim: int = 128,
        use_uncertainty: bool = True,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.latent_dim = latent_dim
        self.use_uncertainty = use_uncertainty
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Step embedding (tells predictor which step we're predicting)
        self.step_embed = nn.Embedding(num_steps, hidden_dim)
        
        # Prediction heads (one per step, or shared with uncertainty)
        if use_uncertainty:
            self.predictors = nn.ModuleList([
                UncertaintyHead(hidden_dim, latent_dim, hidden_dim // 2)
                for _ in range(num_steps)
            ])
        else:
            self.predictors = nn.ModuleList([
                nn.Linear(hidden_dim, latent_dim)
                for _ in range(num_steps)
            ])
        
    def forward(
        self, 
        context: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            context: (B, latent_dim) - context latent state
            
        Returns:
            dict with:
                'predictions': (B, num_steps, latent_dim)
                'log_vars': (B, num_steps, latent_dim) if use_uncertainty
        """
        B = context.shape[0]
        
        # Encode context
        h = self.context_encoder(context)  # (B, hidden_dim)
        
        predictions = []
        log_vars = []
        
        for k in range(self.num_steps):
            # Add step embedding
            step_emb = self.step_embed(
                torch.tensor([k], device=context.device)
            ).expand(B, -1)
            h_k = h + step_emb
            
            if self.use_uncertainty:
                pred, logvar = self.predictors[k](h_k)
                predictions.append(pred)
                log_vars.append(logvar)
            else:
                pred = self.predictors[k](h_k)
                predictions.append(pred)
        
        result = {
            'predictions': torch.stack(predictions, dim=1),  # (B, K, D)
        }
        
        if self.use_uncertainty:
            result['log_vars'] = torch.stack(log_vars, dim=1)
        
        return result


# =============================================================================
# FULL A-JEPA V2 ENCODER (FIXED)
# =============================================================================

class AJEPAv2Encoder(nn.Module):
    """
    Complete A-JEPA v2 encoder combining cognitive innovations.
    
    FIXES APPLIED:
    1. SlotAttention now operates on SPATIAL TOKENS (H*W positions) not fake slots
    2. Simplified temporal processing: use EITHER multi-timescale OR memory (not both)
    3. Softened sparsity via exposed lambda parameter
    
    Architecture:
    1. Conv encoder → spatial feature map (B, C, H, W)
    2. Reshape to tokens (B, N, C) where N = H*W
    3. SlotAttention discovers objects from spatial tokens
    4. Sparse bottleneck compresses slot representation
    5. Temporal memory for video (simplified: just GRU, no multi-timescale)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        img_size: int = 32,
        # Slot attention
        num_slots: int = 4,
        slot_dim: int = 32,
        # Bottleneck
        bottleneck_dim: int = 32,
        use_vq: bool = False,
        use_vae: bool = False,
        sparsity_lambda: float = 0.002,  # Exposed, softened
        # Output dimension
        output_dim: int = 48,  # Simplified: single output dim
        # Memory
        memory_dim: int = 64,
    ):
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.total_dim = output_dim  # Simplified output
        
        # Conv encoder - outputs spatial feature map
        # 32x32 -> 16x16 -> 8x8 -> 4x4
        self.conv_channels = 64  # Channel dim for conv output
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.conv_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True),
        )
        
        # Spatial size after conv: 32 -> 4
        self.conv_spatial = img_size // 8
        self.num_tokens = self.conv_spatial * self.conv_spatial  # 16 tokens
        
        # FIX: SlotAttention now takes spatial tokens as input
        # input_dim = conv_channels (64), not slot_dim
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=self.conv_channels,  # FIX: actual conv channel dim
            num_iters=3,
            hidden_dim=64,
        )
        
        # Sparse bottleneck (from all slots)
        slot_total_dim = num_slots * slot_dim
        self.bottleneck = SparseBottleneck(
            input_dim=slot_total_dim,
            bottleneck_dim=bottleneck_dim,
            use_vq=use_vq,
            use_vae=use_vae,
            sparsity_lambda=sparsity_lambda,  # Use softened default
        )
        
        # Project bottleneck to output dim
        self.to_output = nn.Linear(bottleneck_dim, output_dim)
        
        # SIMPLIFIED: Just temporal memory, no multi-timescale
        # (Remove redundant processing that was slowing learning)
        self.memory = TemporalMemoryCore(
            state_dim=output_dim,
            memory_dim=memory_dim,
        )
        
    def encode_frame(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode single frame using proper spatial SlotAttention.
        
        FIX: Conv features are now kept as spatial tokens (B, N, C)
        so SlotAttention can discover objects from the H*W grid.
        """
        B = x.shape[0]
        
        # Conv features -> spatial map
        h = self.conv(x)  # (B, C, H, W) where H=W=4
        
        # FIX: Reshape to spatial tokens for SlotAttention
        # (B, C, H, W) -> (B, H*W, C) = (B, 16, 64)
        h = h.view(B, self.conv_channels, -1)  # (B, C, N)
        h = h.transpose(1, 2)  # (B, N, C) - spatial tokens
        
        # SlotAttention discovers objects from spatial tokens
        slots = self.slot_attention(h)  # (B, num_slots, slot_dim)
        
        # Flatten slots and apply bottleneck
        slots_flat = slots.view(B, -1)  # (B, num_slots * slot_dim)
        z, _ = self.bottleneck(slots_flat)  # (B, bottleneck_dim)
        
        # Project to output dimension
        z = self.to_output(z)  # (B, output_dim)
        
        # L2 normalize
        z = F.normalize(z, dim=-1)
        
        return z
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode single image."""
        return self.encode_frame(x)
    
    def encode_video(
        self, 
        video: torch.Tensor,
        return_all: bool = False,
    ) -> torch.Tensor:
        """
        Encode video with temporal memory.
        
        SIMPLIFIED: Just per-frame encoding + GRU memory.
        No multi-timescale (was adding complexity without benefit at this scale).
        """
        B, T, C, H, W = video.shape
        
        # Encode each frame
        frames = video.reshape(B * T, C, H, W)
        z_frames = self.encode_frame(frames)  # (B*T, output_dim)
        z_frames = z_frames.view(B, T, -1)  # (B, T, output_dim)
        
        # Apply temporal memory
        z_mem, _ = self.memory(z_frames)  # (B, T, output_dim)
        
        if return_all:
            return z_mem
        else:
            return z_mem.mean(dim=1)  # (B, output_dim)
    
    def get_aux_loss(self) -> torch.Tensor:
        """Get auxiliary losses (just sparsity now, simplified)."""
        return self.bottleneck.get_aux_loss()


# =============================================================================
# FULL A-JEPA V2 MODEL (ENCODER + PREDICTOR) - SIMPLIFIED
# =============================================================================

class AJEPAv2(nn.Module):
    """
    Complete A-JEPA v2 system with encoder and multi-step predictor.
    
    SIMPLIFIED: Uses new fixed encoder with proper spatial SlotAttention.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        img_size: int = 32,
        num_slots: int = 4,
        slot_dim: int = 32,
        bottleneck_dim: int = 32,
        output_dim: int = 48,  # Simplified: single output dim
        num_pred_steps: int = 5,
        use_uncertainty: bool = True,
        sparsity_lambda: float = 0.002,  # Exposed for tuning
    ):
        super().__init__()
        
        self.total_dim = output_dim  # Simplified
        
        # Encoder (with fixes)
        self.encoder = AJEPAv2Encoder(
            in_channels=in_channels,
            img_size=img_size,
            num_slots=num_slots,
            slot_dim=slot_dim,
            bottleneck_dim=bottleneck_dim,
            output_dim=output_dim,
            sparsity_lambda=sparsity_lambda,
        )
        
        # Multi-step predictor
        self.predictor = MultiStepPredictor(
            latent_dim=self.total_dim,
            num_steps=num_pred_steps,
            hidden_dim=self.total_dim * 2,
            use_uncertainty=use_uncertainty,
        )
        
        self.use_uncertainty = use_uncertainty
        
    def forward(
        self, 
        context_video: torch.Tensor,
        target_video: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            context_video: (B, T_ctx, C, H, W) - context frames
            target_video: (B, T_tgt, C, H, W) - target frames to predict
            
        Returns:
            dict with predictions, targets, losses
        """
        # Encode context (mean pooled)
        z_context = self.encoder.encode_video(context_video)  # (B, total_dim)
        
        # Encode targets individually
        B, T_tgt, C, H, W = target_video.shape
        z_targets = self.encoder.encode_video(target_video, return_all=True)  # (B, T, total_dim)
        
        # Multi-step prediction
        pred_output = self.predictor(z_context)
        
        # Compute loss
        num_steps = min(pred_output['predictions'].shape[1], T_tgt)
        pred = pred_output['predictions'][:, :num_steps]  # (B, K, D)
        target = z_targets[:, :num_steps]  # (B, K, D)
        
        if self.use_uncertainty:
            logvar = pred_output['log_vars'][:, :num_steps]
            # Heteroscedastic loss
            precision = torch.exp(-logvar)
            diff = (pred - target.detach()) ** 2
            pred_loss = 0.5 * (precision * diff + logvar).mean()
        else:
            # Simple cosine similarity
            pred_norm = F.normalize(pred, dim=-1)
            target_norm = F.normalize(target.detach(), dim=-1)
            pred_loss = -torch.mean(torch.sum(pred_norm * target_norm, dim=-1))
        
        # Add auxiliary loss (sparsity/KL)
        aux_loss = self.encoder.get_aux_loss()
        
        return {
            'loss': pred_loss + aux_loss,
            'pred_loss': pred_loss,
            'aux_loss': aux_loss,
            'predictions': pred,
            'targets': target,
        }
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode single image."""
        return self.encoder(x)
    
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video to single vector."""
        return self.encoder.encode_video(video)


# =============================================================================
# V-JEPA V2 (FAIR COMPARISON - Same temporal features, no cognitive constraints)
# =============================================================================

class VJEPAv2Encoder(nn.Module):
    """
    V-JEPA v2: Same temporal features as A-JEPA v2, but:
    - RGB input (3 channels)
    - No slot attention (single vector)
    - No sparsity constraints
    - Higher dimensionality
    
    This ensures fair comparison - both have temporal memory + multi-step prediction.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 32,
        emb_dim: int = 128,
        memory_dim: int = 128,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.total_dim = emb_dim
        
        # Standard conv encoder (RGB)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Spatial size after conv (32 -> 2)
        conv_spatial = img_size // 16
        conv_dim = 256 * conv_spatial * conv_spatial
        
        # Project to embedding
        self.fc = nn.Linear(conv_dim, emb_dim)
        
        # Temporal memory (same as A-JEPA v2)
        self.memory = TemporalMemoryCore(
            state_dim=emb_dim,
            memory_dim=memory_dim,
        )
        
    def encode_frame(self, x: torch.Tensor) -> torch.Tensor:
        """Encode single frame."""
        B = x.shape[0]
        h = self.conv(x)
        h = h.view(B, -1)
        z = self.fc(h)
        z = F.normalize(z, dim=-1)
        return z
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode single image."""
        return self.encode_frame(x)
    
    def encode_video(
        self, 
        video: torch.Tensor,
        return_all: bool = False,
    ) -> torch.Tensor:
        """Encode video with temporal memory."""
        B, T, C, H, W = video.shape
        
        # Encode each frame
        frames = video.reshape(B * T, C, H, W)
        z_frames = self.encode_frame(frames)
        z_frames = z_frames.view(B, T, -1)
        
        # Apply temporal memory
        z_mem, _ = self.memory(z_frames)
        
        if return_all:
            return z_mem
        else:
            return z_mem.mean(dim=1)


class VJEPAv2(nn.Module):
    """
    V-JEPA v2: Fair comparison with same temporal features.
    - Temporal memory core ✓
    - Multi-step prediction ✓
    - RGB input (no edge preprocessing)
    - No slot attention, no sparsity
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 32,
        emb_dim: int = 128,
        memory_dim: int = 128,
        num_pred_steps: int = 5,
        use_uncertainty: bool = True,
    ):
        super().__init__()
        
        self.total_dim = emb_dim
        
        # Encoder
        self.encoder = VJEPAv2Encoder(
            in_channels=in_channels,
            img_size=img_size,
            emb_dim=emb_dim,
            memory_dim=memory_dim,
        )
        
        # Multi-step predictor (same as A-JEPA v2)
        self.predictor = MultiStepPredictor(
            latent_dim=emb_dim,
            num_steps=num_pred_steps,
            hidden_dim=emb_dim * 2,
            use_uncertainty=use_uncertainty,
        )
        
        self.use_uncertainty = use_uncertainty
        
    def forward(
        self, 
        context_video: torch.Tensor,
        target_video: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with multi-step prediction loss."""
        # Encode context
        z_context = self.encoder.encode_video(context_video)
        
        # Encode targets
        B, T_tgt, C, H, W = target_video.shape
        z_targets = self.encoder.encode_video(target_video, return_all=True)
        
        # Multi-step prediction
        pred_output = self.predictor(z_context)
        
        # Compute loss
        num_steps = min(pred_output['predictions'].shape[1], T_tgt)
        pred = pred_output['predictions'][:, :num_steps]
        target = z_targets[:, :num_steps]
        
        if self.use_uncertainty:
            logvar = pred_output['log_vars'][:, :num_steps]
            precision = torch.exp(-logvar)
            diff = (pred - target.detach()) ** 2
            pred_loss = 0.5 * (precision * diff + logvar).mean()
        else:
            pred_norm = F.normalize(pred, dim=-1)
            target_norm = F.normalize(target.detach(), dim=-1)
            pred_loss = -torch.mean(torch.sum(pred_norm * target_norm, dim=-1))
        
        return {
            'loss': pred_loss,
            'pred_loss': pred_loss,
            'aux_loss': 0.0,
            'predictions': pred,
            'targets': target,
        }
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode_video(video)


def get_vjepa_v2(
    in_channels: int = 3,
    img_size: int = 32,
    config: str = 'default'
) -> VJEPAv2:
    """Create V-JEPA v2 with matched architecture to A-JEPA v2."""
    
    configs = {
        'default': {
            'emb_dim': 128,
            'memory_dim': 128,
            'num_pred_steps': 5,
            'use_uncertainty': True,
        },
        'small': {
            'emb_dim': 64,
            'memory_dim': 64,
            'num_pred_steps': 3,
            'use_uncertainty': False,
        },
    }
    
    cfg = configs.get(config, configs['default'])
    
    return VJEPAv2(
        in_channels=in_channels,
        img_size=img_size,
        **cfg,
    )


# =============================================================================
# FACTORY FUNCTION (UPDATED FOR SIMPLIFIED ARCHITECTURE)
# =============================================================================

def get_ajepa_v2(
    in_channels: int = 1,
    img_size: int = 32,
    config: str = 'default'
) -> AJEPAv2:
    """
    Create A-JEPA v2 model with preset configurations.
    
    UPDATED: Uses simplified architecture with proper spatial SlotAttention.
    
    Configs:
    - 'default': balanced config (~300K params)
    - 'tiny': minimal for testing
    - 'medium': slightly larger for better capacity
    - 'large': scaled up for fair comparison with V-JEPA
    """
    
    configs = {
        'default': {
            'num_slots': 4,
            'slot_dim': 32,
            'bottleneck_dim': 32,
            'output_dim': 48,
            'num_pred_steps': 5,
            'use_uncertainty': True,
            'sparsity_lambda': 0.002,  # Softened
        },
        'tiny': {
            'num_slots': 2,
            'slot_dim': 16,
            'bottleneck_dim': 16,
            'output_dim': 24,
            'num_pred_steps': 3,
            'use_uncertainty': False,
            'sparsity_lambda': 0.001,
        },
        'medium': {
            'num_slots': 4,
            'slot_dim': 48,
            'bottleneck_dim': 48,
            'output_dim': 64,
            'num_pred_steps': 5,
            'use_uncertainty': True,
            'sparsity_lambda': 0.002,
        },
        'large': {
            'num_slots': 6,
            'slot_dim': 64,
            'bottleneck_dim': 64,
            'output_dim': 96,
            'num_pred_steps': 5,
            'use_uncertainty': True,
            'sparsity_lambda': 0.001,  # Even softer for larger model
        },
    }
    
    cfg = configs.get(config, configs['default'])
    
    return AJEPAv2(
        in_channels=in_channels,
        img_size=img_size,
        **cfg,
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    print("Testing A-JEPA v2 components...")
    
    # Test individual components
    B, T, C, H, W = 4, 10, 1, 32, 32
    
    print("\n1. TemporalMemoryCore:")
    memory = TemporalMemoryCore(state_dim=48, memory_dim=64)
    states = torch.randn(B, T, 48)
    out, hidden = memory(states)
    print(f"   Input: {states.shape} -> Output: {out.shape}, Hidden: {hidden.shape}")
    
    print("\n2. SlotAttention:")
    slot_attn = SlotAttention(num_slots=4, slot_dim=32, input_dim=64)
    inputs = torch.randn(B, 16, 64)  # 16 spatial positions
    slots = slot_attn(inputs)
    print(f"   Input: {inputs.shape} -> Slots: {slots.shape}")
    
    print("\n3. SparseBottleneck:")
    bottleneck = SparseBottleneck(input_dim=128, bottleneck_dim=32)
    x = torch.randn(B, 128)
    z, x_recon = bottleneck(x)
    print(f"   Input: {x.shape} -> Bottleneck: {z.shape}, Aux loss: {bottleneck.get_aux_loss():.4f}")
    
    print("\n4. MultiTimescaleLatent:")
    multiscale = MultiTimescaleLatent(input_dim=32, fast_dim=32, slow_dim=16)
    seq = torch.randn(B, T, 32)
    z_comb, z_fast, slow_h = multiscale(seq)
    print(f"   Input: {seq.shape} -> Combined: {z_comb.shape}, Fast: {z_fast.shape}, Slow: {slow_h.shape}")
    
    print("\n5. UncertaintyHead:")
    unc_head = UncertaintyHead(input_dim=48, output_dim=32)
    x = torch.randn(B, 48)
    mean, logvar = unc_head(x)
    print(f"   Input: {x.shape} -> Mean: {mean.shape}, LogVar: {logvar.shape}")
    
    print("\n6. MultiStepPredictor:")
    predictor = MultiStepPredictor(latent_dim=48, num_steps=5)
    ctx = torch.randn(B, 48)
    preds = predictor(ctx)
    print(f"   Context: {ctx.shape} -> Predictions: {preds['predictions'].shape}")
    
    print("\n7. Full AJEPAv2:")
    model = get_ajepa_v2(in_channels=1, img_size=32, config='default')
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {params:,}")
    
    # Test forward pass
    context = torch.randn(B, 5, C, H, W)
    target = torch.randn(B, 5, C, H, W)
    output = model(context, target)
    print(f"   Context: {context.shape}, Target: {target.shape}")
    print(f"   Loss: {output['loss']:.4f} (pred: {output['pred_loss']:.4f}, aux: {output['aux_loss']:.4f})")
    
    print("\n✅ All A-JEPA v2 components working!")

