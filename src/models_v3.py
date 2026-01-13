"""
A-JEPA v3 and V-JEPA v3: Enhanced Architectures for Cognitive World Modeling

Key upgrades from v2:

A-JEPA v3 (Cognitive Architecture):
1. 8 slots (up from 4) - more abstract concepts
2. RelationalBlock - explicit slot-to-slot reasoning
3. PerSlotBottleneck - preserves object identity (no flattening)
4. SlotTemporalMemory - per-slot GRU for individual object tracking
5. 4-channel input: multi-scale edges (3) + motion (1)

V-JEPA v3 (Fair Baseline with shared improvements):
1. 4-channel input: RGB (3) + motion (1)
2. Same improved temporal model
3. No slots/relational reasoning (tests the aphantasia hypothesis)

Design Philosophy:
- A-JEPA v3 embodies aphantasia-like reasoning: abstract slots, relations, no pixels
- V-JEPA v3 is a strong baseline with engineering improvements but pixel-based
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict


# =============================================================================
# SLOT ATTENTION (Reused from v2, but with more slots)
# =============================================================================

class SlotAttention(nn.Module):
    """
    Slot Attention module for object-centric representation.
    Based on: Locatello et al., "Object-Centric Learning with Slot Attention"
    """
    
    def __init__(
        self, 
        num_slots: int = 8,  # Increased from 4
        slot_dim: int = 48,  # Increased from 32
        input_dim: int = 128,
        num_iters: int = 4,  # Increased from 3
        hidden_dim: int = 96,
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
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: (B, N, input_dim) - N spatial positions
            
        Returns:
            slots: (B, num_slots, slot_dim)
            attn: (B, num_slots, N)
        """
        B, N, _ = inputs.shape
        
        # Project inputs
        inputs = self.project_input(inputs)
        inputs = self.norm_input(inputs)
        
        # Initialize slots
        slots = self.slots_mu + self.slots_sigma * torch.randn(
            B, self.num_slots, self.slot_dim, device=inputs.device
        )
        
        # Iterative attention
        attn = None
        for _ in range(self.num_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            q = self.q(slots)
            k = self.k(inputs)
            v = self.v(inputs)
            
            scale = self.slot_dim ** -0.5
            attn = torch.softmax(
                torch.bmm(q, k.transpose(1, 2)) * scale,
                dim=-1
            )
            
            updates = torch.bmm(attn, v)
            
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(B, self.num_slots, self.slot_dim)
            
            slots = slots + self.mlp(self.norm_mlp(slots))
        
        return slots, attn


# =============================================================================
# NEW: RELATIONAL BLOCK (A-JEPA v3 only)
# =============================================================================

class RelationalBlock(nn.Module):
    """
    Computes pairwise relationships between slots.
    
    Aphantasia Insight: Physics reasoning is about relationships between objects,
    not their visual appearance. This block explicitly models slot interactions.
    
    Architecture:
    1. Multi-head self-attention (global context)
    2. Pairwise MLP (explicit relation computation)
    3. Residual connection
    """
    
    def __init__(self, slot_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention for global slot context
        self.self_attn = nn.MultiheadAttention(
            embed_dim=slot_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(slot_dim)
        
        # Pairwise relation MLP
        self.relation_mlp = nn.Sequential(
            nn.Linear(slot_dim * 2, slot_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        self.norm2 = nn.LayerNorm(slot_dim)
        
        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim, slot_dim),
        )
        self.norm3 = nn.LayerNorm(slot_dim)
        
    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slots: (B, K, D) - K slots
            
        Returns:
            slots: (B, K, D) - slots enriched with relational information
        """
        B, K, D = slots.shape
        
        # 1. Self-attention: each slot attends to all others
        attn_out, _ = self.self_attn(slots, slots, slots)
        slots = self.norm1(slots + attn_out)
        
        # 2. Pairwise relations
        # Expand to compute all pairs: (B, K, K, 2D)
        s_i = slots.unsqueeze(2).expand(-1, -1, K, -1)  # (B, K, K, D)
        s_j = slots.unsqueeze(1).expand(-1, K, -1, -1)  # (B, K, K, D)
        pairs = torch.cat([s_i, s_j], dim=-1)  # (B, K, K, 2D)
        
        # Compute relations
        relations = self.relation_mlp(pairs)  # (B, K, K, D)
        
        # Mask out self-relations and aggregate
        mask = 1.0 - torch.eye(K, device=slots.device)
        mask = mask.view(1, K, K, 1)
        relations = relations * mask
        slot_updates = relations.sum(dim=2) / (K - 1)  # Average over other slots
        
        slots = self.norm2(slots + slot_updates)
        
        # 3. Final projection
        slots = self.norm3(slots + self.proj(slots))
        
        return slots


# =============================================================================
# NEW: PER-SLOT BOTTLENECK (A-JEPA v3 only)
# =============================================================================

class PerSlotBottleneck(nn.Module):
    """
    Apply bottleneck compression to each slot independently.
    
    Aphantasia Insight: Objects remain distinct concepts, not merged into one.
    By keeping slots separate, we preserve object identity through the pipeline.
    
    Key difference from v2: 
    - v2: slots → flatten → single bottleneck → one vector
    - v3: slots → per-slot bottleneck → K separate compressed slots
    """
    
    def __init__(
        self,
        slot_dim: int,
        bottleneck_dim: int = 32,
        sparsity_lambda: float = 0.001,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.bottleneck_dim = bottleneck_dim
        self.sparsity_lambda = sparsity_lambda
        
        # Shared bottleneck (applied independently to each slot)
        self.encoder = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim, bottleneck_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim, slot_dim),
        )
        
        self.aux_loss = 0.0
        
    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            slots: (B, K, slot_dim)
            
        Returns:
            z: (B, K, bottleneck_dim) - compressed slots (still K of them!)
            slots_recon: (B, K, slot_dim) - for residual
        """
        B, K, D = slots.shape
        
        # Flatten for processing
        slots_flat = slots.view(B * K, D)
        
        # Encode
        z = self.encoder(slots_flat)  # (B*K, bottleneck_dim)
        
        # L1 sparsity penalty
        self.aux_loss = self.sparsity_lambda * torch.abs(z).mean()
        
        # Decode for residual
        slots_recon = self.decoder(z)  # (B*K, slot_dim)
        
        # Reshape back
        z = z.view(B, K, self.bottleneck_dim)
        slots_recon = slots_recon.view(B, K, D)
        
        return z, slots_recon
    
    def get_aux_loss(self) -> torch.Tensor:
        return self.aux_loss


# =============================================================================
# NEW: SLOT TEMPORAL MEMORY (A-JEPA v3 only)
# =============================================================================

class SlotTemporalMemory(nn.Module):
    """
    Maintain temporal memory for each slot independently.
    
    Aphantasia Insight: When tracking multiple objects mentally, we maintain
    SEPARATE temporal histories. "Ball A was moving left, Ball B was moving right."
    A shared GRU can't do this - we need per-slot memory.
    
    Architecture:
    - Shared GRU weights (parameter efficient)
    - Applied independently to each slot's temporal sequence
    """
    
    def __init__(self, slot_dim: int, memory_dim: int = 64):
        super().__init__()
        self.slot_dim = slot_dim
        self.memory_dim = memory_dim
        
        # Shared GRU (applied to each slot independently)
        self.gru = nn.GRU(
            input_size=slot_dim,
            hidden_size=memory_dim,
            num_layers=1,
            batch_first=True,
        )
        
        # Project back to slot dimension
        self.project = nn.Linear(memory_dim, slot_dim)
        
    def forward(
        self, 
        slots_seq: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            slots_seq: (B, T, K, D) - K slots over T timesteps
            hidden: Optional initial hidden state
            
        Returns:
            output: (B, T, K, D) - temporal slots with per-object memory
            hidden: (B, K, memory_dim) - final hidden state
        """
        B, T, K, D = slots_seq.shape
        
        # Reshape: treat each slot as independent sequence
        # (B, T, K, D) -> (B*K, T, D)
        slots_flat = slots_seq.permute(0, 2, 1, 3).reshape(B * K, T, D)
        
        # Apply GRU
        gru_out, h_n = self.gru(slots_flat)  # (B*K, T, memory_dim)
        
        # Project back
        output = self.project(gru_out)  # (B*K, T, D)
        
        # Residual connection
        output = output + slots_flat
        
        # Reshape back: (B*K, T, D) -> (B, T, K, D)
        output = output.reshape(B, K, T, D).permute(0, 2, 1, 3)
        h_n = h_n.view(B, K, self.memory_dim)
        
        return output, h_n


# =============================================================================
# NEW: SLOT-AWARE MULTI-STEP PREDICTOR (A-JEPA v3)
# =============================================================================

class SlotPredictor(nn.Module):
    """
    Predict future slot states while maintaining slot identity.
    
    Unlike v2 which predicted a single future vector, this predicts
    K future slot states, preserving object-level predictions.
    """
    
    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        num_steps: int = 5,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_steps = num_steps
        
        # Cross-slot context (knows about all slots)
        self.slot_mixer = nn.Sequential(
            nn.Linear(num_slots * slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Per-slot predictor (conditioned on global context)
        self.step_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(slot_dim + hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, slot_dim),
            )
            for _ in range(num_steps)
        ])
        
        # Uncertainty heads
        self.uncertainty_heads = nn.ModuleList([
            nn.Linear(slot_dim, slot_dim)
            for _ in range(num_steps)
        ])
        
    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            slots: (B, K, D) - current slot states
            
        Returns:
            dict with 'predictions' (B, num_steps, K, D) and 'log_vars'
        """
        B, K, D = slots.shape
        
        # Global context from all slots
        slots_flat = slots.view(B, -1)  # (B, K*D)
        context = self.slot_mixer(slots_flat)  # (B, hidden_dim)
        
        predictions = []
        log_vars = []
        
        current_slots = slots
        for step in range(self.num_steps):
            # Condition each slot on global context
            context_expanded = context.unsqueeze(1).expand(-1, K, -1)  # (B, K, hidden_dim)
            slot_input = torch.cat([current_slots, context_expanded], dim=-1)  # (B, K, D+hidden)
            
            # Flatten, predict, unflatten
            slot_input_flat = slot_input.view(B * K, -1)
            pred_flat = self.step_predictors[step](slot_input_flat)  # (B*K, D)
            pred = pred_flat.view(B, K, D)
            
            # Normalize
            pred = F.normalize(pred, dim=-1)
            predictions.append(pred)
            
            # Uncertainty
            log_var = self.uncertainty_heads[step](pred.view(B * K, D)).view(B, K, D)
            log_vars.append(log_var)
            
            # Autoregressive: next step uses predicted slots
            current_slots = pred
        
        return {
            'predictions': torch.stack(predictions, dim=1),  # (B, num_steps, K, D)
            'log_vars': torch.stack(log_vars, dim=1),
        }


# =============================================================================
# A-JEPA V3 ENCODER
# =============================================================================

class AJEPAv3Encoder(nn.Module):
    """
    A-JEPA v3 Encoder: Full cognitive architecture.
    
    Pipeline:
    1. Multi-scale edge + motion input (4 channels)
    2. Conv encoder -> spatial tokens
    3. Slot Attention (8 slots)
    4. Relational Block (slot-to-slot reasoning)
    5. Per-Slot Bottleneck (preserves object identity)
    
    Output: (B, K, bottleneck_dim) where K=8 slots
    """
    
    def __init__(
        self,
        in_channels: int = 4,  # 3 edge scales + 1 motion
        img_size: int = 32,
        num_slots: int = 8,
        slot_dim: int = 48,
        bottleneck_dim: int = 32,
        sparsity_lambda: float = 0.001,
    ):
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.bottleneck_dim = bottleneck_dim
        
        # Conv encoder: 32x32 -> 4x4
        self.conv_channels = 128
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, self.conv_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True),
        )
        
        self.conv_spatial = img_size // 8  # 4
        self.num_tokens = self.conv_spatial * self.conv_spatial  # 16
        
        # Slot Attention
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=self.conv_channels,
            num_iters=4,
            hidden_dim=96,
        )
        
        # Relational reasoning between slots
        self.relational = RelationalBlock(slot_dim=slot_dim, num_heads=4)
        
        # Per-slot bottleneck
        self.bottleneck = PerSlotBottleneck(
            slot_dim=slot_dim,
            bottleneck_dim=bottleneck_dim,
            sparsity_lambda=sparsity_lambda,
        )
        
    def encode_frame(
        self, 
        x: torch.Tensor, 
        return_attn: bool = False,
    ) -> torch.Tensor:
        """
        Encode single frame to slots.
        
        Args:
            x: (B, C, H, W) - 4-channel input
            
        Returns:
            z: (B, K, bottleneck_dim) - K compressed slots
        """
        B = x.shape[0]
        
        # Conv features -> spatial tokens
        h = self.conv(x)  # (B, C, 4, 4)
        h = h.view(B, self.conv_channels, -1).transpose(1, 2)  # (B, 16, C)
        
        # Slot Attention
        slots, attn = self.slot_attention(h)  # (B, K, slot_dim)
        
        # Relational reasoning
        slots = self.relational(slots)  # (B, K, slot_dim)
        
        # Per-slot bottleneck
        z, _ = self.bottleneck(slots)  # (B, K, bottleneck_dim)
        
        # Normalize each slot
        z = F.normalize(z, dim=-1)
        
        if return_attn:
            return z, attn
        return z
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode single image."""
        return self.encode_frame(x)
    
    def get_aux_loss(self) -> torch.Tensor:
        return self.bottleneck.get_aux_loss()


# =============================================================================
# A-JEPA V3 FULL MODEL
# =============================================================================

class AJEPAv3(nn.Module):
    """
    Complete A-JEPA v3: Encoder + Temporal Memory + Slot Predictor.
    
    Key differences from v2:
    - Slots preserved throughout (never flattened to single vector)
    - Relational reasoning between slots
    - Per-slot temporal memory
    - Slot-aware multi-step prediction
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        img_size: int = 32,
        num_slots: int = 8,
        slot_dim: int = 48,
        bottleneck_dim: int = 32,
        memory_dim: int = 64,
        num_pred_steps: int = 5,
        sparsity_lambda: float = 0.001,
    ):
        super().__init__()
        
        self.num_slots = num_slots
        self.bottleneck_dim = bottleneck_dim
        
        # Encoder
        self.encoder = AJEPAv3Encoder(
            in_channels=in_channels,
            img_size=img_size,
            num_slots=num_slots,
            slot_dim=slot_dim,
            bottleneck_dim=bottleneck_dim,
            sparsity_lambda=sparsity_lambda,
        )
        
        # Per-slot temporal memory
        self.temporal = SlotTemporalMemory(
            slot_dim=bottleneck_dim,
            memory_dim=memory_dim,
        )
        
        # Slot-aware predictor
        self.predictor = SlotPredictor(
            num_slots=num_slots,
            slot_dim=bottleneck_dim,
            num_steps=num_pred_steps,
            hidden_dim=128,
        )
        
    def encode_video(
        self, 
        video: torch.Tensor,
        return_all: bool = False,
    ) -> torch.Tensor:
        """
        Encode video to slot representations.
        
        Args:
            video: (B, T, C, H, W)
            
        Returns:
            If return_all: (B, T, K, D) - all temporal slots
            Else: (B, K, D) - mean-pooled slots
        """
        B, T, C, H, W = video.shape
        
        # Encode each frame
        frames = video.reshape(B * T, C, H, W)
        slots = self.encoder.encode_frame(frames)  # (B*T, K, D)
        slots = slots.view(B, T, self.num_slots, self.bottleneck_dim)
        
        # Temporal memory
        slots, _ = self.temporal(slots)  # (B, T, K, D)
        
        if return_all:
            return slots
        else:
            return slots.mean(dim=1)  # (B, K, D)
    
    def forward(
        self,
        context_video: torch.Tensor,
        target_video: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            context_video: (B, T_ctx, C, H, W)
            target_video: (B, T_tgt, C, H, W)
        """
        # Encode context (mean-pooled)
        z_context = self.encode_video(context_video)  # (B, K, D)
        
        # Encode targets (all frames)
        z_targets = self.encode_video(target_video, return_all=True)  # (B, T, K, D)
        
        # Predict future slots
        pred_output = self.predictor(z_context)  # (B, num_steps, K, D)
        
        # Compute loss
        num_steps = min(pred_output['predictions'].shape[1], z_targets.shape[1])
        pred = pred_output['predictions'][:, :num_steps]  # (B, S, K, D)
        target = z_targets[:, :num_steps]  # (B, S, K, D)
        
        # Cosine similarity loss (per slot, per step)
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target.detach(), dim=-1)
        pred_loss = -torch.mean(torch.sum(pred_norm * target_norm, dim=-1))
        
        # Auxiliary loss
        aux_loss = self.encoder.get_aux_loss()
        
        return {
            'loss': pred_loss + aux_loss,
            'pred_loss': pred_loss,
            'aux_loss': aux_loss,
            'predictions': pred,
            'targets': target,
        }
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode single frame to slots, then flatten for linear probe."""
        slots = self.encoder.encode_frame(x)  # (B, K, D)
        return slots.view(x.shape[0], -1)  # (B, K*D)


# =============================================================================
# V-JEPA V3 ENCODER (Fair baseline with shared improvements)
# =============================================================================

class VJEPAv3Encoder(nn.Module):
    """
    V-JEPA v3 Encoder: RGB + motion, no slots.
    
    Shared improvements:
    - Motion channel input (4 channels: RGB + motion)
    - Same temporal architecture as A-JEPA v3
    
    NOT included (A-JEPA specific):
    - Slot Attention
    - Relational Block
    - Per-slot processing
    """
    
    def __init__(
        self,
        in_channels: int = 4,  # RGB + motion
        img_size: int = 32,
        emb_dim: int = 256,
        conv_channels: tuple = (64, 128, 256, 256),
    ):
        super().__init__()
        self.emb_dim = emb_dim
        
        c1, c2, c3, c4 = conv_channels
        
        # Conv encoder
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, 3, stride=2, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        
        conv_spatial = img_size // 16  # 2
        conv_dim = c4 * conv_spatial * conv_spatial
        
        self.fc = nn.Linear(conv_dim, emb_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - 4-channel input
            
        Returns:
            z: (B, emb_dim)
        """
        h = self.conv(x)
        h = h.view(h.shape[0], -1)
        z = self.fc(h)
        z = F.normalize(z, dim=-1)
        return z


# =============================================================================
# V-JEPA V3 FULL MODEL
# =============================================================================

class VJEPAv3(nn.Module):
    """
    Complete V-JEPA v3: Fair baseline with shared improvements.
    
    - Motion channel (same as A-JEPA v3)
    - Dense embedding (no slots)
    - Standard temporal GRU
    - Standard multi-step predictor
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        img_size: int = 32,
        emb_dim: int = 256,
        memory_dim: int = 128,
        num_pred_steps: int = 5,
    ):
        super().__init__()
        
        self.emb_dim = emb_dim
        
        # Encoder
        self.encoder = VJEPAv3Encoder(
            in_channels=in_channels,
            img_size=img_size,
            emb_dim=emb_dim,
        )
        
        # Temporal GRU
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=memory_dim,
            num_layers=1,
            batch_first=True,
        )
        self.gru_proj = nn.Linear(memory_dim, emb_dim)
        
        # Multi-step predictor
        self.num_pred_steps = num_pred_steps
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, emb_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(emb_dim * 2, emb_dim),
            )
            for _ in range(num_pred_steps)
        ])
        
    def encode_video(
        self, 
        video: torch.Tensor,
        return_all: bool = False,
    ) -> torch.Tensor:
        """
        Encode video.
        
        Args:
            video: (B, T, C, H, W)
            
        Returns:
            If return_all: (B, T, D)
            Else: (B, D)
        """
        B, T, C, H, W = video.shape
        
        # Encode each frame
        frames = video.reshape(B * T, C, H, W)
        z = self.encoder(frames)  # (B*T, D)
        z = z.view(B, T, self.emb_dim)
        
        # Temporal GRU
        gru_out, _ = self.gru(z)
        z = z + self.gru_proj(gru_out)
        
        if return_all:
            return z
        else:
            return z.mean(dim=1)
    
    def forward(
        self,
        context_video: torch.Tensor,
        target_video: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            context_video: (B, T_ctx, C, H, W)
            target_video: (B, T_tgt, C, H, W)
        """
        # Encode
        z_context = self.encode_video(context_video)  # (B, D)
        z_targets = self.encode_video(target_video, return_all=True)  # (B, T, D)
        
        # Predict
        predictions = []
        current = z_context
        for step in range(self.num_pred_steps):
            pred = self.predictors[step](current)
            pred = F.normalize(pred, dim=-1)
            predictions.append(pred)
            current = pred
        
        predictions = torch.stack(predictions, dim=1)  # (B, S, D)
        
        # Loss
        num_steps = min(self.num_pred_steps, z_targets.shape[1])
        pred = predictions[:, :num_steps]
        target = z_targets[:, :num_steps]
        
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target.detach(), dim=-1)
        pred_loss = -torch.mean(torch.sum(pred_norm * target_norm, dim=-1))
        
        return {
            'loss': pred_loss,
            'pred_loss': pred_loss,
            'aux_loss': torch.tensor(0.0),
            'predictions': pred,
            'targets': target,
        }
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode single frame."""
        return self.encoder(x)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def get_ajepa_v3(config: str = 'default') -> AJEPAv3:
    """Create A-JEPA v3 model."""
    configs = {
        'default': {
            'in_channels': 4,
            'num_slots': 8,
            'slot_dim': 48,
            'bottleneck_dim': 32,
            'memory_dim': 64,
            'num_pred_steps': 5,
            'sparsity_lambda': 0.001,
        },
        'small': {
            'in_channels': 4,
            'num_slots': 4,
            'slot_dim': 32,
            'bottleneck_dim': 24,
            'memory_dim': 48,
            'num_pred_steps': 5,
            'sparsity_lambda': 0.001,
        },
        'large': {
            'in_channels': 4,
            'num_slots': 12,
            'slot_dim': 64,
            'bottleneck_dim': 48,
            'memory_dim': 96,
            'num_pred_steps': 5,
            'sparsity_lambda': 0.0005,
        },
    }
    return AJEPAv3(**configs[config])


def get_vjepa_v3(config: str = 'default') -> VJEPAv3:
    """Create V-JEPA v3 model."""
    configs = {
        'default': {
            'in_channels': 4,
            'emb_dim': 256,
            'memory_dim': 128,
            'num_pred_steps': 5,
        },
        'small': {
            'in_channels': 4,
            'emb_dim': 128,
            'memory_dim': 64,
            'num_pred_steps': 5,
        },
    }
    return VJEPAv3(**configs[config])


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("A-JEPA v3 and V-JEPA v3 Architecture Test")
    print("=" * 60)
    
    # Test A-JEPA v3
    print("\n[A-JEPA v3]")
    model_a = get_ajepa_v3('default')
    params_a = sum(p.numel() for p in model_a.parameters())
    print(f"  Parameters: {params_a:,}")
    
    # Test forward pass
    B, T, C, H, W = 2, 10, 4, 32, 32
    context = torch.randn(B, 5, C, H, W)
    target = torch.randn(B, 5, C, H, W)
    
    output = model_a(context, target)
    print(f"  Loss: {output['loss'].item():.4f}")
    print(f"  Predictions shape: {output['predictions'].shape}")
    print(f"  (Expected: B, num_steps, K, D = {B}, 5, 8, 32)")
    
    # Test encode for linear probe
    x = torch.randn(B, C, H, W)
    z = model_a.encode(x)
    print(f"  Encode output: {z.shape} (for linear probe)")
    
    # Test V-JEPA v3
    print("\n[V-JEPA v3]")
    model_v = get_vjepa_v3('default')
    params_v = sum(p.numel() for p in model_v.parameters())
    print(f"  Parameters: {params_v:,}")
    
    output = model_v(context, target)
    print(f"  Loss: {output['loss'].item():.4f}")
    print(f"  Predictions shape: {output['predictions'].shape}")
    
    # Component tests
    print("\n[Component Tests]")
    
    # RelationalBlock
    slots = torch.randn(B, 8, 48)
    rel = RelationalBlock(slot_dim=48)
    out = rel(slots)
    print(f"  RelationalBlock: {slots.shape} -> {out.shape}")
    
    # PerSlotBottleneck
    bottleneck = PerSlotBottleneck(slot_dim=48, bottleneck_dim=32)
    z, recon = bottleneck(slots)
    print(f"  PerSlotBottleneck: {slots.shape} -> {z.shape}")
    
    # SlotTemporalMemory
    slots_seq = torch.randn(B, T, 8, 32)
    temporal = SlotTemporalMemory(slot_dim=32)
    out, h = temporal(slots_seq)
    print(f"  SlotTemporalMemory: {slots_seq.shape} -> {out.shape}")
    
    print("\n" + "=" * 60)
    print(f"A-JEPA v3: {params_a:,} params")
    print(f"V-JEPA v3: {params_v:,} params")
    print(f"Ratio: V/A = {params_v/params_a:.1f}x")
    print("=" * 60)

