"""Contextual Position Encoding (CoPE) for TinyPLM.

CoPE computes positions based on content rather than absolute indices.
Each token learns a "gate" that determines how much it counts toward position.
Positions are cumulative sums of gates, making them content-dependent.

This is particularly interesting for proteins because:
- Conserved domains can anchor positions regardless of absolute location
- Insertions/deletions (indels) are handled more gracefully
- Secondary structure periodicity may be captured naturally

Reference: "CoPE: Contextual Position Encoding" (2024)

NOTE: This is an EXPERIMENTAL implementation for future exploration.
      The current training uses YaRN/RoPE (see rope.py).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualPositionEncoding(nn.Module):
    """Contextual Position Encoding (CoPE).

    Unlike RoPE which uses absolute positions, CoPE computes positions
    based on token content. Each token produces a gate value in [0, 1],
    and positions are cumulative sums of these gates.

    For proteins, this could allow:
    - Conserved residues (e.g., catalytic sites) to "anchor" positions
    - Loop regions to have less positional weight
    - Domain-relative positioning
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        max_seq_len: int = 2048,
        gate_type: str = "per_head",  # "per_head" or "shared"
    ):
        """Initialize CoPE.

        Args:
            hidden_dim: Model hidden dimension.
            num_heads: Number of attention heads.
            max_seq_len: Maximum sequence length (for position embedding table).
            gate_type: Whether gates are per-head or shared across heads.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_seq_len = max_seq_len
        self.gate_type = gate_type

        # Gate projection: hidden_states -> gate values
        # Each token produces a scalar gate in [0, 1] per head (or shared)
        if gate_type == "per_head":
            self.gate_proj = nn.Linear(hidden_dim, num_heads, bias=False)
        else:
            self.gate_proj = nn.Linear(hidden_dim, 1, bias=False)

        # Position embedding table (like a learned positional encoding)
        # Indexed by the computed contextual position
        # Using interpolation for non-integer positions
        self.pos_embed = nn.Embedding(max_seq_len, self.head_dim)

        # Temperature for attention (similar to YaRN's mscale)
        self.register_buffer(
            "attn_scale",
            torch.tensor(1.0 / math.sqrt(self.head_dim)),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Initialize gate projection to produce ~0.5 initially
        # This gives positions similar to absolute positions at init
        nn.init.zeros_(self.gate_proj.weight)

        # Initialize position embeddings with sinusoidal pattern
        # Helps with initial training stability
        position = torch.arange(self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.head_dim, 2) *
            (-math.log(10000.0) / self.head_dim)
        )
        pe = torch.zeros(self.max_seq_len, self.head_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pos_embed.weight.data.copy_(pe)

    def compute_gates(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute gate values for each token.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len] or None

        Returns:
            gates: [batch, seq_len, num_heads] or [batch, seq_len, 1]
        """
        # Project to gate logits
        gate_logits = self.gate_proj(hidden_states)  # [batch, seq, heads/1]

        # Apply sigmoid to get values in [0, 1]
        gates = torch.sigmoid(gate_logits)

        # Mask padded positions (they shouldn't contribute to position)
        if attention_mask is not None:
            gates = gates * attention_mask.unsqueeze(-1)

        return gates

    def compute_positions(
        self,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contextual positions from gate values.

        Args:
            gates: [batch, seq_len, num_heads] or [batch, seq_len, 1]

        Returns:
            positions: [batch, seq_len, num_heads] or [batch, seq_len, 1]
                      Contextual position for each token
        """
        # Cumulative sum gives position
        # Each token's position = sum of gates of all previous tokens
        positions = torch.cumsum(gates, dim=1)

        return positions

    def get_position_embeddings(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Get position embeddings via interpolation.

        Since positions are continuous (not integers), we interpolate
        between adjacent position embeddings.

        Args:
            positions: [batch, seq_len, num_heads]

        Returns:
            pos_embeds: [batch, seq_len, num_heads, head_dim]
        """
        batch_size, seq_len, num_heads = positions.shape

        # Clamp positions to valid range
        positions = positions.clamp(0, self.max_seq_len - 1.001)

        # Get floor and ceil positions for interpolation
        pos_floor = positions.floor().long()
        pos_ceil = (pos_floor + 1).clamp(max=self.max_seq_len - 1)

        # Interpolation weights
        weight_ceil = positions - pos_floor.float()
        weight_floor = 1.0 - weight_ceil

        # Get embeddings for floor and ceil positions
        # pos_embed: [max_seq_len, head_dim]
        embed_floor = self.pos_embed(pos_floor)  # [batch, seq, heads, head_dim]
        embed_ceil = self.pos_embed(pos_ceil)

        # Interpolate
        pos_embeds = (
            weight_floor.unsqueeze(-1) * embed_floor +
            weight_ceil.unsqueeze(-1) * embed_ceil
        )

        return pos_embeds

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply CoPE to queries and keys.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_heads, seq_len, head_dim]
            hidden_states: Input hidden states [batch, seq_len, hidden_dim]
            attention_mask: Optional mask [batch, seq_len]

        Returns:
            q_rope: Position-enhanced queries
            k_rope: Position-enhanced keys
            positions: Computed contextual positions [batch, seq_len, num_heads]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # 1. Compute gates from hidden states
        gates = self.compute_gates(hidden_states, attention_mask)

        # Expand gates to per-head if shared
        if self.gate_type != "per_head":
            gates = gates.expand(-1, -1, num_heads)

        # 2. Compute contextual positions
        positions = self.compute_positions(gates)  # [batch, seq, heads]

        # 3. Get position embeddings via interpolation
        pos_embeds = self.get_position_embeddings(positions)
        # pos_embeds: [batch, seq, heads, head_dim]

        # 4. Transpose for attention format: [batch, heads, seq, head_dim]
        pos_embeds = pos_embeds.transpose(1, 2)

        # 5. Add position information to Q and K
        # Option A: Additive (like absolute position embeddings)
        q_pos = q + pos_embeds
        k_pos = k + pos_embeds

        return q_pos, k_pos, positions


class CoPEWithRoPE(nn.Module):
    """Hybrid: CoPE gates + RoPE-style rotary embeddings.

    This combines the best of both worlds:
    - CoPE's content-aware position computation
    - RoPE's rotary embedding mechanism (better for relative positions)

    The contextual position is used to generate RoPE frequencies,
    rather than the absolute token index.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Gate projection for contextual positions
        self.gate_proj = nn.Linear(hidden_dim, num_heads, bias=False)

        # Precompute inverse frequencies for RoPE
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._init_weights()

    def _init_weights(self):
        # Initialize gates to ~1.0 so initial positions ≈ absolute positions
        nn.init.zeros_(self.gate_proj.weight)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply CoPE-RoPE hybrid.

        Args:
            q: [batch, num_heads, seq_len, head_dim]
            k: [batch, num_heads, seq_len, head_dim]
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: Optional [batch, seq_len]

        Returns:
            q_rot, k_rot: Rotated queries and keys
            positions: Computed contextual positions
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # 1. Compute contextual positions
        gate_logits = self.gate_proj(hidden_states)  # [batch, seq, heads]
        gates = torch.sigmoid(gate_logits)

        if attention_mask is not None:
            gates = gates * attention_mask.unsqueeze(-1)

        positions = torch.cumsum(gates, dim=1)  # [batch, seq, heads]

        # 2. Compute per-position frequencies
        # positions: [batch, seq, heads] -> [batch, seq, heads, 1]
        # inv_freq: [head_dim/2] -> [1, 1, 1, head_dim/2]
        freqs = positions.unsqueeze(-1) * self.inv_freq.view(1, 1, 1, -1)
        # freqs: [batch, seq, heads, head_dim/2]

        # 3. Compute cos/sin
        emb = torch.cat([freqs, freqs], dim=-1)  # [batch, seq, heads, head_dim]
        cos = emb.cos()
        sin = emb.sin()

        # 4. Transpose to attention format: [batch, heads, seq, head_dim]
        cos = cos.transpose(1, 2)
        sin = sin.transpose(1, 2)

        # 5. Apply rotation (same as standard RoPE)
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat([-x2, x1], dim=-1)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)

        return q_rot, k_rot, positions


# =============================================================================
# Example integration with attention (for reference)
# =============================================================================

def example_cope_attention():
    """Example of how CoPE would integrate with the attention mechanism.

    This is NOT meant to be used directly - it's documentation showing
    how to modify MultiHeadAttention to use CoPE.
    """

    # In attention.py, the forward method would change from:
    #
    # def forward(self, hidden_states, attention_mask=None):
    #     q = self.q_proj(hidden_states)
    #     k = self.k_proj(hidden_states)
    #     v = self.v_proj(hidden_states)
    #
    #     # Apply RoPE
    #     cos, sin = self.rope(seq_len)
    #     q, k = apply_rope(q, k, cos, sin)
    #
    #     # Attention...
    #
    # To using CoPE:
    #
    # def forward(self, hidden_states, attention_mask=None):
    #     q = self.q_proj(hidden_states)
    #     k = self.k_proj(hidden_states)
    #     v = self.v_proj(hidden_states)
    #
    #     # Apply CoPE (needs hidden_states for gate computation)
    #     q, k, positions = self.cope(q, k, hidden_states, attention_mask)
    #
    #     # Attention...
    #     # Optionally return positions for visualization

    pass


# =============================================================================
# Protein-specific CoPE variants (ideas for future exploration)
# =============================================================================

class ProteinAwareCoPE(nn.Module):
    """CoPE variant with protein-specific inductive biases.

    Ideas for future exploration:

    1. Amino acid type-specific gates:
       - Hydrophobic residues might get different gate behavior than charged
       - Proline (helix breaker) might reset positions
       - Cysteines involved in disulfide bonds might anchor positions

    2. Secondary structure-aware gates:
       - Could be trained with structure supervision
       - Alpha helices have 3.6 residue periodicity
       - Beta strands have 2-residue periodicity

    3. Domain boundary detection:
       - Learn to identify domain boundaries
       - Reset position counter at boundaries
       - Enable domain-relative positioning

    This is speculative and would need experimental validation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        vocab_size: int = 25,
        head_dim: int = 64,
    ):
        super().__init__()

        # Per-residue type gate bias
        # Each amino acid type gets a learnable gate prior
        self.aa_gate_bias = nn.Embedding(vocab_size, num_heads)

        # Context-dependent gate (as in standard CoPE)
        self.gate_proj = nn.Linear(hidden_dim, num_heads, bias=False)

        # Domain boundary detector (optional auxiliary output)
        self.boundary_head = nn.Linear(hidden_dim, 1)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute protein-aware contextual positions.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            input_ids: [batch, seq_len] - amino acid token IDs
            attention_mask: [batch, seq_len]

        Returns:
            positions: [batch, seq_len, num_heads]
            boundary_logits: [batch, seq_len, 1] - optional boundary predictions
        """
        # Get amino acid-specific gate biases
        aa_bias = self.aa_gate_bias(input_ids)  # [batch, seq, heads]

        # Context-dependent gates
        context_gates = self.gate_proj(hidden_states)  # [batch, seq, heads]

        # Combine: residue type prior + context
        gate_logits = context_gates + aa_bias
        gates = torch.sigmoid(gate_logits)

        if attention_mask is not None:
            gates = gates * attention_mask.unsqueeze(-1)

        # Compute positions
        positions = torch.cumsum(gates, dim=1)

        # Optional: domain boundary prediction
        boundary_logits = self.boundary_head(hidden_states)

        return positions, boundary_logits
