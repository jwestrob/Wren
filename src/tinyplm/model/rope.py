"""Rotary Position Embeddings (RoPE) for TinyPLM.

Implements position encoding via rotation of query/key vectors.
Enables relative position awareness and extrapolation to longer sequences.

Reference: https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding.

    Precomputes and caches sin/cos embeddings for efficiency.
    Supports dynamic extension for longer sequences at inference.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ):
        """Initialize RoPE.

        Args:
            dim: Dimension of the embedding (typically head_dim).
            max_seq_len: Maximum sequence length for precomputation.
            base: Base for frequency computation.
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build sin/cos cache for given sequence length."""
        self.max_seq_len = seq_len

        # Position indices
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)

        # Outer product: [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)

        # Concatenate for full dimension: [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)

        # Cache cos and sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cos/sin embeddings for given sequence length.

        Args:
            seq_len: Sequence length.

        Returns:
            Tuple of (cos, sin) tensors of shape [seq_len, dim].
        """
        # Extend cache if needed
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dims for RoPE.

    Args:
        x: Tensor of shape (..., dim).

    Returns:
        Rotated tensor where first half and second half are swapped with sign flip.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape [batch, heads, seq_len, head_dim].
        k: Key tensor of shape [batch, heads, seq_len, head_dim].
        cos: Cosine embeddings of shape [seq_len, head_dim].
        sin: Sine embeddings of shape [seq_len, head_dim].
        position_ids: Optional position indices for non-contiguous positions.

    Returns:
        Tuple of rotated (q, k) tensors.
    """
    # Handle position_ids if provided
    if position_ids is not None:
        cos = cos[position_ids]  # [batch, seq_len, head_dim]
        sin = sin[position_ids]
    else:
        # Broadcast for batch and heads: [1, 1, seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot


class NTKAwareRotaryEmbedding(RotaryEmbedding):
    """RoPE with NTK-aware interpolation for context extension.

    Allows training at shorter context and extending at inference
    with improved extrapolation compared to linear interpolation.

    Reference: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        scale: float = 1.0,
    ):
        """Initialize NTK-aware RoPE.

        Args:
            dim: Dimension of the embedding.
            max_seq_len: Maximum sequence length.
            base: Base for frequency computation.
            scale: Context extension scale factor (e.g., 4.0 for 4x extension).
        """
        self.scale = scale

        # Adjust base for NTK-aware scaling
        if scale != 1.0:
            base = base * (scale ** (dim / (dim - 2)))

        super().__init__(dim=dim, max_seq_len=max_seq_len, base=base)


class YaRNRotaryEmbedding(nn.Module):
    """YaRN (Yet another RoPE extensioN) for improved context extension.

    Combines multiple techniques for better extrapolation:
    1. NTK-by-parts: Different scaling for high/low frequency components
    2. Attention temperature scaling: Prevents entropy collapse at long contexts
    3. Dynamic scaling: Adjusts based on actual sequence length

    Reference: https://arxiv.org/abs/2309.00071
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        original_max_seq_len: int = 2048,
        base: float = 10000.0,
        scale: float = 1.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
    ):
        """Initialize YaRN RoPE.

        Args:
            dim: Dimension of the embedding (head_dim).
            max_seq_len: Maximum sequence length (extended).
            original_max_seq_len: Original training sequence length.
            base: Base for frequency computation.
            scale: Context extension scale factor.
            beta_fast: Beta parameter for high-frequency interpolation.
            beta_slow: Beta parameter for low-frequency interpolation.
            mscale: Magnitude scaling factor for attention.
            mscale_all_dim: Scale factor applied across all dimensions.
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.original_max_seq_len = original_max_seq_len
        self.base = base
        self.scale = scale
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim

        # Compute the attention scaling factor
        self.attn_scale = self._compute_attn_scale(scale)

        # Build frequency bands with YaRN interpolation
        self._build_yarn_cache(max_seq_len)

    def _compute_attn_scale(self, scale: float) -> float:
        """Compute attention temperature scaling.

        Prevents attention entropy collapse at extended contexts.
        """
        if scale <= 1.0:
            return 1.0

        # YaRN attention scaling: 0.1 * ln(scale) + 1.0
        import math
        if self.mscale_all_dim > 0:
            return (0.1 * math.log(scale) + 1.0) ** self.mscale_all_dim
        return (0.1 * math.log(scale) + 1.0) ** self.mscale

    def _yarn_find_correction_dim(
        self,
        num_rotations: int,
        dim: int,
        base: float,
        max_seq_len: int,
    ) -> float:
        """Find the dimension where interpolation should transition."""
        import math
        return (dim * math.log(max_seq_len / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    def _yarn_find_correction_range(
        self,
        low_rot: float,
        high_rot: float,
        dim: int,
        base: float,
        max_seq_len: int,
    ) -> tuple[int, int]:
        """Find the range of dimensions for interpolation correction."""
        low = max(
            0,
            int(self._yarn_find_correction_dim(low_rot, dim, base, max_seq_len)),
        )
        high = min(
            dim - 1,
            int(self._yarn_find_correction_dim(high_rot, dim, base, max_seq_len)),
        )
        return low, high

    def _yarn_get_mscale(self, scale: float = 1.0) -> float:
        """Get magnitude scale for attention."""
        import math
        if scale <= 1.0:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def _build_yarn_cache(self, seq_len: int) -> None:
        """Build YaRN-interpolated frequency cache."""
        self.max_seq_len = seq_len

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )

        # Apply YaRN interpolation if scaling
        if self.scale > 1.0:
            # Find correction range
            low, high = self._yarn_find_correction_range(
                self.beta_fast,
                self.beta_slow,
                self.dim,
                self.base,
                self.original_max_seq_len,
            )

            # Create interpolation ramp
            inv_freq_extrapolation = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
            )
            inv_freq_interpolation = inv_freq_extrapolation / self.scale

            # Linear ramp for smooth transition between interpolation and extrapolation
            ramp = torch.linspace(0, 1, self.dim // 2)

            # Adjust ramp based on correction range
            ramp_mask = (ramp >= low / (self.dim // 2)) & (ramp <= high / (self.dim // 2))
            ramp = torch.where(ramp < low / (self.dim // 2), torch.zeros_like(ramp), ramp)
            ramp = torch.where(ramp > high / (self.dim // 2), torch.ones_like(ramp), ramp)

            # Smooth interpolation between the two
            inv_freq = inv_freq_interpolation * (1 - ramp) + inv_freq_extrapolation * ramp

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build position embeddings
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Get cos/sin embeddings and attention scale.

        Args:
            seq_len: Sequence length.

        Returns:
            Tuple of (cos, sin, attn_scale).
        """
        if seq_len > self.max_seq_len:
            self._build_yarn_cache(seq_len)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
            self.attn_scale,
        )


def apply_rope_with_scale(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attn_scale: float = 1.0,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings with optional attention scaling (for YaRN).

    Args:
        q: Query tensor [batch, heads, seq_len, head_dim].
        k: Key tensor [batch, heads, seq_len, head_dim].
        cos: Cosine embeddings [seq_len, head_dim].
        sin: Sine embeddings [seq_len, head_dim].
        attn_scale: Attention scaling factor from YaRN.
        position_ids: Optional position indices.

    Returns:
        Tuple of rotated (q, k) tensors with scaling applied to q.
    """
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    # Apply YaRN attention scaling to queries
    if attn_scale != 1.0:
        q_rot = q_rot * attn_scale

    return q_rot, k_rot
