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
