"""Transformer Block for Wren.

Pre-norm architecture combining:
- Multi-head attention (or Differential attention) with YaRN
- SwiGLU feed-forward network
- Residual connections
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from wren.model.attention import AttentionOutput, DifferentialAttention, MultiHeadAttention
from wren.model.ffn import SwiGLUFFN
from wren.model.norm import RMSNorm


@dataclass
class TransformerBlockOutput:
    """Output from transformer block with optional attention maps."""

    hidden_states: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None


class TransformerBlock(nn.Module):
    """Single Transformer Block with pre-norm architecture.

    Architecture:
        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        head_dim: Optional[int] = None,
        use_bitlinear: bool = True,
        use_differential_attn: bool = False,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        gradient_checkpointing: bool = False,
        use_yarn: bool = True,
        rope_scale: float = 1.0,
    ):
        """Initialize Transformer Block.

        Args:
            hidden_dim: Model hidden dimension.
            num_heads: Number of attention heads.
            ffn_dim: FFN intermediate dimension.
            head_dim: Dimension per head.
            use_bitlinear: Whether to use BitLinear.
            use_differential_attn: Whether to use Differential Attention (Phase 1b).
            dropout: Dropout probability.
            max_seq_len: Maximum sequence length.
            gradient_checkpointing: Whether to use gradient checkpointing.
            use_yarn: Whether to use YaRN position encoding.
            rope_scale: Context extension scale factor.
        """
        super().__init__()

        self.gradient_checkpointing = gradient_checkpointing

        # Pre-norm for attention
        self.attn_norm = RMSNorm(hidden_dim)

        # Attention layer
        if use_differential_attn:
            self.attn = DifferentialAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                use_bitlinear=use_bitlinear,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
        else:
            self.attn = MultiHeadAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                use_bitlinear=use_bitlinear,
                dropout=dropout,
                max_seq_len=max_seq_len,
                use_yarn=use_yarn,
                rope_scale=rope_scale,
            )

        # Pre-norm for FFN
        self.ffn_norm = RMSNorm(hidden_dim)

        # FFN
        self.ffn = SwiGLUFFN(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            use_bitlinear=use_bitlinear,
            dropout=dropout,
        )

    def _attn_forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Attention forward for checkpointing."""
        return self.attn(
            self.attn_norm(x),
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def _ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """FFN forward for checkpointing."""
        return self.ffn(self.ffn_norm(x))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through transformer block.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim].
            attention_mask: Optional attention mask.
            position_ids: Optional position indices.

        Returns:
            Output tensor of shape [batch, seq_len, hidden_dim].
        """
        # Attention with residual
        if self.gradient_checkpointing and self.training:
            attn_out = checkpoint(
                self._attn_forward,
                x,
                attention_mask,
                position_ids,
                use_reentrant=False,
            )
        else:
            attn_out = self._attn_forward(x, attention_mask, position_ids)
        x = x + attn_out

        # FFN with residual
        if self.gradient_checkpointing and self.training:
            ffn_out = checkpoint(self._ffn_forward, x, use_reentrant=False)
        else:
            ffn_out = self._ffn_forward(x)
        x = x + ffn_out

        return x
