"""SwiGLU Feed-Forward Network for TinyPLM.

Implements gated linear unit with SiLU (Swish) activation.
Uses BitLinear for weight-quantized layers.

Reference: https://arxiv.org/abs/2002.05202
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyplm.model.bitlinear import get_linear_layer


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    Architecture:
        gate = SiLU(W_gate @ x)
        up = W_up @ x
        down = W_down @ (gate * up)

    Uses 8/3 expansion ratio to match parameter count of standard 4x FFN
    when accounting for the additional gate projection.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        use_bitlinear: bool = True,
        dropout: float = 0.0,
    ):
        """Initialize SwiGLU FFN.

        Args:
            hidden_dim: Model hidden dimension.
            ffn_dim: FFN intermediate dimension (typically ~8/3 * hidden_dim).
            use_bitlinear: Whether to use BitLinear (True) or FP16 (False).
            dropout: Dropout probability (typically 0 for BitNet).
        """
        super().__init__()

        # Gate and up projections (can be fused for efficiency)
        self.gate_proj = get_linear_layer(
            hidden_dim, ffn_dim, bias=False, use_bitlinear=use_bitlinear
        )
        self.up_proj = get_linear_layer(
            hidden_dim, ffn_dim, bias=False, use_bitlinear=use_bitlinear
        )
        self.down_proj = get_linear_layer(
            ffn_dim, hidden_dim, bias=False, use_bitlinear=use_bitlinear
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SwiGLU FFN.

        Args:
            x: Input tensor of shape (..., hidden_dim).

        Returns:
            Output tensor of shape (..., hidden_dim).
        """
        # Gated activation: SiLU(gate) * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up

        # Project back down
        out = self.down_proj(hidden)
        out = self.dropout(out)

        return out
