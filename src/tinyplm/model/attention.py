"""Multi-Head Attention for TinyPLM.

Implements standard multi-head attention with:
- BitLinear for Q, K, V, O projections
- Rotary Position Embeddings (RoPE)
- Optional Flash Attention for efficiency

Phase 1b will add Differential Attention.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyplm.model.bitlinear import get_linear_layer
from tinyplm.model.rope import RotaryEmbedding, apply_rope


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with RoPE.

    Uses BitLinear for all projections when quantization is enabled.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        use_bitlinear: bool = True,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ):
        """Initialize Multi-Head Attention.

        Args:
            hidden_dim: Model hidden dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head (default: hidden_dim // num_heads).
            use_bitlinear: Whether to use BitLinear.
            dropout: Attention dropout probability.
            max_seq_len: Maximum sequence length for RoPE.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_dim // num_heads)
        self.scale = self.head_dim**-0.5

        # Total dimension for all heads
        self.total_head_dim = self.num_heads * self.head_dim

        # Q, K, V projections
        self.q_proj = get_linear_layer(
            hidden_dim, self.total_head_dim, bias=False, use_bitlinear=use_bitlinear
        )
        self.k_proj = get_linear_layer(
            hidden_dim, self.total_head_dim, bias=False, use_bitlinear=use_bitlinear
        )
        self.v_proj = get_linear_layer(
            hidden_dim, self.total_head_dim, bias=False, use_bitlinear=use_bitlinear
        )

        # Output projection
        self.o_proj = get_linear_layer(
            self.total_head_dim, hidden_dim, bias=False, use_bitlinear=use_bitlinear
        )

        # RoPE for position encoding
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through attention.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim].
            attention_mask: Optional mask of shape [batch, seq_len] or
                [batch, 1, seq_len, seq_len]. 1 = attend, 0 = mask.
            position_ids: Optional position indices for RoPE.

        Returns:
            Output tensor of shape [batch, seq_len, hidden_dim].
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(seq_len)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        q, k = apply_rope(q, k, cos, sin, position_ids)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask
        if attention_mask is not None:
            # Convert mask to additive form: 0 -> 0, 1 -> -inf for masked positions
            # Input mask: 1 = attend, 0 = mask
            if attention_mask.dim() == 2:
                # [batch, seq_len] -> [batch, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Convert to float and create additive mask
            attn_mask = (1.0 - attention_mask.to(x.dtype)) * torch.finfo(x.dtype).min
            attn_weights = attn_weights + attn_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, total_head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.total_head_dim)

        # Output projection
        output = self.o_proj(attn_output)

        return output


class DifferentialAttention(nn.Module):
    """Differential Attention for improved long-context modeling.

    Computes two attention maps and subtracts to cancel common-mode noise.
    Phase 1b implementation.

    Reference: https://arxiv.org/abs/2410.05258
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        use_bitlinear: bool = True,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
        lambda_init: float = 0.8,
    ):
        """Initialize Differential Attention.

        Args:
            hidden_dim: Model hidden dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            use_bitlinear: Whether to use BitLinear.
            dropout: Attention dropout probability.
            max_seq_len: Maximum sequence length for RoPE.
            lambda_init: Initial value for learnable lambda.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_dim // num_heads)
        self.scale = self.head_dim**-0.5

        # For differential attention, we need 2x the Q/K dimension
        # to compute two attention maps
        self.total_head_dim = self.num_heads * self.head_dim

        # Q, K projections (2x for differential)
        self.q_proj = get_linear_layer(
            hidden_dim, self.total_head_dim * 2, bias=False, use_bitlinear=use_bitlinear
        )
        self.k_proj = get_linear_layer(
            hidden_dim, self.total_head_dim * 2, bias=False, use_bitlinear=use_bitlinear
        )
        self.v_proj = get_linear_layer(
            hidden_dim, self.total_head_dim, bias=False, use_bitlinear=use_bitlinear
        )

        # Output projection
        self.o_proj = get_linear_layer(
            self.total_head_dim, hidden_dim, bias=False, use_bitlinear=use_bitlinear
        )

        # RoPE for position encoding
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

        # Learnable lambda per head
        self.lambda_q1 = nn.Parameter(torch.ones(self.num_heads) * lambda_init)
        self.lambda_k1 = nn.Parameter(torch.ones(self.num_heads) * lambda_init)
        self.lambda_q2 = nn.Parameter(torch.ones(self.num_heads) * lambda_init)
        self.lambda_k2 = nn.Parameter(torch.ones(self.num_heads) * lambda_init)
        self.lambda_init = nn.Parameter(torch.tensor(lambda_init))

        # GroupNorm for output (per DiffAttn paper)
        self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=self.total_head_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through differential attention."""
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V (Q and K are 2x for differential)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split Q and K into two groups for differential attention
        q = q.view(batch_size, seq_len, 2, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, 2, self.num_heads, self.head_dim)

        # Reshape: [batch, 2, num_heads, seq_len, head_dim]
        q = q.permute(0, 2, 3, 1, 4)
        k = k.permute(0, 2, 3, 1, 4)

        # Separate into two groups
        q1, q2 = q[:, 0], q[:, 1]  # Each: [batch, num_heads, seq_len, head_dim]
        k1, k2 = k[:, 0], k[:, 1]

        # Apply RoPE to both groups
        cos, sin = self.rotary_emb(seq_len)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        q1, k1 = apply_rope(q1, k1, cos, sin, position_ids)
        q2, k2 = apply_rope(q2, k2, cos, sin, position_ids)

        # Reshape v: [batch, num_heads, seq_len, head_dim]
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute two attention maps
        attn1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scale
        attn2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scale

        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = (1.0 - attention_mask.to(x.dtype)) * torch.finfo(x.dtype).min
            attn1 = attn1 + attn_mask
            attn2 = attn2 + attn_mask

        # Softmax
        attn1 = F.softmax(attn1, dim=-1)
        attn2 = F.softmax(attn2, dim=-1)

        # Compute lambda for this layer
        lambda_1 = torch.exp(self.lambda_q1 * self.lambda_k1).view(1, -1, 1, 1)
        lambda_2 = torch.exp(self.lambda_q2 * self.lambda_k2).view(1, -1, 1, 1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Differential attention: A1 - lambda * A2
        attn_diff = attn1 - lambda_full * attn2
        attn_diff = self.dropout(attn_diff)

        # Apply to values
        attn_output = torch.matmul(attn_diff, v)

        # Reshape: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, total_head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.total_head_dim)

        # GroupNorm (reshape for group norm: [batch * seq_len, total_head_dim])
        attn_output = attn_output.view(-1, self.total_head_dim)
        attn_output = self.group_norm(attn_output)
        attn_output = attn_output.view(batch_size, seq_len, self.total_head_dim)

        # Scale by (1 - lambda_init) as per paper
        attn_output = attn_output * (1 - self.lambda_init)

        # Output projection
        output = self.o_proj(attn_output)

        return output
