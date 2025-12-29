"""BitLinear layer with ternary weight quantization.

Implements BitNet b1.58 quantization-aware training with:
- Ternary weights: {-1, 0, +1}
- Int8 activation quantization
- Straight-Through Estimator (STE) for gradients
- RMSNorm before quantization (per BitNet spec)

Reference: https://arxiv.org/abs/2402.17764
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyplm.model.norm import RMSNorm


class BitLinear(nn.Module):
    """Linear layer with ternary weight quantization.

    During training:
    - FP16 shadow weights are updated by optimizer
    - Forward pass uses quantized weights (ternary) and activations (int8)
    - STE passes gradients through quantization

    During inference:
    - Weights can be stored as int2 (ternary: 2 bits per weight)
    - ~10x memory reduction compared to FP16
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-5,
    ):
        """Initialize BitLinear layer.

        Args:
            in_features: Size of input features.
            out_features: Size of output features.
            bias: Whether to include bias term.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        # FP16 shadow weights for optimizer updates
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # RMSNorm for input activation normalization (BitNet requirement)
        self.input_norm = RMSNorm(in_features)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights using Kaiming uniform."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def ternary_quantize(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize weights to ternary {-1, 0, +1}.

        Uses mean absolute value as threshold for determining zeros.
        Implements Straight-Through Estimator for gradient flow.

        Args:
            w: Weight tensor to quantize.

        Returns:
            Tuple of (quantized_weights, scale_factor).
        """
        # Compute scale (mean absolute value)
        scale = w.abs().mean()

        # Normalize by scale
        w_scaled = w / (scale + self.eps)

        # Ternary quantization with threshold at 0.5
        # Values > 0.5 -> +1, values < -0.5 -> -1, else -> 0
        w_ternary = torch.sign(w_scaled) * (w_scaled.abs() > 0.5).float()

        # STE: forward uses quantized, backward uses continuous
        w_ternary = w_ternary.detach() + w_scaled - w_scaled.detach()

        return w_ternary, scale

    def quantize_activations(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize activations to int8 range [-127, 127].

        Uses per-tensor absmax scaling.

        Args:
            x: Activation tensor to quantize.

        Returns:
            Tuple of (quantized_activations, scale_factor).
        """
        # Per-token absmax scaling (scale per last dim)
        scale = x.abs().max(dim=-1, keepdim=True).values / 127.0

        # Quantize to int8 range
        x_quant = (x / (scale + self.eps)).round().clamp(-128, 127)

        # STE for gradient flow
        x_quant = x_quant.detach() + x - x.detach()

        return x_quant, scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized weights and activations.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        # Normalize input activations (per BitNet spec)
        x = self.input_norm(x)

        # Quantize activations to int8
        x_q, x_scale = self.quantize_activations(x)

        # Quantize weights to ternary
        w_q, w_scale = self.ternary_quantize(self.weight)

        # Linear transformation with scaling
        # Output = (x_q @ w_q.T) * (x_scale * w_scale)
        out = F.linear(x_q, w_q)

        # Apply combined scale
        out = out * (x_scale * w_scale)

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias

        return out

    def extra_repr(self) -> str:
        """String representation for printing."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


class FP16Linear(nn.Module):
    """Standard FP16 linear layer for baseline comparison.

    Drop-in replacement for BitLinear without quantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        """Initialize FP16 linear layer."""
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm = RMSNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with standard FP16 computation."""
        x = self.norm(x)
        return self.linear(x)


def get_linear_layer(
    in_features: int,
    out_features: int,
    bias: bool = False,
    use_bitlinear: bool = True,
) -> nn.Module:
    """Factory function to get appropriate linear layer.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        bias: Whether to include bias.
        use_bitlinear: If True, return BitLinear. If False, return FP16Linear.

    Returns:
        Linear layer module.
    """
    if use_bitlinear:
        return BitLinear(in_features, out_features, bias=bias)
    else:
        return FP16Linear(in_features, out_features, bias=bias)
