"""Protein Language Model (TinyPLM).

Full model combining:
- Token embeddings (FP16, not quantized)
- Transformer blocks with BitLinear
- MLM head for masked language modeling
- Embedding pooling for sequence representations
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyplm.config import ModelConfig
from tinyplm.model.norm import RMSNorm
from tinyplm.model.transformer import TransformerBlock


class TinyPLM(nn.Module):
    """TinyPLM: BitNet-MRL Protein Language Model.

    Architecture:
        - FP16 token embeddings (not quantized for semantic discrimination)
        - Stack of transformer blocks with BitLinear
        - FP16 output projection (not quantized for MRL loss)
        - Mean pooling for sequence embeddings

    Supports:
        - Masked Language Modeling (MLM) for pretraining
        - Matryoshka Representation Learning (MRL) for flexible embeddings
    """

    def __init__(self, config: ModelConfig):
        """Initialize TinyPLM.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        # Token embeddings (FP16 - not quantized)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                head_dim=config.head_dim,
                use_bitlinear=config.use_bitlinear,
                use_differential_attn=False,  # Phase 1a: standard attention
                dropout=config.dropout,
                max_seq_len=config.max_seq_len,
                gradient_checkpointing=config.gradient_checkpointing,
                use_yarn=config.use_yarn,
                rope_scale=config.rope_scale,
            )
            for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.final_norm = RMSNorm(config.hidden_dim)

        # MLM head (FP16 - not quantized)
        self.mlm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Embedding projection for MRL (FP16 - not quantized)
        # Projects to embedding_dim which should be max(mrl_dims)
        self.embed_proj = nn.Linear(config.hidden_dim, config.embedding_dim, bias=False)

        # Tie MLM head weights to embeddings
        self.mlm_head.weight = self.embeddings.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with small values for stability."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through TinyPLM.

        Args:
            input_ids: Token IDs of shape [batch, seq_len].
            attention_mask: Attention mask of shape [batch, seq_len].
                1 = attend, 0 = mask.
            position_ids: Optional position indices.
            output_hidden_states: Whether to return all hidden states.

        Returns:
            Dictionary containing:
                - 'logits': MLM logits of shape [batch, seq_len, vocab_size]
                - 'embeddings': Sequence embeddings of shape [batch, embedding_dim]
                - 'last_hidden_state': Final hidden states [batch, seq_len, hidden_dim]
                - 'hidden_states': (optional) All layer hidden states
        """
        # Token embeddings
        hidden_states = self.embeddings(input_ids)

        all_hidden_states = [hidden_states] if output_hidden_states else None

        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        # Final layer norm
        hidden_states = self.final_norm(hidden_states)

        # MLM logits
        logits = self.mlm_head(hidden_states)

        # Sequence embeddings via mean pooling
        if attention_mask is not None:
            # Mask padding tokens for mean pooling
            mask = attention_mask.unsqueeze(-1).float()
            sum_hidden = (hidden_states * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            pooled = sum_hidden / sum_mask
        else:
            pooled = hidden_states.mean(dim=1)

        # Project to embedding dimension
        embeddings = self.embed_proj(pooled)
        embeddings = F.normalize(embeddings, dim=-1)

        result = {
            "logits": logits,
            "embeddings": embeddings,
            "last_hidden_state": hidden_states,
        }

        if output_hidden_states:
            result["hidden_states"] = all_hidden_states

        return result

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dim: Optional[int] = None,
    ) -> torch.Tensor:
        """Get sequence embeddings, optionally truncated for MRL.

        Args:
            input_ids: Token IDs of shape [batch, seq_len].
            attention_mask: Attention mask of shape [batch, seq_len].
            dim: Embedding dimension to truncate to (for MRL).
                If None, returns full embedding.

        Returns:
            Embeddings of shape [batch, dim] or [batch, embedding_dim].
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask=attention_mask)
            embeddings = outputs["embeddings"]

            if dim is not None:
                embeddings = embeddings[:, :dim]
                embeddings = F.normalize(embeddings, dim=-1)

            return embeddings

    def compute_mlm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Compute MLM cross-entropy loss.

        Args:
            logits: Model logits of shape [batch, seq_len, vocab_size].
            labels: Target labels of shape [batch, seq_len].
                Use ignore_index for non-masked positions.
            ignore_index: Index to ignore in loss computation.

        Returns:
            Scalar loss tensor.
        """
        # Flatten for cross-entropy
        logits_flat = logits.view(-1, self.config.vocab_size)
        labels_flat = labels.view(-1)

        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index)
        return loss

    @property
    def num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
