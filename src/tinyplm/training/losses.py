"""Loss functions for TinyPLM training.

Implements:
- MLM cross-entropy loss
- MRL InfoNCE contrastive loss at multiple dimensions
- Combined loss with configurable weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLMLoss(nn.Module):
    """Masked Language Modeling loss.

    Standard cross-entropy loss for masked token prediction.
    """

    def __init__(self, ignore_index: int = -100):
        """Initialize MLM loss.

        Args:
            ignore_index: Label index to ignore (unmasked positions).
        """
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MLM loss.

        Args:
            logits: Model predictions [batch, seq_len, vocab_size].
            labels: Target labels [batch, seq_len]. Use -100 for unmasked.

        Returns:
            Scalar loss tensor.
        """
        # Flatten for cross-entropy
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=self.ignore_index,
        )
        return loss


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss.

    Used for MRL training where we want embeddings to be similar
    for augmented views of the same sequence.
    """

    def __init__(self, temperature: float = 0.07):
        """Initialize InfoNCE loss.

        Args:
            temperature: Temperature for softmax scaling.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            embeddings: Normalized embeddings [2N, dim] where consecutive pairs
                (2i, 2i+1) are positive pairs from same sequence.
            labels: Optional labels for supervised contrastive. If None,
                uses consecutive pairs as positives.

        Returns:
            Scalar loss tensor.
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)

        batch_size = embeddings.size(0)
        n_pairs = batch_size // 2

        # Compute similarity matrix
        sim = embeddings @ embeddings.T / self.temperature  # [2N, 2N]

        # Mask out self-similarity
        mask = torch.eye(batch_size, device=sim.device, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))

        if labels is None:
            # Consecutive pairs are positives: (0,1), (2,3), (4,5), ...
            # For row 2i, positive is 2i+1; for row 2i+1, positive is 2i
            targets = torch.arange(batch_size, device=sim.device)
            targets[0::2] = targets[0::2] + 1  # Even rows: point to next
            targets[1::2] = targets[1::2] - 1  # Odd rows: point to previous
        else:
            targets = labels

        loss = F.cross_entropy(sim, targets)
        return loss


class MRLLoss(nn.Module):
    """Matryoshka Representation Learning loss.

    Computes contrastive loss at multiple embedding dimensions,
    enabling flexible truncation at inference time.
    """

    def __init__(
        self,
        dims: list[int] = [64, 128, 256, 512],
        temperature: float = 0.07,
        weights: list[float] | None = None,
    ):
        """Initialize MRL loss.

        Args:
            dims: List of embedding dimensions to train.
            temperature: Temperature for InfoNCE.
            weights: Optional weights for each dimension. If None, uniform.
        """
        super().__init__()
        self.dims = dims
        self.infonce = InfoNCELoss(temperature=temperature)

        if weights is None:
            weights = [1.0] * len(dims)
        assert len(weights) == len(dims)
        self.weights = weights

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute MRL loss at all dimensions.

        Args:
            embeddings: Full embeddings [2N, full_dim] where consecutive
                pairs are positive pairs.

        Returns:
            Tuple of (total_loss, dict of per-dimension losses).
        """
        total_loss = 0.0
        loss_dict = {}

        for dim, weight in zip(self.dims, self.weights):
            # Truncate to this dimension
            truncated = embeddings[:, :dim]
            truncated = F.normalize(truncated, dim=-1)

            # Compute loss at this dimension
            loss = self.infonce(truncated)
            total_loss = total_loss + weight * loss
            loss_dict[f"mrl_loss_{dim}"] = loss.item()

        # Average by number of dimensions
        total_loss = total_loss / len(self.dims)

        return total_loss, loss_dict


class CombinedLoss(nn.Module):
    """Combined MLM + MRL loss for joint training."""

    def __init__(
        self,
        mlm_weight: float = 1.0,
        mrl_weight: float = 0.1,
        mrl_dims: list[int] = [64, 128, 256, 512],
        temperature: float = 0.07,
    ):
        """Initialize combined loss.

        Args:
            mlm_weight: Weight for MLM loss.
            mrl_weight: Weight for MRL contrastive loss.
            mrl_dims: Dimensions for MRL training.
            temperature: Temperature for contrastive loss.
        """
        super().__init__()
        self.mlm_weight = mlm_weight
        self.mrl_weight = mrl_weight

        self.mlm_loss = MLMLoss()
        self.mrl_loss = MRLLoss(dims=mrl_dims, temperature=temperature)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        view1_embeddings: torch.Tensor,
        view2_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Args:
            logits: MLM logits [batch, seq_len, vocab_size].
            labels: MLM labels [batch, seq_len].
            view1_embeddings: Embeddings for first augmented view [batch, dim].
            view2_embeddings: Embeddings for second augmented view [batch, dim].

        Returns:
            Tuple of (total_loss, dict of component losses).
        """
        # MLM loss
        mlm = self.mlm_loss(logits, labels)

        # Interleave views for MRL: [v1_0, v2_0, v1_1, v2_1, ...]
        batch_size = view1_embeddings.size(0)
        embeddings = torch.stack([view1_embeddings, view2_embeddings], dim=1)
        embeddings = embeddings.view(batch_size * 2, -1)

        # MRL loss
        mrl, mrl_dict = self.mrl_loss(embeddings)

        # Combine
        total = self.mlm_weight * mlm + self.mrl_weight * mrl

        loss_dict = {
            "mlm_loss": mlm.item(),
            "mrl_loss": mrl.item(),
            "total_loss": total.item(),
            **mrl_dict,
        }

        return total, loss_dict
