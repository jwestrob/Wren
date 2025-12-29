"""Tests for Wren training components."""

import pytest
import torch
from torch.optim import AdamW

from wren.training.losses import MLMLoss, InfoNCELoss, MRLLoss, CombinedLoss
from wren.training.scheduler import (
    get_cosine_schedule_with_warmup,
    get_wsd_schedule,
    get_linear_decay_schedule,
    get_scheduler,
)


class TestMLMLoss:
    """Tests for MLM loss."""

    def test_loss_computation(self):
        """Test basic loss computation."""
        loss_fn = MLMLoss()

        logits = torch.randn(4, 32, 25)  # [batch, seq, vocab]
        labels = torch.randint(0, 25, (4, 32))
        labels[:, ::2] = -100  # Mask half

        loss = loss_fn(logits, labels)

        assert loss.shape == ()
        assert loss.item() > 0

    def test_ignores_masked_positions(self):
        """Test that -100 labels are ignored."""
        loss_fn = MLMLoss()

        logits = torch.randn(4, 32, 25)

        # All masked - returns NaN (no valid targets)
        labels = torch.full((4, 32), -100)
        loss = loss_fn(logits, labels)

        # Loss is NaN when all positions are ignored (expected)
        import math
        assert math.isnan(loss.item())


class TestInfoNCELoss:
    """Tests for InfoNCE contrastive loss."""

    def test_loss_with_pairs(self):
        """Test loss with positive pairs."""
        loss_fn = InfoNCELoss(temperature=0.07)

        # 4 pairs = 8 embeddings
        embeddings = torch.randn(8, 128)

        loss = loss_fn(embeddings)

        assert loss.shape == ()
        assert loss.item() > 0

    def test_identical_pairs_low_loss(self):
        """Test that identical pairs have lower loss."""
        loss_fn = InfoNCELoss(temperature=0.07)

        # Create embeddings where pairs are identical
        base = torch.randn(4, 128)
        embeddings = torch.cat([base, base], dim=0)
        # Reorder to interleave: [0, 0, 1, 1, 2, 2, 3, 3]
        embeddings = embeddings.reshape(2, 4, 128).transpose(0, 1).reshape(8, 128)

        loss_similar = loss_fn(embeddings)

        # Random embeddings should have higher loss
        random_embeddings = torch.randn(8, 128)
        loss_random = loss_fn(random_embeddings)

        assert loss_similar.item() < loss_random.item()


class TestMRLLoss:
    """Tests for Matryoshka loss."""

    def test_multi_dim_loss(self):
        """Test loss at multiple dimensions."""
        loss_fn = MRLLoss(dims=[32, 64, 128], temperature=0.07)

        embeddings = torch.randn(8, 128)  # 4 pairs

        loss, loss_dict = loss_fn(embeddings)

        assert loss.shape == ()
        assert "mrl_loss_32" in loss_dict
        assert "mrl_loss_64" in loss_dict
        assert "mrl_loss_128" in loss_dict

    def test_truncation_happens(self):
        """Test that truncation actually happens."""
        loss_fn = MRLLoss(dims=[32, 64], temperature=0.07)

        # Large embeddings
        embeddings = torch.randn(8, 256)

        loss, _ = loss_fn(embeddings)

        # Should work even though dims < embedding size
        assert loss.shape == ()


class TestCombinedLoss:
    """Tests for combined MLM + MRL loss."""

    def test_combined_output(self):
        """Test combined loss returns all components."""
        loss_fn = CombinedLoss(
            mlm_weight=1.0,
            mrl_weight=0.1,
            mrl_dims=[32, 64],
        )

        logits = torch.randn(4, 32, 25)
        labels = torch.randint(0, 25, (4, 32))
        view1 = torch.randn(4, 64)
        view2 = torch.randn(4, 64)

        loss, metrics = loss_fn(logits, labels, view1, view2)

        assert loss.shape == ()
        assert "mlm_loss" in metrics
        assert "mrl_loss" in metrics
        assert "total_loss" in metrics


class TestSchedulers:
    """Tests for learning rate schedulers."""

    @pytest.fixture
    def optimizer(self):
        """Create dummy optimizer."""
        model = torch.nn.Linear(10, 10)
        return AdamW(model.parameters(), lr=1e-3)

    def test_cosine_warmup(self, optimizer):
        """Test cosine schedule with warmup."""
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=1000,
        )

        # Check warmup
        lrs = []
        for _ in range(100):
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        # LR should increase during warmup
        assert lrs[0] < lrs[50] < lrs[99]

        # Check decay
        peak_lr = scheduler.get_last_lr()[0]
        for _ in range(900):
            scheduler.step()

        # LR should decrease
        assert scheduler.get_last_lr()[0] < peak_lr

    def test_wsd_schedule(self, optimizer):
        """Test WSD schedule."""
        scheduler = get_wsd_schedule(
            optimizer,
            num_warmup_steps=100,
            num_stable_steps=500,
            num_decay_steps=400,
        )

        # Warmup
        for _ in range(100):
            scheduler.step()

        peak_lr = scheduler.get_last_lr()[0]

        # Stable phase - LR should stay at peak
        for _ in range(250):
            scheduler.step()

        assert abs(scheduler.get_last_lr()[0] - peak_lr) < 1e-6

        # Decay phase
        for _ in range(400):
            scheduler.step()

        assert scheduler.get_last_lr()[0] < peak_lr

    def test_linear_decay(self, optimizer):
        """Test linear decay schedule."""
        scheduler = get_linear_decay_schedule(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=1000,
        )

        for _ in range(100):
            scheduler.step()

        peak_lr = scheduler.get_last_lr()[0]

        # Should decay linearly
        for _ in range(450):
            scheduler.step()

        # Should be about half
        assert scheduler.get_last_lr()[0] < peak_lr
        assert scheduler.get_last_lr()[0] > 0.3 * peak_lr

    def test_get_scheduler_factory(self, optimizer):
        """Test scheduler factory function."""
        for name in ["cosine", "wsd", "linear", "d2z"]:
            scheduler = get_scheduler(
                name,
                optimizer,
                num_warmup_steps=100,
                num_training_steps=1000,
            )
            assert scheduler is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
