"""Tests for TinyPLM data pipeline."""

import tempfile
from pathlib import Path

import pytest
import torch

from tinyplm.data.tokenizer import ProteinTokenizer
from tinyplm.data.dataset import (
    ProteinDataset,
    MLMDataset,
    ContrastiveMLMDataset,
    MLMMasker,
    ContrastiveAugmenter,
)
from tinyplm.data.dataloader import (
    LengthBucketSampler,
    collate_fn,
    collate_contrastive_fn,
    create_dataloader,
)


@pytest.fixture
def sample_fasta():
    """Create a temporary FASTA file for testing."""
    sequences = [
        ("seq1", "MKTAYIAKQRQISFVKSHFSRQLEER"),
        ("seq2", "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"),
        ("seq3", "GLSDGEWQQVLNVWGKVEADIAGHGQEVLIRLFTGHPETLEKFDKFKHLKTEAEMK"),
        ("seq4", "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL"),
        ("seq5", "MKWVTFISLLLLFSSAYS"),  # Short sequence
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        for name, seq in sequences:
            f.write(f">{name}\n{seq}\n")
        fasta_path = Path(f.name)

    yield fasta_path

    # Cleanup
    fasta_path.unlink()


@pytest.fixture
def tokenizer():
    """Create tokenizer for tests."""
    return ProteinTokenizer()


class TestProteinDataset:
    """Tests for ProteinDataset."""

    def test_load_sequences(self, sample_fasta, tokenizer):
        """Test loading sequences from FASTA."""
        dataset = ProteinDataset(
            sample_fasta,
            tokenizer=tokenizer,
            min_length=10,
            max_length=200,
        )

        # Should load all 5 sequences (all within length range 10-200)
        assert len(dataset) == 5

    def test_getitem_returns_tensors(self, sample_fasta, tokenizer):
        """Test that __getitem__ returns proper tensors."""
        dataset = ProteinDataset(sample_fasta, tokenizer=tokenizer, min_length=10)

        sample = dataset[0]

        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert isinstance(sample["input_ids"], torch.Tensor)
        assert isinstance(sample["attention_mask"], torch.Tensor)

    def test_special_tokens_added(self, sample_fasta, tokenizer):
        """Test that CLS and SEP tokens are added."""
        dataset = ProteinDataset(sample_fasta, tokenizer=tokenizer, min_length=10)

        sample = dataset[0]
        input_ids = sample["input_ids"]

        assert input_ids[0].item() == tokenizer.CLS_ID
        assert input_ids[-1].item() == tokenizer.SEP_ID


class TestMLMMasker:
    """Tests for MLM masking."""

    def test_mask_ratio(self, tokenizer):
        """Test that approximately 15% of tokens are masked."""
        masker = MLMMasker(tokenizer, mask_prob=0.15)

        # Create a long sequence for statistical testing
        input_ids = torch.tensor(
            [tokenizer.CLS_ID] + [5] * 100 + [tokenizer.SEP_ID]  # 100 A's
        )
        attention_mask = torch.ones_like(input_ids)

        masked_ids, labels = masker(input_ids, attention_mask)

        # Count masked positions (where label != -100)
        n_masked = (labels != -100).sum().item()

        # Should be roughly 15% of non-special tokens (100 tokens)
        # Allow for random variance
        assert 5 <= n_masked <= 25

    def test_mask_token_distribution(self, tokenizer):
        """Test 80/10/10 distribution of mask types."""
        masker = MLMMasker(tokenizer, mask_prob=1.0)  # Mask everything

        input_ids = torch.tensor([5] * 1000)  # 1000 A's
        attention_mask = torch.ones_like(input_ids)

        masked_ids, labels = masker(input_ids, attention_mask)

        # Count each type
        n_mask_token = (masked_ids == tokenizer.MASK_ID).sum().item()
        n_unchanged = (masked_ids == input_ids).sum().item()
        n_random = 1000 - n_mask_token - n_unchanged

        # Check rough proportions (allow variance)
        assert 700 <= n_mask_token <= 900  # ~80%
        assert 50 <= n_random <= 150  # ~10%
        assert 50 <= n_unchanged <= 150  # ~10%

    def test_special_tokens_preserved(self, tokenizer):
        """Test that special tokens are never masked."""
        masker = MLMMasker(tokenizer, mask_prob=1.0)

        input_ids = torch.tensor([
            tokenizer.CLS_ID, 5, 6, 7, tokenizer.SEP_ID
        ])
        attention_mask = torch.ones_like(input_ids)

        masked_ids, labels = masker(input_ids, attention_mask)

        # CLS and SEP should be unchanged
        assert masked_ids[0].item() == tokenizer.CLS_ID
        assert masked_ids[-1].item() == tokenizer.SEP_ID

        # Labels should be -100 for special tokens
        assert labels[0].item() == -100
        assert labels[-1].item() == -100


class TestContrastiveAugmenter:
    """Tests for contrastive augmentations."""

    def test_random_crop_length(self, tokenizer):
        """Test that crops are within expected range."""
        augmenter = ContrastiveAugmenter(
            tokenizer,
            crop_min_ratio=0.7,
            crop_max_ratio=0.9,
        )

        sequence = "A" * 100

        for _ in range(10):
            cropped = augmenter.random_crop(sequence)
            assert 70 <= len(cropped) <= 90

    def test_create_pair_returns_different_views(self, tokenizer):
        """Test that two views are created."""
        augmenter = ContrastiveAugmenter(tokenizer)

        sequence = "MKTAYIAKQRQISFVKSHFSRQLEER" * 4  # Long enough for meaningful crops

        view1, view2 = augmenter.create_pair(sequence)

        assert "input_ids" in view1
        assert "input_ids" in view2
        assert "attention_mask" in view1
        assert "attention_mask" in view2

        # Views should have different lengths (usually)
        # or at least be valid tensors
        assert len(view1["input_ids"]) > 0
        assert len(view2["input_ids"]) > 0

    def test_extra_masking_applied(self, tokenizer):
        """Test that extra masking is applied."""
        augmenter = ContrastiveAugmenter(
            tokenizer,
            extra_mask_prob=1.0,  # Mask everything
        )

        input_ids = torch.tensor([tokenizer.CLS_ID, 5, 6, 7, tokenizer.SEP_ID])
        masked = augmenter.random_mask(input_ids)

        # All non-special tokens should be masked
        assert masked[1].item() == tokenizer.MASK_ID
        assert masked[2].item() == tokenizer.MASK_ID
        assert masked[3].item() == tokenizer.MASK_ID

        # Special tokens preserved
        assert masked[0].item() == tokenizer.CLS_ID
        assert masked[-1].item() == tokenizer.SEP_ID


class TestMLMDataset:
    """Tests for MLMDataset wrapper."""

    def test_returns_masked_data(self, sample_fasta, tokenizer):
        """Test that MLMDataset applies masking."""
        base_dataset = ProteinDataset(sample_fasta, tokenizer=tokenizer, min_length=10)
        dataset = MLMDataset(base_dataset, tokenizer, mask_prob=0.15)

        sample = dataset[0]

        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample

        # Labels should have some -100 values (unmasked positions)
        assert (sample["labels"] == -100).any()


class TestContrastiveMLMDataset:
    """Tests for ContrastiveMLMDataset."""

    def test_returns_all_fields(self, sample_fasta, tokenizer):
        """Test that all required fields are returned."""
        base_dataset = ProteinDataset(sample_fasta, tokenizer=tokenizer, min_length=10)
        dataset = ContrastiveMLMDataset(base_dataset, tokenizer)

        sample = dataset[0]

        # MLM fields
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample

        # Contrastive fields
        assert "view1_input_ids" in sample
        assert "view1_attention_mask" in sample
        assert "view2_input_ids" in sample
        assert "view2_attention_mask" in sample


class TestLengthBucketSampler:
    """Tests for length bucketing sampler."""

    def test_groups_by_length(self, sample_fasta, tokenizer):
        """Test that sampler groups sequences by length."""
        dataset = ProteinDataset(sample_fasta, tokenizer=tokenizer, min_length=10)
        sampler = LengthBucketSampler(
            dataset,
            batch_size=2,
            num_buckets=2,
            shuffle=False,
        )

        batches = list(sampler)
        assert len(batches) > 0

        # Each batch should have indices
        for batch in batches:
            assert len(batch) <= 2

    def test_sampler_length(self, sample_fasta, tokenizer):
        """Test that sampler reports correct length."""
        dataset = ProteinDataset(sample_fasta, tokenizer=tokenizer, min_length=10)
        sampler = LengthBucketSampler(dataset, batch_size=2, shuffle=False)

        # Should have ceil(len(dataset) / batch_size) batches
        expected_batches = (len(dataset) + 1) // 2
        assert len(sampler) == expected_batches


class TestCollateFunctions:
    """Tests for collate functions."""

    def test_collate_fn_pads_correctly(self, tokenizer):
        """Test that collate_fn pads sequences."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
            },
        ]

        result = collate_fn(batch, pad_token_id=0)

        assert result["input_ids"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 3)

        # Check padding
        assert result["input_ids"][1, 2].item() == 0
        assert result["attention_mask"][1, 2].item() == 0

    def test_collate_contrastive_fn(self, tokenizer):
        """Test contrastive collate function."""
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([-100, 2, -100]),
                "view1_input_ids": torch.tensor([1, 2]),
                "view1_attention_mask": torch.tensor([1, 1]),
                "view2_input_ids": torch.tensor([2, 3, 4]),
                "view2_attention_mask": torch.tensor([1, 1, 1]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([4, -100]),
                "view1_input_ids": torch.tensor([4]),
                "view1_attention_mask": torch.tensor([1]),
                "view2_input_ids": torch.tensor([5, 6]),
                "view2_attention_mask": torch.tensor([1, 1]),
            },
        ]

        result = collate_contrastive_fn(batch, pad_token_id=0)

        assert "input_ids" in result
        assert "view1_input_ids" in result
        assert "view2_input_ids" in result

        # Check shapes match batch size
        assert result["input_ids"].shape[0] == 2
        assert result["view1_input_ids"].shape[0] == 2


class TestCreateDataloader:
    """Tests for create_dataloader factory."""

    def test_creates_dataloader(self, sample_fasta, tokenizer):
        """Test that create_dataloader returns a working DataLoader."""
        dataset = ProteinDataset(sample_fasta, tokenizer=tokenizer, min_length=10)

        dataloader = create_dataloader(
            dataset,
            batch_size=2,
            tokenizer=tokenizer,
            use_bucketing=True,
            shuffle=False,
            num_workers=0,  # For testing
        )

        # Should be able to iterate
        batch = next(iter(dataloader))

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].shape[0] == 2

    def test_contrastive_dataloader(self, sample_fasta, tokenizer):
        """Test contrastive dataloader."""
        base_dataset = ProteinDataset(sample_fasta, tokenizer=tokenizer, min_length=10)
        dataset = ContrastiveMLMDataset(base_dataset, tokenizer)

        dataloader = create_dataloader(
            dataset,
            batch_size=2,
            tokenizer=tokenizer,
            use_bucketing=False,  # Can't use bucketing with wrapper
            contrastive=True,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(dataloader))

        assert "view1_input_ids" in batch
        assert "view2_input_ids" in batch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
