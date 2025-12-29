"""Tests for TinyPLM model components."""

import pytest
import torch

from tinyplm.config import ModelConfig
from tinyplm.model.bitlinear import BitLinear
from tinyplm.model.rope import RotaryEmbedding, YaRNRotaryEmbedding, apply_rope, apply_rope_with_scale
from tinyplm.model.norm import RMSNorm
from tinyplm.model.ffn import SwiGLUFFN
from tinyplm.model.attention import MultiHeadAttention
from tinyplm.model.transformer import TransformerBlock
from tinyplm.model.plm import TinyPLM
from tinyplm.data.tokenizer import ProteinTokenizer


class TestBitLinear:
    """Tests for BitLinear layer."""

    def test_forward_shape(self):
        """Test that output shape matches expected."""
        layer = BitLinear(256, 512)
        x = torch.randn(4, 32, 256)
        out = layer(x)
        assert out.shape == (4, 32, 512)

    def test_ternary_quantization(self):
        """Test that weights are quantized to ternary."""
        layer = BitLinear(64, 64)
        w_q, scale = layer.ternary_quantize(layer.weight)

        # Check values are in {-1, 0, 1}
        unique_vals = torch.unique(w_q.detach().round())
        assert all(v in [-1, 0, 1] for v in unique_vals.tolist())

    def test_gradient_flows(self):
        """Test that gradients flow through STE."""
        layer = BitLinear(64, 64)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert layer.weight.grad is not None
        assert x.grad is not None


class TestRoPE:
    """Tests for Rotary Position Embeddings."""

    def test_cache_building(self):
        """Test that sin/cos cache is built correctly."""
        rope = RotaryEmbedding(dim=64, max_seq_len=128)
        cos, sin = rope(64)

        assert cos.shape == (64, 64)
        assert sin.shape == (64, 64)

    def test_apply_rope_shape(self):
        """Test that RoPE application preserves shape."""
        rope = RotaryEmbedding(dim=64, max_seq_len=128)
        cos, sin = rope(32)

        q = torch.randn(2, 8, 32, 64)
        k = torch.randn(2, 8, 32, 64)

        q_rot, k_rot = apply_rope(q, k, cos, sin)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_cache_extension(self):
        """Test that cache extends for longer sequences."""
        rope = RotaryEmbedding(dim=64, max_seq_len=64)
        cos, sin = rope(128)  # Longer than initial max

        assert cos.shape == (128, 64)


class TestYaRN:
    """Tests for YaRN Rotary Position Embeddings."""

    def test_yarn_forward_shape(self):
        """Test that YaRN returns correct shapes."""
        yarn = YaRNRotaryEmbedding(dim=64, max_seq_len=128, scale=1.0)
        cos, sin, attn_scale = yarn(64)

        assert cos.shape == (64, 64)
        assert sin.shape == (64, 64)
        assert isinstance(attn_scale, float)

    def test_yarn_scaling(self):
        """Test that YaRN with scale > 1 produces attention scaling."""
        yarn_scaled = YaRNRotaryEmbedding(dim=64, max_seq_len=256, scale=2.0)
        _, _, attn_scale = yarn_scaled(64)

        # Scale > 1 should produce attn_scale > 1
        assert attn_scale > 1.0

    def test_yarn_no_scaling(self):
        """Test that YaRN with scale=1 has attn_scale=1."""
        yarn = YaRNRotaryEmbedding(dim=64, max_seq_len=128, scale=1.0)
        _, _, attn_scale = yarn(64)

        assert attn_scale == 1.0

    def test_apply_rope_with_scale(self):
        """Test apply_rope_with_scale preserves shapes."""
        yarn = YaRNRotaryEmbedding(dim=64, max_seq_len=128, scale=1.0)
        cos, sin, attn_scale = yarn(32)

        q = torch.randn(2, 8, 32, 64)
        k = torch.randn(2, 8, 32, 64)

        q_rot, k_rot = apply_rope_with_scale(q, k, cos, sin, attn_scale)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_yarn_cache_extension(self):
        """Test that YaRN cache extends for longer sequences."""
        yarn = YaRNRotaryEmbedding(dim=64, max_seq_len=64, scale=1.0)
        cos, sin, _ = yarn(128)  # Longer than initial max

        assert cos.shape == (128, 64)


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_forward_shape(self):
        """Test that output shape matches input."""
        norm = RMSNorm(256)
        x = torch.randn(4, 32, 256)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization(self):
        """Test that RMS is normalized to ~1."""
        norm = RMSNorm(256)
        x = torch.randn(4, 32, 256) * 10  # Large values
        out = norm(x)

        rms = out.pow(2).mean(-1).sqrt()
        # After normalization and weighting, should be close to 1
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


class TestSwiGLUFFN:
    """Tests for SwiGLU FFN."""

    def test_forward_shape(self):
        """Test that output shape matches hidden_dim."""
        ffn = SwiGLUFFN(hidden_dim=256, ffn_dim=683, use_bitlinear=True)
        x = torch.randn(4, 32, 256)
        out = ffn(x)
        assert out.shape == x.shape

    def test_fp16_variant(self):
        """Test FP16 variant without BitLinear."""
        ffn = SwiGLUFFN(hidden_dim=256, ffn_dim=683, use_bitlinear=False)
        x = torch.randn(4, 32, 256)
        out = ffn(x)
        assert out.shape == x.shape


class TestMultiHeadAttention:
    """Tests for Multi-Head Attention."""

    def test_forward_shape(self):
        """Test that output shape matches input."""
        attn = MultiHeadAttention(
            hidden_dim=256, num_heads=8, use_bitlinear=True
        )
        x = torch.randn(2, 32, 256)
        out = attn(x)
        assert out.shape == x.shape

    def test_with_attention_mask(self):
        """Test attention with padding mask."""
        attn = MultiHeadAttention(hidden_dim=256, num_heads=8)
        x = torch.randn(2, 32, 256)
        mask = torch.ones(2, 32)
        mask[:, 16:] = 0  # Mask second half

        out = attn(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_output_attentions(self):
        """Test that attention weights can be extracted."""
        attn = MultiHeadAttention(hidden_dim=256, num_heads=8)
        x = torch.randn(2, 32, 256)

        out = attn(x, output_attentions=True)

        # Should return AttentionOutput dataclass
        assert hasattr(out, 'hidden_states')
        assert hasattr(out, 'attention_weights')
        assert out.hidden_states.shape == x.shape
        assert out.attention_weights.shape == (2, 8, 32, 32)  # [batch, heads, seq, seq]

    def test_yarn_mode(self):
        """Test attention with YaRN position encoding."""
        attn = MultiHeadAttention(
            hidden_dim=256, num_heads=8, use_yarn=True, rope_scale=1.0
        )
        x = torch.randn(2, 32, 256)
        out = attn(x)
        assert out.shape == x.shape

    def test_rope_mode(self):
        """Test attention with standard RoPE (no YaRN)."""
        attn = MultiHeadAttention(
            hidden_dim=256, num_heads=8, use_yarn=False
        )
        x = torch.randn(2, 32, 256)
        out = attn(x)
        assert out.shape == x.shape


class TestTransformerBlock:
    """Tests for Transformer Block."""

    def test_forward_shape(self):
        """Test that output shape matches input."""
        block = TransformerBlock(
            hidden_dim=256, num_heads=8, ffn_dim=683, use_bitlinear=True
        )
        x = torch.randn(2, 32, 256)
        out = block(x)
        assert out.shape == x.shape

    def test_with_gradient_checkpointing(self):
        """Test gradient checkpointing."""
        block = TransformerBlock(
            hidden_dim=256,
            num_heads=8,
            ffn_dim=683,
            gradient_checkpointing=True,
        )
        block.train()  # Checkpointing only active in training mode

        x = torch.randn(2, 32, 256, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None


class TestTinyPLM:
    """Tests for full TinyPLM model."""

    @pytest.fixture
    def small_config(self):
        """Create small config for testing."""
        return ModelConfig(
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            ffn_dim=341,
            vocab_size=25,
            max_seq_len=64,
            mrl_dims=[32, 64, 128],
            embedding_dim=128,
            use_bitlinear=True,
            gradient_checkpointing=False,
        )

    def test_forward_shape(self, small_config):
        """Test that all output shapes are correct."""
        model = TinyPLM(small_config)
        input_ids = torch.randint(0, 25, (2, 32))

        outputs = model(input_ids)

        assert outputs["logits"].shape == (2, 32, 25)
        assert outputs["embeddings"].shape == (2, 128)
        assert outputs["last_hidden_state"].shape == (2, 32, 128)

    def test_with_attention_mask(self, small_config):
        """Test forward with attention mask."""
        model = TinyPLM(small_config)
        input_ids = torch.randint(0, 25, (2, 32))
        attention_mask = torch.ones(2, 32)
        attention_mask[:, 16:] = 0

        outputs = model(input_ids, attention_mask=attention_mask)
        assert outputs["logits"].shape == (2, 32, 25)

    def test_get_embeddings_mrl(self, small_config):
        """Test MRL truncation of embeddings."""
        model = TinyPLM(small_config)
        model.eval()
        input_ids = torch.randint(0, 25, (2, 32))

        # Full embeddings
        full_emb = model.get_embeddings(input_ids)
        assert full_emb.shape == (2, 128)

        # Truncated embeddings (MRL)
        trunc_emb = model.get_embeddings(input_ids, dim=32)
        assert trunc_emb.shape == (2, 32)

        # Check normalization
        assert torch.allclose(
            trunc_emb.norm(dim=-1), torch.ones(2), atol=1e-5
        )

    def test_mlm_loss(self, small_config):
        """Test MLM loss computation."""
        model = TinyPLM(small_config)
        input_ids = torch.randint(0, 25, (2, 32))
        labels = input_ids.clone()
        labels[:, ::2] = -100  # Ignore even positions

        outputs = model(input_ids)
        loss = model.compute_mlm_loss(outputs["logits"], labels)

        assert loss.shape == ()
        assert loss.item() > 0

    def test_parameter_count(self, small_config):
        """Test parameter counting."""
        model = TinyPLM(small_config)
        total = model.num_parameters
        trainable = model.num_trainable_parameters

        assert total > 0
        assert total == trainable  # All params should be trainable


class TestTokenizer:
    """Tests for protein tokenizer."""

    def test_encode_decode(self):
        """Test round-trip encoding/decoding."""
        tokenizer = ProteinTokenizer(add_special_tokens=False)
        sequence = "MKTAYIAKQRQISFVKSHFSRQLEER"

        encoded = tokenizer.encode(sequence)
        decoded = tokenizer.decode(encoded)

        assert decoded == sequence

    def test_special_tokens(self):
        """Test that special tokens are added."""
        tokenizer = ProteinTokenizer(add_special_tokens=True)
        sequence = "MKTAY"

        encoded = tokenizer.encode(sequence)

        assert encoded[0] == tokenizer.CLS_ID
        assert encoded[-1] == tokenizer.SEP_ID
        assert len(encoded) == len(sequence) + 2

    def test_nonstandard_amino_acids(self):
        """Test handling of non-standard amino acids."""
        tokenizer = ProteinTokenizer(add_special_tokens=False)

        # X should map to UNK
        encoded = tokenizer.encode("AXA")
        assert encoded[1] == tokenizer.UNK_ID

        # B should map to D
        encoded = tokenizer.encode("ABA")
        decoded = tokenizer.decode(encoded)
        assert decoded == "ADA"

    def test_batch_encode(self):
        """Test batch encoding with padding."""
        tokenizer = ProteinTokenizer()
        sequences = ["MKT", "MKTAYIAK"]

        batch = tokenizer.batch_encode(sequences, padding=True)

        assert len(batch["input_ids"]) == 2
        assert len(batch["attention_mask"]) == 2
        # Should be padded to same length
        assert len(batch["input_ids"][0]) == len(batch["input_ids"][1])

    def test_truncation(self):
        """Test sequence truncation."""
        tokenizer = ProteinTokenizer(add_special_tokens=True)
        sequence = "MKTAYIAKQRQISFVK"

        encoded = tokenizer.encode(sequence, max_length=10, truncation=True)

        assert len(encoded) == 10
        assert encoded[-1] == tokenizer.SEP_ID  # SEP preserved


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
