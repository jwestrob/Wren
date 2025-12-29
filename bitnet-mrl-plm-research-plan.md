# BitNet-MRL Protein Language Model Research Plan

**Project Codename:** TBD  
**Author:** Jacob (with Claude)  
**Created:** December 28, 2025  
**Status:** Planning → Phase 1a

---

## Executive Summary

This project aims to develop a **small, efficient protein language model** that matches or exceeds larger models in embedding quality for protein retrieval and analysis tasks. We combine several recent advances from the LLM efficiency literature that have not yet been applied to protein models:

1. **BitNet b1.58**: Ternary quantization-aware training (1.58 bits per weight)
2. **Matryoshka Representation Learning (MRL)**: Flexible embedding dimensions
3. **Differential Attention**: Improved long-context modeling via attention noise cancellation
4. **μP (Maximal Update Parameterization)**: Hyperparameter transfer across scales
5. **Modern learning rate schedules**: WSD or Linear Decay-to-Zero

The combination is **genuinely novel**—no one has applied BitNet to protein models, and only one toy example of MRL for proteins exists. The compound architecture stacks 6 independent innovations.

**Target outcome:** A 300M parameter model (with 2x hidden dim, ~600M effective capacity) that:
- Matches ESM-2 650M quality on standard benchmarks
- Fits in <200MB memory (quantized backbone)
- Provides flexible 64-1024 dim embeddings for multi-stage retrieval
- Handles long sequences (10K+ residues) for giant Omnitrophota proteins

---

## Background and Motivation

### The Problem

Jacob works with **giant proteins from Omnitrophota bacteria** (30,000-85,000 amino acids). Current protein language models like ESM-2 struggle with:
- Memory constraints at these sequence lengths
- Fixed embedding dimensions (1280 for ESM-2 650M) that are overkill for initial screening
- Large model sizes requiring expensive GPU infrastructure

### The Opportunity

Recent LLM research has produced techniques that dramatically improve efficiency:
- **Phi-series models**: 2-3B params matching GPT-4 on many tasks via data curation and training innovations
- **BitNet b1.58**: Ternary weights matching FP16 performance at 10x memory reduction
- **Small model revolution**: 2B LLMs running on phones

These techniques have **not been systematically applied to protein language models**.

### Why This Combination?

| Technique | What it compresses | Benefit |
|-----------|-------------------|---------|
| BitNet | Weight precision (16-bit → 1.58-bit) | 10x memory reduction |
| MRL | Embedding dimension (1024 → 64) | 16x retrieval speedup |
| DiffAttention | Attention noise | Better long-context |
| Layer Sharing | Unique parameters | Further compression |

These compress **orthogonal axes**—they stack multiplicatively, not redundantly.

---

## Technical Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    BitNet-MRL-DiffAttn pLM                  │
├─────────────────────────────────────────────────────────────┤
│  Embedding Layer          │  FP16 (required for MRL)        │
├─────────────────────────────────────────────────────────────┤
│  Position Encoding        │  RoPE + NTK scaling             │
├─────────────────────────────────────────────────────────────┤
│  Attention Mechanism      │  Differential Attention          │
│    - Q projection         │    └─ BitLinear (ternary)       │
│    - K projection         │    └─ BitLinear (ternary)       │
│    - V projection         │    └─ BitLinear (ternary)       │
│    - Output projection    │    └─ BitLinear (ternary)       │
├─────────────────────────────────────────────────────────────┤
│  Feed-Forward Network     │  SwiGLU activation              │
│    - Gate projection      │    └─ BitLinear (ternary)       │
│    - Up projection        │    └─ BitLinear (ternary)       │
│    - Down projection      │    └─ BitLinear (ternary)       │
├─────────────────────────────────────────────────────────────┤
│  Normalization            │  RMSNorm (pre-norm style)       │
├─────────────────────────────────────────────────────────────┤
│  Output Projection        │  FP16 (required for MRL)        │
└─────────────────────────────────────────────────────────────┘
```

### BitNet b1.58 Specifics

**Weight quantization:**
```python
# Ternary quantization: weights ∈ {-1, 0, +1}
W_ternary = sign(W) * (|W| > threshold)  # Simplified
# Average 1.58 bits per weight (log2(3) = 1.58)
```

**Training mechanics:**
- Maintain FP16 "shadow weights" for optimizer updates
- Quantize to ternary only in forward pass
- Use Straight-Through Estimator (STE) for gradients
- Activations quantized to int8

**Critical design choices:**
- **Embedding layer stays FP16** — required for semantic discrimination
- **Output projection stays FP16** — required for MRL loss computation
- **Double hidden dimensions** — compensates for ternary capacity reduction

### Differential Attention

```python
def differential_attention(Q, K, V, λ):
    # Split Q and K into two groups
    Q1, Q2 = split(Q, dim=-1)
    K1, K2 = split(K, dim=-1)
    
    # Compute two attention maps
    A1 = softmax(Q1 @ K1.T / sqrt(d))
    A2 = softmax(Q2 @ K2.T / sqrt(d))
    
    # Subtract to cancel common-mode noise
    A_diff = A1 - λ * A2
    
    return A_diff @ V
```

**Benefits for proteins:**
- Addresses "lost-in-the-middle" problem for long sequences
- Better at preserving information across 30K+ residue proteins
- λ is learnable per-head

### Matryoshka Representation Learning

**Loss function:**
```python
def mrl_loss(embeddings, labels, dims=[64, 128, 256, 512, 1024]):
    total_loss = 0
    for d in dims:
        truncated = embeddings[:, :d]
        total_loss += contrastive_loss(truncated, labels)
    return total_loss / len(dims)
```

**Deployment flexibility:**
- Use 64-128 dims for rapid database screening
- Use full 1024 dims for detailed analysis
- Same model, just truncate embeddings

### Position Encoding: RoPE + NTK Extension

**Base:** Rotary Position Embeddings (standard in ESM-2)

**Extension for long context:**
```python
# NTK-aware interpolation for context extension
# Train at 2048, extend to 16K+ at inference
base_freq = 10000
scale = target_length / training_length
adjusted_freq = base_freq * (scale ** (dim / (dim - 2)))
```

---

## Training Configuration

### Model Sizes

| Config | Params (nominal) | Hidden Dim | Layers | Heads | Effective Capacity |
|--------|-----------------|------------|--------|-------|-------------------|
| Tiny | 50M | 1024 (2x512) | 12 | 16 | ~25M FP16 equiv |
| Small | 150M | 1536 (2x768) | 18 | 24 | ~75M FP16 equiv |
| Base | 300M | 2048 (2x1024) | 24 | 32 | ~150M FP16 equiv |

*Hidden dims are 2x standard to compensate for ternary capacity loss*

### Training Hyperparameters

**From μP (tune on Tiny, transfer to Base):**
- Learning rate: Tune on 50M model, scale by width ratio
- Initialization: Scale by 1/sqrt(width)
- Output logit scaling: Divide by width

**Learning rate schedule (WSD):**
```
Warmup: 2000 steps, linear ramp to peak LR
Stable: Train at peak LR for majority of steps  
Decay: Cosine decay in final 20% of training
```

**Alternative (Linear Decay-to-Zero):**
```
Warmup: 2000 steps
Linear decay: From peak LR to 0 over remaining steps
```

### Dataset

**Primary:** UniRef50 (clustered at 50% identity)
- ~45M representative sequences
- Filter to max length 2048 for Phase 1
- Extend to longer sequences in Phase 3

**Preprocessing:**
- Standard amino acid tokenization (20 tokens + special tokens)
- 15% masking probability for MLM

### Loss Function

```python
total_loss = (
    mlm_loss(predictions, targets) +           # Primary objective
    mrl_weight * mrl_loss(embeddings, dims) +  # Matryoshka
    structure_weight * structure_loss(...)     # Phase 4 only
)
```

---

## Implementation Roadmap

### Phase 1a: Minimal Viable Novelty (2-3 weeks)
**Goal:** Validate BitNet + MRL works at all for proteins

**Architecture:**
- 50M parameter BitNet (2x hidden = 1024)
- Standard multi-head attention (not DiffAttn yet)
- 12 layers
- MRL dims: [64, 128, 256, 512]

**Training:**
- UniRef50 subset (~10M sequences)
- ~10B tokens
- Standard cosine LR schedule

**Evaluation:**
- Perplexity on held-out sequences
- TAPE secondary structure prediction
- TAPE remote homology detection
- Embedding quality via kNN SCOP classification

**Success criteria:**
- Perplexity within 1.5x of FP16 baseline
- TAPE scores within 5% of FP16 baseline
- MRL truncation maintains >90% performance at 128 dims

**Compute:** ~1 week on single 4090

---

### Phase 1b: Add Differential Attention (1-2 weeks)
**Goal:** Integrate DiffAttn for long-context improvement

**Changes from 1a:**
- Replace MultiHeadAttention with DifferentialAttention
- BitLinear for all Q/K/V/O projections
- Learnable λ per head

**Additional evaluation:**
- Long-sequence perplexity (2K+ residues)
- Attention pattern analysis (do heads learn structural contacts?)

**Success criteria:**
- No degradation on short sequences
- Improved perplexity on long sequences vs 1a

---

### Phase 2: Training Optimizations (1 week)
**Goal:** Improve training efficiency and final quality

**Changes:**
- Implement μP initialization and scaling
- Tune hyperparameters on 10M parameter model
- Transfer to 50M model
- Compare WSD vs D2Z schedules

**Experiments:**
1. μP vs standard initialization (same compute budget)
2. WSD vs D2Z vs cosine (final loss comparison)

**Success criteria:**
- μP transfer matches direct tuning within 2%
- Better final loss than Phase 1b with same compute

---

### Phase 3: Context Extension (1 week)
**Goal:** Enable giant protein processing

**Changes:**
- Implement NTK-aware RoPE scaling
- Fine-tune on longer sequences (4K-8K residues)

**Evaluation:**
- Perplexity on held-out long sequences
- Embedding quality on giant Omnitrophota proteins
- Domain boundary detection on multi-domain proteins

**Success criteria:**
- Graceful degradation from 2K to 8K context
- Meaningful embeddings for 30K+ residue sequences (via chunking/pooling)

---

### Phase 4: Structure Alignment (Optional, 1-2 weeks)
**Goal:** Inject structural knowledge without structure as input

**Changes:**
- Add contrastive loss between pLM embeddings and GNN embeddings
- Use ESMFold/AlphaFold structures for supervision

**Evaluation:**
- Contact prediction from attention
- Structure-sensitive downstream tasks

---

### Phase 5: Scale Up (If Phases 1-3 Succeed)
**Goal:** Train production-quality model

**Target:** 300M parameter BitNet-DiffAttn-MRL model
- 2048 hidden dim
- 24 layers
- Full UniRef50 training

**Compute:** Likely requires multi-GPU setup or cloud compute

---

## Evaluation Benchmarks

### Primary Benchmarks

| Benchmark | Task | Metric | Why |
|-----------|------|--------|-----|
| Perplexity | Language modeling | PPL | Basic sanity check |
| TAPE-SS | Secondary structure | Accuracy | Standard pLM eval |
| TAPE-Remote | Remote homology | Accuracy | Embedding quality |
| TAPE-Contact | Contact prediction | Precision@L | Structural info |
| SCOP kNN | Fold classification | Accuracy | Retrieval quality |

### MRL-Specific Evaluation

| Embedding Dim | Expected Performance | Use Case |
|--------------|---------------------|----------|
| 64 | 85-90% of full | Fast screening |
| 128 | 92-96% of full | Standard retrieval |
| 256 | 97-99% of full | Detailed analysis |
| 512+ | ~100% of full | Maximum quality |

### Efficiency Metrics

| Metric | Target |
|--------|--------|
| Model size (backbone) | <150MB at 300M params |
| Inference memory | <2GB for 2K sequence |
| Embedding throughput | >1000 seq/sec on 4090 |
| Training tokens/sec | Track for comparison |

---

## Hardware Requirements

### Development (Phases 1-3)

**Primary:** NVIDIA RTX 4090 (24GB VRAM)
- Sufficient for 50-150M parameter training
- Mixed precision with gradient checkpointing

**Secondary:** Apple M4 Max (48GB unified memory)
- Good for inference testing
- PyTorch MPS support for training (slower but works)

### Production Training (Phase 5)

**Option A:** Multi-4090 setup (2-4 GPUs)
**Option B:** Cloud A100 instances

---

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Ternary doesn't work for proteins | Medium | High | Compare to FP16 baseline early |
| MRL degrades with ternary | Low | Medium | Test independently first |
| DiffAttn incompatible with BitLinear | Low | Low | Standard attention fallback |
| Training instability | Medium | Medium | Lower LR, gradient clipping |
| μP transfer fails | Low | Low | Direct tuning fallback |

### Scientific Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Small vocab (20 AA) breaks assumptions | Medium | High | This is novel—we'll find out |
| Protein embeddings need more precision | Medium | High | Compare quantized vs FP16 carefully |
| Results don't generalize past toy scale | Medium | Medium | Progressive scaling |

### Mitigations Built Into Plan

1. **Phase 1a is minimal** — tests core hypothesis with minimum investment
2. **FP16 baselines throughout** — always know what we're comparing to
3. **Ablations at each phase** — understand contribution of each component
4. **Fail-fast evaluation** — TAPE benchmarks are quick

---

## Success Criteria

### Minimum Viable Success (Phase 1)
- [ ] BitNet pLM trains stably on proteins
- [ ] Perplexity within 2x of FP16 baseline
- [ ] MRL truncation works (128d within 90% of full)
- [ ] Published negative result if it doesn't work (still novel!)

### Strong Success (Phases 1-3)
- [ ] 50M BitNet-DiffAttn-MRL matches 50M FP16 standard attention
- [ ] Long-context performance exceeds baseline
- [ ] MRL provides meaningful quality/speed tradeoff

### Home Run (Phase 5)
- [ ] 300M BitNet-DiffAttn-MRL matches ESM-2 650M on benchmarks
- [ ] 5x memory reduction demonstrated
- [ ] Paper-worthy results
- [ ] Practical tool for giant protein analysis

---

## Key References

### BitNet
- Wang et al. (2023). "BitNet: Scaling 1-bit Transformers for Large Language Models"
- Ma et al. (2024). "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
- Nielsen et al. (2024). "BitNet b1.58 Reloaded: State-of-the-art Performance Also on Smaller Networks"

### Matryoshka
- Kusupati et al. (2022). "Matryoshka Representation Learning" (NeurIPS 2022)
- QAMA (2025). "Quantization Aware Matryoshka Adaptation" (CIKM 2025)

### Differential Attention
- Ye et al. (2024). "Differential Transformer" (ICLR 2025)

### Protein Language Models
- Lin et al. (2022). "Language models of protein sequences at the scale of evolution" (ESM-2)
- Elnaggar et al. (2021). "ProtTrans" 
- PTQ4Protein (2023). "Exploring Post-Training Quantization of Protein Language Models"

### Training Optimizations
- Yang et al. (2022). "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" (μP)
- Hu et al. (2024). "MiniCPM: Unveiling the Potential of Small Language Models" (WSD schedule)

---

## Code Repositories

**BitNet Training:**
- https://github.com/kyegomez/BitNet (PyTorch training)
- https://github.com/microsoft/BitNet (inference only)

**MRL:**
- sentence-transformers MatryoshkaLoss
- https://huggingface.co/blog/matryoshka

**Differential Attention:**
- https://github.com/microsoft/unilm (official)

**Protein Baselines:**
- https://github.com/facebookresearch/esm

---

## Notes and Open Questions

### Open Questions
1. Does the tiny protein vocabulary (20 tokens) affect ternary quantization differently than large NLP vocabularies?
2. What's the optimal MRL dimension schedule for proteins? (Powers of 2? Different for proteins?)
3. Should we use MSA information during training, or stick to single-sequence?
4. How does embedding pooling strategy (mean vs CLS) interact with MRL?

### Ideas for Future Work
- BitNet + MRL + Layer Sharing (after validating base approach)
- Multi-task training with auxiliary structure prediction heads
- Extending to nucleotide sequences (DNA/RNA language models)
- Hardware-specific optimizations for ternary inference

---

## Changelog

| Date | Changes |
|------|---------|
| 2024-12-28 | Initial plan created |

---

*This document is a living reference. Update as experiments progress.*
