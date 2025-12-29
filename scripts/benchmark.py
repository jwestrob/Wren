#!/usr/bin/env python3
"""Benchmark TinyPLM vs FP16 baseline vs ESM-2.

Compares:
- Memory usage (model size, peak memory)
- Inference speed (tokens/sec)
- Training throughput (samples/sec)
- Embedding quality (if pretrained weights available)

Usage:
    python scripts/benchmark.py --model tinyplm --checkpoint checkpoints/swissprot_50m/best.pt
    python scripts/benchmark.py --model fp16-baseline --config configs/swissprot_50m.yaml
    python scripts/benchmark.py --model esm2 --esm-model esm2_t6_8M_UR50D
"""

import argparse
import gc
import time
from pathlib import Path

import torch
import torch.nn as nn

from tinyplm.config import Config, ModelConfig
from tinyplm.model import TinyPLM
from tinyplm.data import ProteinTokenizer


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def get_param_count(model: nn.Module) -> dict:
    """Get parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def benchmark_inference(model: nn.Module, input_ids: torch.Tensor,
                        attention_mask: torch.Tensor, num_runs: int = 50,
                        warmup: int = 10) -> dict:
    """Benchmark inference speed."""
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids, attention_mask=attention_mask)

    # Sync before timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_ids, attention_mask=attention_mask)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

            times.append(time.perf_counter() - start)

    batch_size, seq_len = input_ids.shape
    total_tokens = batch_size * seq_len

    return {
        "mean_time_ms": sum(times) / len(times) * 1000,
        "std_time_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5 * 1000,
        "tokens_per_sec": total_tokens / (sum(times) / len(times)),
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def benchmark_training_step(model: nn.Module, input_ids: torch.Tensor,
                            attention_mask: torch.Tensor, labels: torch.Tensor,
                            num_runs: int = 20, warmup: int = 5) -> dict:
    """Benchmark training throughput."""
    model.train()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = nn.functional.cross_entropy(
            outputs["logits"].view(-1, outputs["logits"].size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        loss.backward()
        optimizer.step()

    # Sync
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        optimizer.zero_grad()

        start = time.perf_counter()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = nn.functional.cross_entropy(
            outputs["logits"].view(-1, outputs["logits"].size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        times.append(time.perf_counter() - start)

    batch_size = input_ids.shape[0]

    return {
        "mean_time_ms": sum(times) / len(times) * 1000,
        "std_time_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5 * 1000,
        "samples_per_sec": batch_size / (sum(times) / len(times)),
        "batch_size": batch_size,
    }


def get_peak_memory_mb(device: torch.device) -> float:
    """Get peak memory usage in MB."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    elif device.type == "mps":
        # MPS doesn't have direct memory tracking, estimate from model
        return -1  # Not available
    return -1


def load_tinyplm(checkpoint_path: Path = None, config_path: Path = None,
                 use_bitlinear: bool = True, device: str = "mps") -> TinyPLM:
    """Load TinyPLM model."""
    if config_path:
        config = Config.from_yaml(config_path)
        config.model.use_bitlinear = use_bitlinear
    else:
        # Default config matching our training
        config = Config()
        config.model = ModelConfig(
            hidden_dim=512,
            num_layers=8,
            num_heads=8,
            head_dim=64,
            ffn_dim=1365,
            vocab_size=25,
            max_seq_len=2048,
            use_bitlinear=use_bitlinear,
        )

    model = TinyPLM(config.model)

    if checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")

    return model.to(device)


def load_esm2(model_name: str = "esm2_t6_8M_UR50D", device: str = "mps"):
    """Load ESM-2 model from fair-esm."""
    try:
        import esm
    except ImportError:
        print("ESM not installed. Run: pip install fair-esm")
        return None, None

    # Available models (smallest to largest):
    # esm2_t6_8M_UR50D    - 6 layers, 8M params
    # esm2_t12_35M_UR50D  - 12 layers, 35M params
    # esm2_t30_150M_UR50D - 30 layers, 150M params
    # esm2_t33_650M_UR50D - 33 layers, 650M params

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)
    return model, alphabet


def create_dummy_batch(tokenizer: ProteinTokenizer, batch_size: int = 8,
                       seq_len: int = 512, device: str = "mps") -> tuple:
    """Create dummy batch for benchmarking."""
    # Random protein-like sequences
    import random
    aa = "ACDEFGHIKLMNPQRSTVWY"
    sequences = ["".join(random.choices(aa, k=seq_len)) for _ in range(batch_size)]

    encoded = tokenizer.batch_encode(sequences, max_length=seq_len, padding=True)
    input_ids = torch.tensor(encoded, dtype=torch.long)
    attention_mask = (input_ids != tokenizer.PAD_ID).long()

    # Create labels (MLM style - mask 15%)
    labels = input_ids.clone()
    mask_prob = 0.15
    mask = torch.rand(input_ids.shape) < mask_prob
    mask[:, 0] = False  # Don't mask CLS
    labels[~mask] = -100

    return input_ids, attention_mask, labels


def run_benchmark(model_type: str, checkpoint: Path = None, config: Path = None,
                  esm_model: str = "esm2_t6_8M_UR50D", device: str = "mps",
                  batch_size: int = 8, seq_len: int = 512):
    """Run full benchmark suite."""

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_type}")
    print(f"{'='*60}\n")

    tokenizer = ProteinTokenizer()

    # Load model
    if model_type == "tinyplm":
        model = load_tinyplm(checkpoint, config, use_bitlinear=True, device=device)
        model_name = "TinyPLM (BitNet)"
    elif model_type == "fp16-baseline":
        model = load_tinyplm(checkpoint, config, use_bitlinear=False, device=device)
        model_name = "TinyPLM (FP16)"
    elif model_type == "esm2":
        model, alphabet = load_esm2(esm_model, device)
        if model is None:
            return
        model_name = f"ESM-2 ({esm_model})"
        # ESM uses different tokenization - skip detailed benchmarks for now
        print(f"Model: {model_name}")
        params = get_param_count(model)
        print(f"Parameters: {params['total']:,}")
        print(f"Model size: {get_model_size_mb(model):.2f} MB")
        print("\nNote: ESM-2 uses different tokenization. For fair comparison,")
        print("use the same eval tasks (TAPE) rather than raw speed benchmarks.")
        return
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Model: {model_name}")
    print(f"Device: {device}")

    # Model stats
    params = get_param_count(model)
    print(f"\n--- Model Stats ---")
    print(f"Parameters: {params['total']:,}")
    print(f"Trainable: {params['trainable']:,}")
    print(f"Model size: {get_model_size_mb(model):.2f} MB")

    # Create dummy data
    input_ids, attention_mask, labels = create_dummy_batch(
        tokenizer, batch_size=batch_size, seq_len=seq_len, device=device
    )

    # Inference benchmark
    print(f"\n--- Inference Benchmark (batch={batch_size}, seq={seq_len}) ---")
    gc.collect()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    inf_results = benchmark_inference(model, input_ids, attention_mask)
    print(f"Mean time: {inf_results['mean_time_ms']:.2f} +/- {inf_results['std_time_ms']:.2f} ms")
    print(f"Throughput: {inf_results['tokens_per_sec']:.0f} tokens/sec")

    # Training benchmark
    print(f"\n--- Training Benchmark (batch={batch_size}, seq={seq_len}) ---")
    gc.collect()

    train_results = benchmark_training_step(model, input_ids, attention_mask, labels)
    print(f"Mean step time: {train_results['mean_time_ms']:.2f} +/- {train_results['std_time_ms']:.2f} ms")
    print(f"Throughput: {train_results['samples_per_sec']:.2f} samples/sec")

    # Memory (CUDA only)
    if device == "cuda":
        peak_mem = get_peak_memory_mb(torch.device(device))
        print(f"\n--- Memory ---")
        print(f"Peak memory: {peak_mem:.2f} MB")

    return {
        "model": model_name,
        "params": params,
        "model_size_mb": get_model_size_mb(model),
        "inference": inf_results,
        "training": train_results,
    }


def compare_all(config: Path, checkpoint: Path = None, device: str = "mps",
                batch_size: int = 8, seq_len: int = 512):
    """Compare TinyPLM (BitNet) vs FP16 baseline."""

    print("\n" + "="*60)
    print("COMPARISON: BitNet vs FP16 Baseline")
    print("="*60)

    results = {}

    # BitNet
    results["bitnet"] = run_benchmark(
        "tinyplm", checkpoint=checkpoint, config=config,
        device=device, batch_size=batch_size, seq_len=seq_len
    )

    # FP16 baseline
    results["fp16"] = run_benchmark(
        "fp16-baseline", config=config,
        device=device, batch_size=batch_size, seq_len=seq_len
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if results["bitnet"] and results["fp16"]:
        bitnet = results["bitnet"]
        fp16 = results["fp16"]

        print(f"\n{'Metric':<30} {'BitNet':>15} {'FP16':>15} {'Ratio':>10}")
        print("-" * 70)

        print(f"{'Parameters':.<30} {bitnet['params']['total']:>15,} {fp16['params']['total']:>15,} {'1.00x':>10}")
        print(f"{'Model size (MB)':.<30} {bitnet['model_size_mb']:>15.2f} {fp16['model_size_mb']:>15.2f} {bitnet['model_size_mb']/fp16['model_size_mb']:>10.2f}x")

        inf_ratio = fp16['inference']['mean_time_ms'] / bitnet['inference']['mean_time_ms']
        print(f"{'Inference time (ms)':.<30} {bitnet['inference']['mean_time_ms']:>15.2f} {fp16['inference']['mean_time_ms']:>15.2f} {inf_ratio:>10.2f}x")

        train_ratio = fp16['training']['mean_time_ms'] / bitnet['training']['mean_time_ms']
        print(f"{'Training step (ms)':.<30} {bitnet['training']['mean_time_ms']:>15.2f} {fp16['training']['mean_time_ms']:>15.2f} {train_ratio:>10.2f}x")

        print("\nNote: Ratios > 1.0 mean BitNet is faster/smaller")
        print("Note: Real BitNet speedups require custom CUDA/Triton kernels")
        print("      Current implementation uses 'fake quantization' for training")


def main():
    parser = argparse.ArgumentParser(description="Benchmark TinyPLM models")
    parser.add_argument("--model", type=str, default="tinyplm",
                        choices=["tinyplm", "fp16-baseline", "esm2", "compare"],
                        help="Model to benchmark")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=Path, default=Path("configs/swissprot_50m.yaml"),
                        help="Path to config file")
    parser.add_argument("--esm-model", type=str, default="esm2_t6_8M_UR50D",
                        help="ESM-2 model name")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device (cuda, mps, cpu)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for benchmarking")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length for benchmarking")

    args = parser.parse_args()

    if args.model == "compare":
        compare_all(
            config=args.config,
            checkpoint=args.checkpoint,
            device=args.device,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )
    else:
        run_benchmark(
            model_type=args.model,
            checkpoint=args.checkpoint,
            config=args.config,
            esm_model=args.esm_model,
            device=args.device,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )


if __name__ == "__main__":
    main()
