#!/bin/bash
# Run FP16 baseline for comparison with BitNet
#
# This trains an identical architecture but with standard FP16 linear layers
# instead of BitLinear (ternary quantization).
#
# Usage:
#   ./scripts/run_fp16_baseline.sh          # Full 50K steps
#   ./scripts/run_fp16_baseline.sh --quick  # Quick 5K step test

set -e

cd "$(dirname "$0")/.."

if [ "$1" == "--quick" ]; then
    echo "Running QUICK FP16 baseline (5K steps)..."

    # Create temp config with FP16
    sed 's/use_bitlinear: true/use_bitlinear: false/' \
        configs/quick_comparison.yaml > /tmp/quick_fp16.yaml
    sed -i '' 's|checkpoints/quick_test|checkpoints/quick_fp16|' /tmp/quick_fp16.yaml

    python scripts/train.py \
        --config /tmp/quick_fp16.yaml \
        --train-data data/swissprot/train.fasta \
        --val-data data/swissprot/val.fasta \
        --no-wandb
else
    echo "Running FULL FP16 baseline (50K steps)..."
    echo "Checkpoint dir: checkpoints/swissprot_50m_fp16"
    echo ""

    python scripts/train.py \
        --config configs/swissprot_50m_fp16.yaml \
        --train-data data/swissprot/train.fasta \
        --val-data data/swissprot/val.fasta \
        --no-wandb
fi
