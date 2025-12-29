#!/bin/bash
# One-command setup for Lambda Labs / cloud training
#
# Usage:
#   git clone <repo> tinyplm && cd tinyplm && ./setup_cluster.sh
#
# This will:
#   1. Install Python dependencies
#   2. Install mmseqs2 for data preprocessing
#   3. Download and preprocess UniRef50
#   4. Print next steps

set -e

echo "=============================================="
echo "TinyPLM Cluster Setup"
echo "=============================================="

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Make sure you're on a GPU node!"
fi

echo ""
echo "=== Installing Python dependencies ==="
pip install -e ".[dev]"

echo ""
echo "=== Installing mmseqs2 ==="
if command -v conda &> /dev/null; then
    conda install -y -c conda-forge -c bioconda mmseqs2
elif command -v mamba &> /dev/null; then
    mamba install -y -c conda-forge -c bioconda mmseqs2
else
    echo "No conda/mamba found. Installing mmseqs2 from binary..."
    mkdir -p ~/bin
    wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
    tar xzf mmseqs-linux-avx2.tar.gz
    mv mmseqs/bin/mmseqs ~/bin/
    rm -rf mmseqs mmseqs-linux-avx2.tar.gz
    export PATH="$HOME/bin:$PATH"
    echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
fi

echo ""
echo "=== Downloading UniRef50 (this takes ~30 min) ==="
./scripts/download_uniref50.sh

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To start training on B200:"
echo "  python scripts/train.py --config configs/uniref50_b200.yaml \\"
echo "    --train-data data/uniref50/train.fasta \\"
echo "    --val-data data/uniref50/val.fasta"
echo ""
echo "To start training on multi-GPU A100:"
echo "  torchrun --nproc_per_node=\$(nvidia-smi -L | wc -l) scripts/train.py \\"
echo "    --config configs/uniref50_a100.yaml \\"
echo "    --train-data data/uniref50/train.fasta \\"
echo "    --val-data data/uniref50/val.fasta"
echo ""
