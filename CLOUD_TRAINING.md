# Cloud Training Guide

## Quick Start (Lambda Labs / Vast.ai)

### 1. Rent GPU(s)
- **Recommended:** 1-4x A100-80GB
- **Budget option:** 1x A100-40GB or A10
- **Cost estimate:** ~$1-2/hr per A100

### 2. Setup Instance

```bash
# Clone repo
git clone <your-repo-url> wren
cd wren

# Install dependencies
pip install -e .
pip install wandb  # optional, for logging

# Install mmseqs2 for data preprocessing
conda install -c bioconda mmseqs2
```

### 3. Download Data

```bash
# Takes ~30 min to download, ~1-2 hours to cluster
chmod +x scripts/download_uniref50.sh
./scripts/download_uniref50.sh
```

### 4. Train

**Single GPU:**
```bash
python scripts/train.py \
    --config configs/uniref50_a100.yaml \
    --train-data data/uniref50/train.fasta \
    --val-data data/uniref50/val.fasta
```

**Multi-GPU (4x A100):**
```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/uniref50_a100.yaml \
    --train-data data/uniref50/train.fasta \
    --val-data data/uniref50/val.fasta
```

## Expected Training Time

| Setup | Steps/sec | 100K steps | Cost (~$2/hr) |
|-------|-----------|------------|---------------|
| 1x A100-80GB | ~3-5 | ~6-8 hours | ~$12-16 |
| 4x A100-80GB | ~10-15 | ~2-3 hours | ~$16-24 |
| 1x MPS (M4 Max) | ~0.7 | ~40 hours | $0 |

## Checkpointing for Spot Instances

If using spot/preemptible instances, checkpoints save every 10K steps by default.
Resume training:

```bash
python scripts/train.py \
    --config configs/uniref50_a100.yaml \
    --train-data data/uniref50/train.fasta \
    --val-data data/uniref50/val.fasta \
    --resume checkpoints/uniref50_bitnet/step_10000.pt
```

## Recommended Providers

1. **Lambda Labs** - Simple, good availability, ~$1.10/hr for A100
2. **Vast.ai** - Cheapest, variable quality, ~$0.80-1.50/hr
3. **RunPod** - Good middle ground, ~$1.00/hr
4. **Google Cloud** - Reliable but expensive, ~$3/hr for A100

## After Training

Download checkpoints:
```bash
# From your local machine
scp -r user@instance:/path/to/wren/checkpoints ./
```

Or sync to cloud storage during training (edit train.py to add S3/GCS sync).
