#!/bin/bash
# Download and preprocess UniRef50 for training
#
# Requirements:
#   - ~15GB disk space for download
#   - ~50GB disk space after decompression
#   - mmseqs2 for clustering (optional but recommended)
#
# Usage:
#   ./scripts/download_uniref50.sh [--no-cluster]

set -e

DATA_DIR="data/uniref50"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== Downloading UniRef50 ==="
# UniRef50 FASTA from UniProt
URL="https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"

if [ ! -f "uniref50.fasta.gz" ] && [ ! -f "uniref50.fasta" ]; then
    echo "Downloading from $URL..."
    wget -c "$URL"
fi

if [ ! -f "uniref50.fasta" ]; then
    echo "Decompressing..."
    gunzip -k uniref50.fasta.gz
fi

# Count sequences
echo "Counting sequences..."
SEQ_COUNT=$(grep -c "^>" uniref50.fasta)
echo "Total sequences: $SEQ_COUNT"

# Optional: cluster at 99% identity to remove near-duplicates
if [ "$1" != "--no-cluster" ]; then
    if command -v mmseqs &> /dev/null; then
        echo ""
        echo "=== Clustering at 99% identity with MMseqs2 ==="

        if [ ! -f "uniref50_clustered.fasta" ]; then
            mmseqs easy-cluster uniref50.fasta uniref50_clust tmp \
                --min-seq-id 0.99 \
                -c 0.8 \
                --cov-mode 1 \
                --threads $(nproc)

            mv uniref50_clust_rep_seq.fasta uniref50_clustered.fasta
            rm -rf tmp uniref50_clust_*

            CLUST_COUNT=$(grep -c "^>" uniref50_clustered.fasta)
            echo "Sequences after clustering: $CLUST_COUNT"
        fi

        SOURCE_FASTA="uniref50_clustered.fasta"
    else
        echo "mmseqs2 not found, skipping clustering"
        echo "Install with: conda install -c bioconda mmseqs2"
        SOURCE_FASTA="uniref50.fasta"
    fi
else
    SOURCE_FASTA="uniref50.fasta"
fi

echo ""
echo "=== Filtering by length (50-2048 residues) ==="
python3 << EOF
from pathlib import Path

source = "$SOURCE_FASTA"
min_len, max_len = 50, 2048

kept = 0
skipped = 0

with open(source) as f_in, open("filtered.fasta", "w") as f_out:
    header = None
    seq = []

    for line in f_in:
        if line.startswith(">"):
            if header and min_len <= len("".join(seq)) <= max_len:
                f_out.write(header)
                f_out.write("".join(seq) + "\n")
                kept += 1
            else:
                skipped += 1 if header else 0
            header = line
            seq = []
        else:
            seq.append(line.strip())

    # Last sequence
    if header and min_len <= len("".join(seq)) <= max_len:
        f_out.write(header)
        f_out.write("".join(seq) + "\n")
        kept += 1

print(f"Kept: {kept:,} sequences")
print(f"Filtered out: {skipped:,} sequences")
EOF

echo ""
echo "=== Creating train/val split (95/5) ==="
python3 << EOF
import random
random.seed(42)

sequences = []
with open("filtered.fasta") as f:
    header = None
    seq = []
    for line in f:
        if line.startswith(">"):
            if header:
                sequences.append((header, "".join(seq)))
            header = line
            seq = []
        else:
            seq.append(line.strip())
    if header:
        sequences.append((header, "".join(seq)))

random.shuffle(sequences)
split_idx = int(len(sequences) * 0.95)
train = sequences[:split_idx]
val = sequences[split_idx:]

with open("train.fasta", "w") as f:
    for h, s in train:
        f.write(h)
        f.write(s + "\n")

with open("val.fasta", "w") as f:
    for h, s in val:
        f.write(h)
        f.write(s + "\n")

print(f"Train: {len(train):,} sequences")
print(f"Val: {len(val):,} sequences")
EOF

echo ""
echo "=== Done ==="
echo "Files created in $DATA_DIR:"
ls -lh train.fasta val.fasta
