#!/usr/bin/env python3
"""Data preprocessing script for TinyPLM.

Downloads UniRef50 and dereplicates at 99.5% identity using MMSeqs2
to prevent near-identical sequences from appearing as negatives in MRL training.

Usage:
    python scripts/preprocess_data.py --output data/uniref50_derep
    python scripts/preprocess_data.py --input my_sequences.fasta --output data/custom_derep
"""

import argparse
import gzip
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm


UNIREF50_URL = "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"


def download_uniref50(output_path: Path, force: bool = False) -> Path:
    """Download UniRef50 FASTA file.

    Args:
        output_path: Directory to save the file.
        force: Re-download even if file exists.

    Returns:
        Path to downloaded (gzipped) file.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    fasta_gz = output_path / "uniref50.fasta.gz"

    if fasta_gz.exists() and not force:
        print(f"UniRef50 already exists at {fasta_gz}")
        return fasta_gz

    print(f"Downloading UniRef50 from {UNIREF50_URL}...")
    print("This is a large file (~10GB compressed), please be patient.")

    def progress_hook(count, block_size, total_size):
        percent = count * block_size * 100 // total_size
        print(f"\rProgress: {percent}%", end="", flush=True)

    urlretrieve(UNIREF50_URL, fasta_gz, reporthook=progress_hook)
    print("\nDownload complete!")

    return fasta_gz


def decompress_fasta(fasta_gz: Path, output_path: Path) -> Path:
    """Decompress gzipped FASTA file.

    Args:
        fasta_gz: Path to gzipped FASTA.
        output_path: Directory for output.

    Returns:
        Path to decompressed FASTA.
    """
    fasta = output_path / "uniref50.fasta"

    if fasta.exists():
        print(f"Decompressed FASTA already exists at {fasta}")
        return fasta

    print(f"Decompressing {fasta_gz}...")
    with gzip.open(fasta_gz, "rb") as f_in:
        with open(fasta, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    print("Decompression complete!")
    return fasta


def check_mmseqs2() -> bool:
    """Check if MMSeqs2 is installed."""
    try:
        result = subprocess.run(
            ["mmseqs", "version"],
            capture_output=True,
            text=True,
        )
        print(f"Found MMSeqs2: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        return False


def dereplicate_mmseqs2(
    input_fasta: Path,
    output_path: Path,
    identity: float = 0.995,
    threads: int = 4,
) -> Path:
    """Dereplicate sequences using MMSeqs2.

    Clusters sequences at specified identity threshold and outputs
    representative sequences.

    Args:
        input_fasta: Input FASTA file.
        output_path: Output directory.
        identity: Sequence identity threshold (0.995 = 99.5%).
        threads: Number of threads for MMSeqs2.

    Returns:
        Path to dereplicated FASTA.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    output_fasta = output_path / f"uniref50_derep{int(identity * 100)}.fasta"

    if output_fasta.exists():
        print(f"Dereplicated FASTA already exists at {output_fasta}")
        return output_fasta

    if not check_mmseqs2():
        print("ERROR: MMSeqs2 not found!")
        print("Install with: conda install -c bioconda mmseqs2")
        print("Or: brew install mmseqs2")
        raise RuntimeError("MMSeqs2 not installed")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        db = tmpdir / "seqdb"
        cluster_db = tmpdir / "clusterdb"
        rep_db = tmpdir / "repdb"

        print(f"Creating MMSeqs2 database...")
        subprocess.run(
            ["mmseqs", "createdb", str(input_fasta), str(db)],
            check=True,
        )

        print(f"Clustering at {identity * 100}% identity...")
        subprocess.run(
            [
                "mmseqs", "cluster",
                str(db), str(cluster_db), str(tmpdir),
                "--min-seq-id", str(identity),
                "-c", "0.8",  # Coverage threshold
                "--cov-mode", "1",  # Coverage of target
                "--threads", str(threads),
            ],
            check=True,
        )

        print("Extracting representative sequences...")
        subprocess.run(
            ["mmseqs", "createsubdb", str(cluster_db), str(db), str(rep_db)],
            check=True,
        )

        subprocess.run(
            ["mmseqs", "convert2fasta", str(rep_db), str(output_fasta)],
            check=True,
        )

    print(f"Dereplicated FASTA saved to {output_fasta}")
    return output_fasta


def filter_by_length(
    input_fasta: Path,
    output_fasta: Path,
    min_length: int = 50,
    max_length: int = 2048,
) -> tuple[int, int]:
    """Filter sequences by length.

    Args:
        input_fasta: Input FASTA file.
        output_fasta: Output FASTA file.
        min_length: Minimum sequence length.
        max_length: Maximum sequence length.

    Returns:
        Tuple of (kept_count, filtered_count).
    """
    print(f"Filtering sequences to length {min_length}-{max_length}...")

    kept = 0
    filtered = 0
    current_header = None
    current_seq = []

    with open(input_fasta) as f_in, open(output_fasta, "w") as f_out:
        for line in tqdm(f_in, desc="Filtering"):
            line = line.strip()
            if line.startswith(">"):
                # Write previous sequence if valid
                if current_header is not None:
                    seq = "".join(current_seq)
                    if min_length <= len(seq) <= max_length:
                        f_out.write(f"{current_header}\n{seq}\n")
                        kept += 1
                    else:
                        filtered += 1
                current_header = line
                current_seq = []
            else:
                current_seq.append(line)

        # Handle last sequence
        if current_header is not None:
            seq = "".join(current_seq)
            if min_length <= len(seq) <= max_length:
                f_out.write(f"{current_header}\n{seq}\n")
                kept += 1
            else:
                filtered += 1

    print(f"Kept {kept:,} sequences, filtered {filtered:,}")
    return kept, filtered


def create_train_val_split(
    input_fasta: Path,
    output_dir: Path,
    val_fraction: float = 0.01,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Split FASTA into train and validation sets.

    Args:
        input_fasta: Input FASTA file.
        output_dir: Output directory.
        val_fraction: Fraction of sequences for validation.
        seed: Random seed.

    Returns:
        Tuple of (train_path, val_path).
    """
    import random

    random.seed(seed)

    train_path = output_dir / "train.fasta"
    val_path = output_dir / "val.fasta"

    print(f"Creating train/val split (val_fraction={val_fraction})...")

    with open(input_fasta) as f_in:
        with open(train_path, "w") as f_train, open(val_path, "w") as f_val:
            current_header = None
            current_seq = []

            for line in tqdm(f_in, desc="Splitting"):
                line = line.strip()
                if line.startswith(">"):
                    # Write previous sequence
                    if current_header is not None:
                        seq = "".join(current_seq)
                        if random.random() < val_fraction:
                            f_val.write(f"{current_header}\n{seq}\n")
                        else:
                            f_train.write(f"{current_header}\n{seq}\n")
                    current_header = line
                    current_seq = []
                else:
                    current_seq.append(line)

            # Handle last sequence
            if current_header is not None:
                seq = "".join(current_seq)
                if random.random() < val_fraction:
                    f_val.write(f"{current_header}\n{seq}\n")
                else:
                    f_train.write(f"{current_header}\n{seq}\n")

    print(f"Train set: {train_path}")
    print(f"Val set: {val_path}")
    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess protein sequences for TinyPLM training"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input FASTA file (if not provided, downloads UniRef50)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/uniref50"),
        help="Output directory",
    )
    parser.add_argument(
        "--identity",
        type=float,
        default=0.995,
        help="Dereplication identity threshold (default: 0.995 = 99.5%%)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum sequence length (default: 50)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Threads for MMSeqs2 (default: 4)",
    )
    parser.add_argument(
        "--skip-derep",
        action="store_true",
        help="Skip dereplication step",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.01,
        help="Fraction of data for validation (default: 0.01)",
    )

    args = parser.parse_args()

    # Get input FASTA
    if args.input is None:
        fasta_gz = download_uniref50(args.output)
        input_fasta = decompress_fasta(fasta_gz, args.output)
    else:
        input_fasta = args.input

    # Dereplicate
    if not args.skip_derep:
        derep_fasta = dereplicate_mmseqs2(
            input_fasta,
            args.output,
            identity=args.identity,
            threads=args.threads,
        )
    else:
        derep_fasta = input_fasta

    # Filter by length
    filtered_fasta = args.output / "filtered.fasta"
    filter_by_length(
        derep_fasta,
        filtered_fasta,
        min_length=args.min_length,
        max_length=args.max_length,
    )

    # Train/val split
    create_train_val_split(
        filtered_fasta,
        args.output,
        val_fraction=args.val_fraction,
    )

    print("\nPreprocessing complete!")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
