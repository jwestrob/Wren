#!/usr/bin/env python3
"""Training script for Wren.

Usage:
    python scripts/train.py --config configs/tiny_50m.yaml
    python scripts/train.py --config configs/tiny_50m.yaml --resume checkpoints/step_5000.pt
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from wren.config import Config
from wren.data import (
    ProteinTokenizer,
    ProteinDataset,
    MLMDataset,
    ContrastiveMLMDataset,
    create_dataloader,
)
from wren.model import Wren
from wren.training.trainer import Trainer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train Wren")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help="Override training data path",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=None,
        help="Override validation data path",
    )
    parser.add_argument(
        "--use-contrastive",
        action="store_true",
        help="Use contrastive MRL training (requires more memory)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config)

    # Override wandb if requested
    if args.no_wandb:
        config.training.wandb_project = None

    # Set seed
    set_seed(config.training.seed)

    # Print config
    print("=" * 60)
    print("Wren Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Model: {config.model.hidden_dim}d, {config.model.num_layers}L, {config.model.num_heads}H")
    print(f"Device: {config.training.device}")
    print(f"Batch size: {config.training.batch_size} x {config.training.gradient_accumulation_steps} = {config.training.batch_size * config.training.gradient_accumulation_steps}")
    print(f"Total steps: {config.training.total_steps}")
    print("=" * 60)

    # Create tokenizer
    tokenizer = ProteinTokenizer()

    # Data paths
    train_path = args.train_data or Path(config.data.dataset_path) / "train.fasta"
    val_path = args.val_data or Path(config.data.dataset_path) / "val.fasta"

    # Check data exists
    if not train_path.exists():
        print(f"ERROR: Training data not found at {train_path}")
        print("Run scripts/preprocess_data.py first to prepare the data.")
        return

    print(f"Training data: {train_path}")
    print(f"Validation data: {val_path if val_path.exists() else 'None'}")

    # Create datasets
    print("Loading datasets...")
    base_train = ProteinDataset(
        train_path,
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        min_length=config.data.min_length,
    )
    print(f"Training sequences: {len(base_train):,}", flush=True)

    print("Creating training dataloader...", flush=True)
    if args.use_contrastive:
        train_dataset = ContrastiveMLMDataset(
            base_train,
            tokenizer=tokenizer,
            mask_prob=config.data.mask_prob,
            crop_min_ratio=config.data.crop_min_ratio,
            crop_max_ratio=config.data.crop_max_ratio,
            extra_mask_prob=config.data.extra_mask_prob,
            max_length=config.data.max_length,
        )
    else:
        train_dataset = MLMDataset(
            base_train,
            tokenizer=tokenizer,
            mask_prob=config.data.mask_prob,
        )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        tokenizer=tokenizer,
        use_bucketing=False,  # Disabled for now - bucketing needs optimization for large datasets
        shuffle=True,
        num_workers=config.data.num_workers,
        contrastive=args.use_contrastive,
    )
    print("Training dataloader created.", flush=True)

    # Validation
    val_loader = None
    if val_path.exists():
        print("Loading validation data...", flush=True)
        base_val = ProteinDataset(
            val_path,
            tokenizer=tokenizer,
            max_length=config.data.max_length,
            min_length=config.data.min_length,
        )
        print(f"Validation sequences: {len(base_val):,}", flush=True)

        val_dataset = MLMDataset(base_val, tokenizer=tokenizer)
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config.training.batch_size,
            tokenizer=tokenizer,
            use_bucketing=False,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        print("Validation dataloader created.", flush=True)

    # Create model
    print("Creating model...", flush=True)
    model = Wren(config.model)
    print(f"Parameters: {model.num_parameters:,}", flush=True)
    print(f"Trainable: {model.num_trainable_parameters:,}", flush=True)

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config.training,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        mrl_dims=config.model.mrl_dims,
    )

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
