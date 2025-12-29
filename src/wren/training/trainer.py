"""Trainer class for Wren.

Implements:
- Training loop with gradient accumulation
- Mixed precision training
- Checkpointing and resume
- Logging to wandb
- Evaluation during training
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from wren.config import TrainingConfig
from wren.model.plm import Wren
from wren.training.losses import CombinedLoss
from wren.training.scheduler import get_scheduler


@dataclass
class TrainState:
    """Training state for checkpointing."""

    step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")
    samples_seen: int = 0


class Trainer:
    """Trainer for Wren."""

    def __init__(
        self,
        model: Wren,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        mrl_dims: list[int] = [64, 128, 256, 512],
    ):
        """Initialize trainer.

        Args:
            model: Wren model to train.
            config: Training configuration.
            train_dataloader: Training data loader.
            val_dataloader: Optional validation data loader.
            mrl_dims: MRL dimensions for contrastive loss.
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Move model to device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = get_scheduler(
            config.lr_schedule,
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.total_steps,
        )

        # Loss function
        self.loss_fn = CombinedLoss(
            mlm_weight=1.0,
            mrl_weight=0.1,
            mrl_dims=mrl_dims,
        )

        # Mixed precision
        self.use_amp = config.mixed_precision and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.state = TrainState()

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.wandb_run = None
        if config.wandb_project:
            self._init_wandb()

    def _init_wandb(self) -> None:
        """Initialize wandb logging."""
        try:
            import wandb

            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    "model": self.model.config.__dict__,
                    "training": self.config.__dict__,
                },
            )
        except ImportError:
            print("wandb not installed, skipping logging")
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")

    def _log(self, metrics: dict, step: int) -> None:
        """Log metrics to wandb."""
        if self.wandb_run:
            import wandb

            wandb.log(metrics, step=step)

    def train(self) -> None:
        """Run full training loop."""
        print(f"Starting training for {self.config.total_steps} steps")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")

        self.model.train()
        accumulated_loss = 0.0
        accumulated_metrics = {}

        # Progress bar
        pbar = tqdm(total=self.config.total_steps, initial=self.state.step)

        data_iter = iter(self.train_dataloader)

        while self.state.step < self.config.total_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                self.state.epoch += 1
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            loss, metrics = self._training_step(batch)

            # Backward pass (scaled for gradient accumulation)
            scaled_loss = loss / self.config.gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accumulated_loss += loss.item()
            for k, v in metrics.items():
                accumulated_metrics[k] = accumulated_metrics.get(k, 0) + v

            # Optimizer step
            if (self.state.step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            self.state.step += 1
            self.state.samples_seen += batch["input_ids"].size(0)
            pbar.update(1)

            # Logging
            if self.state.step % self.config.log_every_steps == 0:
                avg_loss = accumulated_loss / self.config.log_every_steps
                avg_metrics = {
                    k: v / self.config.log_every_steps
                    for k, v in accumulated_metrics.items()
                }

                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                self._log(
                    {
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/epoch": self.state.epoch,
                        "train/samples_seen": self.state.samples_seen,
                        **{f"train/{k}": v for k, v in avg_metrics.items()},
                    },
                    step=self.state.step,
                )

                accumulated_loss = 0.0
                accumulated_metrics = {}

            # Evaluation
            if (
                self.val_dataloader is not None
                and self.state.step % self.config.eval_every_steps == 0
            ):
                val_loss = self.evaluate()
                self.model.train()

                if val_loss < self.state.best_loss:
                    self.state.best_loss = val_loss
                    self.save_checkpoint("best.pt")

            # Checkpointing
            if self.state.step % self.config.save_every_steps == 0:
                self.save_checkpoint(f"step_{self.state.step}.pt")

        pbar.close()
        self.save_checkpoint("final.pt")
        print("Training complete!")

    def _training_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict]:
        """Single training step.

        Args:
            batch: Batch from dataloader.

        Returns:
            Tuple of (loss, metrics dict).
        """
        with autocast(enabled=self.use_amp):
            # Forward pass for MLM
            outputs = self.model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            # Check if we have contrastive views
            if "view1_input_ids" in batch:
                # Get embeddings for contrastive views
                view1_out = self.model(
                    batch["view1_input_ids"],
                    attention_mask=batch["view1_attention_mask"],
                )
                view2_out = self.model(
                    batch["view2_input_ids"],
                    attention_mask=batch["view2_attention_mask"],
                )

                # Combined loss
                loss, metrics = self.loss_fn(
                    outputs["logits"],
                    batch["labels"],
                    view1_out["embeddings"],
                    view2_out["embeddings"],
                )
            else:
                # MLM only
                loss = self.model.compute_mlm_loss(
                    outputs["logits"], batch["labels"]
                )
                metrics = {"mlm_loss": loss.item()}

        return loss, metrics

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation on validation set.

        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = self.model.compute_mlm_loss(
                    outputs["logits"], batch["labels"]
                )

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        self._log({"val/loss": avg_loss}, step=self.state.step)
        print(f"Validation loss: {avg_loss:.4f}")

        return avg_loss

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "state": self.state.__dict__,
            "config": self.config.__dict__,
            "model_config": self.model.config.__dict__,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        for k, v in checkpoint["state"].items():
            setattr(self.state, k, v)

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"Loaded checkpoint from {path}")
        print(f"Resuming from step {self.state.step}")
