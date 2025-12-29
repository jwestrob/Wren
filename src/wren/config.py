"""Configuration dataclasses for Wren."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Architecture
    hidden_dim: int = 1024  # 2x standard to compensate for ternary
    num_layers: int = 12
    num_heads: int = 16
    head_dim: int = 64  # hidden_dim // num_heads
    ffn_dim: int = 2730  # ~8/3 * hidden_dim for SwiGLU
    vocab_size: int = 25  # 20 AA + 5 special tokens
    max_seq_len: int = 2048

    # MRL dimensions for flexible embeddings
    mrl_dims: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    embedding_dim: int = 512  # Final embedding dimension

    # Regularization
    dropout: float = 0.0  # BitNet typically uses no dropout

    # Quantization
    use_bitlinear: bool = True  # False for FP16 baseline

    # Position encoding
    use_yarn: bool = True  # Use YaRN (True) or standard RoPE (False)
    rope_scale: float = 1.0  # Context extension scale (1.0 = no scaling)

    # Memory optimization
    gradient_checkpointing: bool = True

    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
        if self.head_dim != self.hidden_dim // self.num_heads:
            self.head_dim = self.hidden_dim // self.num_heads


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Optimization
    learning_rate: float = 1e-3  # BitNet uses higher LR
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Schedule
    warmup_steps: int = 2000
    total_steps: int = 100000
    lr_schedule: str = "cosine"  # cosine, wsd, or d2z

    # Batching
    batch_size: int = 32
    gradient_accumulation_steps: int = 8  # Effective batch = 256

    # Checkpointing
    save_every_steps: int = 5000
    eval_every_steps: int = 500
    checkpoint_dir: str = "checkpoints"

    # Logging
    log_every_steps: int = 100
    wandb_project: str = "wren"
    wandb_run_name: Optional[str] = None

    # Device
    device: str = "cuda"  # cuda, mps, or cpu
    mixed_precision: bool = True

    # Reproducibility
    seed: int = 42


@dataclass
class DataConfig:
    """Data loading configuration."""

    # Dataset
    dataset_path: str = "data/uniref50"
    max_length: int = 2048
    min_length: int = 50  # Skip very short sequences

    # MLM
    mask_prob: float = 0.15
    mask_token_prob: float = 0.8  # 80% [MASK]
    random_token_prob: float = 0.1  # 10% random
    # Remaining 10% unchanged

    # MRL contrastive
    crop_min_ratio: float = 0.7
    crop_max_ratio: float = 0.9
    extra_mask_prob: float = 0.2  # Additional masking for augmentation

    # DataLoader
    num_workers: int = 4
    prefetch_factor: int = 2


@dataclass
class Config:
    """Combined configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        model_cfg = ModelConfig(**data.get("model", {}))
        training_cfg = TrainingConfig(**data.get("training", {}))
        data_cfg = DataConfig(**data.get("data", {}))

        return cls(model=model_cfg, training=training_cfg, data=data_cfg)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        from dataclasses import asdict

        data = {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "data": asdict(self.data),
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Preset configurations
TINY_50M = ModelConfig(
    hidden_dim=1024,
    num_layers=12,
    num_heads=16,
    ffn_dim=2730,
)

SMALL_150M = ModelConfig(
    hidden_dim=1536,
    num_layers=18,
    num_heads=24,
    ffn_dim=4096,
)

BASE_300M = ModelConfig(
    hidden_dim=2048,
    num_layers=24,
    num_heads=32,
    ffn_dim=5461,
)
