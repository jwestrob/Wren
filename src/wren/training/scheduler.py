"""Learning rate schedulers for Wren.

Implements:
- Cosine decay with warmup (Phase 1a)
- WSD: Warmup-Stable-Decay (Phase 2)
- Linear decay to zero (Phase 2 alternative)
"""

import math
from typing import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Create cosine decay schedule with linear warmup.

    Args:
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum LR as fraction of peak (default 0).

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Scale to [min_lr_ratio, 1.0]
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def get_wsd_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Create Warmup-Stable-Decay (WSD) schedule.

    Used in MiniCPM and other efficient training recipes.
    Maintains peak LR for majority of training, then decays.

    Args:
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_stable_steps: Number of steps at peak LR.
        num_decay_steps: Number of decay steps.
        min_lr_ratio: Minimum LR as fraction of peak.

    Returns:
        LambdaLR scheduler.
    """
    total_steps = num_warmup_steps + num_stable_steps + num_decay_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_warmup_steps + num_stable_steps:
            # Stable phase at peak LR
            return 1.0
        else:
            # Cosine decay
            decay_step = current_step - num_warmup_steps - num_stable_steps
            progress = float(decay_step) / float(max(1, num_decay_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def get_linear_decay_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Create linear decay to zero schedule.

    Simple alternative to cosine - just linearly decay LR to 0.

    Args:
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        # Linear decay to zero
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    **kwargs,
) -> LambdaLR:
    """Factory function to create scheduler by name.

    Args:
        name: Scheduler name ("cosine", "wsd", "linear").
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total training steps.
        **kwargs: Additional scheduler-specific arguments.

    Returns:
        LR scheduler.
    """
    if name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=kwargs.get("min_lr_ratio", 0.0),
        )
    elif name == "wsd":
        # Default: 80% stable, 20% decay
        num_stable = kwargs.get(
            "num_stable_steps",
            int(0.8 * (num_training_steps - num_warmup_steps)),
        )
        num_decay = num_training_steps - num_warmup_steps - num_stable
        return get_wsd_schedule(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_stable_steps=num_stable,
            num_decay_steps=num_decay,
            min_lr_ratio=kwargs.get("min_lr_ratio", 0.0),
        )
    elif name == "linear" or name == "d2z":
        return get_linear_decay_schedule(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")
