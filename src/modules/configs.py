"""
Collecion of configuration objects

Types of configs supported:
* Model config
* Training config
"""

from dataclasses import dataclass


@dataclass
class GPTModelConfig:
    n_layers: int
    n_head: int
    n_embed: int
    block_size: int
    batch_size: int
    device: str
    device_type: str
    bias: bool
    dropout: float
    compile: bool


@dataclass
class LossConfig:
    """Nested class for training config"""

    weight_decay: float
    learning_rate: float
    beta1: float
    beta2: float


@dataclass
class LearningRateScheduleConfig:
    """Nested class for training config"""

    decay_lr: bool
    warmup_iters: int
    lr_decay_iters: int
    base_lr: float
    min_lr: float


@dataclass
class TrainingConfig:
    loss_config: LossConfig
    lr_schedule_config: LearningRateScheduleConfig
    max_iters: int
    eval_interval: int
    eval_iters: int
    gradient_accumulation_steps: int
    grad_clip: float
