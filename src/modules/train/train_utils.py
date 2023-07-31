"""
Utility functions used by the trainer
"""

import math

import numpy as np
import torch


def get_batch(
    data: np.memmap, block_size: int, batch_size: int, device: str, device_type: str
):
    """Get a batch of X and y"""
    idx_split = torch.randint(len(data) - block_size, (batch_size,))
    x: torch.Tensor = torch.stack(
        [
            torch.from_numpy((data[i : i + block_size]).astype(np.int64))
            for i in idx_split
        ]
    )
    y: torch.Tensor = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in idx_split
        ]
    )
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model,
    ctx,
    train_data,
    val_data,
    block_size,
    batch_size,
    device,
    device_type,
    eval_iters: int,
):
    """Estimate an arbitrarily accurate loss over train or val split using many batches"""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        data = train_data if split == "train" else val_data
        for k in range(eval_iters):
            x, y = get_batch(
                data=data,
                block_size=block_size,
                batch_size=batch_size,
                device=device,
                device_type=device_type,
            )
            with ctx:
                logits, loss = model(x, y)
            losses[k]: np.ndarray = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_learning_rate(
    it,
    warmup_iters,
    lr_decay_iters,
    base_lr,
    min_lr,
):
    """Get learning rate based on a schedule"""

    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return base_lr * it / warmup_iters

    # 2) If it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr

    # 3) otherwise, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # range in between 0..1
    return min_lr + coeff * (base_lr - min_lr)
