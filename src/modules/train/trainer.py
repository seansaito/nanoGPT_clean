"""
Main trainer module

The training method runs the following steps

* Initialize configurations
* Load data
* Initialize model
* Compile the model
* Run training loop

"""
import logging
import os
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch

from src import CHECKPOINT_DIR, DATA_DIR
from src.modules.model.gpt import GPT
from src.modules.train.train_utils import estimate_loss, get_batch, get_learning_rate
from src.utils import gen_path, timeit

logger = logging.getLogger(__name__)


class GPTTrainer:
    def __init__(
        self,
        # Model
        n_layers,
        n_head,
        n_embed,
        block_size,
        batch_size,
        device,
        device_type,
        dataset,
        bias,
        dropout,
        compile,
        # Loss
        weight_decay,
        learning_rate,
        beta1,
        beta2,
        # Learning rate
        decay_lr: bool,
        warmup_iters,
        lr_decay_iters,
        base_lr,
        min_lr,
        # Training
        max_iters,
        eval_interval,
        eval_iters,
        gradient_accumulation_steps,
        grad_clip,
        log_interval,
        # Others
        quick_test,
        vocab_size,
    ):
        self.dataset = dataset
        self.n_layers = n_layers
        self.n_head = n_head
        self.n_embed = n_embed
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.device_type = device_type
        self.bias = bias
        self.dropout = dropout
        self.compile = compile
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay_lr = decay_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        # TODO add wandb support
        self.quick_test = quick_test
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.max_iters = max_iters
        self.vocab_size = vocab_size

    @timeit
    def train(
        self,
    ):
        # Initialize the configurations
        dtype = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
        ctx = (
            nullcontext()
            if self.device_type == "cpu"
            else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        )

        # Load the data
        train_data, val_data, dict_meta = self.load_data()

        if self.vocab_size is None:
            assert dict_meta["vocab_size"] is not None
            self.vocab_size = dict_meta["vocab_size"]

        # Initialize the model
        model = self.init_model(
            n_layers=self.n_layers,
            n_head=self.n_head,
            n_embed=self.n_embed,
            block_size=self.block_size,
            bias=self.bias,
            vocab_size=self.vocab_size,
            dropout=self.dropout,
            device=self.device,
            compile=self.compile,
        )

        # Initialize the optimizer
        optimizer = self.init_optimizer(
            model=model,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            device_type=self.device_type,
        )

        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

        # Model args
        dict_model_args = dict(
            n_layers=self.n_layers,
            n_head=self.n_head,
            n_embed=self.n_embed,
            block_size=self.block_size,
            bias=self.bias,
            vocab_size=self.vocab_size,
            dropout=self.dropout,
        )

        _ = self.run_training_loop(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            ctx=ctx,
            train_data=train_data,
            val_data=val_data,
            dict_model_args=dict_model_args,
        )

    def run_training_loop(
        self,
        model,
        optimizer,
        scaler,
        ctx,
        train_data,
        val_data,
        dict_model_args,
    ):
        # Run the training loop
        iter_num = 0
        best_val_loss = 1e9
        local_iter_num = 0
        running_mfu = -1.0
        X, Y = get_batch(
            data=train_data,
            block_size=self.block_size,
            batch_size=self.batch_size,
            device=self.device,
            device_type=self.device_type,
        )

        t0 = time.time()
        while True:
            # Set the learning rate for this iteration
            lr = get_learning_rate(
                it=iter_num,
                warmup_iters=self.warmup_iters,
                lr_decay_iters=self.lr_decay_iters,
                base_lr=self.base_lr,
                min_lr=self.min_lr,
            )

            # Update the learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if iter_num % self.eval_interval == 0:
                losses = estimate_loss(
                    model=model,
                    ctx=ctx,
                    train_data=train_data,
                    val_data=val_data,
                    block_size=self.block_size,
                    batch_size=self.batch_size,
                    device=self.device,
                    device_type=self.device_type,
                    eval_iters=self.eval_iters,
                )

                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    if iter_num > 0:
                        _ = self.save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            dict_model_args=dict_model_args,
                            iter_num=iter_num,
                            best_val_loss=best_val_loss,
                        )
            if self.quick_test and iter_num == 0:
                break

            for microstep in range(self.gradient_accumulation_steps):
                with ctx:
                    logits, loss = model(X, Y)
                    # TODO understand the role of gradient accumulation steps
                    loss = loss / self.gradient_accumulation_steps

                # Get next batch
                X, Y = get_batch(
                    train_data,
                    block_size=self.block_size,
                    batch_size=self.batch_size,
                    device=self.device,
                    device_type=self.device_type,
                )

                # Backward pass, with gradient scaling if training in fp16
                # TODO understand further
                scaler.scale(loss).backward()

            if self.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=self.grad_clip
                )

            # Step the optimizer
            scaler.step(optimizer)
            scaler.update()

            # Flush gradients
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % self.log_interval == 0:
                loss_float = loss.item() * self.gradient_accumulation_steps
                # TODO understand
                if local_iter_num >= 5:  # let the training loop settle a bit
                    mfu = model.estimate_mfu(
                        self.batch_size * self.gradient_accumulation_steps, dt
                    )
                    running_mfu = (
                        mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                    )
                logger.info(
                    f"iter {iter_num}: loss {loss_float:.4f} time {dt * 1000: .2f} ms, mfu {running_mfu*100:.2f}%"
                )

            iter_num += 1
            local_iter_num += 1

            if iter_num > self.max_iters:
                break

    def save_checkpoint(
        self, model, optimizer, dict_model_args, iter_num, best_val_loss
    ):
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "dict_model_args": dict_model_args,
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
        }
        path_checkpoint = gen_path(
            path_dir=CHECKPOINT_DIR,
            fname="checkpoint.pt",
        )
        logger.info("Saving checkpoint to {}".format(path_checkpoint))
        torch.save(checkpoint, path_checkpoint)
        return path_checkpoint

    def load_data(self):
        path_dataset = DATA_DIR / "dataset" / self.dataset
        train_data = np.memmap(
            str(path_dataset / "train.bin"), dtype=np.uint16, mode="r"
        )
        val_data = np.memmap(str(path_dataset / "val.bin"), dtype=np.uint16, mode="r")

        path_meta = str(path_dataset / "meta.pkl")
        if os.path.exists(path_meta):
            with open(path_meta, "rb") as fp:
                dict_meta = pickle.load(fp)
        else:
            dict_meta = None

        logger.info("Meta: {}".format(dict_meta))
        return train_data, val_data, dict_meta

    def init_model(
        self,
        n_layers,
        n_head,
        n_embed,
        block_size,
        bias,
        vocab_size,
        dropout,
        device,
        compile,
    ):
        # TODO implement checkpoint reloading
        # initialize the model
        dict_model_args = {
            "n_layers": n_layers,
            "n_head": n_head,
            "n_embed": n_embed,
            "block_size": block_size,
            "bias": bias,
            "vocab_size": vocab_size,
            "dropout": dropout,
        }
        model = GPT(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layers=n_layers,
            n_head=n_head,
            n_embed=n_embed,
            dropout=dropout,
            bias=bias,
        )

        if block_size < model.block_size:
            model.crop_block_size(block_size)
            dict_model_args["block_size"] = block_size

        # Set the model device
        model.to(device)

        if compile:
            model = torch.compile(model)

        return model

    def init_optimizer(
        self,
        model,
        weight_decay,
        learning_rate,
        beta1,
        beta2,
        device_type,
    ):
        # Create the optimizer
        optimizer = model.configure_optimizers(
            weight_decay, learning_rate, (beta1, beta2), device_type
        )

        return optimizer
