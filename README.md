# nanoGPT_clean
A reimplementation of Andrej Karpathy's nanoGPT for educational purposes

## TODO

- [ ] overall refactoring and rearchitecting
- [x] add wandb
- [ ] add checkpoint loading
- [ ] add flash attention 2
- [ ] add RoPE
- [ ] type checking with mypy

## Running

Data preparation

```shell
$ make prepare-data
```

Training

```shell
$ make train
```

Training with wandb integration

```shell
$ make train --wandb_project=nano_gpt_clean --wandb_run_name=run_1
```
