"""
Based on Andrej Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT/tree/master)
which is also based on the following:

1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import logging
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    """LayerNorm layer"""

    def __init__(self, ndim: int, bias: bool, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)


class CausalSelfAttention(nn.Module):
    """Causal self attention layer"""

    def __init__(self, n_embed, n_head, bias, dropout, block_size):
        super().__init__()
        # Key, query, value projections for all heads in a batch
        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=bias)
        # Output projection
        self.c_proj = nn.Linear(n_embed, n_embed, bias=bias)

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embed = n_embed
        self.dropout = dropout

        # Flash attention for PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_production_attention")
        if not self.flash:
            logger.warning(
                "Using slow attention. Flash attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size)).view(
                    1, 1, block_size, block_size
                ),
            )

    def forward(self, x):
        batch_size, seq_len, n_embed = x.size()

        # Calculate q, k, v
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        k = k.view(batch_size, seq_len, self.n_head, n_embed // self.n_head).transpose(
            1, 2
        )  # (batch_size, num heads, seq_len, head dim)
        q = q.view(batch_size, seq_len, self.n_head, n_embed // self.n_head).transpose(
            1, 2
        )
        v = v.view(batch_size, seq_len, self.n_head, n_embed // self.n_head).transpose(
            1, 2
        )

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(
                self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf")
            )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embed)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embed, bias, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embed, 4 * n_embed, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embed, n_embed, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embed, n_head, bias, dropout, block_size, eps=1e-5):
        super().__init__()
        self.layer_norm_1 = LayerNorm(ndim=n_embed, bias=bias, eps=eps)
        self.attention_layer = CausalSelfAttention(
            n_embed=n_embed,
            n_head=n_head,
            bias=bias,
            dropout=dropout,
            block_size=block_size,
        )
        self.layer_norm_2 = LayerNorm(ndim=n_embed, bias=bias, eps=eps)
        self.mlp = MLP(n_embed=n_embed, bias=bias, dropout=dropout)

    def forward(self, x):
        # Residual
        x = x + self.attention_layer(self.layer_norm_1(x))
        x = x + self.mlp(self.layer_norm_2(x))
        return x
