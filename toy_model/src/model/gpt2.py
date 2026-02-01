"""
GPT-2-like Transformer model with Rotary Position Embedding (RoPE).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    d = x.size(-1)
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to input tensor."""
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Precomputes cos/sin values for efficient position encoding.
    """
    
    def __init__(self, head_dim: int, max_len: int = 1024, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"
        self.head_dim = head_dim
        self.max_len = max_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        """Build cos/sin cache up to max_len."""
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.einsum("l,d->ld", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self.max_len = max_len

    def forward(self, seq_len: int, device=None, dtype=None):
        """Get cos/sin embeddings for given sequence length."""
        if seq_len > self.max_len:
            self._build_cache(seq_len)
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        if device is not None:
            cos = cos.to(device)
            sin = sin.to(device)
        if dtype is not None:
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
        return cos, sin


class CausalSelfAttentionRoPE(nn.Module):
    """Causal Self-Attention with Rotary Position Embedding."""
    
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float,
        max_len: int,
        rope_base: float = 10000.0
    ):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim, max_len=max_len, base=rope_base)

        causal = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", causal[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None):
        """
        Args:
            x: Input tensor of shape (B, L, C)
            pad_mask: Padding mask of shape (B, L), True for PAD positions
        """
        B, L, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(seq_len=L, device=x.device, dtype=x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.causal_mask[:, :, :L, :L], float("-inf"))

        if pad_mask is not None:
            att = att.masked_fill(pad_mask[:, None, None, :], float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, L, C)
        y = self.resid_drop(self.out(y))
        return y


class GPT2BlockRoPE(nn.Module):
    """GPT-2-like Transformer block with RoPE (Pre-LN variant)."""
    
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float,
        max_len: int,
        rope_base: float = 10000.0
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionRoPE(d_model, n_head, dropout, max_len, rope_base=rope_base)
        self.ln_2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None):
        x = x + self.attn(self.ln_1(x), pad_mask=pad_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2LikeEncoder(nn.Module):
    """
    GPT-2-like Transformer encoder with Rotary Position Embedding.
    Uses pre-LN architecture and weight tying between embedding and output layers.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layer: int = 8,
        n_head: int = 12,
        dropout: float = 0.0,
        max_len: int = 1024,
        rope_base: float = 100.0
    ):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Model dimension
            n_layer: Number of transformer layers
            n_head: Number of attention heads
            dropout: Dropout rate
            max_len: Maximum sequence length
            rope_base: Base for rotary position embedding
        """
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            GPT2BlockRoPE(d_model, n_head, dropout, max_len, rope_base=rope_base)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # self.head.weight = self.tok_emb.weight  # Weight tying

        self.max_len = max_len
        self.apply(self._init)

    def _init(self, m):
        """Initialize weights."""
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor, pad_mask: torch.Tensor = None):
        """
        Args:
            input_ids: Input token IDs of shape (B, L)
            pad_mask: Padding mask of shape (B, L), True for PAD positions
            
        Returns:
            Logits of shape (B, L, V)
        """
        B, L = input_ids.shape
        if L > self.max_len:
            raise ValueError(f"seq_len {L} exceeds max_len {self.max_len}")

        x = self.drop(self.tok_emb(input_ids))

        for blk in self.blocks:
            x = blk(x, pad_mask=pad_mask)

        x = self.ln_f(x)
        return self.head(x)
