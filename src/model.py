"""
A minimal, fully-from-scratch decoder-only transformer.

Built deliberately simple (no HuggingFace) so every tensor is inspectable:
this is a research tool, not a production model. Every submodule exposes
its intermediate activations via forward hooks so the interp/ tools can
cache and patch them.

Architecture follows the setup used in Nanda et al. "Progress measures
for grokking via mechanistic interpretability" (2023): a single-layer
(or few-layer) attention-only-or-MLP transformer trained on modular
arithmetic, small enough to fully reverse-engineer.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    d_vocab: int = 114          # p=113 tokens + 1 "=" token
    d_model: int = 128
    n_ctx: int = 3               # a, b, = -> predict a+b mod p
    n_layers: int = 1
    n_heads: int = 4
    d_head: int = 32
    d_mlp: int = 512
    act_fn: str = "relu"
    use_ln: bool = False         # grokking papers often train w/o LayerNorm
    seed: int = 0


class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(cfg.d_vocab, cfg.d_model) / cfg.d_model**0.5)

    def forward(self, tokens):
        return self.W_E[tokens]


class Unembed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(cfg.d_model, cfg.d_vocab) / cfg.d_model**0.5)

    def forward(self, x):
        return x @ self.W_U


class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(cfg.n_ctx, cfg.d_model) / cfg.d_model**0.5)

    def forward(self, tokens):
        return self.W_pos[: tokens.shape[-1]]


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        shape = (cfg.n_heads, cfg.d_model, cfg.d_head)
        self.W_Q = nn.Parameter(torch.randn(shape) / cfg.d_model**0.5)
        self.W_K = nn.Parameter(torch.randn(shape) / cfg.d_model**0.5)
        self.W_V = nn.Parameter(torch.randn(shape) / cfg.d_model**0.5)
        self.W_O = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_head, cfg.d_model) / cfg.d_model**0.5)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx)).bool())

    def forward(self, x):
        # x: [batch, pos, d_model]
        q = torch.einsum("bpd,hdk->bphk", x, self.W_Q)
        k = torch.einsum("bpd,hdk->bphk", x, self.W_K)
        v = torch.einsum("bpd,hdk->bphk", x, self.W_V)

        attn_scores = torch.einsum("bphk,bqhk->bhpq", q, k) / (self.cfg.d_head**0.5)
        attn_scores = attn_scores.masked_fill(~self.mask[: x.shape[1], : x.shape[1]], float("-inf"))
        pattern = F.softmax(attn_scores, dim=-1)  # [batch, head, q_pos, k_pos]
        self.pattern = pattern.detach()  # cached for interpretability

        z = torch.einsum("bhpq,bqhk->bphk", pattern, v)
        out = torch.einsum("bphk,hkd->bpd", z, self.W_O)
        return out


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(cfg.d_model, cfg.d_mlp) / cfg.d_model**0.5)
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.W_out = nn.Parameter(torch.randn(cfg.d_mlp, cfg.d_model) / cfg.d_mlp**0.5)
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, x):
        pre = x @ self.W_in + self.b_in
        self.pre_act = pre.detach()  # cached: neuron activations pre-nonlinearity
        post = F.relu(pre)
        self.post_act = post.detach()
        return post @ self.W_out + self.b_out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Transformer(nn.Module):
    """Minimal decoder-only transformer. Call with return_cache=True to get
    every intermediate activation back for interpretability work."""

    def __init__(self, cfg: Config):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.unembed = Unembed(cfg)

    def forward(self, tokens, return_cache=False):
        x = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            x = block(x)
        logits = self.unembed(x)

        if not return_cache:
            return logits

        cache = {}
        for i, block in enumerate(self.blocks):
            cache[f"blocks.{i}.attn.pattern"] = block.attn.pattern
            cache[f"blocks.{i}.mlp.pre_act"] = block.mlp.pre_act
            cache[f"blocks.{i}.mlp.post_act"] = block.mlp.post_act
        return logits, cache

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
