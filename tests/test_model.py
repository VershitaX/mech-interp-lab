"""Basic sanity tests. Run with: pytest tests/"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from model import Config, Transformer
from data import make_dataset, train_test_split


def test_model_forward_shape():
    cfg = Config(d_vocab=14, d_model=32, n_ctx=3, n_layers=1, n_heads=4, d_head=8, d_mlp=64)
    model = Transformer(cfg)
    tokens = torch.randint(0, 14, (5, 3))
    logits = model(tokens)
    assert logits.shape == (5, 3, 14)


def test_model_forward_with_cache():
    cfg = Config(d_vocab=14, d_model=32, n_ctx=3, n_layers=1, n_heads=4, d_head=8, d_mlp=64)
    model = Transformer(cfg)
    tokens = torch.randint(0, 14, (5, 3))
    logits, cache = model(tokens, return_cache=True)
    assert "blocks.0.attn.pattern" in cache
    assert "blocks.0.mlp.post_act" in cache
    # attention pattern should sum to 1 over keys (it's a softmax)
    pattern = cache["blocks.0.attn.pattern"]
    sums = pattern.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_causal_mask_is_causal():
    """Query at position 0 should never attend to positions 1 or 2."""
    cfg = Config(d_vocab=14, d_model=32, n_ctx=3, n_layers=1, n_heads=4, d_head=8, d_mlp=64)
    model = Transformer(cfg)
    tokens = torch.randint(0, 14, (2, 3))
    _, cache = model(tokens, return_cache=True)
    pattern = cache["blocks.0.attn.pattern"]  # [batch, head, q_pos, k_pos]
    assert torch.allclose(pattern[:, :, 0, 1], torch.zeros_like(pattern[:, :, 0, 1]))
    assert torch.allclose(pattern[:, :, 0, 2], torch.zeros_like(pattern[:, :, 0, 2]))


def test_dataset_shapes():
    p = 13
    inputs, labels = make_dataset(p=p)
    assert inputs.shape == (p * p, 3)
    assert labels.shape == (p * p,)
    assert (inputs[:, 2] == p).all()  # "=" token
    assert (labels == (inputs[:, 0] + inputs[:, 1]) % p).all()


def test_train_test_split_no_overlap():
    inputs, labels = make_dataset(p=13)
    tr_in, tr_lb, te_in, te_lb = train_test_split(inputs, labels, train_frac=0.3)
    assert tr_in.shape[0] + te_in.shape[0] == inputs.shape[0]
    # crude overlap check via set of tuples
    tr_set = {tuple(x.tolist()) for x in tr_in}
    te_set = {tuple(x.tolist()) for x in te_in}
    assert tr_set.isdisjoint(te_set)


def test_model_can_overfit_tiny_batch():
    """Sanity check that gradients flow and loss decreases."""
    cfg = Config(d_vocab=14, d_model=32, n_ctx=3, n_layers=1, n_heads=4, d_head=8, d_mlp=64)
    model = Transformer(cfg)
    tokens = torch.randint(0, 13, (8, 3))
    tokens[:, 2] = 13
    labels = torch.randint(0, 13, (8,))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(50):
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(logits[:, -1, :], labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5
