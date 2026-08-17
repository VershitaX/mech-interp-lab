"""
Data generation for the modular addition task: a + b (mod p).

Input sequence: [a, b, "="]  (token p represents "=")
Target: (a + b) mod p, predicted at the final position.

This task is the canonical grokking benchmark: with a small held-out
fraction of examples used for training, the model first memorizes the
training set (train acc -> 100%, test acc stays near random) and then,
much later in training, suddenly generalizes (test acc jumps to 100%).
That delayed generalization is "grokking", and reverse-engineering what
changes in the weights when it happens is the interpretability payoff.
"""
import torch


def make_dataset(p: int = 113, seed: int = 0):
    """All p*p pairs (a, b), 0 <= a, b < p."""
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    equals_token = torch.full_like(a, p)  # "=" token id
    inputs = torch.stack([a, b, equals_token], dim=1)  # [p*p, 3]
    labels = (a + b) % p
    return inputs, labels


def train_test_split(inputs, labels, train_frac: float = 0.3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    n = inputs.shape[0]
    perm = torch.randperm(n, generator=g)
    n_train = int(n * train_frac)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    return (
        inputs[train_idx], labels[train_idx],
        inputs[test_idx], labels[test_idx],
    )
