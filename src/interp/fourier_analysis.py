"""
Fourier analysis of the learned embeddings and neuron activations.

This is THE key technique from Nanda et al.'s grokking interpretability
work: a model trained on modular addition, once it has "grokked", turns
out to represent numbers as points on a circle and compute (a+b) mod p
using trig identities (cos/sin addition formulas) implemented via its
attention and MLP weights. That structure is invisible by staring at
raw weights, but jumps out immediately in the frequency domain.

This module lets you check, for a given checkpoint:
  1. Which Fourier frequencies the token embedding concentrates on
  2. Whether neuron activations are periodic in (a+b) mod p
  3. A visual "does this model look like it grokked" sanity check
"""
import torch
import numpy as np


def fourier_basis(p: int):
    """Returns a [p, p] orthonormal Fourier basis: constant, then
    (cos, sin) pairs for each frequency 1..(p-1)//2."""
    basis = [np.ones(p) / np.sqrt(p)]
    labels = ["const"]
    for freq in range(1, p // 2 + 1):
        cos_v = np.cos(2 * np.pi * freq * np.arange(p) / p)
        sin_v = np.sin(2 * np.pi * freq * np.arange(p) / p)
        cos_v = cos_v / np.linalg.norm(cos_v)
        basis.append(cos_v)
        labels.append(f"cos_{freq}")
        if freq != p / 2:
            sin_v = sin_v / np.linalg.norm(sin_v)
            basis.append(sin_v)
            labels.append(f"sin_{freq}")
    return np.stack(basis), labels  # [n_basis, p]


def embedding_frequency_norms(model, p: int):
    """For each Fourier frequency, how much of the token embedding's
    variance (across the p number-tokens, excluding '=') lies in that
    frequency. A model that has grokked concentrates its energy on a
    small handful of frequencies rather than spreading uniformly."""
    W_E = model.embed.W_E.detach().cpu().numpy()[:p]  # [p, d_model]
    basis, labels = fourier_basis(p)  # [n_basis, p]
    coeffs = basis @ W_E  # [n_basis, d_model]
    norms = np.linalg.norm(coeffs, axis=1)  # [n_basis]
    order = np.argsort(-norms)
    return [(labels[i], float(norms[i])) for i in order]


def top_frequencies(model, p: int, k: int = 5):
    """Convenience: the k Fourier frequencies (by cos/sin pair, merged)
    with the most embedding energy."""
    ranked = embedding_frequency_norms(model, p)
    freq_energy = {}
    for label, norm in ranked:
        if label == "const":
            continue
        freq = int(label.split("_")[1])
        freq_energy[freq] = freq_energy.get(freq, 0.0) + norm**2
    ranked_freqs = sorted(freq_energy.items(), key=lambda kv: -kv[1])
    return ranked_freqs[:k]


def neuron_periodicity(model, p: int, layer: int = 0, top_k: int = 10):
    """Runs the model on all (a,b) pairs and checks, for each MLP neuron,
    how periodic its activation is as a function of (a+b) mod p. Neurons
    implementing the addition circuit fire in a clean periodic pattern;
    ones that were pruned away / irrelevant look noisy.

    Returns the top_k neurons ranked by periodicity score (fraction of
    variance explained by the single best Fourier frequency).
    """
    from data import make_dataset

    inputs, labels = make_dataset(p=p)
    device = next(model.parameters()).device
    with torch.no_grad():
        _, cache = model(inputs.to(device), return_cache=True)
    acts = cache[f"blocks.{layer}.mlp.post_act"][:, -1, :].cpu().numpy()  # [p*p, d_mlp]

    # average activation as a function of (a+b) mod p, per neuron
    ab_mod = labels.numpy()
    d_mlp = acts.shape[1]
    avg_by_target = np.zeros((p, d_mlp))
    for t in range(p):
        mask = ab_mod == t
        avg_by_target[t] = acts[mask].mean(axis=0)

    basis, _ = fourier_basis(p)  # [n_basis, p]
    coeffs = basis @ avg_by_target  # [n_basis, d_mlp]
    energy = coeffs**2
    total_energy = energy.sum(axis=0) + 1e-9
    best_freq_energy = energy[1:].max(axis=0)  # exclude constant term
    periodicity_score = best_freq_energy / total_energy

    order = np.argsort(-periodicity_score)[:top_k]
    return [(int(n), float(periodicity_score[n])) for n in order]
