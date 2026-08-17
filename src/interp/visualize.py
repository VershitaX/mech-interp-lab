"""
Visualization utilities. Every function saves a PNG to figures/ and
also returns the matplotlib Figure so it can be shown inline in a
notebook.
"""
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curve(metrics_csv: str, save_path: str = "figures/training_curve.png"):
    import csv

    epochs, train_accs, test_accs = [], [], []
    with open(metrics_csv) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_accs.append(float(row["train_acc"]))
            test_accs.append(float(row["test_acc"]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_accs, label="train accuracy", linewidth=2)
    ax.plot(epochs, test_accs, label="test accuracy", linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Grokking: train acc saturates fast, test acc lags then jumps")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_embedding_fourier_norms(model, p: int, save_path: str = "figures/embedding_fourier_norms.png"):
    from fourier_analysis import embedding_frequency_norms

    ranked = embedding_frequency_norms(model, p)
    labels = [r[0] for r in ranked[:20]]
    norms = [r[1] for r in ranked[:20]]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(norms)), norms)
    ax.set_xticks(range(len(norms)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("embedding energy (L2 norm)")
    ax.set_title("Which Fourier frequencies does the token embedding use?")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_attention_pattern(cache, layer: int, head: int, tokens_labels, save_path: str = "figures/attention_pattern.png"):
    pattern = cache[f"blocks.{layer}.attn.pattern"][0, head].cpu().numpy()  # [q_pos, k_pos]
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(pattern, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(tokens_labels)))
    ax.set_xticklabels(tokens_labels)
    ax.set_yticks(range(len(tokens_labels)))
    ax.set_yticklabels(tokens_labels)
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    ax.set_title(f"Attention pattern: layer {layer}, head {head}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_neuron_periodicity(model, p: int, save_path: str = "figures/neuron_periodicity.png"):
    from fourier_analysis import neuron_periodicity

    top = neuron_periodicity(model, p, top_k=15)
    neurons = [str(n) for n, _ in top]
    scores = [s for _, s in top]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(neurons, scores)
    ax.set_xlabel("MLP neuron index")
    ax.set_ylabel("periodicity score (0-1)")
    ax.set_title("Most periodic MLP neurons (candidates for the addition circuit)")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
