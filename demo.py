"""
End-to-end demo: trains a small model on modular addition, then runs
the full interpretability pipeline on it and saves all figures.

This is the fastest way to see the whole project work: one command,
a few minutes, produces a checkpoint + a full set of interpretability
plots in figures/.

Usage:
    python demo.py                # quick settings (~1-2 min on CPU)
    python demo.py --full         # real grokking run (p=113, slower, but shows the actual phenomenon)
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "interp"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="run the real grokking setup (p=113, 15k epochs)")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 1/3: Training model")
    print("=" * 60)
    train_cmd = [sys.executable, "src/train.py"]
    if not args.full:
        train_cmd.append("--quick")
    subprocess.run(train_cmd, check=True)

    print()
    print("=" * 60)
    print("STEP 2/3: Loading checkpoint")
    print("=" * 60)
    import torch
    from model import Transformer
    from data import make_dataset

    ckpt = torch.load("checkpoints/model.pt", weights_only=False)
    cfg = ckpt["config"]
    model = Transformer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    p = cfg.d_vocab - 1
    print(f"Loaded model trained on mod {p} addition, {model.num_params():,} params")

    print()
    print("=" * 60)
    print("STEP 3/3: Running interpretability analysis + saving figures")
    print("=" * 60)
    from visualize import plot_training_curve, plot_embedding_fourier_norms, plot_attention_pattern, plot_neuron_periodicity
    from fourier_analysis import top_frequencies, neuron_periodicity

    plot_training_curve("checkpoints/metrics.csv")
    print("  saved figures/training_curve.png")

    plot_embedding_fourier_norms(model, p)
    print("  saved figures/embedding_fourier_norms.png")

    top_freqs = top_frequencies(model, p, k=5)
    print(f"  top embedding frequencies: {top_freqs}")

    inputs, labels = make_dataset(p=p)
    example = inputs[:1]
    with torch.no_grad():
        _, cache = model(example, return_cache=True)
    plot_attention_pattern(cache, layer=0, head=0, tokens_labels=["a", "b", "="])
    print("  saved figures/attention_pattern.png")

    plot_neuron_periodicity(model, p)
    print("  saved figures/neuron_periodicity.png")

    print()
    print("Done. Check the figures/ directory and checkpoints/metrics.csv")
    print("Note: with --quick settings the model likely only MEMORIZED (not groked) -- ")
    print("run with --full for the real grokking phenomenon (test acc jumping late in training).")


if __name__ == "__main__":
    main()
