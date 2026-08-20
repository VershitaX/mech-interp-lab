"""
Interactive demo for the Mech Interp Lab project.

Lets a visitor:
  - Type in two numbers and watch the model predict their sum mod p
  - See the model's attention pattern for that specific input, live
  - Browse the training curve (if a full run has been logged)
  - See which Fourier frequencies the model's embedding relies on
  - See which MLP neurons look most "periodic" (candidates for the
    addition circuit)

Run locally with:
    streamlit run app.py

If no trained checkpoint is found in checkpoints/model.pt, this app
trains a small one on the fly (takes ~10-20 sec, cached after that) so
the demo always works out of the box, even on a fresh clone.
"""
import os
import sys

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "interp"))

from model import Config, Transformer
from data import make_dataset, train_test_split
from fourier_analysis import top_frequencies, embedding_frequency_norms, neuron_periodicity

st.set_page_config(page_title="Mech Interp Lab", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner=False)
def get_model():
    """Load a saved checkpoint if one exists; otherwise train a small
    model on the fly so the app works out of the box."""
    ckpt_path = "checkpoints/model.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = Transformer(cfg)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, cfg.d_vocab - 1, True

    # No checkpoint -> train a quick one now
    p = 13
    cfg = Config(d_vocab=p + 1, d_model=32, n_ctx=3, n_layers=1, n_heads=4, d_head=8, d_mlp=128)
    model = Transformer(cfg)
    inputs, labels = make_dataset(p=p)
    train_in, train_lb, _, _ = train_test_split(inputs, labels, train_frac=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.5)
    for _ in range(300):
        logits = model(train_in)
        loss = F.cross_entropy(logits[:, -1, :], train_lb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model, p, False


model, p, has_real_checkpoint = get_model()

st.title("🧠 Mech Interp Lab: Interactive Demo")
st.caption("Companion demo for the [Mech Interp Lab](https://github.com/VershitaX/mech-interp-lab) project")

if not has_real_checkpoint:
    st.info(
        f"No trained checkpoint found in `checkpoints/model.pt`, so this demo trained a small "
        f"model on the fly (mod {p} addition, ~300 steps) so everything below still works. "
        f"For the real grokking result, run `python demo.py --full` first, then reload this app.",
        icon="ℹ️",
    )

tab1, tab2, tab3, tab4 = st.tabs(["🔢 Try it yourself", "📈 Training curve", "🌊 Fourier analysis", "⚡ Neuron periodicity"])

# ---------------------------------------------------------------
with tab1:
    st.subheader("Ask the model to add two numbers (mod p)")
    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input(f"a (0 to {p - 1})", min_value=0, max_value=p - 1, value=3)
    with col2:
        b = st.number_input(f"b (0 to {p - 1})", min_value=0, max_value=p - 1, value=5)

    true_answer = (a + b) % p
    tokens = torch.tensor([[a, b, p]])  # p = "=" token
    with torch.no_grad():
        logits, cache = model(tokens, return_cache=True)
    probs = F.softmax(logits[0, -1, :p], dim=-1).numpy()
    pred = int(np.argmax(probs))

    st.markdown(f"**Model predicts:** `{a} + {b} mod {p} = {pred}`  |  **True answer:** `{true_answer}`")
    if pred == true_answer:
        st.success(f"Correct! (confidence: {probs[pred]:.1%})")
    else:
        st.error(f"Wrong — model said {pred}, correct is {true_answer} (confidence in {pred}: {probs[pred]:.1%})")

    st.markdown("**Prediction distribution** (model's confidence across all possible answers):")
    st.bar_chart({"probability": probs})

    st.markdown("**Attention pattern for this input** (which position attends to which):")
    pattern = cache["blocks.0.attn.pattern"][0].mean(dim=0).numpy()  # average over heads
    st.dataframe(
        {
            "query \\ key": ["a", "b", "="],
            "attends to a": [f"{pattern[i,0]:.2f}" for i in range(3)],
            "attends to b": [f"{pattern[i,1]:.2f}" for i in range(3)],
            "attends to =": [f"{pattern[i,2]:.2f}" for i in range(3)],
        },
        hide_index=True,
    )
    st.caption("Row = query position, columns = how much attention it pays to each key position (averaged over heads).")

# ---------------------------------------------------------------
with tab2:
    st.subheader("Training curve")
    metrics_path = "checkpoints/metrics.csv"
    if os.path.exists(metrics_path):
        import pandas as pd

        df = pd.read_csv(metrics_path)
        st.line_chart(df.set_index("epoch")[["train_acc", "test_acc"]])
        st.caption(
            "If you ran the full training (`python demo.py --full`), you should see test accuracy "
            "stay low for a long stretch, then jump sharply — that's grokking."
        )
    else:
        st.warning("No `checkpoints/metrics.csv` found yet. Run `python demo.py` or `python src/train.py` first.")

# ---------------------------------------------------------------
with tab3:
    st.subheader("Which Fourier frequencies does the embedding use?")
    st.caption(
        "Grokked models on modular addition represent numbers as points on a circle and rely on "
        "a small number of frequencies, rather than spreading energy uniformly. This checks that directly."
    )
    ranked = embedding_frequency_norms(model, p)[:20]
    labels = [r[0] for r in ranked]
    norms = [r[1] for r in ranked]
    st.bar_chart({"energy": norms}, x_label="rank", y_label="embedding energy")
    st.write("Top labels (by energy):", ", ".join(labels[:8]))

    top_freqs = top_frequencies(model, p, k=5)
    st.markdown(f"**Top 5 frequencies by combined cos/sin energy:** {top_freqs}")

# ---------------------------------------------------------------
with tab4:
    st.subheader("Which MLP neurons look periodic?")
    st.caption(
        "A neuron with a high periodicity score fires in a clean periodic pattern as a function of "
        "(a+b) mod p — a strong candidate for being part of the addition circuit, as opposed to noise."
    )
    top_neurons = neuron_periodicity(model, p, top_k=15)
    neuron_ids = [str(n) for n, _ in top_neurons]
    scores = [s for _, s in top_neurons]
    st.bar_chart({"periodicity": scores})
    st.write("Neuron indices (highest periodicity first):", ", ".join(neuron_ids))

st.divider()
st.caption("Built from scratch in PyTorch — no HuggingFace models. See the [full repo](https://github.com/VershitaX/mech-interp-lab) for training code and interpretability tools.")
