"""
Train the transformer on modular addition and reproduce grokking.

Usage:
    python src/train.py                      # full run (p=113), reproduces grokking (~15-30 min on CPU, faster on GPU)
    python src/train.py --quick              # fast smoke-test run (p=13, few epochs) to check everything works

Key ingredient for grokking to happen: a SMALL training fraction (model can
memorize) + weight decay (pushes the model away from memorizing, "grokking"
generalization emerges as training continues past the memorization point).
"""
import argparse
import csv
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import Config, Transformer
from data import make_dataset, train_test_split


def evaluate(model, inputs, labels, batch_size=8192):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, inputs.shape[0], batch_size):
            batch_in = inputs[i : i + batch_size]
            batch_lb = labels[i : i + batch_size]
            logits = model(batch_in)
            preds = logits[:, -1, :].argmax(dim=-1)
            correct += (preds == batch_lb).sum().item()
    model.train()
    return correct / inputs.shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=113, help="modulus (prime works best)")
    parser.add_argument("--train_frac", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--quick", action="store_true", help="fast smoke test: tiny p, few epochs")
    args = parser.parse_args()

    if args.quick:
        args.p = 13
        args.epochs = 500
        args.d_model = 32
        args.log_every = 20

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    inputs, labels = make_dataset(p=args.p)
    train_in, train_lb, test_in, test_lb = train_test_split(
        inputs, labels, train_frac=args.train_frac, seed=args.seed
    )
    print(f"Task: a + b mod {args.p} | train examples: {train_in.shape[0]} | test examples: {test_in.shape[0]}")

    cfg = Config(
        d_vocab=args.p + 1,
        d_model=args.d_model,
        n_ctx=3,
        n_layers=1,
        n_heads=args.n_heads,
        d_head=args.d_model // args.n_heads,
        d_mlp=args.d_model * 4,
        seed=args.seed,
    )
    model = Transformer(cfg).to(args.device)
    print(f"Model params: {model.num_params():,}")

    train_in, train_lb = train_in.to(args.device), train_lb.to(args.device)
    test_in, test_lb = test_in.to(args.device), test_lb.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))

    log_path = os.path.join(args.save_dir, "metrics.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_acc", "elapsed_sec"])

    start = time.time()
    for epoch in range(args.epochs + 1):
        logits = model(train_in)
        loss = F.cross_entropy(logits[:, -1, :], train_lb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % args.log_every == 0 or epoch == args.epochs:
            train_acc = evaluate(model, train_in, train_lb)
            test_acc = evaluate(model, test_in, test_lb)
            elapsed = time.time() - start
            print(f"epoch {epoch:6d} | loss {loss.item():.4f} | train_acc {train_acc:.3f} | test_acc {test_acc:.3f} | {elapsed:.0f}s")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, loss.item(), train_acc, test_acc, elapsed])

    ckpt_path = os.path.join(args.save_dir, "model.pt")
    torch.save({"model_state": model.state_dict(), "config": cfg}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Saved metrics log to {log_path}")


if __name__ == "__main__":
    main()
