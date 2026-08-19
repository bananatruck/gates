import numpy as np
import time
from datasets import load_dataset

# Required external HuggingFace dataset (harness requirement); the numerical experiment itself is numpy-only.
_hf_fixture = load_dataset("poem_sentiment", split="train[:1]")

seed = 42
rng = np.random.default_rng(seed)

X = rng.normal(size=(600, 8))
W_true = rng.normal(size=(8, 3))
scores = X @ W_true
y = np.argmax(scores, axis=1)

idx = rng.permutation(600)
test_idx = idx[:200]
pool_idx = idx[200:]
test_X = X[test_idx]
test_y = y[test_idx]
pool_X = X[pool_idx]
pool_y = y[pool_idx]

lr = 0.1
n_steps = 300

test_X_b = np.hstack([np.ones((200, 1)), test_X])

# Arm A: 25% of the 400-sample pool (100 samples)
n_A = 100
X_b_A = np.hstack([np.ones((n_A, 1)), pool_X[:n_A]])
y_A = pool_y[:n_A]
one_hot_A = np.eye(3)[y_A]
W_A = np.zeros((9, 3))

t0 = time.perf_counter()
for step in range(n_steps + 1):
    logits_A = X_b_A @ W_A
    exp_logits_A = np.exp(logits_A - logits_A.max(axis=1, keepdims=True))
    probs_A = exp_logits_A / exp_logits_A.sum(axis=1, keepdims=True)
    loss_A = -np.mean(np.log(probs_A[np.arange(n_A), y_A] + 1e-12))
    if step % 10 == 0:
        print(f"Arm A step {step} train_loss {loss_A:.6f}")
    if step < n_steps:
        grad_A = (1 / n_A) * X_b_A.T @ (probs_A - one_hot_A)
        W_A -= lr * grad_A
t1 = time.perf_counter()

# Arm B: full 400-sample pool
n_B = 400
X_b_B = np.hstack([np.ones((n_B, 1)), pool_X])
y_B = pool_y
one_hot_B = np.eye(3)[y_B]
W_B = np.zeros((9, 3))

for step in range(n_steps + 1):
    logits_B = X_b_B @ W_B
    exp_logits_B = np.exp(logits_B - logits_B.max(axis=1, keepdims=True))
    probs_B = exp_logits_B / exp_logits_B.sum(axis=1, keepdims=True)
    loss_B = -np.mean(np.log(probs_B[np.arange(n_B), y_B] + 1e-12))
    if step % 10 == 0:
        print(f"Arm B step {step} train_loss {loss_B:.6f}")
    if step < n_steps:
        grad_B = (1 / n_B) * X_b_B.T @ (probs_B - one_hot_B)
        W_B -= lr * grad_B
t2 = time.perf_counter()

train_s = (t1 - t0) + (t2 - t1)

pred_25 = np.argmax(test_X_b @ W_A, axis=1)
acc_at_25 = np.mean(pred_25 == test_y)
pred_100 = np.argmax(test_X_b @ W_B, axis=1)
acc_at_100 = np.mean(pred_100 == test_y)
efficiency_ratio = acc_at_25 / acc_at_100

record_metadata("seed", seed)
record_result("eff.acc_at_25", acc_at_25)
record_result("eff.acc_at_100", acc_at_100)
record_result("eff.efficiency_ratio", efficiency_ratio)
record_result("eff.train_s", train_s)
import numpy as np
import time
from datasets import load_dataset

# Required external HuggingFace dataset (harness requirement); the numerical experiment itself is numpy-only.
_hf_fixture = load_dataset("poem_sentiment", split="train[:1]")

seed = 42
rng = np.random.default_rng(seed)

X = rng.normal(size=(600, 8))
W_true = rng.normal(size=(8, 3))
scores = X @ W_true
y = np.argmax(scores, axis=1)

idx = rng.permutation(600)
test_idx = idx[:200]
pool_idx = idx[200:]
test_X = X[test_idx]
test_y = y[test_idx]
pool_X = X[pool_idx]
pool_y = y[pool_idx]

lr = 0.1
n_steps = 300

test_X_b = np.hstack([np.ones((200, 1)), test_X])

# ---- Experiment 1 (Arm A): 25% of the 400-sample pool (100 labelled samples) ----
print("=" * 90)
print("EXPERIMENT 1 (Arm A): Multinomial logistic regression on 25% of the 400-sample pool.")
print("What this shows: the test accuracy achievable with only 100 labelled training samples on a")
print("fixed 200-sample held-out test set, under a controlled setup (seed=42, lr=0.1, 300 GD steps,")
print("zero-initialised weights, identical model/optimiser/evaluation as Arm B).")
print("Purpose: establish the 'data-poor' baseline for the data-efficiency comparison.")
print("=" * 90)

n_A = 100
X_b_A = np.hstack([np.ones((n_A, 1)), pool_X[:n_A]])
y_A = pool_y[:n_A]
one_hot_A = np.eye(3)[y_A]
W_A = np.zeros((9, 3))

steps_A = []
losses_A = []
t0 = time.perf_counter()
for step in range(n_steps + 1):
    logits_A = X_b_A @ W_A
    exp_logits_A = np.exp(logits_A - logits_A.max(axis=1, keepdims=True))
    probs_A = exp_logits_A / exp_logits_A.sum(axis=1, keepdims=True)
    loss_A = -np.mean(np.log(probs_A[np.arange(n_A), y_A] + 1e-12))
    if step % 10 == 0:
        print(f"Arm A step {step} train_loss {loss_A:.6f}")
        steps_A.append(step)
        losses_A.append(loss_A)
    if step < n_steps:
        grad_A = (1 / n_A) * X_b_A.T @ (probs_A - one_hot_A)
        W_A -= lr * grad_A
t1 = time.perf_counter()

# ---- Experiment 2 (Arm B): 100% of the 400-sample pool ----
print("=" * 90)
print("EXPERIMENT 2 (Arm B): Multinomial logistic regression on 100% of the 400-sample pool.")
print("What this shows: the test accuracy on the same fixed 200-sample test set using all 400")
print("labelled training samples. This is the fully-supervised reference accuracy.")
print("Purpose: provide the denominator for the efficiency ratio acc_at_25 / acc_at_100, and quantify")
print("the performance gain from 4x more labelled data.")
print("=" * 90)

n_B = 400
X_b_B = np.hstack([np.ones((n_B, 1)), pool_X])
y_B = pool_y
one_hot_B = np.eye(3)[y_B]
W_B = np.zeros((9, 3))

steps_B = []
losses_B = []
for step in range(n_steps + 1):
    logits_B = X_b_B @ W_B
    exp_logits_B = np.exp(logits_B - logits_B.max(axis=1, keepdims=True))
    probs_B = exp_logits_B / exp_logits_B.sum(axis=1, keepdims=True)
    loss_B = -np.mean(np.log(probs_B[np.arange(n_B), y_B] + 1e-12))
    if step % 10 == 0:
        print(f"Arm B step {step} train_loss {loss_B:.6f}")
        steps_B.append(step)
        losses_B.append(loss_B)
    if step < n_steps:
        grad_B = (1 / n_B) * X_b_B.T @ (probs_B - one_hot_B)
        W_B -= lr * grad_B
t2 = time.perf_counter()

train_s = (t1 - t0) + (t2 - t1)

# ---- Evaluation on the fixed 200-sample test set ----
pred_25 = np.argmax(test_X_b @ W_A, axis=1)
acc_at_25 = np.mean(pred_25 == test_y)
pred_100 = np.argmax(test_X_b @ W_B, axis=1)
acc_at_100 = np.mean(pred_100 == test_y)
efficiency_ratio = acc_at_25 / acc_at_100

print("=" * 90)
print("FINAL MEASUREMENTS (recorded under eff.* keys):")
print(f"acc_at_25        = {acc_at_25:.6f}")
print(f"acc_at_100       = {acc_at_100:.6f}")
print(f"efficiency_ratio = {efficiency_ratio:.6f}")
print(f"train_s          = {train_s:.6f} seconds (both training runs, wallclock)")
print("=" * 90)

# ---- Record provenance and results ----
record_metadata("seed", seed)
record_result("eff.acc_at_25", acc_at_25, unit="ratio")
record_result("eff.acc_at_100", acc_at_100, unit="ratio")
record_result("eff.efficiency_ratio", efficiency_ratio, unit="ratio")
record_result("eff.train_s", train_s, unit="seconds")

# ---- Figures ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure 1: training loss curves (colourful, artistic)
plt.figure(figsize=(10, 6))
plt.plot(steps_A, losses_A, marker="o", linewidth=2.5, markersize=5,
         color="#e74c3c", label="Arm A (100 labelled samples)")
plt.plot(steps_B, losses_B, marker="s", linewidth=2.5, markersize=5,
         color="#3498db", label="Arm B (400 labelled samples)")
plt.xlabel("Gradient descent step", fontsize=12)
plt.ylabel("Training cross-entropy loss", fontsize=12)
plt.title("Training Loss Curves: 100 vs 400 Labelled Samples", fontsize=14, fontweight="bold")
plt.legend(frameon=True, fancybox=True, shadow=True, loc="upper right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Figure_1.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved Figure_1.png (training loss curves).")

# Figure 2: test accuracy comparison bar chart
fig, ax = plt.subplots(figsize=(8, 6))
arm_names = ["Arm A\n100 labelled\nsamples", "Arm B\n400 labelled\nsamples"]
accs = [acc_at_25, acc_at_100]
colors = ["#e67e22", "#27ae60"]
bars = ax.bar(arm_names, accs, color=colors, alpha=0.9, edgecolor="white", linewidth=2, width=0.55)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Test accuracy on fixed 200-sample set", fontsize=12)
ax.set_title("Data Efficiency: Test Accuracy vs Training-Set Size", fontsize=14, fontweight="bold")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{acc:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=12)
ax.text(0.5, 0.92, f"Efficiency ratio = {efficiency_ratio:.4f}",
        transform=ax.transAxes, ha="center", fontsize=13,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("Figure_2.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved Figure_2.png (test accuracy comparison).")