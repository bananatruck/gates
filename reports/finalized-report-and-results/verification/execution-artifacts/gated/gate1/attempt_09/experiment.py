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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
n_test = 200

test_X_b = np.hstack([np.ones((n_test, 1)), test_X])

print("=" * 72)
print("Experiment purpose:")
print("This controlled data-efficiency gate isolates the effect of training-set")
print("size on test accuracy. Both arms use the exact same multinomial logistic")
print("regression model (softmax + cross-entropy), vanilla gradient descent,")
print("identical seed (42), identical initial zero weights, identical learning")
print("rate (0.1), and identical number of gradient steps (300). The only")
print("difference is the number of training samples: Arm A sees 100 labelled")
print("samples (25% of the 400-sample pool) while Arm B sees all 400 samples.")
print("Both arms are evaluated on the same frozen 200-sample test set, so any")
print("accuracy gap is attributable solely to data quantity. The headline metric,")
print("eff.efficiency_ratio = acc_at_25 / acc_at_100, measures how much of the")
print("full-data accuracy is retained when only a quarter of the labels are used.")
print("=" * 72)

# --- Arm A: train on 25% of the pool (100 labelled samples) ---
n_A = 100
X_b_A = np.hstack([np.ones((n_A, 1)), pool_X[:n_A]])
y_A = pool_y[:n_A]
one_hot_A = np.eye(3)[y_A]
W_A = np.zeros((9, 3))
loss_A_history = []

t0 = time.perf_counter()
for step in range(n_steps + 1):
    logits_A = X_b_A @ W_A
    exp_logits_A = np.exp(logits_A - logits_A.max(axis=1, keepdims=True))
    probs_A = exp_logits_A / exp_logits_A.sum(axis=1, keepdims=True)
    loss_A = -np.mean(np.log(probs_A[np.arange(n_A), y_A] + 1e-12))
    loss_A_history.append(loss_A)
    if step % 10 == 0:
        print(f"Arm A step {step} train_loss {loss_A:.6f}")
    if step < n_steps:
        grad_A = (1 / n_A) * X_b_A.T @ (probs_A - one_hot_A)
        W_A -= lr * grad_A
t1 = time.perf_counter()
arm_A_train_s = t1 - t0

# --- Arm B: train on the full pool (400 labelled samples) ---
n_B = 400
X_b_B = np.hstack([np.ones((n_B, 1)), pool_X])
y_B = pool_y
one_hot_B = np.eye(3)[y_B]
W_B = np.zeros((9, 3))
loss_B_history = []

for step in range(n_steps + 1):
    logits_B = X_b_B @ W_B
    exp_logits_B = np.exp(logits_B - logits_B.max(axis=1, keepdims=True))
    probs_B = exp_logits_B / exp_logits_B.sum(axis=1, keepdims=True)
    loss_B = -np.mean(np.log(probs_B[np.arange(n_B), y_B] + 1e-12))
    loss_B_history.append(loss_B)
    if step % 10 == 0:
        print(f"Arm B step {step} train_loss {loss_B:.6f}")
    if step < n_steps:
        grad_B = (1 / n_B) * X_b_B.T @ (probs_B - one_hot_B)
        W_B -= lr * grad_B
t2 = time.perf_counter()
arm_B_train_s = t2 - t1

train_s = arm_A_train_s + arm_B_train_s

print("=" * 72)
print("Evaluation explanation:")
print("After the full 300-step training, each arm's learned weight matrix is")
print("frozen and applied to the fixed 200-sample test set. Test accuracy is the")
print("fraction of argmax predictions that match the true class labels. From the")
print("two accuracies we compute eff.efficiency_ratio = acc_at_25 / acc_at_100.")
print("A ratio below 1 shows that more data helps; the magnitude tells us how")
print("much classification quality is lost when only 100 of the 400 labels are")
print("used. The wallclock time eff.train_s covers only the two training loops,")
print("so it is a direct measure of the computational cost of the comparison.")
print("=" * 72)

pred_25 = np.argmax(test_X_b @ W_A, axis=1)
acc_at_25 = np.mean(pred_25 == test_y)
pred_100 = np.argmax(test_X_b @ W_B, axis=1)
acc_at_100 = np.mean(pred_100 == test_y)
efficiency_ratio = acc_at_25 / acc_at_100

print(f"Arm A (100 samples) test accuracy: {acc_at_25:.6f}")
print(f"Arm B (400 samples) test accuracy: {acc_at_100:.6f}")
print(f"Efficiency ratio (25%/100%): {efficiency_ratio:.6f}")
print(f"Total training wallclock seconds: {train_s:.6f}")
print(f"  (Arm A training wallclock: {arm_A_train_s:.6f} s; "
      f"Arm B training wallclock: {arm_B_train_s:.6f} s)")

# Provenance metadata (not results; purely for reproducibility).
record_metadata("seed", seed)
record_metadata("model", "multinomial_logistic_regression_softmax")
record_metadata("optimizer", "vanilla_gradient_descent")
record_metadata("learning_rate", lr)
record_metadata("n_gradient_steps", n_steps)
record_metadata("arm_A_train_samples", n_A)
record_metadata("arm_B_train_samples", n_B)
record_metadata("test_set_size", n_test)
record_metadata("arm_A_train_s", arm_A_train_s)
record_metadata("arm_B_train_s", arm_B_train_s)

record_result("eff.acc_at_25", acc_at_25, unit="accuracy")
record_result("eff.acc_at_100", acc_at_100, unit="accuracy")
record_result("eff.efficiency_ratio", efficiency_ratio, unit="ratio")
record_result("eff.train_s", train_s, unit="seconds")

# Figure 1: training loss curves, colorful artistic design
steps_axis = np.arange(n_steps + 1)
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
ax.plot(steps_axis, loss_A_history, color="#ff6b6b", lw=2.5, alpha=0.95,
        label="Arm A (100 labelled samples)")
ax.plot(steps_axis, loss_B_history, color="#4ecdc4", lw=2.5, alpha=0.95,
        label="Arm B (400 labelled samples)")
ax.fill_between(steps_axis, loss_A_history, loss_B_history, color="#a29bfe", alpha=0.18)
ax.axhline(np.log(3), color="#f9c74f", lw=1.8, ls="--", alpha=0.85,
           label="Chance-level loss ln(3)")
marker_steps = np.arange(0, n_steps + 1, 25)
ax.scatter(marker_steps, np.asarray(loss_A_history)[marker_steps], color="white",
           edgecolor="#ff6b6b", s=30, zorder=3)
ax.scatter(marker_steps, np.asarray(loss_B_history)[marker_steps], color="white",
           edgecolor="#4ecdc4", s=30, zorder=3)
ax.set_xlabel("Gradient descent step", fontsize=12, fontweight="bold")
ax.set_ylabel("Cross-entropy training loss", fontsize=12, fontweight="bold")
ax.set_title("Training loss: data-limited (100) vs data-rich (400) Arm",
             fontsize=13, fontweight="bold")
ax.legend(frameon=True, fancybox=True, shadow=True, loc="upper right", fontsize=9)
ax.grid(True, linestyle="--", alpha=0.35)
ax.set_facecolor("#f7f7f7")
fig.patch.set_facecolor("#fdfdfd")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig("Figure_1.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Figure 2: test accuracy comparison with efficiency annotation
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
bar_labels = ["Arm A\n100 samples", "Arm B\n400 samples"]
bar_colors = ["#ff6b6b", "#4ecdc4"]
bars = ax.bar(bar_labels, [acc_at_25, acc_at_100], color=bar_colors,
              edgecolor="black", linewidth=1.2, width=0.55)
ax.axhline(1 / 3, color="#f94144", lw=1.6, ls=":", alpha=0.9)
ax.text(0.5, 1 / 3 + 0.012, f"chance = {1 / 3:.3f}", ha="center",
        fontsize=9, style="italic", color="#f94144")
ax.set_ylim(0, 1.08)
ax.set_ylabel("Test accuracy (fixed 200-sample test set)", fontsize=12, fontweight="bold")
ax.set_title("Data efficiency: test accuracy vs training-set size",
             fontsize=13, fontweight="bold")
for bar, acc in zip(bars, [acc_at_25, acc_at_100]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{acc:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
gap_mid = (acc_at_25 + acc_at_100) / 2
ax.annotate("", xy=(1.32, acc_at_100), xytext=(1.32, acc_at_25),
            arrowprops=dict(arrowstyle="<->", color="#2d3436", lw=1.6))
ax.text(1.36, gap_mid, f"Δ = {acc_at_100 - acc_at_25:.4f}",
        fontsize=10, fontweight="bold", color="#2d3436", va="center")
ax.text(0.5, 0.95,
        f"Efficiency ratio = acc_at_25 / acc_at_100 = {efficiency_ratio:.4f}",
        transform=ax.transAxes, ha="center", fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffe66d",
                  edgecolor="black", alpha=0.9))
ax.grid(True, axis="y", linestyle="--", alpha=0.35)
ax.set_facecolor("#f7f7f7")
fig.patch.set_facecolor("#fdfdfd")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig("Figure_2.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Figures saved: Figure_1.png (loss curves), Figure_2.png (accuracy comparison).")

# Final summary of the exact four-row table the report must present.
print("\nFinal recorded table for the report (all values are recorded results):")
print(f"  eff.acc_at_25        = {acc_at_25:.6f}")
print(f"  eff.acc_at_100       = {acc_at_100:.6f}")
print(f"  eff.efficiency_ratio = {efficiency_ratio:.6f}")
print(f"  eff.train_s          = {train_s:.6f} seconds")
print("Headline: with 100 labels we retain "
      f"{100 * efficiency_ratio:.2f}% of the 400-label test accuracy.")
print("\nNOTE: eff.train_s is a wall-clock measurement (time.perf_counter()) and is")
print("therefore not perfectly reproducible across runs. The harness may register")
print("more than one entry for eff.train_s (e.g., a verification pass followed by")
print("the final run); the value to report is the LAST one recorded, which is the")
print("training-loop time of this final execution.")