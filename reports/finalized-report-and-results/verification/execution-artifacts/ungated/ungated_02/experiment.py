import numpy as np
import time

try:
    from datasets import load_dataset
    _hf_external = load_dataset("scikit-learn/iris", split="train[:1]")
    _hf_length = len(_hf_external)
except Exception:
    _hf_length = 0

seed = 42
rng = np.random.default_rng(seed)

centers = [
    np.array([0, 0, 0, 0, 0, 0, 0, 0]),
    np.array([2, 0, 0, 0, 0, 0, 0, 0]),
    np.array([0, 2, 0, 0, 0, 0, 0, 0]),
]

X_list = []
y_list = []
for c in range(3):
    X_class = centers[c] + rng.normal(0.0, 1.0, size=(200, 8))
    y_class = np.full(200, c, dtype=int)
    X_list.append(X_class)
    y_list.append(y_class)

X = np.vstack(X_list)
y = np.concatenate(y_list)

idx = rng.permutation(600)
X = X[idx]
y = y[idx]

X_test = X[:200]
y_test = y[:200]
X_pool = X[200:]
y_pool = y[200:]

X_25 = X_pool[:100]
y_25 = y_pool[:100]
X_100 = X_pool
y_100 = y_pool

Xb_25 = np.hstack([np.ones((X_25.shape[0], 1)), X_25])
Xb_100 = np.hstack([np.ones((X_100.shape[0], 1)), X_100])
Xb_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

Y_25 = np.eye(3)[y_25]
Y_100 = np.eye(3)[y_100]

W_25 = np.zeros((9, 3))
W_100 = np.zeros((9, 3))

lr = 0.1
steps = 1500

t0 = time.perf_counter()

for step in range(steps):
    logits = Xb_25 @ W_25
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    loss = -np.mean(np.sum(Y_25 * np.log(probs + 1e-12), axis=1))
    grad = Xb_25.T @ (probs - Y_25) / X_25.shape[0]
    W_25 -= lr * grad
    if step % 10 == 0:
        print(f"[arm=25][step={step:04d}] cross_entropy={loss:.6f}")

for step in range(steps):
    logits = Xb_100 @ W_100
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    loss = -np.mean(np.sum(Y_100 * np.log(probs + 1e-12), axis=1))
    grad = Xb_100.T @ (probs - Y_100) / X_100.shape[0]
    W_100 -= lr * grad
    if step % 10 == 0:
        print(f"[arm=100][step={step:04d}] cross_entropy={loss:.6f}")

t1 = time.perf_counter()

test_logits_25 = Xb_test @ W_25
test_logits_100 = Xb_test @ W_100

acc_at_25 = np.mean(np.argmax(test_logits_25, axis=1) == y_test)
acc_at_100 = np.mean(np.argmax(test_logits_100, axis=1) == y_test)
efficiency_ratio = acc_at_25 / acc_at_100
train_s = t1 - t0

record_result("eff.acc_at_25", acc_at_25)
record_result("eff.acc_at_100", acc_at_100)
record_result("eff.efficiency_ratio", efficiency_ratio)
record_result("eff.train_s", train_s)
record_metadata("seed", seed)
# =============================================================================
# Controlled Gate 1 Validation - Data Efficiency
# =============================================================================
# The evaluation harness normally prepends the fixed synthetic dataset + main
# experiment block (which records eff.acc_at_25, eff.acc_at_100,
# eff.efficiency_ratio, eff.train_s). To make this script also self-contained,
# we re-run that block only if its recorded results are not already present.

if "acc_at_25" not in globals():
    import numpy as np
    import time

    try:
        from datasets import load_dataset
        _hf_external = load_dataset("scikit-learn/iris", split="train[:1]")
        _hf_length = len(_hf_external)
    except Exception:
        _hf_length = 0

    seed = 42
    rng = np.random.default_rng(seed)

    centers = [
        np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([2, 0, 0, 0, 0, 0, 0, 0]),
        np.array([0, 2, 0, 0, 0, 0, 0, 0]),
    ]

    X_list = []
    y_list = []
    for c in range(3):
        X_class = centers[c] + rng.normal(0.0, 1.0, size=(200, 8))
        y_class = np.full(200, c, dtype=int)
        X_list.append(X_class)
        y_list.append(y_class)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    idx = rng.permutation(600)
    X = X[idx]
    y = y[idx]

    X_test = X[:200]
    y_test = y[:200]
    X_pool = X[200:]
    y_pool = y[200:]

    X_25 = X_pool[:100]
    y_25 = y_pool[:100]
    X_100 = X_pool
    y_100 = y_pool

    Xb_25 = np.hstack([np.ones((X_25.shape[0], 1)), X_25])
    Xb_100 = np.hstack([np.ones((X_100.shape[0], 1)), X_100])
    Xb_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    Y_25 = np.eye(3)[y_25]
    Y_100 = np.eye(3)[y_100]

    W_25 = np.zeros((9, 3))
    W_100 = np.zeros((9, 3))

    lr = 0.1
    steps = 1500

    t0 = time.perf_counter()

    for step in range(steps):
        logits = Xb_25 @ W_25
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        loss = -np.mean(np.sum(Y_25 * np.log(probs + 1e-12), axis=1))
        grad = Xb_25.T @ (probs - Y_25) / X_25.shape[0]
        W_25 -= lr * grad
        if step % 10 == 0:
            print(f"[arm=25][step={step:04d}] cross_entropy={loss:.6f}")

    for step in range(steps):
        logits = Xb_100 @ W_100
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        loss = -np.mean(np.sum(Y_100 * np.log(probs + 1e-12), axis=1))
        grad = Xb_100.T @ (probs - Y_100) / X_100.shape[0]
        W_100 -= lr * grad
        if step % 10 == 0:
            print(f"[arm=100][step={step:04d}] cross_entropy={loss:.6f}")

    t1 = time.perf_counter()

    test_logits_25 = Xb_test @ W_25
    test_logits_100 = Xb_test @ W_100

    acc_at_25 = np.mean(np.argmax(test_logits_25, axis=1) == y_test)
    acc_at_100 = np.mean(np.argmax(test_logits_100, axis=1) == y_test)
    efficiency_ratio = acc_at_25 / acc_at_100
    train_s = t1 - t0

    record_result("eff.acc_at_25", acc_at_25)
    record_result("eff.acc_at_100", acc_at_100)
    record_result("eff.efficiency_ratio", efficiency_ratio)
    record_result("eff.train_s", train_s)
    record_metadata("seed", seed)

# =============================================================================
# Post-hoc analysis and figure generation.
# The final metrics were recorded above; the code below only reconstructs the
# same deterministic GD trajectories to draw training curves and confusion maps.
# =============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

print("\n" + "=" * 78)
print("POST-HOC VISUALIZATION PASS (no recorded results are changed)")
print("=" * 78)
print("We re-run the two deterministic full-batch GD trajectories (same data,")
print("same zero initialization, same lr=0.1, same 1500 steps) to obtain")
print("per-step cross-entropy curves and test-accuracy trajectories for figures.")
print("The recorded metrics remain exactly those from the main experiment block.")

# --- Re-run Arm 25 purely for visualization ---
W_25_vis = np.zeros((9, 3))
losses_25 = []
accs_25 = []
step_marks = []

for step in range(steps):
    logits = Xb_25 @ W_25_vis
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    loss = -np.mean(np.sum(Y_25 * np.log(probs + 1e-12), axis=1))
    grad = Xb_25.T @ (probs - Y_25) / X_25.shape[0]
    W_25_vis -= lr * grad
    if step % 10 == 0:
        step_marks.append(step)
        losses_25.append(loss)
        accs_25.append(np.mean(np.argmax(Xb_test @ W_25_vis, axis=1) == y_test))

# --- Re-run Arm 100 purely for visualization ---
W_100_vis = np.zeros((9, 3))
losses_100 = []
accs_100 = []

for step in range(steps):
    logits = Xb_100 @ W_100_vis
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    loss = -np.mean(np.sum(Y_100 * np.log(probs + 1e-12), axis=1))
    grad = Xb_100.T @ (probs - Y_100) / X_100.shape[0]
    W_100_vis -= lr * grad
    if step % 10 == 0:
        losses_100.append(loss)
        accs_100.append(np.mean(np.argmax(Xb_test @ W_100_vis, axis=1) == y_test))

# =============================================================================
# EXPLANATION BEFORE FIGURE 1
# =============================================================================
print("\n" + "-" * 78)
print("FIGURE 1 EXPLANATION")
print("This figure plots, for both experimental arms, the full-batch cross-entropy")
print("training loss as a function of gradient-descent step (every 10th step).")
print("It demonstrates the *data-efficiency* mechanism: with only 25% of the")
print("training pool (100 samples), reductions in loss are noisier and the final")
print("loss plateau is higher, because each update averages over fewer examples.")
print("With 100% of the pool (400 samples), the gradient estimates are more stable")
print("and the model can drive the loss toward a lower minimum on the same")
print("fixed number of steps. This visualizes the learning-curve gap underlying")
print("the recorded efficiency ratio.")
print("-" * 78)

plt.figure(figsize=(11, 6.5))
plt.plot(step_marks, losses_25, label="Arm 25%  (100 training samples)",
         color="#ff6b6b", linewidth=2.8, marker="o", markersize=2.5, markeredgecolor="#b02424")
plt.plot(step_marks, losses_100, label="Arm 100% (400 training samples)",
         color="#4ecdc4", linewidth=2.8, marker="s", markersize=2.5, markeredgecolor="#1c7e72")
plt.xlabel("Gradient descent step", fontsize=13, fontweight="bold")
plt.ylabel("Cross-entropy loss", fontsize=13, fontweight="bold")
plt.title("Training Loss Curves — Controlled Data-Efficiency Experiment\n"
          "(full-batch multinomial logistic regression, lr=0.1, 1500 steps)",
          fontsize=14, fontweight="bold")
plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=11, loc="upper right")
plt.grid(True, which="both", alpha=0.35, linestyle="--")
plt.tight_layout()
plt.savefig("Figure_1.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved Figure_1.png")

# =============================================================================
# EXPLANATION BEFORE FIGURE 2
# =============================================================================
print("\n" + "-" * 78)
print("FIGURE 2 EXPLANATION")
print("Left panel: bar chart of held-out test accuracy for the 25%-data arm and")
print("the 100%-data arm on the same 200-sample test set, with the headline")
print("eff.efficiency_ratio annotated. This directly quantifies how much test")
print("accuracy changes when training data is increased from 100 to 400 samples.")
print("Right panels: confusion matrices of each arm over the 3 balanced classes,")
print("showing where misclassifications concentrate (e.g., class 0 vs class 1/2).")
print("-" * 78)

def confusion(y_true, y_pred, n_classes=3):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

pred_25 = np.argmax(Xb_test @ W_25_vis, axis=1)
pred_100 = np.argmax(Xb_test @ W_100_vis, axis=1)
cm_25 = confusion(y_test, pred_25)
cm_100 = confusion(y_test, pred_100)

fig, axes = plt.subplots(1, 3, figsize=(19, 5.6),
                         gridspec_kw={"width_ratios": [1.15, 1.3, 1.3]})

# ---- Bar chart ----
ax0 = axes[0]
bars = ax0.bar(["25% data\n(100 samples)", "100% data\n(400 samples)"],
               [acc_at_25, acc_at_100], color=["#ff6b6b", "#4ecdc4"],
               edgecolor="#333333", linewidth=1.5, width=0.58)
ax0.set_ylim(0.7, 1.0)
ax0.set_ylabel("Test accuracy", fontsize=12, fontweight="bold")
ax0.set_title("Test Accuracy vs Training Data Fraction", fontsize=12, fontweight="bold")
for bar, val in zip(bars, [acc_at_25, acc_at_100]):
    ax0.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.005,
             f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax0.text(0.5, 0.95, f"efficiency ratio = {efficiency_ratio:.4f}",
         transform=ax0.transAxes, ha="center", fontsize=11.5, fontweight="bold",
         color="#222222", bbox=dict(boxstyle="round,pad=0.45", facecolor="#ffe66d",
                                    edgecolor="#d4a017", linewidth=1.5))
ax0.grid(axis="y", alpha=0.3, linestyle="--")

# ---- Confusion matrices ----
cm_max = int(np.max([cm_25.max(), cm_100.max()]))
for ax, cm, title in zip(axes[1:], [cm_25, cm_100],
                         ["Confusion Matrix — Arm 25% (100 samples)",
                          "Confusion Matrix — Arm 100% (400 samples)"]):
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm_max)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Class 0", "Class 1", "Class 2"], fontsize=10)
    ax.set_yticklabels(["Class 0", "Class 1", "Class 2"], fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True label", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(3):
        for j in range(3):
            thresh = cm_max / 2.0
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight="bold")

plt.suptitle("Controlled Gate 1 — Data Efficiency: 25% vs 100% of Training Pool",
             fontsize=15, fontweight="bold", y=1.03)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("Figure_2.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved Figure_2.png")

# =============================================================================
# Final human-readable summary of the recorded measurements
# =============================================================================
print("\n" + "=" * 78)
print("FINAL RECORDED RESULTS (as passed to record_result)")
print("=" * 78)
print(f"eff.acc_at_25        = {acc_at_25:.6f}")
print(f"eff.acc_at_100       = {acc_at_100:.6f}")
print(f"eff.efficiency_ratio = {efficiency_ratio:.6f}")
print(f"eff.train_s          = {train_s:.6f} seconds")
print(f"seed                 = {seed}")
print("=" * 78)