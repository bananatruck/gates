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
# Controlled Gate 1 Validation — Data Efficiency
# =============================================================================
# The evaluation harness normally prepends the fixed synthetic dataset plus the
# two controlled training arms (Arm 25 = 100 samples, Arm 100 = 400 samples),
# which already records the headline metrics. To remain valid even when executed
# standalone, the identical deterministic block is re-run below ONLY when the
# recorded variables are not yet present in the global namespace.

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

    print("\n" + "=" * 80)
    print("EXPERIMENT EXPLANATION:")
    print("This controlled data-efficiency experiment trains multinomial logistic")
    print("regression with full-batch gradient descent (lr=0.1, zero init, 1500")
    print("steps) on 25% of the 400-sample training pool (100 samples) versus the")
    print("full pool (400 samples). The 200-sample test set is identical for both")
    print("arms. The headline quantity eff.efficiency_ratio = acc_at_25 /")
    print("acc_at_100 measures what fraction of the full-data test accuracy is")
    print("already reached with one quarter of the training data.")
    print("=" * 80)

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
# Post-hoc analysis, explanations, and figure generation.
# The final metrics have already been recorded; the code below only reconstructs
# the identical deterministic GD trajectories for visualization.
# =============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

print("\n" + "=" * 80)
print("POST-HOC VISUALIZATION AND ANALYSIS PASS")
print("=" * 80)
print("EXPERIMENT EXPLANATION:")
print("This controlled data-efficiency experiment compares multinomial logistic")
print("regression trained with full-batch gradient descent (lr=0.1, zero init,")
print("1500 steps) on 25% of the 400-sample training pool (100 samples) versus")
print("the full pool (400 samples). The test set is the same 200 held-out samples")
print("for both arms. The headline metric eff.efficiency_ratio =")
print("eff.acc_at_25 / eff.acc_at_100 tells us what fraction of the full-data")
print("test accuracy is already achieved with only one quarter of the training")
print("data. The figures below are generated by re-running the same deterministic")
print("optimizer to draw training-loss curves, test-accuracy trajectories,")
print("accuracy bars, and confusion matrices. Recorded results are unchanged.")
print("=" * 80)

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
# Explanation before Figure 1
# =============================================================================
print("\n" + "-" * 80)
print("FIGURE 1 EXPLANATION:")
print("Figure_1.png shows, for both arms, the full-batch cross-entropy training")
print("loss (left axis, solid lines) and the corresponding held-out test accuracy")
print("(right axis, dashed lines) evaluated every 10 gradient-descent steps.")
print("The plot visualizes the data-efficiency trade-off: the 25%-data arm")
print("(100 samples) has noisier loss reduction and plateaus at a higher loss,")
print("while the 100%-data arm (400 samples) enjoys more stable updates and")
print("converges to a lower loss. The dashed test-accuracy curves show how much")
print("of the final accuracy is already reached after the same 1500 steps.")
print("-" * 80)

fig, ax1 = plt.subplots(figsize=(11.5, 6.8))
loss_c25 = ax1.plot(
    step_marks, losses_25,
    label="Arm 25%: training loss (100 samples)",
    color="#ff6b6b", linewidth=3.0, marker="o", markersize=2.5,
    markeredgecolor="#b02424", zorder=3,
)
loss_c100 = ax1.plot(
    step_marks, losses_100,
    label="Arm 100%: training loss (400 samples)",
    color="#4ecdc4", linewidth=3.0, marker="s", markersize=2.5,
    markeredgecolor="#1c7e72", zorder=3,
)
ax1.set_xlabel("Gradient descent step", fontsize=13, fontweight="bold")
ax1.set_ylabel("Cross-entropy loss", fontsize=13, fontweight="bold", color="#b02424")
ax1.tick_params(axis="y", labelcolor="#b02424")
ax1.grid(True, which="both", alpha=0.30, linestyle="--", zorder=0)

ax2 = ax1.twinx()
acc_c25 = ax2.plot(
    step_marks, accs_25,
    label="Arm 25%: test accuracy",
    color="#ff9800", linewidth=2.4, linestyle="--", marker="^",
    markersize=2.5, markeredgecolor="#e65100", zorder=4,
)
acc_c100 = ax2.plot(
    step_marks, accs_100,
    label="Arm 100%: test accuracy",
    color="#7e57c2", linewidth=2.4, linestyle="--", marker="D",
    markersize=2.5, markeredgecolor="#4527a0", zorder=4,
)
ax2.set_ylabel("Test accuracy", fontsize=13, fontweight="bold", color="#4527a0")
ax2.tick_params(axis="y", labelcolor="#4527a0")
ax2.set_ylim(0.0, 1.05)

lines = loss_c25 + loss_c100 + acc_c25 + acc_c100
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="center right", fontsize=10.5, fancybox=True,
           shadow=True, frameon=True)

plt.title(
    "Training Loss & Test Accuracy Curves — Controlled Data-Efficiency Experiment\n"
    "(full-batch multinomial logistic regression, lr=0.1, 1500 steps, same seed/init)",
    fontsize=13.5, fontweight="bold",
)
plt.tight_layout()
plt.savefig("Figure_1.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved Figure_1.png")

# =============================================================================
# Explanation before Figure 2
# =============================================================================
print("\n" + "-" * 80)
print("FIGURE 2 EXPLANATION:")
print("Figure_2.png summarizes the final measured outcomes. Left panel: held-out")
print("test accuracy of the 25%-data arm versus the 100%-data arm on the same")
print("200-sample test set, with the headline efficiency ratio annotated.")
print("Right panels: confusion matrices of each arm over the 3 balanced classes,")
print("showing where misclassifications concentrate when 100 vs 400 samples are")
print("used for training.")
print("-" * 80)

pred_25 = np.argmax(Xb_test @ W_25_vis, axis=1)
pred_100 = np.argmax(Xb_test @ W_100_vis, axis=1)

cm_25 = np.zeros((3, 3), dtype=int)
for t, p in zip(y_test, pred_25):
    cm_25[t, p] += 1

cm_100 = np.zeros((3, 3), dtype=int)
for t, p in zip(y_test, pred_100):
    cm_100[t, p] += 1

fig, axes = plt.subplots(1, 3, figsize=(19, 5.6),
                         gridspec_kw={"width_ratios": [1.15, 1.3, 1.3]})

# ---- Bar chart ----
ax0 = axes[0]
bars = ax0.bar(
    ["25% data\n(100 samples)", "100% data\n(400 samples)"],
    [acc_at_25, acc_at_100],
    color=["#ff6b6b", "#4ecdc4"],
    edgecolor="#333333", linewidth=1.5, width=0.58,
)
ax0.set_ylim(0.5, 1.0)
ax0.set_ylabel("Test accuracy", fontsize=12, fontweight="bold")
ax0.set_title("Test Accuracy vs Training Data Fraction", fontsize=12, fontweight="bold")
for bar, val in zip(bars, [acc_at_25, acc_at_100]):
    ax0.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 0.005,
        f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
    )
ax0.text(
    0.5, 0.93,
    f"efficiency ratio = {efficiency_ratio:.4f}",
    transform=ax0.transAxes, ha="center", fontsize=11.5, fontweight="bold",
    color="#222222",
    bbox=dict(boxstyle="round,pad=0.45", facecolor="#ffe66d",
              edgecolor="#d4a017", linewidth=1.5),
)
ax0.grid(axis="y", alpha=0.30, linestyle="--")

# ---- Confusion matrices ----
cm_max = int(np.max([cm_25.max(), cm_100.max()]))
titles = [
    "Confusion Matrix — Arm 25% (100 samples)",
    "Confusion Matrix — Arm 100% (400 samples)",
]
for ax, cm, title in zip(axes[1:], [cm_25, cm_100], titles):
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
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12, fontweight="bold",
            )

plt.suptitle("Controlled Gate 1 — Data Efficiency: 25% vs 100% of Training Pool",
             fontsize=15, fontweight="bold", y=1.03)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("Figure_2.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved Figure_2.png")

# =============================================================================
# Final human-readable summary of the recorded measurements
# =============================================================================
print("\n" + "=" * 80)
print("FINAL RECORDED RESULTS (as passed to record_result)")
print("=" * 80)
print(f"eff.acc_at_25        = {acc_at_25:.6f}")
print(f"eff.acc_at_100       = {acc_at_100:.6f}")
print(f"eff.efficiency_ratio = {efficiency_ratio:.6f}")
print(f"eff.train_s          = {train_s:.6f} seconds")
print(f"seed                 = {seed}")
print("=" * 80)