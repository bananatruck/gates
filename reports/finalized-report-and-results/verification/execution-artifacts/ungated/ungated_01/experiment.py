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
# The experiment preamble (dataset + two training arms) is injected by the harness.
# To be robust if that preamble is not present, we re-create it behind a guard.
if 'Xb_25' not in globals():
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

# ---- Controlled experiment analysis / presentation ----
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 78)
print("CONTROLLED GATE 1 VALIDATION — DATA EFFICIENCY")
print("=" * 78)
print("This experiment isolates the effect of training-data quantity on test accuracy.")
print("Two multinomial logistic-regression models are trained on the same 3-class")
print("synthetic dataset (8 features, 600 samples, fixed seed 42).")
print("The two arms are identical in every respect except the labelled training-set size:")
print("  Arm 25%: 100 samples from the 400-sample pool.")
print("  Arm 100%: all 400 samples from the same pool.")
print("Both use zero-initialized weights, full-batch gradient descent (lr=0.1),")
print("softmax cross-entropy loss, and exactly 1500 gradient steps.")
print("Both are evaluated on the same held-out 200-sample test set.")
print()
print("The headline metric is the efficiency ratio = acc_at_25 / acc_at_100.")
print("A ratio close to 1.0 means the model is highly data-efficient (little accuracy")
print("is lost when only a quarter of the labelled pool is used), while a ratio well")
print("below 1.0 indicates that the full training pool contributes substantial gains.")
print("-" * 78)
print("MEASURED RESULTS:")
print(f"  eff.acc_at_25        = {acc_at_25:.6f}")
print(f"  eff.acc_at_100       = {acc_at_100:.6f}")
print(f"  eff.efficiency_ratio = {efficiency_ratio:.6f}")
print(f"  eff.train_s          = {train_s:.6f} seconds")
print("-" * 78)
if efficiency_ratio >= 1.0:
    print("Interpretation: 25% of the data matches or beats full-data accuracy; the model")
    print("has already saturated at 100 samples on this synthetic task.")
elif efficiency_ratio >= 0.95:
    print("Interpretation: 25% of the data retains >=95% of full-data accuracy, showing high")
    print("data efficiency and only modest gains from the remaining 300 samples.")
else:
    print("Interpretation: the 100-sample arm loses more than 5% relative accuracy; the")
    print("additional 300 samples provide a meaningful accuracy improvement.")
print("=" * 78)

# Re-run the same training schedule only to collect loss histories for the learning-curve
# figure; the official metrics remain those recorded by the experiment above.
W_25_hist = np.zeros((9, 3))
W_100_hist = np.zeros((9, 3))
losses_25 = []
losses_100 = []
for step in range(steps):
    logits = Xb_25 @ W_25_hist
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    loss = -np.mean(np.sum(Y_25 * np.log(probs + 1e-12), axis=1))
    grad = Xb_25.T @ (probs - Y_25) / X_25.shape[0]
    W_25_hist -= lr * grad
    losses_25.append(loss)

for step in range(steps):
    logits = Xb_100 @ W_100_hist
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    loss = -np.mean(np.sum(Y_100 * np.log(probs + 1e-12), axis=1))
    grad = Xb_100.T @ (probs - Y_100) / X_100.shape[0]
    W_100_hist -= lr * grad
    losses_100.append(loss)

# Figure 1: final accuracy comparison
fig, ax = plt.subplots(figsize=(9, 6))
positions = np.arange(2)
labels = ['25% of pool\n(100 samples)', '100% of pool\n(400 samples)']
accuracies = [acc_at_25, acc_at_100]
colors = ['#FF6B6B', '#4ECDC4']
bars = ax.bar(positions, accuracies, width=0.55, color=colors,
              edgecolor='black', linewidth=1.2, zorder=3)
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2.0, acc + 0.015, f'{acc:.4f}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Test Accuracy', fontsize=14)
ax.set_title('Data-Efficiency Experiment: Test Accuracy by Training-Set Size',
             fontsize=14, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.2, zorder=1)
ax.text(0.5, 0.93, f'Efficiency ratio = {efficiency_ratio:.4f}', transform=ax.transAxes,
        ha='center', va='top', fontsize=15, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFE66D', edgecolor='black', alpha=0.95))
plt.tight_layout()
plt.savefig('Figure_1.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# Figure 2: training learning curves
fig, ax = plt.subplots(figsize=(10, 6))
step_axis = np.arange(steps)
ax.plot(step_axis, losses_25, label='25% pool (100 samples)', color='#FF6B6B', linewidth=2.5)
ax.plot(step_axis, losses_100, label='100% pool (400 samples)', color='#4ECDC4', linewidth=2.5)
ax.fill_between(step_axis, losses_25, losses_100, color='#FFD93D', alpha=0.15)
ax.set_xlabel('Gradient-Descent Step', fontsize=13)
ax.set_ylabel('Cross-Entropy Loss', fontsize=13)
ax.set_title('Training Loss Convergence: 25% vs 100% of Training Pool',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.grid(alpha=0.3)
ax.set_xlim(0, steps)
# Annotate final losses
ax.annotate(f'Final: {losses_25[-1]:.4f}', xy=(steps - 1, losses_25[-1]),
            xytext=(steps * 0.55, losses_25[0] * 0.30),
            arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=1.5),
            fontsize=11, color='#FF6B6B', fontweight='bold')
ax.annotate(f'Final: {losses_100[-1]:.4f}', xy=(steps - 1, losses_100[-1]),
            xytext=(steps * 0.68, losses_100[0] * 0.72),
            arrowprops=dict(arrowstyle='->', color='#4ECDC4', lw=1.5),
            fontsize=11, color='#4ECDC4', fontweight='bold')
plt.tight_layout()
plt.savefig('Figure_2.png', dpi=200, bbox_inches='tight')
plt.close(fig)

print("Saved Figure_1.png (accuracy comparison) and Figure_2.png (learning curves).")