# Controlled Gate 1 Validation — Data Efficiency

This repository contains a fully reproducible, controlled validation of **data-efficiency scaling** on a synthetic classification task. It measures how much test accuracy is retained when a model is trained on only 25% of the available labels compared with the full label set.

The task is deliberately simple: three-class linear classification on eight-dimensional Gaussian features using a matched multinomial logistic-regression model. The only difference between the two training arms is the number of labelled samples:

| Arm | Training samples | Fraction of dataset |
|-----|-----------------|---------------------|
| `25%` | 100 | 25% of the 400-sample pool |
| `100%` | 400 | 100% of the 400-sample pool |

Both arms share the same random seed, data split, zero weight initialization, optimizer, learning rate, step count, and test set. This ensures that any observed accuracy gap is due **solely to data quantity**.

---

## Key Results

| Metric | Value |
|--------|-------|
| Test accuracy with 100 labelled samples | `0.89` |
| Test accuracy with 400 labelled samples | `0.97` |
| Efficiency ratio (`acc_at_25 / acc_at_100`) | `0.9175257731958764` |
| Total training time (both arms) | `0.017149006998806726` seconds |

**Interpretation:** Using a quarter of the available labels preserves **91.75%** of the test accuracy achieved with the full label set.

In error-rate terms:

- Error at 100 samples: `0.11`
- Error at 400 samples: `0.03`
- Implied power-law scaling exponent `β ≈ 0.94` under `E(n) ∝ n^{-β}`

This is consistent with a well-specified parametric model whose variance decays at nearly the theoretical `O(1/n)` rate.

---

## Experimental Setup

- **Synthetic data:** 600 samples, 8 features, 3 classes  
  - 200 samples used as a frozen test set  
  - 400 samples used as the training pool  
- **Data generation:** `np.random.default_rng(42)`  
  - `X ~ N(0, I₈)`  
  - Ground-truth weights `W_true ~ N(0, I₈ₓ₃)`  
  - Labels: `y = argmax(X W_true)`
- **Model:** Multinomial logistic regression with a bias term (27 parameters)
- **Loss:** Softmax cross-entropy
- **Optimizer:** Vanilla gradient descent, learning rate `0.1`, 300 steps
- **Initialization:** Zero weights for both arms
- **Evaluation:** Top-1 test accuracy on the fixed 200-sample test set
- **Dependencies:** NumPy only (Python standard library used only for timing)

---

## Repository Structure

```text
.
├── README.md              # This file
├── experiment.py          # Main controlled data-efficiency experiment
├── requirements.txt       # Python dependencies (numpy)
├── figures/
│   ├── Figure_1.png       # Training loss curves
│   └── Figure_2.png       # Test accuracy bar chart
├── results/
│   └── metrics.json       # Recorded experiment metrics
└── reports/
    └── data_efficiency_report.pdf   # Written research report
```

---

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` contains:

```
numpy
```

### 2. Run the experiment

```bash
python experiment.py
```

The script will:

1. Generate the synthetic dataset with seed `42`.
2. Split into a 200-sample test set and a 400-sample training pool.
3. Train Arm A on the first 100 samples of the pool.
4. Train Arm B on the full 400-sample pool.
5. Evaluate both arms on the frozen test set.
6. Record the accuracies, efficiency ratio, and training time.
7. Save loss-curve and accuracy plots to `figures/`.

### 3. Inspect results

The script prints metrics to stdout and saves them to `results/metrics.json`.

---

## Reproducibility

The experiment is **deterministic** for seed `42`. Re-running the script produces identical:

- Data generation
- Data split
- Weight initialization
- Optimization trajectories
- Final test accuracies
- Reported metrics

This is essential for isolating the effect of training-set size from sampling variance.

---

## Why This Matters

The result establishes an **empirical upper bound** on data efficiency for a well-specified linear model. It provides a clean, reproducible baseline for future experiments that introduce:

- Model misspecification
- Label noise
- Irrelevant features
- Non-linear decision boundaries
- Compositional task structure

In such settings, the efficiency ratio is expected to decrease, and the power-law exponent is expected to be smaller, making this repository a useful reference point for more realistic data-efficiency studies.

---

## Citation

If you use this repository in your research, please refer to it as:

> Agent Laboratory. *Research Report: Controlled Gate 1 Validation — Data Efficiency*. 2026.

---

## License

MIT License. See `LICENSE` for details.