"""
ROC and PR curves with bootstrap CIs, plus cost-based threshold selection
on Breast Cancer Wisconsin (Diagnostic).

Models: Logistic Regression and MLP.
Cost:   one false negative (missed cancer) = 10 x one false positive.
The threshold is chosen on out-of-fold training predictions, then
evaluated once on the held-out test set.

Positive class = malignant (sklearn uses 0 = malignant, so we flip it).
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, cross_val_predict,
                                     StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score, confusion_matrix)

RANDOM_STATE = 42   # reproducibility
TEST_SIZE = 0.25    # 25% held out for testing
N_BOOT = 2000       # bootstrap resamples for the AUC CIs
COST_FN = 10        # cost of a missed cancer
COST_FP = 1         # cost of a false alarm
FIG_DIR = Path(__file__).parent / "figures"

COLORS = {"Logistic Regression": "#2a78d6", "MLP": "#eb6834"}


def load_data():
    """Load the dataset; return X (569x30) and y (1 = malignant, 0 = benign)."""
    data = load_breast_cancer()
    X, y = data.data, 1 - data.target
    print(f"Samples: {X.shape[0]}, features: {X.shape[1]}, "
          f"malignant: {y.sum()}, benign: {(y == 0).sum()}")
    return X, y


def build_models():
    """
    Return the two models, each with a StandardScaler in a pipeline
    (scaler is fit on training data only -> no test leakage).
    """
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                          alpha=1e-3, max_iter=2000,
                          random_state=RANDOM_STATE)),
    }


def bootstrap_ci(y_true, y_score, metric, n_boot=N_BOOT, alpha=0.05):
    """
    Percentile bootstrap CI for a score metric (e.g. ROC AUC).

    Draws n_boot resamples of the test set (with replacement), recomputes
    the metric on each, and takes the 2.5th / 97.5th percentiles.
    Resamples containing a single class are skipped (AUC undefined).

    Returns (point estimate, lower, upper).
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        stats.append(metric(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return metric(y_true, y_score), lo, hi


def expected_cost(y_true, y_score, threshold):
    """Total misclassification cost: COST_FN * FN + COST_FP * FP."""
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return COST_FN * fn + COST_FP * fp


def choose_threshold(y_true, y_score):
    """
    Scan thresholds in [0, 1] and return (best threshold, thresholds, costs).

    A case is called malignant when P(malignant) >= threshold. Lower
    thresholds catch more cancers (fewer FN) but raise more false alarms
    (more FP); the cost 10·FN + FP decides the trade-off.
    Ties are broken by the highest threshold (fewest false alarms).
    """
    thresholds = np.linspace(0, 1, 1001)
    costs = np.array([expected_cost(y_true, y_score, t) for t in thresholds])
    best = thresholds[np.flatnonzero(costs == costs.min())[-1]]
    return best, thresholds, costs


def evaluate_at(y_true, y_score, threshold):
    """Return confusion-matrix counts, Se, Sp and total cost at a threshold."""
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "Se": tp / (tp + fn), "Sp": tn / (tn + fp),
            "Cost": COST_FN * fn + COST_FP * fp}


def plot_roc_pr(results, y_test):
    """
    Draw ROC and PR curves for both models (AUC + 95% CI in the legend).

    ROC: Se vs. 1 - Sp; the diagonal is a random classifier.
    PR:  PPV vs. Se; the baseline is the prevalence of malignant cases.
    Dots mark each model's operating point at the cost-based threshold.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5.2))

    for name, r in results.items():
        c = COLORS[name]
        fpr, tpr, _ = roc_curve(y_test, r["score"])
        auc, lo, hi = r["roc_auc"]
        ax_roc.plot(fpr, tpr, color=c, lw=2,
                    label=f"{name}  AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]")
        m = r["at_best"]
        ax_roc.plot(1 - m["Sp"], m["Se"], "o", ms=9, color=c, mec="white",
                    mew=2, zorder=3)

        prec, rec, _ = precision_recall_curve(y_test, r["score"])
        ap, lo, hi = r["pr_auc"]
        ax_pr.plot(rec, prec, color=c, lw=2,
                   label=f"{name}  AP {ap:.3f} [{lo:.3f}, {hi:.3f}]")

    ax_roc.plot([0, 1], [0, 1], ls="--", lw=1, color="#9a9a94",
                label="Random classifier")
    prevalence = y_test.mean()
    ax_pr.axhline(prevalence, ls="--", lw=1, color="#9a9a94",
                  label=f"Random classifier (prevalence {prevalence:.2f})")

    ax_roc.set(title="ROC curve (test set)", xlabel="False positive rate (1 - Sp)",
               ylabel="True positive rate (Se)")
    ax_pr.set(title="Precision-Recall curve (test set)", xlabel="Recall (Se)",
              ylabel="Precision (PPV)")
    # zoomed inset on the top-left corner, where the curves differ
    ins = ax_roc.inset_axes([0.42, 0.32, 0.54, 0.42])
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["score"])
        ins.plot(fpr, tpr, color=COLORS[name], lw=2)
        m = r["at_best"]
        ins.plot(1 - m["Sp"], m["Se"], "o", ms=8, color=COLORS[name],
                 mec="white", mew=2, zorder=3)
    ins.set(xlim=(-0.01, 0.25), ylim=(0.85, 1.01))
    ins.set_title("zoom · dots = chosen cost threshold", fontsize=8,
                  color="#555550")
    ins.tick_params(labelsize=7)
    _style(ins)

    ax_pr.set_ylim(0, 1.05)
    for ax in (ax_roc, ax_pr):
        _style(ax)
        ax.legend(loc="lower right" if ax is ax_roc else "lower left",
                  fontsize=9, frameon=False)
    fig.tight_layout()
    path = FIG_DIR / "roc_pr_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cost(results):
    """Draw training (out-of-fold) cost vs. threshold, marking the chosen one."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, (name, r) in enumerate(results.items()):
        c = COLORS[name]
        ax.plot(r["thresholds"], r["costs"], color=c, lw=2, label=name)
        ax.plot(r["best_t"], r["costs"].min(), "o", ms=9, color=c,
                mec="white", mew=2, zorder=3)
        ax.annotate(f"{name}: t = {r['best_t']:.3f}, cost = {r['costs'].min()}",
                    (r["best_t"], r["costs"].min()),
                    xytext=(30, 70 + 28 * i), textcoords="offset points",
                    fontsize=9, color="#333330",
                    arrowprops=dict(arrowstyle="-", color="#9a9a94", lw=0.8))
    ax.set(title=f"Cost vs. threshold  (cost = {COST_FN}·FN + {COST_FP}·FP, "
                 f"out-of-fold on training set)",
           xlabel="Decision threshold on P(malignant)", ylabel="Total cost")
    ax.set_ylim(0, 300)  # cost explodes near t = 1; keep the minimum readable
    _style(ax)
    ax.legend(loc="upper center", frameon=False, fontsize=9)
    fig.tight_layout()
    path = FIG_DIR / "cost_vs_threshold.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _style(ax):
    """Recessive grid and axes."""
    ax.grid(True, color="#e6e6e1", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b5ae")


def print_report(name, r):
    """Print AUCs with CIs and the test results at each threshold."""
    print(f"\n=== {name} ===")
    for key, label in (("roc_auc", "ROC AUC"), ("pr_auc", "PR AUC (AP)")):
        v, lo, hi = r[key]
        print(f"{label:<12}: {v:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"Chosen threshold (min cost on training OOF): {r['best_t']:.3f}")
    print("Test set:   threshold    TP  FN  FP  TN     Se      Sp   Cost")
    for label, m, t in (("default", r["at_05"], 0.5),
                        ("cost-based", r["at_best"], r["best_t"])):
        print(f"{label:<11} {t:>9.3f}  {m['TP']:>4}{m['FN']:>4}{m['FP']:>4}"
              f"{m['TN']:>4}  {m['Se']:.4f}  {m['Sp']:.4f}  {m['Cost']:>4}")


def main():
    """
    1. Stratified train/test split.
    2. Out-of-fold probabilities on training (5-fold CV) -> pick the
       threshold that minimises 10·FN + FP.
    3. Fit on all training data, score the test set.
    4. ROC/PR AUC with bootstrap 95% CIs; compare default 0.5 vs. chosen
       threshold; save figures.
    """
    FIG_DIR.mkdir(exist_ok=True)

    # --- Data split -------------------------------------------------------
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print(f"Bayes-optimal threshold for calibrated probabilities: "
          f"{COST_FP / (COST_FP + COST_FN):.3f}")

    results = {}
    for name, model in build_models().items():
        # --- Threshold: chosen on training data only (out-of-fold) --------
        # Each training case is scored by a model that never saw it, so the
        # threshold is not tuned on optimistic, already-fitted predictions.
        oof = cross_val_predict(model, X_train, y_train, cv=cv,
                                method="predict_proba")[:, 1]
        best_t, thresholds, costs = choose_threshold(y_train, oof)

        # --- Final model: fit on all training data, score the test set ----
        model.fit(X_train, y_train)
        score = model.predict_proba(X_test)[:, 1]

        # --- Evaluation: AUCs with CIs, default 0.5 vs. chosen threshold --
        results[name] = {
            "score": score, "best_t": best_t,
            "thresholds": thresholds, "costs": costs,
            "roc_auc": bootstrap_ci(y_test, score, roc_auc_score),
            "pr_auc": bootstrap_ci(y_test, score, average_precision_score),
            "at_05": evaluate_at(y_test, score, 0.5),
            "at_best": evaluate_at(y_test, score, best_t),
        }
        print_report(name, results[name])

    # --- Figures ---------------------------------------------------------
    print(f"\nSaved: {plot_roc_pr(results, y_test)}")
    print(f"Saved: {plot_cost(results)}")


if __name__ == "__main__":
    main()