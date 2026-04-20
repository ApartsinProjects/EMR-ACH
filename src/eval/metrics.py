"""
Evaluation metrics for EMR-ACH experiments.

Metrics:
  - Macro-averaged precision, recall, F1
  - Accuracy
  - KL divergence (predicted vs empirical label distribution)
  - Expected Calibration Error (ECE)
  - Bootstrap confidence intervals
  - McNemar's test for pairwise comparisons
  - Brier score (for ForecastBench)
"""

import warnings
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from src.data.mirai import HYPOTHESES


@dataclass
class EvalResult:
    precision: float
    recall: float
    f1: float
    accuracy: float
    kl_div: float
    ece: float
    # Bootstrap CIs
    f1_ci: tuple[float, float] | None = None
    precision_ci: tuple[float, float] | None = None
    recall_ci: tuple[float, float] | None = None
    accuracy_ci: tuple[float, float] | None = None
    ece_ci: tuple[float, float] | None = None
    n: int = 0

    def to_dict(self) -> dict:
        return {
            "precision": round(self.precision * 100, 1),
            "recall": round(self.recall * 100, 1),
            "f1": round(self.f1 * 100, 1),
            "accuracy": round(self.accuracy * 100, 1),
            "kl_div": round(self.kl_div, 4),
            "ece": round(self.ece, 4),
            "n": self.n,
            "f1_ci": [round(v * 100, 1) for v in self.f1_ci] if self.f1_ci else None,
        }

    def __str__(self) -> str:
        ci = f" [{self.f1_ci[0]*100:.1f}, {self.f1_ci[1]*100:.1f}]" if self.f1_ci else ""
        return (
            f"P={self.precision*100:.1f}% R={self.recall*100:.1f}% "
            f"F1={self.f1*100:.1f}%{ci} Acc={self.accuracy*100:.1f}% "
            f"KL={self.kl_div:.4f} ECE={self.ece:.4f} (n={self.n})"
        )


def evaluate(
    predictions: list[dict],
    labels: list[str],
    hypotheses: list[str] = HYPOTHESES,
    bootstrap_n: int = 1000,
    bootstrap_seed: int = 42,
    ece_bins: int = 10,
) -> EvalResult:
    """
    Evaluate a list of predictions against ground truth labels.

    Args:
        predictions: list of dicts with keys:
            - "prediction": str — top-1 predicted hypothesis label
            - "probabilities": dict[str, float] — probability over all hypotheses
            - "ranking": list[str] — hypotheses sorted by score
        labels: list of ground truth labels (e.g. "VK")
        hypotheses: ordered list of hypothesis names
        bootstrap_n: number of bootstrap samples for CIs
        bootstrap_seed: random seed
        ece_bins: number of bins for ECE computation
    """
    assert len(predictions) == len(labels), f"Length mismatch: {len(predictions)} vs {len(labels)}"
    n = len(predictions)
    preds = [p["prediction"] for p in predictions]

    # Per-class metrics
    precisions, recalls, f1s = [], [], []
    for h in hypotheses:
        tp = sum(p == h and l == h for p, l in zip(preds, labels))
        fp = sum(p == h and l != h for p, l in zip(preds, labels))
        fn = sum(p != h and l == h for p, l in zip(preds, labels))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    precision = float(np.mean(precisions))
    recall = float(np.mean(recalls))
    f1 = float(np.mean(f1s))
    accuracy = float(sum(p == l for p, l in zip(preds, labels)) / n)

    # KL divergence
    kl = _kl_divergence(predictions, labels, hypotheses)

    # ECE
    ece = _ece_multiclass(predictions, labels, hypotheses, n_bins=ece_bins)

    result = EvalResult(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        kl_div=kl,
        ece=ece,
        n=n,
    )

    # Bootstrap CIs
    if bootstrap_n > 0:
        result.f1_ci, result.precision_ci, result.recall_ci, result.accuracy_ci, result.ece_ci = (
            _bootstrap_cis(predictions, labels, hypotheses, bootstrap_n, bootstrap_seed, ece_bins)
        )

    return result


def per_class_f1(
    predictions: list[dict],
    labels: list[str],
    hypotheses: list[str] = HYPOTHESES,
) -> dict[str, float]:
    preds = [p["prediction"] for p in predictions]
    result = {}
    for h in hypotheses:
        tp = sum(p == h and l == h for p, l in zip(preds, labels))
        fp = sum(p == h and l != h for p, l in zip(preds, labels))
        fn = sum(p != h and l == h for p, l in zip(preds, labels))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        result[h] = float(f1)
    return result


def brier_score(probs: list[float], labels: list[int]) -> float:
    """Binary Brier score for ForecastBench."""
    return float(np.mean([(p - l) ** 2 for p, l in zip(probs, labels)]))


def _kl_divergence(
    predictions: list[dict],
    labels: list[str],
    hypotheses: list[str],
) -> float:
    """KL(empirical || predicted mean distribution)."""
    n = len(predictions)
    # Empirical distribution from ground truth
    counts = Counter(labels)
    empirical = np.array([counts.get(h, 0) / n for h in hypotheses])

    # Mean predicted distribution
    predicted = np.zeros(len(hypotheses))
    for pred in predictions:
        probs = pred.get("probabilities", {})
        for j, h in enumerate(hypotheses):
            predicted[j] += probs.get(h, 1.0 / len(hypotheses))
    predicted /= n

    # KL(empirical || predicted), smoothed
    eps = 1e-9
    empirical = np.clip(empirical, eps, 1.0)
    predicted = np.clip(predicted, eps, 1.0)
    empirical /= empirical.sum()
    predicted /= predicted.sum()

    return float(np.sum(empirical * np.log(empirical / predicted)))


def _ece_multiclass(
    predictions: list[dict],
    labels: list[str],
    hypotheses: list[str],
    n_bins: int = 10,
) -> float:
    """
    ECE for multiclass: flatten confidence/accuracy pairs across all hypotheses.
    For each prediction, use max probability as confidence and correctness as accuracy.
    """
    n = len(predictions)
    confidences = []
    accuracies = []
    for pred, label in zip(predictions, labels):
        probs = pred.get("probabilities", {})
        top_h = max(probs, key=probs.get) if probs else hypotheses[0]
        conf = probs.get(top_h, 1.0 / len(hypotheses))
        correct = 1.0 if top_h == label else 0.0
        confidences.append(conf)
        accuracies.append(correct)

    conf_arr = np.array(confidences)
    acc_arr = np.array(accuracies)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (conf_arr >= low) & (conf_arr < high)
        if mask.sum() == 0:
            continue
        bin_acc = acc_arr[mask].mean()
        bin_conf = conf_arr[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)
    return float(ece)


def _bootstrap_cis(
    predictions: list[dict],
    labels: list[str],
    hypotheses: list[str],
    n_bootstrap: int,
    seed: int,
    ece_bins: int,
) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(predictions)
    f1s, precs, recs, accs, eces = [], [], [], [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_preds = [predictions[i] for i in idx]
        boot_labels = [labels[i] for i in idx]
        r = evaluate(boot_preds, boot_labels, hypotheses, bootstrap_n=0, ece_bins=ece_bins)
        f1s.append(r.f1)
        precs.append(r.precision)
        recs.append(r.recall)
        accs.append(r.accuracy)
        eces.append(r.ece)

    def ci(vals):
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    return ci(f1s), ci(precs), ci(recs), ci(accs), ci(eces)


def mcnemar_test(
    preds_a: list[str],
    preds_b: list[str],
    labels: list[str],
) -> dict:
    """McNemar's test for pairwise comparison of two systems."""
    correct_a = [p == l for p, l in zip(preds_a, labels)]
    correct_b = [p == l for p, l in zip(preds_b, labels)]
    # Discordant pairs
    b = sum(ca and not cb for ca, cb in zip(correct_a, correct_b))
    c = sum(not ca and cb for ca, cb in zip(correct_a, correct_b))
    n_disc = b + c
    if n_disc == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1 - chi2.cdf(statistic, df=1))
    return {
        "statistic": float(statistic),
        "p_value": p_value,
        "b": b, "c": c,
        "significant": p_value < 0.05,
    }
