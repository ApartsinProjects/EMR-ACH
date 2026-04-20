"""
Platt scaling calibration for the heuristic qualitative-to-numeric mapping.

Training: fit logistic regression on a hold-out calibration set.
Inference: phi_cal(x) = sigmoid(a*x + b)

Default values from paper: a=1.42, b=-2.18 (5-fold CV on MIRAI training partition).
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid

from src.config import get_config


# Heuristic mapping from Remez 2025 (used as fallback if calibration not fitted)
HEURISTIC_MAP = {
    "highly likely": 0.9,
    "likely": 0.66,
    "possible": 0.33,
    "unlikely": 0.1,
    "highly unlikely": 0.1,
}


class PlattCalibration:
    """Fitted Platt scaling: phi_cal(x) = sigmoid(a*x + b)."""

    def __init__(self, a: float = 1.42, b: float = -2.18):
        self.a = a
        self.b = b
        self.fitted = False

    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        return expit(self.a * x + self.b)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PlattCalibration":
        """
        Fit a and b via maximum likelihood on calibration data.

        Args:
            x: raw LLM scores, shape (N,)
            y: ground truth binary labels, shape (N,)
        """
        def neg_log_likelihood(params):
            a, b = params
            p = expit(a * x + b)
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        result = minimize(neg_log_likelihood, [self.a, self.b], method="L-BFGS-B")
        self.a, self.b = result.x
        self.fitted = True
        return self

    def fit_cv(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        seed: int = 42,
    ) -> tuple[float, float]:
        """
        Fit with cross-validation. Returns (mean_a, mean_b) across folds.
        Sets self.a, self.b to fold-averaged values.
        """
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(x))
        folds = np.array_split(idx, n_folds)

        all_a, all_b = [], []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            fold_cal = PlattCalibration(self.a, self.b)
            fold_cal.fit(x[train_idx], y[train_idx])
            all_a.append(fold_cal.a)
            all_b.append(fold_cal.b)
            ece_fold = _ece(fold_cal(x[val_idx]), y[val_idx])
            print(f"  Fold {i+1}: a={fold_cal.a:.3f} b={fold_cal.b:.3f} ECE={ece_fold:.4f}")

        self.a = float(np.mean(all_a))
        self.b = float(np.mean(all_b))
        self.fitted = True
        print(f"  CV result: a={self.a:.3f} ± {np.std(all_a):.3f}, b={self.b:.3f} ± {np.std(all_b):.3f}")
        return self.a, self.b

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump({"a": self.a, "b": self.b, "fitted": self.fitted}, f)

    @classmethod
    def load(cls, path: str | Path) -> "PlattCalibration":
        with open(path) as f:
            d = json.load(f)
        cal = cls(d["a"], d["b"])
        cal.fitted = d.get("fitted", True)
        return cal

    @classmethod
    def default(cls, config=None) -> "PlattCalibration":
        cfg = config or get_config()
        a = cfg.get("pipeline", "calibration", "a", default=1.42)
        b = cfg.get("pipeline", "calibration", "b", default=-2.18)
        return cls(float(a), float(b))


def heuristic_to_numeric(label: str) -> float:
    """Map qualitative LLM output to numeric (Remez heuristic)."""
    return HEURISTIC_MAP.get(label.lower().strip(), 0.5)


def apply_calibration_to_matrix(
    M: np.ndarray,
    calibration: PlattCalibration | None,
) -> np.ndarray:
    """Apply Platt calibration element-wise to a matrix of raw LLM scores."""
    if calibration is None:
        return M
    return calibration(M)


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error for a binary probability vector."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (probs >= low) & (probs < high)
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)
    return float(ece)
