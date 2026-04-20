"""
Step 5: Diagnosticity-weighted aggregation.

Score(h_k) = (1 / (n+1)) * sum_i sum_j  d_tilde_j * A_tilde_{ij} * I_{jk}

In matrix form: P_tilde = A_tilde @ diag(d_tilde) @ I
Score(h_k) = mean over rows of P_tilde[:, k]

Also provides uniform-weight fallback (d_j = 1/m) for ablation comparison.
"""

import numpy as np

from src.data.mirai import HYPOTHESES
from src.pipeline.influence import compute_diagnosticity_weights, normalize_diagnosticity


def aggregate(
    A_tilde: np.ndarray,
    I: np.ndarray,
    use_diagnostic_weighting: bool = True,
) -> dict:
    """
    Compute hypothesis scores from augmented analysis matrix and influence matrix.

    Args:
        A_tilde: shape (n+1, m) — augmented analysis matrix (includes absence row)
        I:       shape (m, 4) — influence matrix
        use_diagnostic_weighting: if False, use uniform weights (ablation)

    Returns dict with:
        scores: shape (4,) — raw scores per hypothesis
        probs:  shape (4,) — softmax-normalized probabilities
        ranking: list of hypothesis labels sorted by score descending
        d_weights: shape (m,) — diagnosticity weights used
    """
    n_plus_1, m = A_tilde.shape
    assert I.shape == (m, 4), f"I shape mismatch: {I.shape} vs ({m}, 4)"

    # Compute diagnosticity weights
    d = compute_diagnosticity_weights(I)

    if use_diagnostic_weighting:
        d_tilde = normalize_diagnosticity(d)
    else:
        d_tilde = np.ones(m) / m  # uniform

    # Weighted aggregation: P_tilde = A_tilde @ diag(d_tilde) @ I
    # shape: (n+1, m) @ (m, m) @ (m, 4) = (n+1, 4)
    P_tilde = A_tilde @ np.diag(d_tilde) @ I

    # Mean over rows
    scores = P_tilde.mean(axis=0)  # shape (4,)

    # Normalize to probability distribution via softmax
    probs = _softmax(scores)

    # Rank hypotheses
    ranking = [HYPOTHESES[i] for i in np.argsort(scores)[::-1]]

    return {
        "scores": scores.tolist(),
        "probs": probs.tolist(),
        "ranking": ranking,
        "prediction": ranking[0],
        "d_weights": d_tilde.tolist(),
        "d_raw": d.tolist(),
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def scores_to_prob_dict(scores: list[float]) -> dict[str, float]:
    probs = _softmax(np.array(scores))
    return {h: float(probs[i]) for i, h in enumerate(HYPOTHESES)}


def rank_from_probs(probs: dict[str, float]) -> list[str]:
    return sorted(probs, key=probs.get, reverse=True)
