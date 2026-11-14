"""Paired bootstrap over questions: is config B really better than config A?

Both configs answer the same questions, so we resample question ids (with
replacement) and recompute the mean difference each time. The 95% interval of
those differences tells us whether a gap is larger than question-sampling noise.
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 10_000, seed: int = 0) -> dict:
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    diff = b_arr - a_arr
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean_diff": float(diff.mean()),
        "ci95": (float(lo), float(hi)),
        "p_b_not_better": float((boots <= 0).mean()),
        "wins": int((diff > 0).sum()),
        "losses": int((diff < 0).sum()),
    }
