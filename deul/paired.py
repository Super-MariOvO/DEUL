# -*- coding: utf-8 -*-
"""Paired-advantage GLS supervision for DEUL (paper Section 5.5).

At a paired training snapshot ``d`` the executor runs three reuse branches,
three adaptation branches and one replanning branch from the same prefix and
the same continuation policy, each repeated ``m`` times.  With branch success
means :math:`\\bar Y_{d,a}`, advantages are defined relative to the common
replanning branch:

    Delta_{d,a} = bar Y_{d,a} - bar Y_{d,0}

Because *all* advantages share the same replanning mean, their estimation
errors are **not independent**.  Under conditionally independent branch
noise, the exact covariance is

    R_d = diag(sigma_{d,a}^2 / m_{d,a}) + (sigma_{d,0}^2 / m_{d,0}) * 11^T

so for two different advantages (a != b)

    Cov(Delta_a, Delta_b) = Var(bar Y_0)

-- the "shared-baseline covariance" of paper Section 6.3.  The vector loss is
a generalized least-squares objective over the whole advantage vector:

    L_pair = (1/|D|) sum_d (U_d - Delta_d)^T sg(R_d^{-1}) (U_d - Delta_d)
                             / |A_d^paired|

This module provides:

* :func:`paired_means_to_advantages` -- from branch outcome vectors to Delta.
* :func:`shared_baseline_covariance` -- empirical check of the exact
  covariance identity above (used by unit tests).
* :func:`shrunk_covariance` -- pooled-residual, common-baseline-term,
  diagonal-shrinkage estimate of ``R_d`` with a positive-eigenvalue floor
  (two repeats cannot identify a full covariance matrix).
* :func:`gls_loss` / :func:`gls_gradient` -- the paired objective.
"""

from __future__ import annotations

from typing import Dict, Hashable, Mapping, Optional, Sequence

import numpy as np

__all__ = ["paired_means_to_advantages", "shared_baseline_covariance",
           "shrunk_covariance", "gls_loss", "gls_gradient"]


# ---------------------------------------------------------------------------
# From branch outcomes to advantages
# ---------------------------------------------------------------------------


def paired_means_to_advantages(
        outcomes: Mapping[Hashable, Sequence[float]]) -> Dict[Hashable, float]:
    """Compute :math:`\\Delta_a = \\bar Y_a - \\bar Y_0` for each branch.

    Args:
        outcomes: ``{branch_key: [success indicators...]}``; the replanning
            branch must be keyed by ``0`` (paper convention).

    Returns:
        ``{branch_key: advantage}`` with the replanning branch mapped to 0.0.
    """
    if 0 not in outcomes:
        raise KeyError("outcomes must contain the replanning branch keyed 0")
    y0 = float(np.mean(outcomes[0]))
    return {a: float(np.mean(y)) - y0 for a, y in outcomes.items()}


def shared_baseline_covariance(
        outcomes: Mapping[Hashable, Sequence[float]]) -> np.ndarray:
    """Empirical covariance matrix of the advantage vector.

    Returns the sample covariance of ``(Delta_{a_1}, ..., Delta_{a_k})`` over
    *independent repetitions* of the paired design.  Each element of
    ``outcomes`` must therefore be one *repetition*: ``outcomes[rep]`` is
    itself ``{branch_key: [indicators]}``.  For the exact identity
    ``Cov(Delta_a, Delta_b) = Var(bar Y_0)`` see the unit tests.
    """
    reps = list(outcomes.values())
    # Advantage vector EXCLUDES the replanning baseline itself; the returned
    # matrix is over (Delta_{a_1}, ..., Delta_{a_k}) for non-baseline branches.
    keys = sorted({k for rep in reps for k in rep if k != 0}, key=str)
    mat = []
    for rep in reps:
        y0 = float(np.mean(rep[0]))
        mat.append([float(np.mean(rep[k])) - y0 for k in keys])
    return np.cov(np.array(mat), rowvar=False, ddof=1)


# ---------------------------------------------------------------------------
# Shrunk covariance estimate R_d
# ---------------------------------------------------------------------------


def shrunk_covariance(
        branch_variances: Mapping[Hashable, float],
        baseline_variance: float,
        repeats: Mapping[Hashable, int],
        baseline_repeats: int,
        shrink: float = 0.1,
        eig_floor: float = 1e-3) -> np.ndarray:
    """Estimate the advantage-error covariance ``R_d``.

    Args:
        branch_variances: ``{branch: sigma_a^2}`` pooled residual variances.
        baseline_variance: ``sigma_0^2`` of the common replanning branch.
        repeats: ``{branch: m_a}`` repeats per branch.
        baseline_repeats: ``m_0`` repeats of the replanning branch.
        shrink: Diagonal shrinkage strength in ``[0, 1]``; the estimated
            common-baseline term is scaled by ``(1 - shrink)``.
        eig_floor: Minimum eigenvalue enforced on the returned matrix.

    Returns:
        Positive-definite estimate ``R = diag(sigma_a^2 / m_a)
        + (1 - shrink) * (sigma_0^2 / m_0) * 11^T``.
    """
    branches = sorted(branch_variances, key=str)
    n = len(branches)
    diag = np.array([branch_variances[b] / max(repeats.get(b, 1), 1)
                     for b in branches])
    base = (1.0 - shrink) * baseline_variance / max(baseline_repeats, 1)
    R = np.diag(diag) + base * np.ones((n, n))

    # Positive-eigenvalue floor (paper: 设置正特征值下限)
    w, V = np.linalg.eigh(R)
    w = np.maximum(w, eig_floor)
    return (V * w) @ V.T


# ---------------------------------------------------------------------------
# GLS objective
# ---------------------------------------------------------------------------


def gls_loss(u_hat: Sequence[float], delta: Sequence[float],
             R: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Generalized least-squares paired loss for one snapshot.

    ``(u_hat - delta)^T sg(R^{-1}) (u_hat - delta) / |A_paired|``.
    ``weights`` is the optional stop-gradiented inverse (defaults to
    ``inv(R)``); pass it when the weight matrix itself is estimated.
    """
    u_hat = np.asarray(u_hat, dtype=float)
    delta = np.asarray(delta, dtype=float)
    resid = u_hat - delta
    W = np.linalg.inv(R) if weights is None else weights
    return float(resid @ W @ resid / max(len(resid), 1))


def gls_gradient(u_hat: Sequence[float], delta: Sequence[float],
                 R: np.ndarray) -> np.ndarray:
    """Gradient of :func:`gls_loss` w.r.t. ``u_hat``: ``2 R^{-1} (u - d) / |A|``."""
    u_hat = np.asarray(u_hat, dtype=float)
    delta = np.asarray(delta, dtype=float)
    W = np.linalg.inv(R)
    return 2.0 * W @ (u_hat - delta) / max(len(u_hat), 1)
