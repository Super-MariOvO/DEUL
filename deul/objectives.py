# -*- coding: utf-8 -*-
"""Unified training objective and dual cost-multiplier update (paper Section 5.6).

The unified training objective is

    min_{theta, eta}  L_TD + alpha * L_pair + beta * L_obs
                      + kappa * L_query + xi * R_graph

with the non-negative cost multiplier updated by the standard dual rule

    lambda_{k+1} = [ lambda_k + rho * ( E[C_i / B_i] - bar C ) ]_+

The four losses constrain four different failure modes; writing them on one
line is not itself a contribution (paper Section 5.6: "损失写在同一行本身不是
贡献") -- the point is that the *same* frozen continuation policy and the
*same* filter are used across all of them:

* ``L_TD``    -- actual-transition TD for executed actions;
* ``L_pair``  -- paired-advantage GLS over snapshots (see ``deul.paired``);
* ``L_obs``   -- unique-event predictive likelihood, anchoring the belief to
  observable evidence (each event counted once);
* ``L_query`` -- query-consistency, tying the query head to the continuation
  values produced by the same lagged filter;
* ``R_graph`` -- penalty on unsupported dense dependency edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .paired import gls_loss

__all__ = ["LossWeights", "td_loss", "obs_loss", "query_consistency_loss",
           "graph_penalty", "total_loss", "DualLambda"]


@dataclass(frozen=True)
class LossWeights:
    """Default loss weights (paper Section 5.9 implementation starting point)."""

    alpha: float = 1.0    # L_pair
    beta: float = 0.2     # L_obs
    kappa: float = 0.5    # L_query
    xi: float = 0.001     # R_graph
    omega_c: float = 1.0  # cost term inside L_TD


# ---------------------------------------------------------------------------
# Individual losses
# ---------------------------------------------------------------------------


def td_loss(q_pred: Sequence[float], y_targets: Sequence[float],
            c_pred: Sequence[float], c_targets: Sequence[float],
            omega_c: float = 1.0) -> float:
    """TD loss over actual transitions: success and cost share one next action.

    ``E[(Q^Y - sg(y^Y))^2 + omega_c * (Q^C - sg(y^C))^2]``; both targets are
    defined by the *same* frozen next action ``a^+ = phi_k(b')`` so that the
    two heads evaluate one common continuation (paper Section 5.6).
    """
    q_pred = np.asarray(q_pred, dtype=float)
    y_targets = np.asarray(y_targets, dtype=float)
    c_pred = np.asarray(c_pred, dtype=float)
    c_targets = np.asarray(c_targets, dtype=float)
    return float(np.mean((q_pred - y_targets) ** 2)
                 + omega_c * np.mean((c_pred - c_targets) ** 2))


def obs_loss(log_likelihoods: Sequence[float]) -> float:
    """Unique-event predictive NLL.

    ``-(1 / |E_unique|) sum_u log p(o_u)`` where every event appears exactly
    once -- the ledger of ``deul.filter`` is what makes this count legal.
    """
    ll = np.asarray(log_likelihoods, dtype=float)
    return float(-np.mean(ll))


def query_consistency_loss(q_hat: Sequence[float],
                           q_targets: Sequence[float]) -> float:
    """Query-consistency loss: align predicted query values to continuations
    produced by the same lagged filter, not to imagined rewards."""
    q_hat = np.asarray(q_hat, dtype=float)
    q_targets = np.asarray(q_targets, dtype=float)
    return float(np.mean((q_hat - q_targets) ** 2))


def graph_penalty(edges_supported: Sequence[bool]) -> float:
    """Penalty on unsupported dense dependency edges.

    ``xi * (#unsupported edges)``; the regularizer discourages density
    without forcing all dependencies to zero (paper: "不把所有依赖压到零").
    """
    edges_supported = np.asarray(edges_supported, dtype=bool)
    return float(np.sum(~edges_supported))


# ---------------------------------------------------------------------------
# Total objective
# ---------------------------------------------------------------------------


def total_loss(l_td: float, l_pair: float, l_obs: float, l_query: float,
               l_graph: float, weights: LossWeights = LossWeights()) -> float:
    """Weighted total objective (weights fixed on development data only)."""
    return (l_td + weights.alpha * l_pair + weights.beta * l_obs
            + weights.kappa * l_query + weights.xi * l_graph)


# ---------------------------------------------------------------------------
# Dual cost multiplier
# ---------------------------------------------------------------------------


@dataclass
class DualLambda:
    """Dual update of the non-negative cost multiplier ``lambda``.

    ``lambda_{k+1} = [lambda_k + rho * (normalized_cost - bar_C)]_+``
    evaluated on held-out *training* rollouts -- never on test performance
    (paper Section 5.6: "Update lambda from held-out training rollout cost,
    not from test performance").
    """

    value: float = 0.05
    rho: float = 0.01
    bar_c: float = 1.0
    history: List[float] = field(default_factory=list)

    def update(self, normalized_cost: float) -> float:
        """One dual step given the measured normalized average cost."""
        self.value = max(self.value + self.rho * (normalized_cost - self.bar_c),
                         0.0)
        self.history.append(self.value)
        return self.value
