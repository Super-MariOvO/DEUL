# -*- coding: utf-8 -*-
"""Source-conditioned value vector for DEUL (paper Section 5.4).

The controller keeps *one* shared encoder that reads the goal, the observable
physical state, the source content, the representation residual, the
dependency graph and the remaining budget.  For every execution candidate
``a`` (reuse / adapt / replan) two heads emit a remaining-success value and a
cost value, both integrated under the *same* posterior ``p_t``:

    Q^Y(b, a) = E_{z~p_t}[ Q_theta^Y(g, b_obs, z, a, B) ]
    Q^C(b, a) = E_{z~p_t}[ Q_theta^C(g, b_obs, z, a, B) ]
    Q(b, a)   = Q^Y(b, a) - lambda * Q^C(b, a)

The estimated incremental utility of reusing representation ``r`` of source
``e`` is then

    U(e, r | g, b) = Q^Y(b, reuse(e, r)) - Q^Y(b, replan)

Design obligations encoded here:

* The success head is bounded to ``[0, 1]``; the cost head is non-negative.
* Replanning has its own state-dependent value -- it is **not** a constant.
* Sharing a source does **not** force equal utilities: the representation
  residual lets the same condition affect Trajectory / Plan / Skill with
  different strength, sign and residual (paper: "不要求'同源则同分'").
* Duplicate candidates are deduplicated by entity and representation version
  before any max is taken, so candidate duplication cannot change the
  selection distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, List, Mapping, Optional

import numpy as np

from .filter import JointParticleFilter

__all__ = ["Candidate", "ValueModel", "LinearValueModel", "deduplicate"]


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """An execution candidate.

    Attributes:
        key: Unique key of the candidate itself, e.g. ``("reuse", e, r)``,
            ``("adapt", e, r, J)`` or ``("replan",)``.
        features: Feature vector read by the shared encoder.  Contents are
            adapter-defined (goal/source/representation/graph/budget
            summaries); the model never sees hidden condition labels.
        version: Representation version used for deduplication.  Two
            candidates are duplicates iff their ``(key, version)`` match.
    """

    key: Hashable
    features: np.ndarray
    version: Hashable = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "features",
                           np.asarray(self.features, dtype=float))


def deduplicate(candidates: Iterable[Candidate]) -> List[Candidate]:
    """Drop duplicate ``(key, version)`` candidates, preserving first order.

    The paper requires deduplication *before* any max over candidates:
    copying a candidate must not change its selection probability.
    """
    seen = set()
    out: List[Candidate] = []
    for c in candidates:
        sig = (c.key, c.version)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Value models
# ---------------------------------------------------------------------------


class ValueModel:
    """Interface for per-condition value heads.

    Implementations return, for one candidate and one joint condition
    assignment ``z``, the pair ``(Q^Y, Q^C)``.  A candidate that is infeasible
    under ``z`` should return success 0 and a large cost so it is never
    selected in that world.
    """

    def q_values(self, candidate: Candidate,
                 z: Mapping[Hashable, Hashable]) -> tuple[float, float]:
        raise NotImplementedError  # pragma: no cover - interface

    def integrate(self, candidate: Candidate,
                  posterior: JointParticleFilter) -> tuple[float, float]:
        """Posterior expectation of (Q^Y, Q^C) over joint particles."""
        successes, costs = [], []
        for k in range(posterior.n_particles):
            z = {j: posterior.values[j][posterior.particles[k, i]]
                 for i, j in enumerate(posterior.conditions)}
            qy, qc = self.q_values(candidate, z)
            successes.append(qy)
            costs.append(qc)
        w = posterior.weights
        return float(np.dot(w, successes)), float(np.dot(w, costs))


@dataclass
class LinearValueModel(ValueModel):
    """Linear heads on candidate features with condition interactions.

    ``Q^Y = sigmoid(w_y · x + sum_j u_y[j](z_j))``
    ``Q^C = softplus(w_c · x + sum_j u_c[j](z_j))``

    The per-condition offsets ``u_*[j][value]`` let one condition change the
    value of *different* candidates by different amounts -- the mechanism
    behind representation-conditional utility without forcing same-source
    candidates to share a score.
    """

    w_success: np.ndarray
    w_cost: np.ndarray
    cond_success: Optional[Dict[Hashable, Dict[Hashable, float]]] = None
    cond_cost: Optional[Dict[Hashable, Dict[Hashable, float]]] = None
    lam: float = 0.05

    def _offset(self, table: Optional[Dict], j: Hashable,
                v: Hashable) -> float:
        if not table:
            return 0.0
        return float(table.get(j, {}).get(v, 0.0))

    def q_values(self, candidate: Candidate,
                 z: Mapping[Hashable, Hashable]) -> tuple[float, float]:
        x = candidate.features
        sy = float(self.w_success @ x) + sum(
            self._offset(self.cond_success, j, v) for j, v in z.items())
        sc = float(self.w_cost @ x) + sum(
            self._offset(self.cond_cost, j, v) for j, v in z.items())
        qy = 1.0 / (1.0 + np.exp(-sy))          # success head in [0, 1]
        qc = np.log1p(np.exp(sc))                # cost head non-negative
        return qy, qc

    # -- convenience constructors used by demos/tests -------------------------

    @classmethod
    def constant(cls, success: float, cost: float,
                 dim: int = 1) -> "LinearValueModel":
        """A value model returning the same (Q^Y, Q^C) for any candidate."""
        m = cls(w_success=np.zeros(dim), w_cost=np.zeros(dim))
        base = cls(w_success=np.zeros(dim), w_cost=np.zeros(dim))
        base.q_values = lambda candidate, z: (success, cost)  # type: ignore
        return base  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Posterior-integrated utilities
# ---------------------------------------------------------------------------


def q_hat(model: ValueModel, candidate: Candidate,
          posterior: JointParticleFilter, lam: float = 0.05) -> float:
    """Net value ``Q(b, a) = Q^Y(b, a) - lambda * Q^C(b, a)``."""
    qy, qc = model.integrate(candidate, posterior)
    return qy - lam * qc


def incremental_utility(model: ValueModel, reuse: Candidate,
                        replan: Candidate,
                        posterior: JointParticleFilter) -> float:
    """``U(e, r) = Q^Y(reuse) - Q^Y(replan)`` under the shared posterior.

    The two candidates are evaluated under the *same* joint particles, so a
    single condition change can move the estimate of one representation up
    and of another down within one consistent world (paper Section 5.4,
    "共同积分的作用").
    """
    qy_reuse, _ = model.integrate(reuse, posterior)
    qy_replan, _ = model.integrate(replan, posterior)
    return qy_reuse - qy_replan
