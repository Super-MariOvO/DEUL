# -*- coding: utf-8 -*-
"""Shared dependency-belief filter for DEUL (paper Section 5.3).

This module implements the local non-stationary belief filter used by the
controller.  Its two defining obligations are

1. **One update per genuine event.**  Every raw observation carries an
   immutable ``event_id``.  The ledger (:class:`EventLedger`) guarantees that
   the likelihood of any single response is multiplied into the posterior
   *exactly once*.  Re-retrieving the same response nine times must not yield
   nine independent pieces of evidence (paper: "同一响应被检索九次不提供九份
   信息").

2. **One shared posterior over conditions.**  All candidate representations
   are evaluated under the *same* joint posterior ``p_t`` over latent
   condition states ``z``.  A single observation may therefore update the
   values of several candidates coherently, without ever being treated as
   several independent observations.

The transition model for a persistent condition ``j`` is the
retain-or-reset mixture from change-point modeling (paper Eq. for
:math:`p_\\eta(z_{j,t}\\mid z_{t-1},a_{t-1},b_{t-1})`):

    p(z_j,t) = (1 - h_j) * p_persist(z_j,t | z_{t-1}) + h_j * p_{j,0}(z_j,t)

where the local hazard ``h_j`` is estimated from visible cues, elapsed time
and historical residuals -- never from evaluator change labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Set

import numpy as np

__all__ = ["Event", "EventLedger", "HazardModel", "StaticHazard",
           "JointParticleFilter"]


# ---------------------------------------------------------------------------
# Events and the uniqueness ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """A single genuine observation.

    Attributes:
        event_id: Immutable identifier of the raw response.  Two *renderings*
            of the same response share one ``event_id`` and are one
            measurement; two independent calls returning identical content
            carry different ``event_id`` and remain distinct measurements.
        condition: Condition node ``j`` the observation is about.
        observation: Observed value for that condition (any hashable label).
        timestamp: Logical time of observation (used by the hazard model).
    """

    event_id: Hashable
    condition: Hashable
    observation: Hashable
    timestamp: float = 0.0


@dataclass
class EventLedger:
    """Records which events have already been incorporated into ``p_t``.

    The ledger is the mechanism behind the paper's *duplicate invariance*
    property (Section 6.3, "重复不变性"): duplicating a reference to an event
    changes neither the posterior nor the induced policy distribution, because
    the second presentation of the same ``event_id`` is not a new update.
    """

    _processed: Set[Hashable] = field(default_factory=set)

    def is_new(self, event: Event) -> bool:
        """Return True iff ``event`` has never been processed."""
        return event.event_id not in self._processed

    def mark_processed(self, event: Event) -> None:
        """Mark ``event`` as incorporated; subsequent calls are idempotent."""
        self._processed.add(event.event_id)

    def __contains__(self, event_id: Hashable) -> bool:
        return event_id in self._processed

    def __len__(self) -> int:
        return len(self._processed)


# ---------------------------------------------------------------------------
# Hazard model (local drift rate)
# ---------------------------------------------------------------------------


class HazardModel:
    """Interface for local drift hazards ``h_{j,t}``.

    A hazard model estimates the reset probability of condition ``j`` from
    *visible* cues only: elapsed time, previous residuals and observable
    change indicators.  It must never read hidden evaluator labels.
    """

    def hazard(self, condition: Hashable, dt: float,
               residual: float = 0.0) -> float:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class StaticHazard(HazardModel):
    """Constant per-condition hazard with an elapsed-time ramp.

    ``h_j = clip(base + rate * dt + gain * |residual|, 0, h_max)``

    This is the simplest legal choice: it uses only visible quantities and
    keeps ``h_j`` inside ``[0, h_max]``.
    """

    base: float = 0.01
    rate: float = 0.0
    gain: float = 0.0
    h_max: float = 0.5

    def hazard(self, condition: Hashable, dt: float,
               residual: float = 0.0) -> float:
        h = self.base + self.rate * max(dt, 0.0) + self.gain * abs(residual)
        return float(np.clip(h, 0.0, self.h_max))


# ---------------------------------------------------------------------------
# Joint particle filter
# ---------------------------------------------------------------------------


class JointParticleFilter:
    """A joint particle belief over a small set of condition nodes.

    Each particle is an *entire assignment* ``z`` of every condition to one of
    its admissible values -- "one particle is one consistent possible world"
    (paper Section 5.3).  Cross-condition correlations are therefore retained
    instead of being collapsed to marginals.

    Parameters:
        conditions: Ordered list of condition node identifiers.
        values: ``values[j]`` = ordered list of admissible values of ``j``.
        n_particles: Number of joint particles (paper default: 32).
        hazard_model: Local drift model for persistent conditions.
        reset_prior: Optional callable ``j, value -> probability`` giving the
            reset component :math:`p_{j,0}`.  Defaults to uniform.
        rng: Optional ``numpy.random.Generator``.
    """

    def __init__(
        self,
        conditions: Iterable[Hashable],
        values: Dict[Hashable, Iterable[Hashable]],
        n_particles: int = 32,
        hazard_model: Optional[HazardModel] = None,
        reset_prior: Optional[Callable[[Hashable, Hashable], float]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.conditions: List[Hashable] = list(conditions)
        self.values: Dict[Hashable, List[Hashable]] = {
            j: list(v) for j, v in values.items()
        }
        for j in self.conditions:
            if j not in self.values or not self.values[j]:
                raise ValueError(f"condition {j!r} needs >=1 admissible value")
        self.n_particles = int(n_particles)
        self.hazard_model = hazard_model or StaticHazard()
        self.reset_prior = reset_prior
        self.rng = rng or np.random.default_rng()

        # particles[k, i] = value index of condition i in particle k.
        # Stratified init: tile the cartesian product of all condition values
        # so the empirical joint prior factorizes *exactly* across conditions
        # (required for the local-update factorization property of Section 6.4).
        import itertools
        combos = np.array(list(itertools.product(
            *[range(len(self.values[j])) for j in self.conditions])))
        reps = int(np.ceil(self.n_particles / max(len(combos), 1)))
        tiled = np.tile(combos, (reps, 1))[: self.n_particles]
        self.particles = self.rng.permutation(tiled)
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)
        self.ledger = EventLedger()
        self._last_timestamp: float = 0.0

    # -- initialization ----------------------------------------------------

    @classmethod
    def uniform(cls, conditions, values, **kw) -> "JointParticleFilter":
        """Construct a filter with a uniform joint prior."""
        return cls(conditions, values, **kw)

    # -- transition ----------------------------------------------------------

    def _reset_probs(self, j: Hashable) -> np.ndarray:
        n = len(self.values[j])
        if self.reset_prior is None:
            return np.full(n, 1.0 / n)
        p = np.array([self.reset_prior(j, v) for v in self.values[j]],
                     dtype=float)
        total = p.sum()
        return p / total if total > 0 else np.full(n, 1.0 / n)

    def predict(self, dt: float = 0.0) -> None:
        """Transition step :math:`p_t^-(z)` with retain-or-reset per condition.

        With probability ``1 - h_j`` a particle keeps its value of ``j``;
        with probability ``h_j`` it redraws from the reset prior
        :math:`p_{j,0}`.  Conditions are transitioned independently given the
        current particle, so existing cross-condition structure in the
        empirical particle set is only diluted by drift, never amplified.
        """
        for i, j in enumerate(self.conditions):
            h = self.hazard_model.hazard(j, dt)
            if h <= 0.0:
                continue
            redraw = self.rng.random(self.n_particles) < h
            if not redraw.any():
                continue
            rp = self._reset_probs(j)
            new_idx = self.rng.choice(len(rp), size=int(redraw.sum()), p=rp)
            self.particles[redraw, i] = new_idx
        self._last_timestamp += max(dt, 0.0)

    # -- update --------------------------------------------------------------

    def update(self, event: Event,
               likelihood: Callable[[Hashable, Hashable], float]) -> bool:
        """Incorporate one event via a single likelihood update.

        ``likelihood(j_value, observation)`` = :math:`p_\\eta(o_u \\mid z_j)`.
        The event is applied to *every* particle and weights are
        renormalized; correlations with other conditions are preserved by the
        joint reweighting.

        Returns:
            True if the event was new and applied, False if it was a
            duplicate (duplicate invariance, paper Section 6.3).
        """
        if not self.ledger.is_new(event):
            return False

        i = self.conditions.index(event.condition)
        obs_idx = {v: k for k, v in enumerate(self.values[event.condition])}
        if event.observation not in obs_idx:
            # Unknown observation branch: keep 'unknown' mass instead of
            # collapsing the belief (paper: 类别器保留 unknown 分支).
            return False

        j_vals = [self.values[event.condition][k] for k in self.particles[:, i]]
        lks = np.array([likelihood(v, event.observation) for v in j_vals],
                       dtype=float)
        if np.any(lks < 0):
            raise ValueError("likelihood must be non-negative")
        self.weights *= lks
        total = self.weights.sum()
        if total <= 0.0:
            # All particles got zero likelihood: relax uncertainty instead of
            # trusting stale experience (paper: 校准失败时放宽不确定性并回退).
            self.weights.fill(1.0 / self.n_particles)
        else:
            self.weights /= total

        self.ledger.mark_processed(event)
        dt = max(event.timestamp - self._last_timestamp, 0.0)
        if dt > 0:
            self.predict(dt)
        return True

    def update_many(self, events: Iterable[Event],
                    likelihood: Callable[[Hashable, Hashable], float]) -> int:
        """Apply several events in order; returns the number of *new* events."""
        return sum(self.update(e, likelihood) for e in events)

    # -- queries ---------------------------------------------------------------

    def marginal(self, condition: Hashable) -> Dict[Hashable, float]:
        """Posterior marginal of one condition as ``{value: probability}``."""
        i = self.conditions.index(condition)
        out = {v: 0.0 for v in self.values[condition]}
        for k in range(self.n_particles):
            out[self.values[condition][self.particles[k, i]]] += self.weights[k]
        return out

    def joint_sample(self, n: int = 1) -> List[Dict[Hashable, Hashable]]:
        """Draw ``n`` joint condition assignments from the posterior."""
        idx = self.rng.choice(self.n_particles, size=n, p=self.weights)
        return [
            {j: self.values[j][self.particles[k, i]]
             for i, j in enumerate(self.conditions)}
            for k in idx
        ]

    def probability(self, condition: Hashable, value: Hashable) -> float:
        """Posterior probability that ``condition == value``."""
        return self.marginal(condition)[value]

    def copy(self) -> "JointParticleFilter":
        """Deep copy (particles, weights, ledger)."""
        new = JointParticleFilter(self.conditions, self.values,
                                  self.n_particles, self.hazard_model,
                                  self.reset_prior, self.rng)
        new.particles = self.particles.copy()
        new.weights = self.weights.copy()
        new.ledger._processed = set(self.ledger._processed)
        new._last_timestamp = self._last_timestamp
        return new

    # -- invariance helpers (used by unit tests) -------------------------------

    def state_signature(self) -> bytes:
        """Hash of the full posterior state, for duplicate-invariance checks."""
        return (self.particles.tobytes(), self.weights.tobytes(),
                frozenset(self.ledger._processed)).__repr__().encode()
