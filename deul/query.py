# -*- coding: utf-8 -*-
"""Shared-evidence sequential value of information (paper Section 5.7).

For a query ``q``, the controller first samples *one* outcome ``o`` from a
single joint predictive distribution, then broadcasts that one outcome to the
entire candidate set through the shared posterior.  It must not draw a
mutually contradictory page/protocol per candidate, and it must not sum the
per-candidate values of information, since the task reward can only be
realized once.

With normalized budget ``B`` and at most ``d`` consecutive queries, the
improvement rule is

    V_0(b, B) = max_{a in A_exec} Q(b, a)
    V_d(b, B) = max{ V_0(b, B),
                     max_q [ -lam * c_q + E_{o|b,q} V_{d-1}(T(b,q,o), B-c_q) ] }

and the net value of one query is

    VOI(q | b, B) = E_{o|b,q} V_{d-1}(T(b,q,o), B-c_q) - V_0(b, B) - lam * c_q.

This module implements the recursion over an abstract world model
(:class:`OutcomeModel`) so that the constructions of paper Section 6
(T1: identical marginals, different query values; T2: complementary queries
beating the myopic rule) can be reproduced and unit-tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, Iterable, List, Mapping, Sequence

import numpy as np

__all__ = ["Query", "OutcomeModel", "DictOutcomeModel", "QueryTree"]


# ---------------------------------------------------------------------------
# Queries and outcome models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Query:
    """A paid evidence-acquisition action.

    Attributes:
        key: Identifier of the query.
        cost: Charged normalized cost ``c_q`` (same budget unit as values).
        tag: Free-form payload, e.g. the condition it inspects.
    """

    key: Hashable
    cost: float
    tag: Hashable = None


class OutcomeModel:
    """Interface for the shared predictive world model.

    Given the belief ``b`` and a query ``q``, the model returns the possible
    outcomes with their probabilities and, for each outcome, the updated
    belief state ``T(b, q, o)`` *shared by all candidates*.
    """

    def outcomes(self, belief: Hashable,
                 query: Query) -> Sequence[tuple[Hashable, float, Hashable]]:
        """Return ``[(outcome, probability, next_belief), ...]``."""
        raise NotImplementedError  # pragma: no cover - interface


@dataclass
class DictOutcomeModel(OutcomeModel):
    """Tabular outcome model, convenient for exact constructions.

    ``table[(belief, query.key)] = [(outcome, prob, next_belief), ...]``
    """

    table: Dict = field(default_factory=dict)

    def outcomes(self, belief: Hashable,
                 query: Query) -> Sequence[tuple[Hashable, float, Hashable]]:
        return self.table.get((belief, query.key), [])


# ---------------------------------------------------------------------------
# Query tree
# ---------------------------------------------------------------------------


class QueryTree:
    """Depth-limited shared-posterior query tree.

    Parameters:
        model: The shared predictive outcome model.
        exec_value: Callable ``(belief, budget) -> V_0`` giving the best
            immediate-execution value.
        lam: Non-negative cost multiplier (budget-normalized units).
        max_depth: Maximum number of consecutive paid queries (paper
            default: 2).
        max_queries: Maximum number of legal queries expanded per node
            (paper default: 6).
        max_outcomes: Maximum outcome branches kept per query (paper
            default: 3).
    """

    def __init__(self, model: OutcomeModel,
                 exec_value: Callable[[Hashable, float], float],
                 lam: float = 0.05, max_depth: int = 2,
                 max_queries: int = 6, max_outcomes: int = 3) -> None:
        self.model = model
        self.exec_value = exec_value
        self.lam = float(lam)
        self.max_depth = int(max_depth)
        self.max_queries = int(max_queries)
        self.max_outcomes = int(max_outcomes)

    # -- public API ------------------------------------------------------------

    def value(self, belief: Hashable, budget: float, depth: int,
              queries: Iterable[Query]) -> float:
        """``V_d(b, B)`` = best of executing now vs. querying then continuing."""
        immediate = self.exec_value(belief, budget)
        if depth <= 0 or budget <= 0:
            return immediate

        best = immediate
        for q in list(queries)[: self.max_queries]:
            if q.cost > budget:
                continue
            branch = self._query_branch(belief, budget, depth, q, queries)
            best = max(best, branch)
        return best

    def _query_branch(self, belief: Hashable, budget: float, depth: int,
                      query: Query, queries: Iterable[Query]) -> float:
        """Value of taking ``query`` first, then acting optimally afterwards."""
        total = 0.0
        branches = self.model.outcomes(belief, query)[: self.max_outcomes]
        for outcome, prob, next_belief in branches:
            continuation = self.value(next_belief, budget - query.cost,
                                      depth - 1, queries)
            total += prob * continuation
        return total - self.lam * query.cost

    def voi(self, belief: Hashable, budget: float, depth: int,
            query: Query, queries: Iterable[Query]) -> float:
        """Net value of one query: ``E_o V_{d-1} - V_0 - lam * c_q``."""
        if query.cost > budget or depth <= 0:
            return -self.lam * query.cost
        expected = 0.0
        branches = self.model.outcomes(belief, query)[: self.max_outcomes]
        for _, prob, next_belief in branches:
            expected += prob * self.value(next_belief, budget - query.cost,
                                          depth - 1, queries)
        immediate = self.exec_value(belief, budget)
        return expected - immediate - self.lam * query.cost

    def myopic_voi(self, belief: Hashable, budget: float, query: Query,
                   queries: Iterable[Query]) -> float:
        """One-step VOI used by the myopic comparator (depth = 1).

        The myopic rule commits to execution after at most one query; it is
        the baseline that complementary queries are shown to beat in the
        paper's T2 construction.
        """
        return self.voi(belief, budget, 1, query, queries)


# ---------------------------------------------------------------------------
# Helpers for exact constructions (paper Sections 6.1-6.2)
# ---------------------------------------------------------------------------


def bernoulli_marginal_worlds() -> Dict[str, Dict]:
    """The two worlds of construction T1 (identical marginals).

    Returns ``{"positive": {...}, "negative": {...}}`` where each world is a
    dict of candidate-indexed joint outcome tables.  Candidates 1 and 2 have
    payoff 0/1 each with marginal Bernoulli(0.5); replan is always 0.
    """
    return {
        "positive": {  # outcomes (1,1) or (0,0) with prob 1/2
            "outcomes": [((1, 1), 0.5), ((0, 0), 0.5)],
            "reveal_value": lambda o: 0.5,      # after revealing c1, best = 0.5
        },
        "negative": {  # outcomes (1,0) or (0,1) with prob 1/2
            "outcomes": [((1, 0), 0.5), ((0, 1), 0.5)],
            "reveal_value": lambda o: 1.0,      # after revealing c1, best = 1.0
        },
    }


def xor_world() -> Dict[str, object]:
    """The XOR world of construction T2 (complementary queries).

    ``z1, z2`` are independent fair bits; candidate A pays 1 iff z1 XOR z2 = 0,
    candidate B pays 1 iff z1 XOR z2 = 1.  Each query reveals one bit at cost
    0.1.  No single query changes either candidate's 0.5 expectation, but the
    pair identifies a sure winner with net value 1 - 0.2 = 0.8 > 0.5.
    """
    return {
        "bits": ("z1", "z2"),
        "query_cost": 0.1,
        "myopic_value": 0.5,                    # best without querying
        "joint_value": 1.0 - 2 * 0.1,           # certain winner after two
    }
