# -*- coding: utf-8 -*-
"""DEUL online control loop (paper Algorithm 2, Section 5.7/5.8).

One decision round of the online controller:

1. Observe the current goal and legal environment response; update the
   physical context.
2. Retrieve ``K`` source origins; construct and *deduplicate* their
   representation candidates.
3. Apply the action-conditioned drift prediction and **one update per
   genuinely new event** through the shared filter.
4. Evaluate candidate values and the depth-``d`` paid-query tree under one
   shared posterior.
5. Take only the first feasible action; charge all observation / reasoning /
   execution cost.
6. If querying: broadcast the one actual outcome through the dependency
   graph, then decide again.
   If reusing / adapting / replanning: execute until the next boundary.

The controller never accesses unchosen-branch rewards, future tasks, hidden
drift labels, or free rollback (paper Algorithm 2, step 9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Sequence

import numpy as np

from .filter import Event, JointParticleFilter
from .query import Query, QueryTree
from .values import Candidate, ValueModel, deduplicate, q_hat

__all__ = ["ActionRecord", "DecisionTrace", "OnlineController"]


# ---------------------------------------------------------------------------
# Decision trace
# ---------------------------------------------------------------------------


@dataclass
class ActionRecord:
    """One charged decision in the online loop."""

    kind: str                      # 'query' | 'reuse' | 'adapt' | 'replan'
    target: Hashable               # query key or candidate key
    cost_tokens: float = 0.0       # logical token cost charged
    cost_env: float = 0.0          # native environment primitive cost


@dataclass
class DecisionTrace:
    """The full charged trace of one task."""

    records: List[ActionRecord] = field(default_factory=list)
    token_spent: float = 0.0
    env_spent: float = 0.0

    def charge(self, record: ActionRecord) -> None:
        self.records.append(record)
        self.token_spent += record.cost_tokens
        self.env_spent += record.cost_env


# ---------------------------------------------------------------------------
# Online controller
# ---------------------------------------------------------------------------


class OnlineController:
    """Budget-constrained sequential controller.

    Parameters:
        filter_: Shared joint particle filter over conditions.
        model: Source-conditioned value model.
        query_tree: Shared-posterior VOI query tree.
        lam: Non-negative cost multiplier (normalized budget units).
        token_budget: Hard logical-token cap ``B_i``.
        env_budget: Hard native-action cap ``B_i^env``.
        query_cost_tokens / query_cost_env: Charging convention of one query.
    """

    def __init__(self, filter_: JointParticleFilter, model: ValueModel,
                 query_tree: QueryTree, lam: float = 0.05,
                 token_budget: float = 64_000.0,
                 env_budget: float = 80.0,
                 query_cost_tokens: float = 1_000.0,
                 query_cost_env: float = 1.0) -> None:
        self.filter = filter_
        self.model = model
        self.query_tree = query_tree
        self.lam = float(lam)
        self.token_budget = float(token_budget)
        self.env_budget = float(env_budget)
        self.query_cost_tokens = float(query_cost_tokens)
        self.query_cost_env = float(query_cost_env)

    # -- candidate assembly ----------------------------------------------------

    def build_candidates(self, sources: Iterable[Candidate]) -> List[Candidate]:
        """Retrieve sources and deduplicate their representation candidates.

        Deduplication happens *before* any max is taken (paper Section 5.4:
        "重复候选在取最大值前按实体与表示版本去重").
        """
        return deduplicate(sources)

    # -- one decision round ------------------------------------------------------

    def decide(self, candidates: Sequence[Candidate],
               queries: Sequence[Query],
               trace: Optional[DecisionTrace] = None,
               belief: Hashable = "b") -> ActionRecord:
        """Evaluate candidates and the query tree; return the chosen action.

        The *first feasible* action is chosen by net value
        ``Q(b, a) = Q^Y(b, a) - lam * Q^C(b, a)`` against the query tree.
        Querying is only chosen when its VOI is positive and exceeds the best
        immediate candidate (paper: VOI>0 才付费查证，否则直接复用/适配/重规划).
        """
        trace = trace or DecisionTrace()
        cands = self.build_candidates(candidates)

        best_cand, best_val = None, -np.inf
        for c in cands:
            v = q_hat(self.model, c, self.filter, lam=self.lam)
            if v > best_val:
                best_cand, best_val = c, v

        # Shared query tree under the same posterior.
        tree_val = -np.inf
        best_query = None
        if queries and self.query_tree is not None:
            for q in queries:
                if (self.token_budget - trace.token_spent) < self.query_cost_tokens:
                    break  # hard budget: querying no longer affordable
                v = self.query_tree.voi(belief, 1.0, 2, q, queries)
                if v > tree_val:
                    best_query, tree_val = q, v

        if best_query is not None and tree_val > 0.0 and tree_val > best_val:
            record = ActionRecord("query", best_query.key,
                                  cost_tokens=self.query_cost_tokens,
                                  cost_env=self.query_cost_env)
        else:
            kind = best_cand.key[0] if isinstance(best_cand.key, tuple) else "reuse"
            record = ActionRecord(kind, best_cand.key)

        trace.charge(record)
        return record

    # -- evidence incorporation ------------------------------------------------

    def incorporate(self, event: Event,
                    likelihood: Callable[[Hashable, Hashable], float]) -> bool:
        """Apply one genuine event to the shared filter (single update)."""
        return self.filter.update(event, likelihood)

    # -- budget guards -----------------------------------------------------------

    def within_budget(self, trace: DecisionTrace) -> bool:
        """Hard budget check: no purchase may violate token or env caps."""
        return (trace.token_spent <= self.token_budget
                and trace.env_spent <= self.env_budget)
