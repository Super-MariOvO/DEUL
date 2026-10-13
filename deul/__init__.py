# -*- coding: utf-8 -*-
"""DEUL: Drift-Aware Experience Utility Learning — component library.

This package implements the algorithmic components of the paper
*When Experience Goes Stale: Drift-Aware Experience Utility Learning for
Long-Horizon Agents* as a small, dependency-light library:

* :mod:`deul.filter`      -- shared dependency-belief particle filter with a
  uniqueness ledger (one update per genuine event; duplicate invariance);
* :mod:`deul.values`      -- source-conditioned value vector (success/cost
  heads integrated under one posterior; dedup before max);
* :mod:`deul.paired`      -- paired-advantage GLS supervision with the exact
  shared-baseline covariance  Cov(Delta_a, Delta_b) = Var(bar Y_0);
* :mod:`deul.query`       -- depth-limited shared-posterior VOI query tree
  (one outcome broadcast to the whole candidate set; no VOI summing);
* :mod:`deul.controller`  -- the online control loop of Algorithm 2;
* :mod:`deul.objectives`  -- the unified training objective and the dual
  cost-multiplier update.

Everything is pure Python + NumPy.  The constructions of paper Section 6
(T1/T2) are reproduced in ``tests/`` and in ``scripts/run_two_state_voi.py``.
"""

from .filter import (Event, EventLedger, HazardModel, JointParticleFilter,
                     StaticHazard)
from .values import (Candidate, LinearValueModel, ValueModel, deduplicate,
                     incremental_utility, q_hat)
from .paired import (gls_gradient, gls_loss, paired_means_to_advantages,
                     shared_baseline_covariance, shrunk_covariance)
from .query import (DictOutcomeModel, Query, QueryTree,
                    bernoulli_marginal_worlds, xor_world)
from .controller import ActionRecord, DecisionTrace, OnlineController
from .objectives import (DualLambda, LossWeights, graph_penalty, obs_loss,
                         query_consistency_loss, td_loss, total_loss)

__version__ = "0.1.0"

__all__ = [
    "Event", "EventLedger", "HazardModel", "JointParticleFilter",
    "StaticHazard", "Candidate", "LinearValueModel", "ValueModel",
    "deduplicate", "incremental_utility", "q_hat",
    "gls_gradient", "gls_loss", "paired_means_to_advantages",
    "shared_baseline_covariance", "shrunk_covariance",
    "DictOutcomeModel", "Query", "QueryTree",
    "bernoulli_marginal_worlds", "xor_world",
    "ActionRecord", "DecisionTrace", "OnlineController",
    "DualLambda", "LossWeights", "graph_penalty", "obs_loss",
    "query_consistency_loss", "td_loss", "total_loss",
]
