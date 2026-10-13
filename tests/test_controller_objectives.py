# -*- coding: utf-8 -*-
"""Tests for the online control loop and unified objective (paper Algorithm 2, Section 5.6)."""

import numpy as np
import pytest

from deul import (Candidate, DecisionTrace, DualLambda, DictOutcomeModel,
                  JointParticleFilter, LinearValueModel, OnlineController,
                  Query, QueryTree, StaticHazard, incremental_utility,
                  obs_loss, query_consistency_loss, td_loss, total_loss,
                  LossWeights, graph_penalty)


def make_controller(token_budget=64_000.0):
    f = JointParticleFilter(["layout"], {"layout": ["old", "new"]},
                            n_particles=32, rng=np.random.default_rng(0))
    model = LinearValueModel(w_success=np.array([1.0]), w_cost=np.array([0.1]))
    tree = QueryTree(DictOutcomeModel(table={}), exec_value=lambda b, B: 0.5,
                     lam=0.05)
    return OnlineController(f, model, tree, lam=0.05,
                            token_budget=token_budget)


def cand(name, x, version=0):
    return Candidate(key=name, features=np.array([float(x)]), version=version)


class TestCandidates:
    def test_dedup_before_max_preserves_selection(self):
        c = make_controller()
        cands = [cand("reuse_t", 0.5), cand("reuse_p", 0.8),
                 cand("reuse_p", 0.8)]            # duplicate of the best
        rec1 = c.decide(cands, [])
        rec2 = c.decide([cand("reuse_t", 0.5), cand("reuse_p", 0.8)], [])
        assert rec1.target == rec2.target == "reuse_p"

    def test_utility_identity(self):
        """U(e, r) = Q^Y(reuse) - Q^Y(replan) under one posterior (paper 5.4)."""
        f = JointParticleFilter(["layout"], {"layout": ["old", "new"]},
                                n_particles=32, rng=np.random.default_rng(0))
        model = LinearValueModel(w_success=np.array([1.0]), w_cost=np.array([0.0]))
        reuse = cand("reuse", 2.0)
        replan = cand(("replan",), 0.0)
        u = incremental_utility(model, reuse, replan, f)
        qy_r, _ = model.integrate(reuse, f)
        qy_p, _ = model.integrate(replan, f)
        assert u == pytest.approx(qy_r - qy_p)


class TestBudget:
    def test_charge_accumulates_and_caps(self):
        c = make_controller(token_budget=2_000.0)
        trace = DecisionTrace()
        trace.charge(type("R", (), {"cost_tokens": 1_500.0, "cost_env": 1.0})())
        assert c.within_budget(trace)
        trace.charge(type("R", (), {"cost_tokens": 600.0, "cost_env": 0.0})())
        assert not c.within_budget(trace)


class TestQueryDecision:
    def test_query_only_when_voi_positive(self):
        f = JointParticleFilter(["layout"], {"layout": ["old", "new"]},
                                n_particles=32, rng=np.random.default_rng(0))
        model = LinearValueModel(w_success=np.array([1.0]), w_cost=np.array([0.1]))
        # Query reveals nothing useful (empty outcome table) -> must execute.
        tree = QueryTree(DictOutcomeModel(table={}), exec_value=lambda b, B: 0.5,
                         lam=0.05)
        c = OnlineController(f, model, tree, lam=0.05)
        rec = c.decide([cand("reuse_t", 0.9)], [Query("q_probe", cost=1.0)])
        assert rec.kind == "reuse"


class TestObjectives:
    def test_td_loss_shared_continuation(self):
        l = td_loss([0.5, 0.5], [1.0, 0.0], [0.2, 0.2], [0.0, 0.0], omega_c=1.0)
        assert l == pytest.approx(0.25 + 0.04)

    def test_obs_loss_counts_unique_events_only(self):
        # Two unique events with known likelihoods; duplicates never reach here.
        assert obs_loss([np.log(0.5), np.log(0.25)]) == pytest.approx(
            -np.mean([np.log(0.5), np.log(0.25)]))

    def test_query_consistency_is_mse(self):
        assert query_consistency_loss([1.0, 3.0], [0.0, 1.0]) == pytest.approx(2.5)

    def test_graph_penalty_counts_unsupported(self):
        assert graph_penalty([True, False, False, True]) == pytest.approx(2.0)

    def test_total_loss_weighting(self):
        w = LossWeights(alpha=1.0, beta=0.2, kappa=0.5, xi=0.001)
        l = total_loss(1.0, 1.0, 1.0, 1.0, 1000.0, w)
        assert l == pytest.approx(1 + 1 + 0.2 + 0.5 + 1.0)

    def test_dual_lambda_moves_with_cost(self):
        dl = DualLambda(value=0.05, rho=0.1, bar_c=1.0)
        v1 = dl.update(1.5)                       # over budget -> lambda up
        assert v1 == pytest.approx(0.05 + 0.1 * 0.5)
        v2 = dl.update(0.0)                       # under budget -> lambda down
        assert v2 < v1
        dl2 = DualLambda(value=0.02, rho=0.5, bar_c=1.0)
        assert dl2.update(0.0) == 0.0             # floored at zero, never negative
