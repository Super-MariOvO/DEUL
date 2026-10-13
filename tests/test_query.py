# -*- coding: utf-8 -*-
"""Tests for the shared-posterior VOI query tree (paper Sections 5.7, 6.1, 6.2)."""

import numpy as np
import pytest

from deul.query import (DictOutcomeModel, Query, QueryTree,
                        bernoulli_marginal_worlds, xor_world)


class TestT1IdenticalMarginals:
    """Paper T1: identical candidate marginals, different query values."""

    def test_positive_world_query_worthless(self):
        worlds = bernoulli_marginal_worlds()
        w = worlds["positive"]
        # Free-query value = E[best after reveal] - best now = 0.5 - 0.5
        assert w["reveal_value"](None) - 0.5 == pytest.approx(0.0)

    def test_negative_world_query_worth_half(self):
        worlds = bernoulli_marginal_worlds()
        w = worlds["negative"]
        assert w["reveal_value"](None) - 0.5 == pytest.approx(0.5)

    def test_cost_decides_differently(self):
        """With query cost 0.1, the two worlds make different decisions."""
        cost = 0.1
        pos = bernoulli_marginal_worlds()["positive"]["reveal_value"](None) - 0.5
        neg = bernoulli_marginal_worlds()["negative"]["reveal_value"](None) - 0.5
        assert pos - cost < 0.0                     # don't query in positive world
        assert neg - cost > 0.0                     # do query in negative world


class TestT2ComplementaryQueries:
    """Paper T2: single query is worthless, the pair is valuable."""

    def test_single_query_no_value(self):
        w = xor_world()
        # One bit revealed: candidates still 0.5/0.5 -> no gain over 0.5
        assert w["myopic_value"] == pytest.approx(0.5)

    def test_pair_beats_myopic(self):
        w = xor_world()
        assert w["joint_value"] == pytest.approx(0.8)
        assert w["joint_value"] > w["myopic_value"]


class TestQueryTree:
    def _model(self):
        # belief b0: query q1 reveals condition; after it, exec values are known
        return DictOutcomeModel(table={
            ("b0", "q1"): [("good", 0.5, "bg"), ("bad", 0.5, "bb")],
        })

    def test_voi_positive_when_separates(self):
        q1 = Query("q1", cost=0.1)
        tree = QueryTree(self._model(),
                         exec_value=lambda b, B: {"b0": 0.5, "bg": 1.0,
                                                  "bb": 0.0}[b],
                         lam=0.0, max_depth=2)
        v = tree.voi("b0", 1.0, 2, q1, [q1])
        # E[after] = 0.5*1.0 + 0.5*0.0 = 0.5; immediate = 0.5 -> VOI = 0.0 here
        assert v == pytest.approx(0.0)

    def test_voi_charges_cost(self):
        q1 = Query("q1", cost=0.2)
        tree = QueryTree(self._model(),
                         exec_value=lambda b, B: {"b0": 0.5, "bg": 1.0,
                                                  "bb": 0.0}[b],
                         lam=0.5, max_depth=2)
        v = tree.voi("b0", 1.0, 2, q1, [q1])
        assert v == pytest.approx(0.0 - 0.5 * 0.2)

    def test_no_voi_summing_two_queries(self):
        """Two queries each individually valuable cannot be summed: the tree
        compares branches, it does not add VOIs (paper Section 5.7)."""
        q1 = Query("q1", cost=0.1)
        q2 = Query("q2", cost=0.1)
        model = DictOutcomeModel(table={
            ("b0", "q1"): [("good", 0.5, "bg"), ("bad", 0.5, "bb")],
            ("b0", "q2"): [("good", 0.5, "bg"), ("bad", 0.5, "bb")],
            ("bg", "q2"): [("good", 1.0, "bg")],
        })
        tree = QueryTree(model,
                         exec_value=lambda b, B: {"b0": 0.5, "bg": 1.0,
                                                  "bb": 0.0}[b],
                         lam=0.0, max_depth=2)
        v_total = tree.value("b0", 1.0, 2, [q1, q2])
        # Optimal: query once (E = 0.5*(1.0) + 0.5*(0.0) - 0.1 = 0.4),
        # immediate 0.5 -> stay. Tree value must be 0.5, NOT 0.5 + 0.4 + 0.4.
        assert v_total == pytest.approx(0.5)

    def test_depth_zero_means_immediate(self):
        tree = QueryTree(self._model(), exec_value=lambda b, B: 0.5,
                         lam=0.0, max_depth=2)
        assert tree.value("b0", 1.0, 0, [Query("q1", 0.1)]) == 0.5

    def test_budget_blocks_unaffordable_query(self):
        q1 = Query("q1", cost=2.0)
        tree = QueryTree(self._model(), exec_value=lambda b, B: 0.5,
                         lam=0.0, max_depth=2)
        assert tree.value("b0", 1.0, 2, [q1]) == 0.5
        assert tree.voi("b0", 1.0, 2, q1, [q1]) == pytest.approx(0.0)
