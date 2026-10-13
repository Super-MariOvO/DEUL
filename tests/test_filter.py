# -*- coding: utf-8 -*-
"""Tests for the shared dependency-belief filter (paper Sections 5.3, 6.3, 6.4)."""

import numpy as np
import pytest

from deul.filter import Event, JointParticleFilter, StaticHazard


def make_filter(n_particles=32, seed=0):
    conds = ["layout", "contract"]
    values = {"layout": ["old", "new"], "contract": ["v1", "v2"]}
    return JointParticleFilter(conds, values, n_particles=n_particles,
                               hazard_model=StaticHazard(base=0.0),
                               rng=np.random.default_rng(seed))


def lk(v, o):
    return 0.9 if v == o else 0.1


class TestEventLedger:
    def test_duplicate_event_updates_once(self):
        f = make_filter()
        e = Event(event_id="u1", condition="layout", observation="new", timestamp=1.0)
        before = f.state_signature()
        assert f.update(e, lk) is True
        mid = f.state_signature()
        assert f.update(e, lk) is False           # duplicate: no second update
        assert f.state_signature() == mid          # posterior unchanged
        assert before != mid                       # but first update did happen
        assert len(f.ledger) == 1                  # one unique event

    def test_same_content_different_id_still_counts(self):
        """Two independent calls returning identical content are two measurements."""
        f = make_filter()
        e1 = Event("u1", "layout", "new", 1.0)
        e2 = Event("u2", "layout", "new", 2.0)
        assert f.update(e1, lk) is True
        assert f.update(e2, lk) is True
        assert len(f.ledger) == 2

    def test_marginal_after_evidence(self):
        f = make_filter(n_particles=2000, seed=1)
        for _ in range(5):
            f.update(Event(f"u{_}", "layout", "new", float(_)), lk)
        p_new = f.probability("layout", "new")
        assert p_new > 0.9                      # strong evidence dominates
        assert abs(f.probability("layout", "old") + p_new - 1.0) < 1e-9

    def test_zero_likelihood_relaxes_instead_of_collapse(self):
        f = make_filter()
        f.update(Event("u1", "layout", "new", 1.0), lk)
        # An observation every particle considers impossible:
        f.update(Event("u2", "layout", "old", 2.0), lambda v, o: 0.0)
        w = f.weights
        assert np.allclose(w, 1.0 / f.n_particles)   # relaxed to uniform

    def test_local_update_factorization(self):
        """Independent condition groups: an event on one does not move the other
        (paper Section 6.4 局部证据的影响边界)."""
        f = make_filter()
        before = f.marginal("contract")
        f.update(Event("u1", "layout", "new", 1.0), lk)
        after = f.marginal("contract")
        for v in before:
            assert after[v] == pytest.approx(before[v], abs=1e-12)

    def test_hazard_reset_dilutes_belief(self):
        f = JointParticleFilter(["layout"], {"layout": ["old", "new"]},
                                n_particles=32,
                                hazard_model=StaticHazard(base=1.0),  # always reset
                                rng=np.random.default_rng(0))
        f.update(Event("u1", "layout", "new", 1.0), lk)
        f.predict(dt=1.0)
        # After a full reset the posterior is the reset prior (uniform here),
        # not the evidence-shaped belief.
        p_new = f.probability("layout", "new")
        assert 0.0 < p_new < 1.0


class TestPredictTransition:
    def test_predict_with_zero_hazard_keeps_particles(self):
        f = make_filter()
        before = f.particles.copy()
        f.predict(dt=1.0)                       # base=0.0 hazard
        assert np.array_equal(f.particles, before)
