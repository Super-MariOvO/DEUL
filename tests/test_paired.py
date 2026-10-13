# -*- coding: utf-8 -*-
"""Tests for paired-advantage GLS supervision (paper Sections 5.5, 6.3)."""

import numpy as np
import pytest

from deul.paired import (gls_gradient, gls_loss, paired_means_to_advantages,
                         shared_baseline_covariance, shrunk_covariance)


class TestAdvantages:
    def test_delta_definition(self):
        out = {0: [1, 0, 1, 0], "reuse": [1, 1, 0, 1], "adapt": [0, 0, 0, 0]}
        d = paired_means_to_advantages(out)
        assert d[0] == pytest.approx(0.0)
        assert d["reuse"] == pytest.approx(0.75 - 0.5)
        assert d["adapt"] == pytest.approx(0.0 - 0.5)

    def test_missing_baseline_raises(self):
        with pytest.raises(KeyError):
            paired_means_to_advantages({"reuse": [1, 1]})


class TestSharedBaselineCovariance:
    def test_covariance_identity(self):
        """Cov(Delta_a, Delta_b) = Var(bar Y_0) for a != b (paper Section 6.3).

        Simulate many repetitions of the paired design with independent branch
        noise and check that off-diagonal empirical covariance converges to the
        variance of the shared replanning mean.
        """
        rng = np.random.default_rng(0)
        p0, pa, pb, m = 0.5, 0.7, 0.3, 4
        reps = {}
        for r in range(20000):
            reps[r] = {
                0: rng.random(m) < p0,
                "a": rng.random(m) < pa,
                "b": rng.random(m) < pb,
            }
        C = shared_baseline_covariance(reps)
        var_y0 = p0 * (1 - p0) / m                       # Var(bar Y_0) exact
        # C is 2x2 over (Delta_a, Delta_b); off-diagonal must equal Var(bar Y_0)
        assert C[0, 1] == pytest.approx(var_y0, abs=0.002)
        assert C[1, 0] == pytest.approx(var_y0, abs=0.002)
        # Diagonals: Var(bar Y_0) + Var(bar Y_a)/m_a（抽样噪声更大，容差放宽）
        assert C[0, 0] == pytest.approx(var_y0 + pa * (1 - pa) / m, abs=0.01)
        assert C[1, 1] == pytest.approx(var_y0 + pb * (1 - pb) / m, abs=0.01)

    def test_shrunk_covariance_is_pd_with_floor(self):
        R = shrunk_covariance({"a": 0.25, "b": 0.25}, baseline_variance=0.25,
                              repeats={"a": 2, "b": 2}, baseline_repeats=2,
                              shrink=0.0, eig_floor=1e-3)
        w = np.linalg.eigvalsh(R)
        assert np.all(w >= 1e-3 - 1e-12)
        # Structure: diag(sigma_a^2/m_a) + base * 11^T
        base = 0.25 / 2
        expected = np.array([[0.125 + base, base], [base, 0.125 + base]])
        assert np.allclose(R, expected, atol=1e-12)


class TestGLS:
    def test_loss_zero_at_target(self):
        R = np.eye(3)
        assert gls_loss([1, 2, 3], [1, 2, 3], R) == pytest.approx(0.0)

    def test_loss_matches_manual(self):
        R = np.diag([1.0, 2.0, 4.0])
        val = gls_loss([1, 0, 0], [0, 0, 0], R)   # = (1/1 + 0 + 0)/3
        assert val == pytest.approx(1.0 / 3.0)

    def test_gradient_points_uphill(self):
        R = np.eye(2)
        g = gls_gradient([1.0, 0.0], [0.0, 0.0], R)
        assert np.allclose(g, [1.0, 0.0])          # 2*R^{-1}(u-d)/|A|

    def test_correlated_R_outperforms_scalar_when_aligned(self):
        """GLS with the correct R scores the *aligned* advantage vector better
        than a scalar MSE when the shift is along the common-baseline term."""
        base = np.full((3, 3), 0.1) + np.eye(3) * 0.05
        delta = np.zeros(3)
        u_aligned = np.ones(3) * 0.1               # shift along 11^T direction
        u_orth = np.array([0.1, -0.1, 0.0])
        l_aligned = gls_loss(u_aligned, delta, base)
        l_orth = gls_loss(u_orth, delta, base)
        assert l_aligned < l_orth                   # common-mode is cheaper
