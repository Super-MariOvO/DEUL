# DEUL Code Supplement

Complete companion code for *When Experience Goes Stale: Drift-Aware Experience Utility Learning for Long-Horizon Agents*.

## Directory Structure

```
deul/
├── filter.py             # JointParticleFilter + EventLedger
│                         #   - stratified init: exact factorization of the joint prior
│                         #   - retain-or-reset hazard transition p = (1-h)·persist + h·p0
│                         #   - one likelihood update per event_id (duplicate invariance)
│                         #   - zero-likelihood fallback relaxes to uniform instead of collapsing
├── values.py             # Candidate / ValueModel / LinearValueModel
│                         #   - success head in [0,1], cost head non-negative (softplus)
│                         #   - posterior integration under the *same* joint particles
│                         #   - deduplicate() before any max (duplication cannot change selection)
│                         #   - incremental_utility: U(e,r) = Q^Y(reuse) - Q^Y(replan)
├── paired.py             # Paired-advantage GLS
│                         #   - Delta_{d,a} = bar Y_{d,a} - bar Y_{d,0}
│                         #   - shared_baseline_covariance(): empirical Cov(Delta_a,Delta_b)
│                         #     reproduced against the exact identity Var(bar Y_0)
│                         #   - shrunk_covariance(): diag + (1-shrink)·(sigma0^2/m0)·11^T
│                         #     with positive-eigenvalue floor
│                         #   - gls_loss / gls_gradient
├── query.py              # QueryTree over an abstract OutcomeModel
│                         #   - V_d(b,B) recursion, depth 2, ≤6 queries/node, ≤3 outcomes/query
│                         #   - voi() = E_o V_{d-1} - V_0 - lambda·c_q
│                         #   - myopic_voi() for the depth-1 comparator
│                         #   - bernoulli_marginal_worlds() / xor_world(): exact T1/T2 data
├── controller.py         # OnlineController (Algorithm 2)
│                         #   - build_candidates() = retrieve + dedup
│                         #   - decide(): best candidate vs. positive-VOI query, hard budgets
│                         #   - DecisionTrace charges tokens and env primitives separately
├── objectives.py         # td_loss (one shared continuation for success & cost),
│                         # obs_loss (unique-event NLL), query_consistency_loss,
│                         # graph_penalty, total_loss, DualLambda
└── training/             # Integration seam — intentionally empty (see below)
scripts/
├── build_bench_data.py   # Deterministic scenario generator (SHA-256 seeded;
│                         #   pinned centers + documented perturbations; regenerates
│                         #   every S01–S45 table and the event-level CSVs)
├── validate_consistency.py # 92 checks: path means = main, PT-NT = Delta,
│                         #   aliases (A0=M13, A7=M11, J0=M13), 1x budget = main,
│                         #   sums-to-100 rows, per-seed recomputation of Mean±SD
├── run_two_state_voi.py  # Prints the T1/T2 constructions
└── make_figures.py       # Recomputes the 4 figures from data/seed_metrics.csv
tests/                    # 35 tests (see below)
paper/                    # main_en.tex (submission), main_zh.tex (reading copy), latexmkrc
data/                     # seed_metrics.csv, *_synthetic_events.csv.gz, ledgers (JSON)
docs/images/              # README/manuscript figures
```

## Key Implementation Details

- **Filter (`filter.py`)**: particles are full joint assignments initialized by tiling the cartesian product of condition values, so the empirical prior factorizes *exactly*; this is what makes the local-update factorization property (Section 6.4) hold in the unit tests, not just in expectation. The hazard model (`StaticHazard`) uses only visible cues (elapsed time, residuals), never evaluator labels.
- **Duplicate invariance**: `EventLedger` records processed `event_id`s; `update()` returns `False` on duplicates and leaves posterior and policy distribution bitwise unchanged (tested via `state_signature()`).
- **GLS covariance (`paired.py`)**: `shrunk_covariance` implements `R_d = diag(σ_a²/m_a) + (1-shrink)·(σ_0²/m_0)·11ᵀ` and re-projects with an eigenvalue floor; `shared_baseline_covariance` is validated against the closed-form identity `Cov(Δa, Δb) = Var(Ȳ₀)` using 20,000 simulated repetitions of the paired design.
- **Query tree (`query.py`)**: branch values are expectations over *shared* outcomes; the test `test_no_voi_summing_two_queries` proves the tree returns a max, never a sum of per-query VOIs.
- **Consistency audit (`validate_consistency.py`)**: all 92 cross-table identities from Section 13.4 of the manuscript pass on the current `data/` snapshot.

## Running Tests

```bash
pip install -e . && pip install pytest
python -m pytest tests/ -v
```

35 tests covering: event-ledger uniqueness, duplicate invariance, hazard reset behavior, local-update factorization, zero-likelihood relaxation, advantage definitions, the shared-baseline covariance identity, covariance shrinkage properties, GLS loss/gradient, T1 marginal-identical query values, T2 complementary queries, no-VOI-summing, budget guards, query-only-if-VOI-positive decisions, TD/obs/query/graph losses, total-loss weighting, and the dual lambda update.

## Known Limitations

This is an **algorithm-level component library plus a documented scenario pipeline**, not an end-to-end trained system. The following integration components are NOT included:

- Environment adapters that turn WebArena / WorkArena++ / AppWorld / ALFWorld responses into typed observations and conditions (the `deul/training/` seam);
- The neural encoder for source-conditioned value heads (the library ships a linear `ValueModel` interface instead);
- Distributed rollout / vLLM serving and the frozen-continuation policy trainer;
- Real environment logs, trained weights, or GPU/compute ledgers.

**Data status**: the S01–S45 protocol numbers in the current manuscript are *documented scenario data* from `scripts/build_bench_data.py` (pinned centers, documented perturbations, SHA-256-derived seeds), not real benchmark executions. They exist to pin down the evaluation protocol, the required evidence shape, and the consistency/audit machinery. Before any empirical submission, every table must be recomputed from real environment logs and the consistency audit rerun against them.
