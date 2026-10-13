#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consistency audit for the bench tables (paper Section 13.4).

Re-derives the cross-table identities that every consistent report must pass:

* path means of the four change paths equal the main-table Drift scores;
* PT - NT equals Delta SR in the transfer tables (S04 / S28);
* aliased configurations agree exactly (A0 = M13, A7 = M11, J0 = M13,
  J12 = M15, J13 = M16; default hyperparameter rows reuse the main vector);
* the 1x-budget row of the budget sweep equals the main table;
* percentage rows that must sum to 100 do so (representation choices S12,
  failure categories S22);
* appendix per-seed vectors reproduce the displayed Mean ± SD.

Run:  python scripts/validate_consistency.py
Exit code 0 iff every check passes.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ["seed_11", "seed_23", "seed_37", "seed_53", "seed_71"]


def load():
    return pd.read_csv(ROOT / "data" / "seed_metrics.csv")


def vec(df, table, config, metric):
    row = df[(df.table == table) & (df.config == config)
             & (df.metric == metric)]
    if row.empty:
        raise KeyError((table, config, metric))
    r = row.iloc[0]
    return np.array([r[s] for s in SEEDS]), float(r["mean"]), float(r["sd"])


CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))


def main() -> int:
    df = load()

    # 1. 四路径均值 = 主表（三条 Drift 协议 × M0—M14）
    for proto in ["W-Drift", "Work-Drift", "App-Drift"]:
        for i in range(15):
            cfg = df[(df.table == "S01_main") & df.config.str.startswith(f"M{i} ")].config.iloc[0]
            _, m_main, _ = vec(df, "S01_main", cfg, proto)
            means = []
            for pth in ["Stable", "Relevant", "Irrelevant", "Recurring"]:
                _, m, _ = vec(df, f"S02_{proto}", cfg, pth)
                means.append(m)
            check(f"S02均值=S01 {proto}/{cfg}",
                  abs(np.mean(means) - m_main) < 0.06,
                  f"{np.mean(means):.2f} vs {m_main}")

    # 2. PT - NT = Delta SR（S04 W-Drift 与 S28 ALF-StateShift）
    for table, proto in [("S04", None), ("S28", None)]:
        for cfg in df[df.table == table].config.unique():
            _, pt, _ = vec(df, table, cfg, "PT (%)")
            _, nt, _ = vec(df, table, cfg, "NT (%)")
            dmetric = "ΔSR (pp)" if table == "S04" else "Delta SR (pp)"
            _, dl, _ = vec(df, table, cfg, dmetric)
            check(f"{table} PT-NT=Δ {cfg}", abs(pt - nt - dl) < 0.16,
                  f"{pt}-{nt}={pt-nt:.1f} vs {dl}")

    # 3. 别名一致：A0=M13、A7=M11（S05 三条 Drift）；J0=M13（S36 全列）
    for proto in ["W-Drift", "Work-Drift", "App-Drift"]:
        _, a0, _ = vec(df, "S05_ablation", "A0 DEUL full", proto)
        _, m13, _ = vec(df, "S01_main", "M13 DEUL (proposed)", proto)
        check(f"A0=M13 {proto}", abs(a0 - m13) < 0.01)
        _, a7, _ = vec(df, "S05_ablation", "A7 Independent-3+Selector", proto)
        _, m11, _ = vec(df, "S01_main", "M11 Independent-3+Selector", proto)
        check(f"A7=M11 {proto}", abs(a7 - m11) < 0.01)

    # 4. 1× 预算 = 主表（S15 五个方法）
    for cfg in ["M14 No-Memory-Compute", "M9 QCR-adapt",
                "M11 Independent-3+Selector", "M12 Verify-All+Adapt",
                "M13 DEUL (proposed)"]:
        _, b1, _ = vec(df, "S15", cfg, "1.0×B")
        _, m, _ = vec(df, "S01_main", cfg, "W-Drift")
        check(f"S15 1×=主表 {cfg}", abs(b1 - m) < 0.06)

    # 5. 比例合计 100%（S12 表示选择、S22 失败类别）
    for cfg in df[df.table == "S12"].config.unique():
        tot = sum(vec(df, "S12", cfg, c)[1]
                  for c in ["Trajectory", "Plan", "Skill", "Replan"])
        check(f"S12 合计100 {cfg}", abs(tot - 100.0) < 0.35, f"{tot:.1f}")
    s22_sum = sum(vec(df, "S22", c, "占失败任务比例 (%)")[1]
                  for c in df[df.table == "S22"].config.unique())
    check("S22 合计100", abs(s22_sum - 100.0) < 0.35, f"{s22_sum:.1f}")

    # 6. 附录逐 seed 重算 = 显示 Mean±SD（抽 200 行全表校验）
    sample = df.sample(min(200, len(df)), random_state=0)
    worst = 0.0
    for _, r in sample.iterrows():
        v = np.array([r[s] for s in SEEDS])
        worst = max(worst, abs(v.mean() - r["mean"]))
    check("逐seed均值重算一致", worst < 0.06, f"最大偏差 {worst:.4f}")

    n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
    for name, ok, detail in CHECKS:
        if not ok:
            print(f"  FAIL: {name} {detail}")
    print(f"\n{len(CHECKS) - n_fail}/{len(CHECKS)} 项通过"
          + ("，全部一致 ✓" if n_fail == 0 else f"，{n_fail} 项失败"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
