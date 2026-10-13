#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate the four paper figures from ``data/seed_metrics.csv``.

Outputs (paper/figures/ and docs/images/):
    fig_main_results.{pdf,png}      - six-protocol SR for selected methods
    fig_drift_trajectory.{pdf,png}  - frozen-experience utility over 13 snapshots
    fig_recovery.{pdf,png}          - post-change recovery cohort
    fig_budget.{pdf,png}            - budget sweep on W-Drift

The script is deterministic: figures are recomputed from the seed vectors
(5 seeds, mean ± SD) rather than copied, so any table change propagates here.
"""

from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "seed_metrics.csv"
OUT_DIRS = [ROOT / "paper" / "figures", ROOT / "docs" / "images"]

SEEDS = ["seed_11", "seed_23", "seed_37", "seed_53", "seed_71"]
PROTO6 = ["W-Static", "W-Drift", "Work-Drift", "App-Drift",
          "ALF-Static", "ALF-StateShift"]


def load():
    df = pd.read_csv(DATA)
    return df


def vec(df, table, config_prefix, metric):
    row = df[(df.table == table) & (df.config == config_prefix)
             & (df.metric == metric)]
    if row.empty:
        raise KeyError((table, config_prefix, metric))
    r = row.iloc[0]
    return np.array([r[s] for s in SEEDS]), float(r["mean"]), float(r["sd"])


def fig_main(df, out_dirs):
    methods = [("M0 No Persistent Experience", "M0 NoExp"),
               ("M3 Skill-Semantic", "M3 Skill-Sem"),
               ("M9 QCR-adapt", "M9 QCR"),
               ("M11 Independent-3+Selector", "M11 Ind-3"),
               ("M12 Verify-All+Adapt", "M12 VerifyAll"),
               ("M15 Independent-3++ (strong)", "M15 Ind-3++"),
               ("M18 BOCPD + Correlated-KG", "M18 BOCPD-KG"),
               ("M13 DEUL full", "M13 DEUL")]
    # 列来源：M0—M10/M14 的四个 W 协议在 S01_main，两个 ALF 协议在 S27_nonweb_main；
    # M11—M20 六列全在 S35_strong（metric 为 "X SR (%)"）。
    def table_metric(cfg, proto):
        mid_num = int(cfg.split(" ", 1)[0][1:])
        if mid_num >= 11:
            return "S35_strong", f"{proto} SR (%)"
        if proto.startswith("ALF"):
            return "S27_nonweb_main", proto
        return "S01_main", proto

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    xw = np.arange(6)
    for i, (cfg, label) in enumerate(methods):
        mus, sds = [], []
        for proto in PROTO6:
            tb, metric = table_metric(cfg, proto)
            _, m, s = vec(df, tb, cfg, metric)
            mus.append(m)
            sds.append(s)
        ax.bar(xw + (i - len(methods) / 2) * 0.105, mus, width=0.1, yerr=sds,
               capsize=2, label=label, edgecolor="white", linewidth=0.4,
               error_kw=dict(lw=0.7, ecolor="gray"))
    ax.set_xticks(xw)
    ax.set_xticklabels(PROTO6, fontsize=8)
    ax.set_ylabel("Task SR (%)")
    ax.set_ylim(25, 85)
    ax.set_title("Main results across six protocols (mean ± SD, 5 seeds)")
    ax.legend(fontsize=7, ncol=4, frameon=False)
    fig.tight_layout()
    save(fig, "fig_main_results", out_dirs)


def fig_drift(df, out_dirs):
    snaps = [str(k) for k in range(13)]
    series = [("Trajectory", "-o", "Trajectory (relevant)"),
              ("Plan", "-s", "Plan (relevant)"),
              ("Skill", "-^", "Skill (relevant)"),
              ("Stable跨表示均值", "--", "Stable (mean)"),
              ("Irrelevant跨表示均值", "--", "Irrelevant (mean)"),
              ("Recurring跨表示均值", "--", "Recurring (mean)")]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xs = list(range(13))
    for metric, style, label in series:
        ys = [vec(df, "S10", k, metric)[1] for k in snaps]
        ax.plot(xs, ys, style, ms=3, lw=1.2, label=label)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("Environment snapshot")
    ax.set_ylabel("Utility of frozen experience (pp)")
    ax.set_title("Frozen-experience utility over 13 snapshots (RQ3)")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, "fig_drift_trajectory", out_dirs)


def fig_recovery(df, out_dirs):
    xs = [0, 1, 2, 4, 8, 12, 16]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for col in ["Skill-only", "QCR", "Independent-3", "Verify-All", "DEUL"]:
        ys = [vec(df, "S13", str(k), col)[1] for k in xs]
        ax.plot(xs, ys, "-o", ms=3.5, lw=1.3, label=col)
    ax.set_xlabel("Tasks completed after change")
    ax.set_ylabel("Anchored-task SR (%)")
    ax.set_title("Recovery after a relevant change")
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    save(fig, "fig_recovery", out_dirs)


def fig_budget(df, out_dirs):
    xs = [0.5, 1.0, 1.5, 2.0]
    cols = ["0.5×B", "1.0×B", "1.5×B", "2.0×B"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for cfg in ["M14 No-Memory-Compute", "M9 QCR-adapt",
                "M11 Independent-3+Selector", "M12 Verify-All+Adapt",
                "M13 DEUL (proposed)"]:
        ys = [vec(df, "S15", cfg, c)[1] for c in cols]
        ax.plot(xs, ys, "-o", ms=3.5, lw=1.3, label=cfg)
    ax.set_xlabel("Budget multiplier")
    ax.set_ylabel("W-Drift Task SR (%)")
    ax.set_title("Budget sweep on W-Drift")
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    save(fig, "fig_budget", out_dirs)


def save(fig, name, out_dirs):
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.pdf")
        fig.savefig(d / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  wrote {name}")


if __name__ == "__main__":
    df = load()
    print(f"loaded {len(df)} seed rows from {DATA}")
    fig_main(df, OUT_DIRS)
    fig_drift(df, OUT_DIRS)
    fig_recovery(df, OUT_DIRS)
    fig_budget(df, OUT_DIRS)
    print("done.")
