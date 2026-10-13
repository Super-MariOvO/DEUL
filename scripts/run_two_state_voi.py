import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runnable demo of the paper's Section 6 constructions (T1 and T2).

T1 (Section 6.1): two worlds have identical candidate marginals, but the same
free query is worth 0 in one and 0.5 in the other -- marginal value
distributions alone cannot determine the value of a query.

T2 (Section 6.2): in the XOR world, any single query is worthless, but the
complementary pair of queries has positive net value -- a depth-two query can
beat the myopic "query once then execute" rule.

Run:  python scripts/run_two_state_voi.py
"""

from deul.query import bernoulli_marginal_worlds, xor_world


def demo_t1() -> None:
    worlds = bernoulli_marginal_worlds()
    print("=" * 68)
    print("T1 — 相同边际效用，不保证相同的最优验证决策（paper §6.1）")
    print("=" * 68)
    print(f"{'世界':<10}{'联合结果':<18}{'未查询最优':<12}{'查询后最优':<12}{'免费查询价值':<12}")
    for name, label in [("positive", "正相关 (1,1)/(0,0)"),
                        ("negative", "负相关 (1,0)/(0,1)")]:
        w = worlds[name]
        unqueried, revealed = 0.5, w["reveal_value"](None)
        print(f"{label:<18}{str([o for o, _ in w['outcomes']]):<18}"
              f"{unqueried:<12.1f}{revealed:<12.1f}{revealed - unqueried:<12.1f}")
    cost = 0.1
    print(f"\n查询成本 = {cost} 时：正相关世界不查（0 - 0.1 < 0），"
          f"负相关世界值得查（0.5 - 0.1 > 0）。")
    print("=> 仅保留边际价值分布不足以决定查询价值；联合结构必须进入模型。\n")


def demo_t2() -> None:
    w = xor_world()
    print("=" * 68)
    print("T2 — 互补查询胜过短视单步 VOI（paper §6.2）")
    print("=" * 68)
    print("z1, z2 为独立公平二元变量；A 在 z1 XOR z2 = 0 时收益 1，B 相反。")
    print("q1、q2 各揭示一个变量，各成本 0.1。")
    print(f"  只查一次：两个候选期望仍各 0.5，单步 VOI 拒绝查询（价值 {w['myopic_value']}）")
    print(f"  连查两次：确定获胜候选，净收益 1 - 0.2 = {w['joint_value']}")
    print(f"=> 互补查询的净价值 {w['joint_value']} > 短视规则 {w['myopic_value']}；"
          f"单步 VOI 会错过互补证据。\n")


if __name__ == "__main__":
    demo_t1()
    demo_t2()
    print("两个构造均与论文 §6.1/§6.2 的表格数值一致；"
          "它们解释机制，不构成任何独立网络必然失败的保证。")
