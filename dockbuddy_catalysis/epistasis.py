# -*- coding: utf-8 -*-
"""libdock.epistasis —— 双突变上位性(epistasis)统计(纯计算, numpy 仅类型用)。

================================================================================
概念与口径:
- 给定若干单点突变的 ΔΔG 与若干双突变的 ΔΔG, 上位性 ε 量化"双突变效应是否等于
  各单点效应之和":
      ε = ΔΔG_observed(double) − ΔΔG_expected(double)
  其中 ΔΔG_expected 由单点加和(加性模型)或乘积(乘性模型, 用于概率/比例量)估计。
- 符号约定(与本仓 FoldX 一致): ΔΔG > 0 = 不稳定(destabilizing), ΔΔG < 0 =
  稳定(stabilizing)。故:
      ε > 0  (观测比加性预期更不稳定) → 协同(synergistic) destabilization;
      ε < 0  (观测比加性预期更稳定)   → 拮抗(antagonistic);
      |ε| ≤ threshold                  → 近加性(neutral / 无显著上位性)。
- 显著性: 仅当提供了单点/双点的 ddG_std(重复或系综)时才能做; 否则 z=None,
  显著与否**未知**, 不得臆断。

诚实声明: 上位性结论高度依赖 ΔΔG 估计质量。FoldX 单 run 噪声地板约 ±(数~十余)
kcal/mol, 在此量级上"加性 vs 协同"极易被噪声淹没 —— 本模块只做算术与分类,
  不替用户宣称上位性存在; verdict 强制此免责。
================================================================================
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from dockbuddy_dock._core import LOG

# 双突变字符串中组件分隔符(接受 '/' 与 ',')
_COMPONENT_SEP = re.compile(r"[/,]")

# 单点突变 token: 可选链前缀 + 数字残基号 + 1~3 字母目标残基(兼容 'F205G' 与
# 'FA205G'; 链/源残基歧义由调用方保证 singles 与 doubles 用同一记法, 本模块只做
# 一致性归一化而非语义解析)。
_SINGLE_MUT_RE = re.compile(r"([A-Z]?)(\d+)([A-Z]{1,3})", re.IGNORECASE)


def _normalize_label(label: str) -> str:
    """把 'FA205G' / 'A205G' / 'F205G' 归一为 '<链?>205G' 小写形式便于匹配。"""
    m = _SINGLE_MUT_RE.search(label)
    if not m:
        return label.strip().upper()
    chain, num, res = m.group(1), m.group(2), m.group(3).upper()
    return f"{chain}{num}{res}" if chain else f"{num}{res}"


def _parse_components(mutation_str: str) -> List[str]:
    """从双突变字符串拆出组件单点标签列表(归一化)。"""
    parts = [p for p in _COMPONENT_SEP.split(mutation_str) if p.strip()]
    return [_normalize_label(p) for p in parts]


def _expected(single_vals: List[float], mode: str) -> Optional[float]:
    """加性: Σ; 乘性: Π (对 ΔΔG 这种差值量不常规, 仅当用户显式选 'multiplicative')。"""
    if not single_vals:
        return None
    if mode == "multiplicative":
        prod = 1.0
        for v in single_vals:
            prod *= (v + 1.0)   # 防止 0 相乘丢失符号信息(保守映射)
        return prod - 1.0
    # 默认 additive
    return float(sum(single_vals))


def epistasis(singles: Dict[str, float],
              doubles: Dict[str, Dict[str, Any]], *,
              mode: str = "additive",
              threshold: float = 0.5,
              single_std: Optional[Dict[str, float]] = None,
              double_std: Optional[Dict[str, float]] = None,
              confidence: float = 1.96,
              label_key: str = "mutation") -> Dict[str, Any]:
    """计算双突变上位性。

    参数:
      singles    {label: ddG}  —— 单点突变 ΔΔG(归一化标签为键)
      doubles    {label: {"mutation": "F205G/Y209A", "ddG": x, ...}} 或
                 {label: {"ddG": x}} 且 label 本身含组件(如 'F205G/Y209A')
      mode       'additive' (默认) | 'multiplicative'
      threshold  近加性判定阈值 kcal/mol(默认 0.5)
      single_std / double_std  各突变 ΔΔG 标准差(用于显著性; 缺则 z=None)
      confidence z 判定临界倍率(默认 1.96 ≈ 95%)
      label_key  doubles 值中取组件字符串的键(默认 'mutation')

    返回: {ok, mode, threshold_kcal, items[], summary{}, verdict}
    """
    if mode not in ("additive", "multiplicative"):
        return {"ok": False, "error": f"未知 mode: {mode}", "items": [],
                "verdict": _verdict([], reason="未知 mode")}

    norm_singles = {_normalize_label(k): v for k, v in (singles or {}).items()}
    norm_single_std = {_normalize_label(k): v for k, v in (single_std or {}).items()}

    items: List[Dict[str, Any]] = []
    for dlabel, dblob in (doubles or {}).items():
        mut_str = dblob.get(label_key) or dlabel
        components = _parse_components(mut_str)
        # 取各组件单点 ddG
        comp_vals = [(c, norm_singles.get(c)) for c in components]
        missing = [c for c, v in comp_vals if v is None]
        if missing:
            items.append({
                "label": dlabel, "mutation": mut_str, "components": components,
                "observed": dblob.get("ddG"),
                "expected": None, "epistasis": None,
                "classification": "indeterminate",
                "missing_components": missing, "z": None, "significant": None,
                "note": "缺少组件单点 ΔΔG: " + ", ".join(missing),
            })
            continue
        obs = dblob.get("ddG")
        exp = _expected([v for _, v in comp_vals], mode)
        if obs is None or exp is None:
            classification = "indeterminate"
            eps = None
        else:
            eps = float(obs - exp)
            classification = ("synergistic" if eps > threshold
                             else "antagonistic" if eps < -threshold
                             else "neutral")
        # 显著性: 需要误差
        z = None
        significant = None
        if (norm_single_std or double_std) and obs is not None and exp is not None:
            dstd = (double_std or {}).get(dlabel)
            sstd = 0.0
            if norm_single_std:
                sstd = float(sum(norm_single_std.get(c, 0.0) ** 2 for c, _ in comp_vals)) ** 0.5
            if dstd is not None and (dstd ** 2 + sstd ** 2) > 0:
                z = eps / (dstd ** 2 + sstd ** 2) ** 0.5
                significant = abs(z) >= confidence
        items.append({
            "label": dlabel, "mutation": mut_str, "components": components,
            "observed": obs, "expected": exp, "epistasis": (round(eps, 4)
                                                           if eps is not None else None),
            "classification": classification, "z": (round(z, 3)
                                                   if z is not None else None),
            "significant": significant, "missing_components": [],
        })

    classified = [it for it in items if it["classification"] != "indeterminate"]
    n_syn = sum(1 for it in classified if it["classification"] == "synergistic")
    n_ant = sum(1 for it in classified if it["classification"] == "antagonistic")
    n_neu = sum(1 for it in classified if it["classification"] == "neutral")
    n_sig = sum(1 for it in items if it.get("significant") is True)
    summary = {
        "n_doubles": len(items),
        "n_classified": len(classified),
        "n_synergistic": n_syn, "n_antagonistic": n_ant, "n_neutral": n_neu,
        "n_significant": n_sig,
        "note": ("含误差数据时可判显著; 无误差时显著性一律未知, 不可宣称。"),
    }
    return {
        "ok": True, "mode": mode, "threshold_kcal": threshold,
        "n_singles": len(norm_singles), "items": items, "summary": summary,
        "verdict": _verdict(items, mode=mode, threshold=threshold),
    }


def _verdict(items: List[Dict[str, Any]], mode: str = "additive",
             threshold: float = 0.5, reason: str = "") -> Dict[str, Any]:
    n_class = sum(1 for it in items if it["classification"] != "indeterminate")
    n_sig = sum(1 for it in items if it.get("significant") is True)
    narrative = (
        f"双突变上位性(基于 {mode} 模型, 阈值 {threshold} kcal/mol): 对 "
        f"{n_class} 个可分类双突变做了 ε=观测−加性预期 分解。上位性**仅在 ΔΔG "
        f"估计误差远小于 |ε| 时成立**; 当前有 {n_sig} 个通过误差显著性检验, 其余"
        f"显著性未知。ε 符号: >0 协同(更不稳定)/ <0 拮抗/ |ε|≤阈值 近加性。"
        f"{('注意: ' + reason) if reason else ''} 结论须以湿实验(多点突变体热力学"
        f"循环)收口, 不得由 FoldX 单 run 直接宣称。")
    return {
        "ok": n_class > 0,
        "applicable": True,
        "narrative": narrative,
        "banned_phrases": [
            "FoldX 单 run 已证实上位性",
            "ε≠0 即存在真实上位性",
            "双突变协同 = 功能增益",
        ],
        "allowed_phrases": [
            "上位性需在 ΔΔG 误差远小于 |ε| 时成立",
            "以多点突变体热力学循环实验独立验证",
        ],
    }


__all__ = ["epistasis", "_normalize_label", "_parse_components"]
