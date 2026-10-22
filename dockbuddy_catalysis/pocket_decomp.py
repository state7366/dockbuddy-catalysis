# -*- coding: utf-8 -*-
"""pocket_decomp.py —— 空腔增益的**逐残基分解**(非 MD 的机理性计算, 2026-10-05)
====================================================================================
动机
----
设计源头案例(Phe→Gly 单残基突变)的正式口径结论是「催化腔 +32.8 Å³ / +23.2%
(Ser OG 球心 R=8 Å, 两侧均松弛)」。
但 Phe→Gly 腾出的侧链体积在 **100~135 Å³ 量级**(数值随是否计入 CB 而变), 实测却只
**净增 32.8 Å³** —— 大约四分之三被邻域回填。谁回填的、回填多少, 在此之前完全没量化,
于是「+23%」只能讲成"少了个苯环", 讲不成"结构重排"。

本模块把差值拆到残基: 球内"被原子占据"的格点按**最贴近的那个原子**归属到残基, 于是

    ΔV_free = V_free(mut) − V_free(wt)
            = ΔN_grid·step³  +  Σ_r [V_occ_wt(r) − V_occ_mut(r)]
              └ 网格离散项      └ 残基 r 的贡献: **正 = 腾出, 负 = 回填**
              (显式报告, 别当误差吃掉)

口径红线(与 pocket_tunnel 必须同源, 否则两个数字不能并列)
----------------------------------------------------------
- 球心 = **催化亲核原子 Ser OG / Cys SG**, 调 ``pocket_tunnel.nucleophile_seed``
  (**唯一来源**, 本模块不得自己推); 不是三联体 Cα 质心 —— 那是 ``pocket_volume_qc``
  的 mutate QC 口径, 两口径的数字**禁止相减**。
- R / step / probe / vdW 表全部取 pocket_tunnel / pocketgeom 的同名常量。
- 两侧**各用自己的球心**(局部同口径, 与 pocket_tunnel 的 seed_A 同理)。球心漂移
  (``seed_gap_A``) 超过 ``CROSS_SEED_WARN_A`` 时告警: 漂移会把"参照系位移"混进
  "腾出/回填"里。

判读注意(已写进产物, 防下游误用)
--------------------------------
- 单对结构的**几何**分解, 不是 MD 系综平均; 回填量依赖"局部最小化路径", 需
  ≥2 个独立松弛模型重复才可写成性质(与喉部那条 ``replicate_required`` 同理)。
- contribution 的**符号**(腾出/回填)可靠; 单个残基的 Å³ 值只做排序与量级讨论,
  不要报"某残基精确回填 X.X Å³" —— 分辨率受归属规则影响(格点归给最贴近原子)。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dockbuddy_dock._core import LOG
from dockbuddy_dock.pocketgeom import POCKET_RADIUS_A, _parse_pdb_heavy
from .pocket_tunnel import (PROBE, SOLVENT, STEP, nucleophile_seed,
                            mutation_sites_from_pdbs, resolve_receptor)

# 球心漂移告警阈值(Å): mutate 产物骨架基本不动, 正常 <<1 Å
CROSS_SEED_WARN_A = 1.0
# 自洽残差容差(Å³): 理论应为 0, 放宽到浮点/离散噪声量级
SELF_CONSISTENCY_TOL_A3 = 0.5
# 默认只回报 |贡献| 达到这个体积的残基(≈2 个格点)
MIN_REPORT_A3 = 0.5


def _sphere_points(center: np.ndarray, radius: float, step: float) -> np.ndarray:
    """半径 radius 的球内均匀格点(相对 center 的绝对坐标)。"""
    n = int(radius / step)
    axes = np.arange(-n, n + 1, dtype=float) * step
    gx, gy, gz = np.meshgrid(axes, axes, axes, indexing="ij")
    m = (gx ** 2 + gy ** 2 + gz ** 2) <= radius * radius
    return np.stack([gx[m], gy[m], gz[m]], axis=1) + np.asarray(center, float)


def occupancy_by_residue(pdb_path: str, center: np.ndarray,
                         radius: float = POCKET_RADIUS_A,
                         step: float = STEP, probe: float = PROBE,
                         chain: str = "", block: int = 256) -> Dict[str, Any]:
    """单结构: 球内格点的**逐残基占用体积**分解。

    占用判据与 pocket_tunnel 完全一致: ``min_i(|p − atom_i| − vdw_i) <= probe``
    即该格点水探针进不去 ⇒ 占用, 归给使该值最小的那个原子所属残基。
    """
    atoms = _parse_pdb_heavy(pdb_path)
    if chain:
        atoms = [a for a in atoms if a[0] == chain]
    if not atoms:
        return {"error": f"{pdb_path} 解析到 0 个重原子(检查 --chain 与格式收敛)"}
    # 预筛: 只有离球心够近的原子才可能占据域内格点(+12 Å 覆盖 vdW+probe 上界)
    keep = [a for a in atoms
            if np.linalg.norm(np.array([a[4], a[5], a[6]], float) - np.asarray(center, float))
            <= radius + 12.0]
    if not keep:
        return {"error": "预筛后 0 个原子(球心远离蛋白?)"}
    xyz = np.array([[a[4], a[5], a[6]] for a in keep], dtype=float)
    vdw = np.array([float(a[7]) for a in keep], dtype=float)
    # label = 残基号+链(**不带残基名**): WT 侧 PHE101 与突变体侧 GLY101 是同一个
    # 位点, 带名字会把一个残基拆成"腾出+回填"两行, 直接误导归因。
    labels = [f"{a[1]}{a[0]}" for a in keep]
    resnames = [str(a[3]).upper() for a in keep]

    pts = _sphere_points(center, radius, step)
    owner = np.full(len(pts), -1, dtype=np.int64)
    for i in range(0, len(pts), block):
        blk = pts[i:i + block]
        d = np.linalg.norm(blk[:, None, :] - xyz[None, :, :], axis=2) - vdw[None, :]
        jmin = np.argmin(d, axis=1)
        dmin = d[np.arange(len(blk)), jmin]
        owner[i:i + block] = np.where(dmin > probe, -1, jmin)

    occ_idx = owner[owner >= 0]
    counts = np.bincount(occ_idx, minlength=len(labels))
    cell = float(step) ** 3
    # 🔴 同一残基有多个原子 ⇒ 索引→label 是**多对一**, 必须累加而不是 dict 覆盖
    # (2026-10-05 自洽校验抓到: 覆盖写法让 Σ逐残基 = 103 Å³ 而实测占用 = 826 Å³,
    #  差了 8 倍, 而"分解和 vs 腔差"只差 82 Å³ —— 靠肉眼根本看不出来)。
    by_res: Dict[str, float] = {}
    for i in range(len(labels)):
        if counts[i]:
            by_res[labels[i]] = by_res.get(labels[i], 0.0) + float(counts[i]) * cell
    return {
        "n_grid": int(len(pts)),
        "sphere_grid_volume_A3": round(float(len(pts)) * cell, 2),
        "occupied_A3": round(float(len(occ_idx)) * cell, 2),
        "free_A3": round(float(len(pts) - len(occ_idx)) * cell, 2),
        "by_residue_A3": by_res,
        "resname_by_label": {labels[i]: resnames[i]
                             for i in range(len(labels)) if counts[i]},
    }


def _seed_of(pdb_path: str, chain: str = "") -> Tuple[Optional[np.ndarray], str]:
    """球心(唯一来源 nucleophile_seed) + 来源串。"""
    atoms = _parse_pdb_heavy(pdb_path)
    if chain:
        atoms = [a for a in atoms if a[0] == chain]
    if not atoms:
        return None, f"0 重原子(chain={chain or '全部'})"
    sel_xyz = np.array([[a[4], a[5], a[6]] for a in atoms
                        if str(a[3]).upper() not in SOLVENT], dtype=float)
    cen = (sel_xyz.mean(axis=0) if len(sel_xyz)
           else np.array([[a[4], a[5], a[6]] for a in atoms], dtype=float).mean(axis=0))
    xyz, src = nucleophile_seed(pdb_path, atoms, fallback_centroid=cen, chain=chain)
    return np.asarray(xyz, dtype=float), src


def cavity_residue_decomposition(wt_pdb: str, mut_pdb: str, chain: str = "",
                                 radius: float = POCKET_RADIUS_A,
                                 step: float = STEP, probe: float = PROBE,
                                 min_report_A3: float = MIN_REPORT_A3,
                                 top_n: int = 8) -> Dict[str, Any]:
    """WT vs 突变体的空腔增益逐残基分解。

    返回字段见模块 docstring; 关键几项: ``cavity_delta_A3``(与 pocket_tunnel 同源
    的自由体积差)、``per_residue``(按 contribution 降序)、``sum_contributions_A3``、
    ``grid_count_delta_A3``、``residual_A3`` / ``self_consistent``(自洽校验)。
    """
    wt_c = resolve_receptor(wt_pdb)
    mut_c = resolve_receptor(mut_pdb)

    sw, src_w = _seed_of(wt_c, chain)
    sm, src_m = _seed_of(mut_c, chain)
    if sw is None or sm is None:
        return {"error": "球心不可定位(任一侧 0 重原子)",
                "wt_source": src_w, "mut_source": src_m}

    ow = occupancy_by_residue(wt_c, sw, radius, step, probe, chain)
    om = occupancy_by_residue(mut_c, sm, radius, step, probe, chain)
    if ow.get("error") or om.get("error"):
        return {"error": ow.get("error") or om.get("error")}

    seed_gap = float(np.linalg.norm(sw - sm))
    cell = float(step) ** 3
    grid_count_delta = (int(om["n_grid"]) - int(ow["n_grid"])) * cell
    delta_cav = round(float(om["free_A3"]) - float(ow["free_A3"]), 2)

    labels = sorted(set(ow["by_residue_A3"]) | set(om["by_residue_A3"]))
    rn_w = ow.get("resname_by_label", {})
    rn_m = om.get("resname_by_label", {})
    per_res: List[Dict[str, Any]] = []
    sum_contrib = 0.0
    for lab in labels:
        vw = float(ow["by_residue_A3"].get(lab, 0.0))
        vm = float(om["by_residue_A3"].get(lab, 0.0))
        contrib = vw - vm                      # 正 = 腾出(占用减少), 负 = 回填
        sum_contrib += contrib
        per_res.append({"label": lab,
                        "resname_wt": rn_w.get(lab, ""), "resname_mut": rn_m.get(lab, ""),
                        "occupancy_wt_A3": round(vw, 2),
                        "occupancy_mut_A3": round(vm, 2),
                        "delta_occupancy_A3": round(vm - vw, 2),
                        "contribution_A3": round(contrib, 2)})
    per_res.sort(key=lambda d: -d["contribution_A3"])
    reported = [d for d in per_res if abs(d["contribution_A3"]) >= float(min_report_A3)]

    residual = delta_cav - round(sum_contrib, 2) - round(grid_count_delta, 2)
    self_ok = bool(abs(residual) <= SELF_CONSISTENCY_TOL_A3)

    mut_sites = mutation_sites_from_pdbs(wt_c, mut_c, chain)

    def _norm_site(s: str) -> str:
        s = str(s).upper()
        i = 0
        while i < len(s) and s[i].isalpha():
            i += 1
        return s[i:]                          # "PHE101A" -> "101A"

    mut_set = {_norm_site(x) for x in mut_sites}

    def _is_mut(lab: str) -> bool:
        return str(lab).upper() in mut_set

    released = [d for d in reported if d["contribution_A3"] > 0]
    backfill = [d for d in reported if d["contribution_A3"] < 0][::-1][:int(top_n)]
    mut_rows = [d for d in per_res if _is_mut(d["label"])]
    released_total = round(sum(d["contribution_A3"] for d in per_res
                               if d["contribution_A3"] > 0), 2)
    backfill_total = round(sum(d["contribution_A3"] for d in per_res
                               if d["contribution_A3"] < 0), 2)

    banned: List[str] = []
    allowed: List[str] = []
    if not self_ok:
        banned.append(f"本分解的逐残基数值(自洽残差 {residual:.2f} Å³ 超容差 "
                      f"{SELF_CONSISTENCY_TOL_A3} Å³, 分解与实测腔差对不上)")
    if seed_gap > CROSS_SEED_WARN_A:
        banned.append(f"腾出/回填的定点归因(两侧球心相距 {seed_gap:.2f} Å > "
                      f"{CROSS_SEED_WARN_A} Å, 分解里混有参照系位移)")
    allowed.append("空腔净增益的**符号与量级**(须与 pocket_tunnel 同源球心)")
    allowed.append("腾出/回填的**排序**(哪些残基贡献最大), 而非单点精确 Å³")
    banned.append("单对结构当性质: 回填量依赖局部最小化路径, 需 ≥2 个独立松弛模型重复")

    mut_desc = ", ".join(f"{d['label']}({d['resname_wt'] or '?'}→{d['resname_mut'] or '?'})"
                         f" {d['contribution_A3']:+.1f}" for d in mut_rows) or "未定位"
    narrative = (f"空腔净增 {delta_cav:+.1f} Å³ = 腾出合计 {released_total:+.1f} Å³ "
                 f"+ 回填合计 {backfill_total:+.1f} Å³(网格离散项 {grid_count_delta:+.1f} Å³); "
                 f"突变位点: {mut_desc}; 主要回填: "
                 + (", ".join(f"{d['label']} {d['contribution_A3']:+.1f}" for d in backfill[:5]) or "无"))

    LOG.info("[pocket_decomp] Δ腔 %+.1f Å³ | 腾出 %+.1f / 回填 %+.1f | 残差 %.2f",
             delta_cav, released_total, backfill_total, residual)

    return {
        "ok": True,
        "radius_A": float(radius), "step_A": float(step), "probe_A": float(probe),
        "wt": str(wt_c), "mut": str(mut_c),
        "seed_wt": [round(float(v), 2) for v in sw],
        "seed_mut": [round(float(v), 2) for v in sm],
        "seed_source_wt": src_w, "seed_source_mut": src_m,
        "seed_gap_A": round(seed_gap, 2),
        "cavity_wt_A3": float(ow["free_A3"]),
        "cavity_mut_A3": float(om["free_A3"]),
        "cavity_delta_A3": delta_cav,
        "sphere_grid_volume_wt_A3": ow["sphere_grid_volume_A3"],
        "sphere_grid_volume_mut_A3": om["sphere_grid_volume_A3"],
        "grid_count_delta_A3": round(grid_count_delta, 2),
        "sum_contributions_A3": round(sum_contrib, 2),
        "residual_A3": round(residual, 2),
        "self_consistent": self_ok,
        "mutated_residues": list(mut_sites),
        "mutated_rows": mut_rows,
        "released_total_A3": released_total,
        "backfill_total_A3": backfill_total,
        "released_top": released[:int(top_n)],
        "backfill_top": backfill,
        "per_residue": reported,
        "n_residue_reported": len(reported),
        "n_residue_total": len(per_res),
        "min_report_A3": float(min_report_A3),
        "banned_phrases": banned,
        "allowed_phrases": allowed,
        "replicate_required": True,
        "narrative": narrative,
    }
