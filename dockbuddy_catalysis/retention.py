# -*- coding: utf-8 -*-
"""retention.py —— 芳香产物滞留解除定量(机理 3 的定量支撑)。

为什么需要它(能量与取向都答不了的那一维)
-----------------------------------------
某丝氨酸水解酶对接聚酯寡聚物时, 已有两条证据线:
  - ΔE(对接能量差): 只反映"结合强弱", 不反映"底物/产物是否被卡在芳香口袋里";
  - oriented_frac(催化取向占比): 反映"底物有没有被摆成可水解姿态", 不反映
    "水解后的芳香产物有没有被 PHE205/PHE56 这类芳香锚定残基卡住出不去"。

机理 3(芳香产物滞留解除 / 产物抑制缓解)要的是第三条线:
  芳香产物(TPA=对苯二甲酸 / MHET=单(2-羟乙基)对苯二甲酸酯)的**芳香环**
  在 WT 与突变体口袋里的**滞留占比** —— 即有多少对接 pose 里, 产物的芳香碳
  仍落在受体芳香锚定残基的接触距离内(π-π 堆叠 / 疏水芳香接触)。F205G 把
  笨重的 PHE205 换成 Gly, 芳香锚定的一壁被拆掉 → 产物滞留占比应下降 →
  产物抑制缓解。

统计口径(与 oriented 同款: 必须多种子 × 全 pose, 不能只看 top1)
------------------------------------------------------------------
单次对接 9 个 pose 自己就散 6.92 Å, 单个 pose 的"卡没卡住"意义不大。故
aromatic_retention_frac 统计的是**全 pose** 的滞留占比, 并随种子维度逐种子
汇总成 mean±std。本模块与 comparative.py 复用同一套底层原语(prepare_ligand /
prepare_receptor / VinaBackend / parse_pdbqt_poses / load_receptor_aromatic_carbons /
orientation), **单次对接只跑一遍**, 同时产出 ΔE / oriented_frac / aromatic_retention_frac
三项指标(避免分别对接导致不同随机种子下结论不可比)。

自否证(`retention_sides`)
-------------------------
滞留占比是比例量, 小样本(n=9/侧)天然噪声大, 判词与 orientation / pocket_tunnel
同款: `retention_sides(wt, mut)` 是**唯一**产出 banned/allowed/narrative 的地方,
report 照抄不得改写。规则(见函数 docstring):
  - 两侧产物均无芳香碳 → 指标不适用, 禁写"滞留解除/加重";
  - 两侧占比差 < 噪声地板 → 禁写"滞留解除/加重"(与抽样抖动不可分辨);
  - 超地板且突变体更低 → 允许写"产物滞留解除"的**几何前提**, 但必须带占比
    数字与地板, 且不得把占比差读成 kcat 倍数、不得与 ΔE 互相印证。
"""
import math
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dockbuddy_dock._core import LOG
# 阈值单一事实源(2026-10-07 证据门槛注册表)
from dockbuddy_dock.evidence import THRESHOLDS as _EV, frac_delta_verdict
from .product_egress import PRODUCT_REGISTRY, ProductSpec, _as_spec  # #281 芳香产物集(type 元数据)
from dockbuddy_dock.triad import detect_catalytic_triad
from dockbuddy_dock.receptor import prepare_receptor
from dockbuddy_dock.ligand import prepare_ligand
from dockbuddy_dock.backends import VinaBackend
from dockbuddy_dock.poses import parse_pdbqt_poses, load_receptor_aromatic_carbons
from dockbuddy_dock.orientation import (find_scissile_sites, receptor_anchors,
                          pose_orientation, summarize_side,
                          NUC_TO_SCISSILE_C_MAX, HIS_TO_LEAVING_O_MAX,
                          ORIENT_NOISE_FRAC)

# 芳香滞留接触阈值(Å): 配体芳香碳 → 受体芳香碳 的最小距离 ≤ 此值即判"滞留"(仍
# 处在芳香锚定接触壳层内, 被卡住/出不去)。π-π 堆叠中心距 ~3.3-3.8 Å, 边缘-面
# C-C ~3.4-5.0 Å; 取 5.0 Å 作"仍处在芳香接触壳层内"的宽松判据(被卡住 = 至少还在
# 接触)。取值见 libdock.evidence.THRESHOLDS["retention_stack_A"]。
RETENTION_STACK_A = _EV["retention_stack_A"]
# 滞留占比差的噪声地板(比例量, 与 ORIENT_NOISE_FRAC 同源): n=9 pose 二项标准误
# ~0.16, 取 0.15 作 1σ 地板; 低于它禁写方向性结论。
RETENTION_NOISE_FRAC = _EV["retention_noise_frac"]
# 跨坐标系守卫(与 comparative 同源): ref 盒心用于 alt 时, 若 alt 自身三联体距
# ref 盒心过远, 该侧能量不能相减。
CROSS_FRAME_WARN_A = _EV["cross_frame_warn_A"]

# 芳香产物集(#281): 机理 3(芳香产物滞留解除)真正关心的是**含芳香环**的产物 ——
# TPA(二酸终产物)与 MHET(单羧酸中间体)。非芳香产物(EG/ADIPIC)对"芳香滞留"
# 指标不适用(retention_sides 会据此禁写), 不进默认集。
DEFAULT_RETENTION_PRODUCTS: List[ProductSpec] = [
    p for p in PRODUCT_REGISTRY if p.aromatic
]
# 仅贡献给机理 3 的"产物集"语义; 单产物对照仍可用 retention_dock(smiles=...) 任意指定。


# ----------------------------------------------------------------------------
# 单 pose 芳香滞留度量
# ----------------------------------------------------------------------------
def _pose_aromatic_stack(pose_atoms: Sequence[Sequence[Any]],
                         arom_names: set,
                         rec_arom: Dict[str, List[Tuple[float, float, float]]]
                         ) -> Tuple[Optional[bool], Optional[float]]:
    """配体芳香碳 → 受体芳香碳 的最小距离度量。

    返回 (retained, nearest_A):
      - retained = 最小距离 ≤ RETENTION_STACK_A(仍在芳香接触壳层内 = 滞留);
      - nearest_A = 最小距离(Å, 诊断用)。
    配体无芳香碳 或 受体无芳香残基 → (None, None)(指标不适用)。
    """
    lig_arom = [a[:3] for a in pose_atoms if a[4] in arom_names]
    if not lig_arom or not rec_arom:
        return None, None
    rec_pts = [p for pts in rec_arom.values() for p in pts]
    best = min(math.dist(l, r) for l in lig_arom for r in rec_pts)
    return (best <= RETENTION_STACK_A), round(best, 2)


# ----------------------------------------------------------------------------
# 单侧 / 跨种子汇总(纯函数, 可单测)
# ----------------------------------------------------------------------------
def _summarize_retention_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """逐种子 runs(ref/alt + delta_E + 各侧 orientation/retention) → 汇总。

    输入 = retention_dock 的 runs(逐种子 {seed, ref, alt, delta_E_alt_minus_ref});
    ref/alt 含 orientation(summarize_side 结果)与 aromatic_retention_frac。
    纯聚合, 不改动既有字段。
    """
    import statistics as _st

    def _mm(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return {"mean": None, "std": 0.0, "n": 0}
        return {"mean": round(_st.mean(vals), 3),
                "std": round(_st.pstdev(vals), 3) if len(vals) > 1 else 0.0,
                "n": len(vals)}

    def _orient(side):
        return _mm([(r[side].get("orientation") or {}).get("oriented_frac")
                    for r in runs])

    def _ret(side):
        return _mm([r[side].get("aromatic_retention_frac") for r in runs])

    d_es = [r["delta_E_alt_minus_ref"] for r in runs]
    ref_e = [r["ref"]["energy_kcal_mol"] for r in runs]
    alt_e = [r["alt"]["energy_kcal_mol"] for r in runs]
    return {
        "n_seeds": len(runs),
        "delta_E_mean": round(_st.mean(d_es), 3),
        "delta_E_std": round(_st.pstdev(d_es), 3) if len(d_es) > 1 else 0.0,
        "delta_E_min": min(d_es), "delta_E_max": max(d_es),
        "ref_energy_mean": _mm(ref_e)["mean"],
        "alt_energy_mean": _mm(alt_e)["mean"],
        "oriented": {"wt": _orient("ref"), "mut": _orient("alt")},
        "aromatic_retention": {"wt": _ret("ref"), "mut": _ret("alt")},
    }


# ----------------------------------------------------------------------------
# 判词单点(唯一允许产出"滞留结论"的地方)
# ----------------------------------------------------------------------------
def retention_sides(wt: Optional[Dict[str, Any]],
                    mut: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """WT / 突变体两侧 retention 汇总 → 可引用对比 + **自否证判词**。

    入参 wt/mut 各为 {"oriented_frac", "aromatic_retention_frac", "n_pose",
    "n_scissile_sites", "arom_name_count"}。oriented_frac / aromatic_retention_frac
    既可给 float 也可给 {"mean":...}(跨种子聚合), 二者都按数值读。

    返回齐 {"ok", "retention_applicable", "noise_floor",
    "oriented_frac_wt/mut", "oriented_frac_delta",
    "retention_frac_wt/mut", "retention_frac_delta",
    "banned_phrases", "allowed_phrases", "narrative"}。

    禁写规则(顺序即优先级, 命中即止):
      1. 两侧产物均无芳香碳 → 指标不适用, 禁写任何滞留方向结论。
      2. 任一侧缺失数据 / 占比 None → 不得写方向性结论。
      3. |Δ滞留占比| < 噪声地板 → 禁写"滞留解除/加重"; 可写"无实质差异"。
      4. 超地板且突变体更低(Δ<0) → 允许写"产物滞留解除"的**几何前提**,
         但必须带占比数字与地板, 且不得读成 kcat 倍数、不得与 ΔE 互相印证。
      5. 超地板且突变体更高(Δ>0) → 与"滞留解除"预期相反, 需复核盒心/坐标系。

    注意: 本函数**不判断能量**(ΔE 由 summary 独立报告)。滞留与能量是两条独立
    证据线, 合并叙事须显式组合, 且各自先过自己的自否证。
    """
    def _brief(tag, s):
        if not s:
            return {"tag": tag, "ok": False, "error": "该侧无数据"}
        return {"tag": tag, "ok": True,
                "oriented_frac": s.get("oriented_frac"),
                "aromatic_retention_frac": s.get("aromatic_retention_frac"),
                "n_pose": s.get("n_pose"),
                "arom_name_count": s.get("arom_name_count")}

    out: Dict[str, Any] = {
        "ok": True, "retention_applicable": True,
        "noise_floor": RETENTION_NOISE_FRAC,
        "banned_phrases": [], "allowed_phrases": [],
        "wt": _brief("WT", wt), "mut": _brief("MUT", mut),
        "oriented_frac_wt": None, "oriented_frac_mut": None,
        "oriented_frac_delta": None,
        "retention_frac_wt": None, "retention_frac_mut": None,
        "retention_frac_delta": None, "narrative": "",
    }

    def _num(v):
        if isinstance(v, dict):
            v = v.get("mean")
        return v if isinstance(v, (int, float)) else None

    # 1. 适用性: 两侧产物均无芳香碳
    arom_w = (wt or {}).get("arom_name_count") or 0
    arom_m = (mut or {}).get("arom_name_count") or 0
    o_fw, o_fm = _num((wt or {}).get("oriented_frac")), _num((mut or {}).get("oriented_frac"))
    r_fw, r_fm = (_num((wt or {}).get("aromatic_retention_frac")),
                  _num((mut or {}).get("aromatic_retention_frac")))
    out["oriented_frac_wt"], out["oriented_frac_mut"] = o_fw, o_fm
    out["retention_frac_wt"], out["retention_frac_mut"] = r_fw, r_fm
    nsw = (wt or {}).get("n_scissile_sites") or 0
    nsm = (mut or {}).get("n_scissile_sites") or 0
    if arom_w == 0 and arom_m == 0:
        out["retention_applicable"] = False
        out["banned_phrases"].append(
            "禁写『突变解除/加重了芳香产物滞留』: 两侧产物均无芳香碳(无芳香环可滞留), "
            "芳香滞留指标不适用")
        out["allowed_phrases"].append(
            "可写: 该产物不含芳香环, 芳香滞留指标不适用, 机理 3 不能由此数据支撑")
        out["narrative"] = "产物无芳香环, 芳香滞留指标不适用"
        return out

    # 取向差(信息性, 仅在两侧确有酯型位点时有意义; 产物 TPA/MHET 多为羧酸端 →
    # n_scissile_sites=0, oriented_frac=0, 此支自动跳过)。
    if not (nsw == 0 and nsm == 0) and o_fw is not None and o_fm is not None:
        d_o = round(o_fm - o_fw, 3)
        out["oriented_frac_delta"] = d_o
        if abs(d_o) < ORIENT_NOISE_FRAC:
            out["banned_phrases"].append(
                f"禁写『突变改善/恶化了催化取向』(滞留分析内): |Δoriented|={abs(d_o):.3f} "
                f"< 噪声地板 {ORIENT_NOISE_FRAC}(n={out['wt'].get('n_pose')}/"
                f"{out['mut'].get('n_pose')} pose, 二项标准误~0.16)")

    # 2. 缺数据 → 不可写方向
    if r_fw is None or r_fm is None:
        out["ok"] = False
        out["banned_phrases"].append("存在一侧芳香滞留占比为 None, 不得写方向性结论")
        out["narrative"] = "芳香滞留对比不可用: 存在侧无有效占比"
        return out

    # 3/4/5. 滞留占比差
    d_r = round(r_fm - r_fw, 3)
    out["retention_frac_delta"] = d_r
    _fmt_w, _fmt_m = f"{r_fw:.2f}", f"{r_fm:.2f}"
    if not frac_delta_verdict(r_fw, r_fm, floor=RETENTION_NOISE_FRAC)["passes"]:
        out["banned_phrases"].append(
            f"禁写『突变解除/加重芳香产物滞留』: |Δ滞留占比|={abs(d_r):.3f} < 噪声地板 "
            f"{RETENTION_NOISE_FRAC}(n={out['wt'].get('n_pose')}/{out['mut'].get('n_pose')} "
            f"pose, 二项标准误~0.16) —— 两侧占比差与抽样抖动不可分辨")
        out["allowed_phrases"].append(
            f"可写: 两侧芳香产物滞留占比无实质差异(WT {_fmt_w} → 突变体 {_fmt_m})")
        out["allowed_phrases"].append(
            "可写: 若 ΔE 也在噪声内(±0.15 kcal/mol), 本轮对接对该突变无滞留层面的区分力")
        out["narrative"] = (f"芳香产物滞留占比无实质变化(Δ={d_r:+.3f}); "
                            f"不能归因于『产物抑制缓解』")
        return out

    if d_r < 0:
        out["allowed_phrases"].append(
            f"可写(带限定): 突变体侧芳香产物滞留占比更低(WT {_fmt_w} → 突变体 "
            f"{_fmt_m}, Δ={d_r:+.3f} < 地板 {RETENTION_NOISE_FRAC} 的负值方向) —— "
            f"支持『F205G 拆掉芳香锚定壁 → 芳香产物滞留解除、产物抑制缓解』的**几何前提**, "
            f"但仍需多种子独立重复确认(单轮 9 pose 检出占比的可复现性未验)")
        out["allowed_phrases"].append(
            "可写时必须同句带占比分母与地板数字, 不得只写形容词『产物释放更顺畅』")
    else:
        out["allowed_phrases"].append(
            f"可写(带限定): 突变体侧芳香产物滞留占比更高(WT {_fmt_w} → 突变体 "
            f"{_fmt_m}, Δ={d_r:+.3f}) —— 与『滞留解除』预期相反, 需复核盒心/坐标系")
    out["banned_phrases"].append(
        "禁止把滞留占比差直接读成 kcat 倍数或活性比例 —— 滞留是几何可用性, 不含速率常数")
    out["banned_phrases"].append(
        "禁止把滞留占比差与 ΔE 能量差互相印证(对接不要求物理通路, 二者不可互相佐证)")
    out["narrative"] = (
        f"芳香产物滞留占比 突变体 vs WT: {_fmt_m} vs {_fmt_w}(Δ={d_r:+.3f}, "
        f"超地板 {RETENTION_NOISE_FRAC})"
        + (" —— 支持产物滞留解除的几何前提" if d_r < 0 else " —— 与滞留解除预期相反, 需复核"))
    return out


# ----------------------------------------------------------------------------
# 主入口: 芳香产物 WT vs 突变体 双边对照对接
# ----------------------------------------------------------------------------
def retention_dock(ref_pdb: str, alt_pdb: str, smiles: str, chain: str = "",
                   seeds=(42,), exhaustiveness: int = 8, num_modes: int = 9,
                   ph: float = 7.4, box_size: float = 24.0,
                   retention_stack_A: float = RETENTION_STACK_A) -> Optional[Dict[str, Any]]:
    """芳香产物双边对照对接: 同一产物/盒子, 逐种子分别对接 ref(WT) 与 alt(突变体)。

    与 comparative.smiles_comparative_dock 同用途(无共晶配体时的对照), 但**一次性**
    在同一次对接里同时产出三项指标, 保证 ΔE / oriented_frac / aromatic_retention_frac
    来自同一随机种子下的同一批 pose(不分别对接导致不可比):
      - ΔE(alt−ref, 同盒心可比)
      - oriented_frac(可攻击姿态占比, 复用 orientation 判据)
      - aromatic_retention_frac(芳香产物滞留占比, 本模块新增)

    返回 None 表示无法评估(三联体缺失/配体失败/全部种子对接失败)。
    """
    import numpy as np

    smiles = (smiles or "").strip()
    if not smiles:
        return None
    _seeds = [int(s) for s in (seeds or [42])]
    _seen: set = set()
    seeds = [s for s in _seeds if not (s in _seen or _seen.add(s))]
    triad = detect_catalytic_triad(ref_pdb, chain_filter=chain)
    if not triad:
        LOG.info("[retention] ref 侧未检出催化三联体, 盒心不可定位, 跳过")
        return None
    center = np.asarray(triad["ca_centroid"], float)
    size = (box_size, box_size, box_size)
    try:
        from dockbuddy_dock.tools import find_vina
        vina = find_vina("")
    except Exception as e:
        LOG.warning("[retention] 定位 vina 失败, 跳过: %s", e)
        return None

    workdir = tempfile.mkdtemp(prefix="retain_")
    try:
        lig_pdbqt = os.path.join(workdir, "lig.pdbqt")
        ok, arom_names = prepare_ligand(smiles, lig_pdbqt, flexible=True, ph=ph)
        if not ok:
            LOG.warning("[retention] 柔性配体制备失败, 退回刚性重试")
            ok, arom_names = prepare_ligand(smiles, lig_pdbqt, flexible=False, ph=ph)
            if not ok:
                return None
        arom_name_set = set(arom_names or [])

        def _dock_one(tag: str, pdb: str, seed: int) -> Optional[Dict[str, Any]]:
            rec_pdbqt = os.path.join(workdir, f"{tag}_rec.pdbqt")
            ok_r, used_chain = prepare_receptor(pdb, rec_pdbqt, receptor_chain=chain)
            if not ok_r:
                return None
            out_pdbqt = os.path.join(workdir, f"{tag}_{seed}_out.pdbqt")
            backend = VinaBackend(vina)
            if not backend.dock(rec_pdbqt, lig_pdbqt, center, size, out_pdbqt,
                                num_modes=num_modes, exhaustiveness=exhaustiveness,
                                seed=seed, cpu=1):
                return None
            poses = parse_pdbqt_poses(out_pdbqt)
            if not poses or poses[0][0] is None:
                LOG.warning("[retention] %s 侧无 pose 或无能量标注, 跳过该边", tag)
                return None
            rec_arom = load_receptor_aromatic_carbons(rec_pdbqt)
            # 取向(全 pose)
            try:
                _sites = find_scissile_sites(poses[0][1]) if poses[0][1] else []
                _anch = receptor_anchors(rec_pdbqt, triad)
                _rows = []
                for _e, _at in poses:
                    if _e is None:
                        continue
                    _o = pose_orientation(_at, _sites, _anch["nuc_pts"], _anch["his_pts"])
                    _o["energy"] = _e
                    _rows.append(_o)
                _or = summarize_side(_rows)
                _or["sites_total"] = len(_sites)
                _or["sites_acid"] = sum(1 for s in _sites if s["acid"])
            except Exception as e:  # 取向附加, 失败不得拖垮主流程
                LOG.warning("[retention] %s 侧催化取向统计失败(%s: %s)",
                            tag, type(e).__name__, e)
                _or = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            top_energy = round(float(poses[0][0]), 2)
            # 芳香滞留(全 pose)
            n_ret, nearest_list = 0, []
            for _e, _at in poses:
                if _e is None:
                    continue
                retained, nearest = _pose_aromatic_stack(_at, arom_name_set, rec_arom)
                if retained:
                    n_ret += 1
                if nearest is not None:
                    nearest_list.append(nearest)
            n_pose = len([p for p in poses if p[0] is not None])
            retention_frac = round(n_ret / n_pose, 3) if n_pose else None
            nearest_A = round(min(nearest_list), 2) if nearest_list else None
            return {"energy_kcal_mol": top_energy,
                    "orientation": _or,
                    "aromatic_retention_frac": retention_frac,
                    "aromatic_retention_nearest_A": nearest_A,
                    "n_pose": n_pose, "n_retained": n_ret,
                    "arom_name_count": len(arom_name_set),
                    "chain_used": used_chain}

        LOG.info("[retention] 多种子双边对照 (盒心=三联体质心 %s, 尺寸 %s, seeds=%s, "
                 "exh=%d, cpu=1, 滞留阈值 %.1fÅ)", np.round(center, 2).tolist(),
                 [round(s, 1) for s in size], seeds, exhaustiveness, retention_stack_A)
        runs: List[Dict[str, Any]] = []
        for s in seeds:
            ref = _dock_one("ref", ref_pdb, s)
            alt = _dock_one("alt", alt_pdb, s)
            if not ref or not alt:
                LOG.warning("[retention] seed=%d 双边未完成(ref=%s, alt=%s), 跳过该种子",
                            s, "ok" if ref else "fail", "ok" if alt else "fail")
                continue
            d_e = round(alt["energy_kcal_mol"] - ref["energy_kcal_mol"], 2)
            runs.append({"seed": s, "ref": ref, "alt": alt,
                         "delta_E_alt_minus_ref": d_e})
        if not runs:
            LOG.warning("[retention] 全部种子双边对接失败")
            return None
        summary = _summarize_retention_runs(runs)
        wt = {"oriented_frac": summary["oriented"]["wt"],
              "aromatic_retention_frac": summary["aromatic_retention"]["wt"],
              "n_pose": sum(r["ref"]["n_pose"] for r in runs),
              "n_scissile_sites": (runs[0]["ref"]["orientation"] or {}).get("n_scissile_sites"),
              "arom_name_count": runs[0]["ref"]["arom_name_count"]}
        mut = {"oriented_frac": summary["oriented"]["mut"],
               "aromatic_retention_frac": summary["aromatic_retention"]["mut"],
               "n_pose": sum(r["alt"]["n_pose"] for r in runs),
               "n_scissile_sites": (runs[0]["alt"]["orientation"] or {}).get("n_scissile_sites"),
               "arom_name_count": runs[0]["alt"]["arom_name_count"]}
        verdict = retention_sides(wt, mut)
        return {
            "available": True, "mode": "smiles",
            "smiles": smiles,
            "box_center": [round(float(c), 3) for c in center],
            "box_size": [round(float(s), 2) for s in size],
            "box_source": "triad_ca_centroid(ref)",
            "seeds": [r["seed"] for r in runs],
            "exhaustiveness": exhaustiveness, "num_modes": num_modes,
            "vina_cpu": 1,
            "retention_stack_A": round(retention_stack_A, 2),
            "retention_noise_floor": RETENTION_NOISE_FRAC,
            "runs": runs,
            "summary": summary,
            "verdict": verdict,
            "note": ("外源产物无晶体基准, 不做等效判定; 对照必须同机同环境产出"
                     "(跨 OpenBabel 版本存在 0.2~0.5 kcal/mol 系统差); "
                     "滞留占比为比例量, 低于噪声地板不得写方向性结论"),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def retention_dock_set(ref_pdb: str, alt_pdb: str, products=None, chain: str = "",
                       seeds=(42,), exhaustiveness: int = 8, num_modes: int = 9,
                       ph: float = 7.4, box_size: float = 24.0) -> Dict[str, Any]:
    """芳香产物集批量对照: 对 ``products`` 中每个产物分别跑 ``retention_dock``。

    不传 ``products`` 时默认 ``DEFAULT_RETENTION_PRODUCTS``(TPA + MHET), 即机理 3
    关心的芳香产物全集 —— 一次命令把"F205G 对两种芳香产物的滞留解除"都跑了,
    避免分多次对接导致随机种子不可比。返回 ``{"mode": "product_set",
    "products": {name: retention_dock 结果或 {available:False,...}} }``; 单产物
    对照仍走 ``retention_dock``(返回结构与单产物 retention_summary 一致)。
    """
    specs = [_as_spec(p) for p in (products if products is not None
                                    else DEFAULT_RETENTION_PRODUCTS)]
    out: Dict[str, Any] = {
        "mode": "product_set",
        "product_set": [p.name for p in specs],
        "products": {},
    }
    for spec in specs:
        res = retention_dock(ref_pdb, alt_pdb, spec.smiles, chain=chain,
                             seeds=seeds, exhaustiveness=exhaustiveness,
                             num_modes=num_modes, ph=ph, box_size=box_size)
        if res is None:
            out["products"][spec.name] = {"available": False,
                                          "error": "对接失败/三联体缺失/全部种子失败"}
        else:
            out["products"][spec.name] = res
    return out


__all__ = [
    "RETENTION_STACK_A", "RETENTION_NOISE_FRAC",
    "_pose_aromatic_stack", "_summarize_retention_runs",
    "retention_sides", "retention_dock", "retention_dock_set",
    "DEFAULT_RETENTION_PRODUCTS",
]
