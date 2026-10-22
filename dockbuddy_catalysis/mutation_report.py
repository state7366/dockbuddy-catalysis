# -*- coding: utf-8 -*-
"""mutation_report.py —— #282 突变综合报告聚合(装配核心, 纯函数)。

装配原则(与 compare / pocket-tunnel 同源的自否证纪律):
- 本模块**只做装配**: 接受各证据线已算好的子报告 dict, 收拢成一份
  mutation_comprehensive_report.json。不调用任何对接 / MM 计算——那部分在
  cli/mutation_report_cmd.py 的编排层。
- 每条证据线的 verdict(verdict.narrative / banned_phrases / allowed_phrases)
  **原样引用**, 综合报告不另造措辞——避免口径漂移导致过度声称。
- 顶层 verdict.narrative 只做保守元叙事: 罗列参与了哪些证据线 + 统一免责
  (各线均为相对/重排序量, 非绝对自由能; 结论须以湿实验 kcat/KM 收口),
  不把任何单线 Δ 值直接翻译成"活性↑/↓"。
"""
from __future__ import annotations

import datetime as _dt
import re
from typing import Any, Dict, List, Optional

from .epistasis import _normalize_label

# 对齐 evidence.py 的 7 类证据维度(剔除不适用的 multiconf):
# reliability(对接可靠性, 这里落为 structure_validity) / orientation /
# retention / deacylation / conformation_entropy / pocket_tunnel / multiconf。
# 几何类 pocket_tunnel 在 compare 里又细拆出 pocket_decomp / product_egress,
# 故证据线在报告里以这几条 key 出现。
EVIDENCE_LINE_ORDER = [
    "structure_validity", "pocket_tunnel", "pocket_decomp",
    "product_egress", "deacylation", "conformation_entropy",
    "retention", "orientation", "msa", "foldx_ddg",
]

# 每条证据线对应的子报告 doc_type(用于溯源与 field_docs 对齐)
LINE_SOURCE_DOC = {
    "structure_validity": "mutate",
    "pocket_tunnel": "pocket_tunnel",
    "pocket_decomp": "pocket_decomp",
    "product_egress": "product_egress",
    "deacylation": "deacylation",
    "conformation_entropy": "conformation_entropy",
    "retention": "retention_summary",
    "orientation": "orientation",
    "msa": "msa_site_report",
    "foldx_ddg": "foldx_ddg_summary",
}


def _verdict_of(block: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """从子报告里取 verdict 子 dict(同构: ok/applicable/narrative/
    banned_phrases/allowed_phrases), 缺失则给保守空 verdict。"""
    if not block:
        return {"ok": False, "applicable": False, "narrative": "",
                "banned_phrases": [], "allowed_phrases": []}
    v = block.get("verdict") or {}
    return {
        "ok": bool(v.get("ok", True)),
        "applicable": bool(v.get("applicable", True)),
        "narrative": v.get("narrative", "") or "",
        "banned_phrases": list(v.get("banned_phrases") or []),
        "allowed_phrases": list(v.get("allowed_phrases") or []),
    }


def _emit(line: str, block: Optional[Dict[str, Any]], *,
          detail: Optional[Any] = None,
          available_override: Optional[bool] = None) -> Dict[str, Any]:
    present = block is not None
    if available_override is not None:
        avail = bool(available_override)
    else:
        avail = bool(block.get("available", present)) if block else False
    entry: Dict[str, Any] = {
        "available": avail,
        "source_doc": LINE_SOURCE_DOC.get(line, ""),
        "verdict": _verdict_of(block),
    }
    if detail is not None:
        entry["detail"] = detail
    return entry


_MUT_TOKEN_RE = re.compile(r"[A-Z]?\d+[A-Z]{1,3}")


def _derive_mut_label(s: str) -> str:
    """从路径/字符串里尽力提取突变 token(如 F205G / FA205G); 失败返回 ''。"""
    m = _MUT_TOKEN_RE.search(s or "")
    return m.group(0) if m else ""


def _match_foldx(fx: Dict[str, Any], label: str, mut_path: str) -> Optional[Dict[str, Any]]:
    """在 foldx_ddg_summary 的 mutations[] 里找匹配 label 的条目。
    优先精确/归一化匹配 label; label 为空时回退到 mut_path 的 stem 提取 token。"""
    muts = fx.get("mutations") or []
    if not muts:
        return None
    if not label:
        label = _derive_mut_label(mut_path)
    norm = _normalize_label(label) if label else None
    for m in muts:
        if norm and _normalize_label(m.get("mutation", "")) == norm:
            return m
        if label and label in (m.get("mutation"),):
            return m
    return None


def assemble_mutation_report(*, wt: str, mut: str, symmetry: str = "",
                             geometry: Optional[Dict[str, Any]] = None,
                             conf_entropy: Optional[Dict[str, Any]] = None,
                             retention: Optional[Dict[str, Any]] = None,
                             msa: Optional[Dict[str, Any]] = None,
                             orientation: Optional[Dict[str, Any]] = None,
                             structure_validity: Optional[Dict[str, Any]] = None,
                             foldx_ddg: Optional[Dict[str, Any]] = None,
                             foldx_mut_label: str = "",
                             meta: Optional[Dict[str, Any]] = None
                             ) -> Dict[str, Any]:
    """聚合各证据线为一份突变综合报告。

    参数均为「已算好的子报告」或 None(表示该证据线未提供/不可用):
      - geometry: run_compare_suite 返回的 compare_summary(含 4 件几何)
      - conf_entropy: configurational_entropy_difference 返回
      - retention: retention_dock 返回
      - msa: site_report 返回({site: {...}})
      - orientation: 取向判词 verdict dict(来自 .mutate.json 的
            eval_smiles_dock.orientation.verdict; 独立取向无 doc_type)
      - structure_validity: .mutate.json 的 qc 子 dict

    返回 mutation_comprehensive_report.json 全量结构。
    """
    evidence: Dict[str, Any] = {}

    # 1) 结构有效性(WT/突变体建模 QC)—— 结论的"前提门槛"
    if structure_validity is not None:
        sv = structure_validity
        ca = sv.get("ca_rmsd_vs_wt_A")
        clashes = sv.get("clashes_total_mutant", sv.get("clashes_total_wt"))
        sv_ok = ca is not None and isinstance(ca, (int, float)) and ca < 2.0 \
            and (clashes in (0, None))
        sv_narr = (f"突变体建模 QC: CA-RMSD vs WT = {ca} Å, "
                   f"总 clash = {clashes}。" if (ca is not None) else
                   "突变体结构 QC 未提供 CA-RMSD。")
        sv_verdict = {
            "ok": bool(sv_ok), "applicable": True,
            "narrative": sv_narr,
            "banned_phrases": ["突变体结构 QC 通过 → 活性提升(结构合理 ≠ 功能提升)"],
            "allowed_phrases": ["结构 QC 仅验证突变体建模质量(无严重 clash/折叠崩溃), 不推断功能"],
        }
        evidence["structure_validity"] = _emit(
            "structure_validity", structure_validity,
            detail={"ca_rmsd_vs_wt_A": ca,
                    "clashes_total": clashes,
                    "triad": sv.get("triad"),
                    "pocket_geom": sv.get("pocket_geom"),
                    "disulfide": sv.get("disulfide")},
            available_override=True)
        evidence["structure_validity"]["verdict"] = sv_verdict

    # 2) 几何四件套(来自 compare_summary)
    if geometry is not None:
        for key in ("pocket_tunnel", "pocket_decomp", "product_egress",
                    "deacylation"):
            sub = geometry.get(key)
            if sub is None:
                continue
            evidence[key] = _emit(key, sub, detail=sub)

    # 3) 局部构象熵差
    if conf_entropy is not None:
        evidence["conformation_entropy"] = _emit(
            "conformation_entropy", conf_entropy,
            detail={"delta_S_J_per_mol_K": conf_entropy.get("delta_S_J_per_mol_K"),
                    "selection": conf_entropy.get("selection"),
                    "n_selected": conf_entropy.get("n_selected"),
                    "noise_floor_J_per_mol_K": conf_entropy.get("noise_floor_J_per_mol_K")})

    # 4) 芳香产物滞留解除(机理 3 直接证据)
    if retention is not None:
        evidence["retention"] = _emit(
            "retention", retention,
            detail={"smiles": retention.get("smiles"),
                    "summary": retention.get("summary"),
                    "seeds": retention.get("seeds")})

    # 5) 进化保守性 / 小残基可行性(理性设计依据)
    if msa is not None:
        msa_lines = []
        for site, d in sorted((msa or {}).items()):
            small = d.get("small_residue_frac")
            msa_lines.append(
                f"位点 {site}: 保守性={d.get('verdict')}, "
                f"小残基占比={small if small is None else f'{small:.1%}'}")
        msa_verdict = {
            "ok": True, "applicable": True,
            "narrative": "进化保守性/小残基可行性证据(信息性, 非 pass/fail); "
                         + ("; ".join(msa_lines) if msa_lines else "无位点数据。"),
            "banned_phrases": [],
            "allowed_phrases": ["保守位点突变需更审慎; 天然小残基占比高=缩残基突变有家族先例"],
        }
        evidence["msa"] = _emit("msa", None, detail={"sites": msa},
                                available_override=True)
        evidence["msa"]["verdict"] = msa_verdict

    # 6) 催化取向判词(来自 eval_smiles_dock.orientation.verdict)
    if orientation is not None:
        evidence["orientation"] = _emit(
            "orientation", orientation, detail=orientation,
            available_override=True)
        evidence["orientation"]["verdict"] = _verdict_of(orientation)

    # 7) FoldX 5.1 ΔΔG 稳定性证据线(A3 接入: 来自 foldx_ddg_summary JSON;
    #    经 --foldx_json / --foldx_mut 接入, 亦可现算)。ΔΔG 是折叠稳定性代理,
    #    非催化活性 —— verdict 强制此免责, 禁止"ΔΔG<0 = 活性提升"。
    if foldx_ddg is not None:
        label = (foldx_mut_label or "").strip()
        hit = _match_foldx(foldx_ddg, label, mut)
        ddg = hit.get("ddG_kcal") if hit else None
        if hit and ddg is not None:
            nar = (f"FoldX 5.1 ΔΔG 稳定性预测: 突变 {label or hit.get('mutation')} "
                   f"的 ΔΔG = {ddg:+.3f} kcal/mol (WT "
                   f"{hit.get('wt_energy')}, MUT {hit.get('mut_energy')}, "
                   f"repair {hit.get('repair_energy')})。该值为折叠稳定性代理, "
                   f"非催化活性, 不得据其直接推断活性升降。")
        else:
            nar = (f"FoldX ΔΔG: 提供的 foldx 结果中未找到突变 "
                   f"{label or '(空标签)'} 的匹配项(已检索 mutations[] 的 "
                   f"mutation 字段); 该证据线不可用, 不计入 verdict。")
        fx_v = {
            "ok": bool(hit and ddg is not None),
            "applicable": True,
            "narrative": nar,
            "banned_phrases": [
                "ΔΔG<0 = 活性提升/增强/增高/上升",
                "FoldX 单 run = 真实稳定性定量值",
                "ΔΔG 直接推断催化活性升降",
            ],
            "allowed_phrases": [
                "ΔΔG 提示突变对折叠稳定性的影响, 非催化活性",
                "FoldX ΔΔG 须先实验标定再写入论文",
            ],
        }
        evidence["foldx_ddg"] = _emit(
            "foldx_ddg", foldx_ddg,
            detail={
                "mutation": label or (hit.get("mutation") if hit else None),
                "matched": hit.get("mutation") if hit else None,
                "ddG_kcal": ddg,
                "wt_energy": (hit or {}).get("wt_energy"),
                "mut_energy": (hit or {}).get("mut_energy"),
                "repair_energy": (hit or {}).get("repair_energy"),
                "n_mutations_in_summary": len(foldx_ddg.get("mutations") or []),
            },
            available_override=True)
        evidence["foldx_ddg"]["verdict"] = fx_v

    # ---- 顶层 verdict: 并集禁写/可写 + 保守元叙事 ----
    banned: List[str] = []
    allowed: List[str] = []
    present_lines: List[str] = []
    all_ok = True
    for line in EVIDENCE_LINE_ORDER:
        e = evidence.get(line)
        if e is None:
            continue
        present_lines.append(line)
        v = e.get("verdict") or {}
        banned.extend(v.get("banned_phrases") or [])
        allowed.extend(v.get("allowed_phrases") or [])
        if not bool(v.get("ok", True)):
            all_ok = False

    # 几何块是核心(compare_summary.ok), 若缺失/失败则整份不可用
    geo_ok = bool((geometry or {}).get("ok", False)) if geometry else False
    overall_ok = geo_ok and all_ok

    lines_dir = "; ".join(
        f"{ln}: {(evidence[ln].get('verdict') or {}).get('narrative', '')}"
        for ln in present_lines if (evidence[ln].get('verdict') or {}).get('narrative'))
    narrative = (
        f"突变综合报告聚合了 {len(present_lines)} 条证据线: "
        f"{', '.join(present_lines)}。各线均为相对/重排序量(几何差、ΔE、ΔS、取向占比、"
        f"进化保守性), 非绝对结合自由能预测; 任一方向的机制结论须以湿实验 kcat/KM 与"
        f"酰基酶捕获质谱收口, 不得由单点 Δ 值直接推断活性升降。" +
        (f" 各线方向: {lines_dir}" if lines_dir else ""))

    report = {
        "ok": overall_ok,
        "available": overall_ok,
        "wt": wt,
        "mut": mut,
        "symmetry": symmetry or "",
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "evidence_categories": present_lines,
        "evidence": evidence,
        "verdict": {
            "ok": overall_ok,
            "applicable": True,
            "banned_phrases": sorted(set(banned)),
            "allowed_phrases": sorted(set(allowed)),
            "narrative": narrative,
        },
        "note": ("本报告是证据聚合层, 不重新计算任何量; 每条证据线的数值与判词均引自"
                 "其来源子报告(见 evidence.<line>.source_doc 与 .verdict)。几何证据须喂"
                 "两侧均松弛的对称组合(如 F205F 对照), 不得把未松弛 WT 参照当正式结论。"),
    }
    if meta:
        report["meta"] = meta
    return report


__all__ = ["assemble_mutation_report", "EVIDENCE_LINE_ORDER", "LINE_SOURCE_DOC"]
