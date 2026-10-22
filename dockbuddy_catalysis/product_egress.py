# -*- coding: utf-8 -*-
"""libdock.product_egress —— 产物释放可及性 core(2026-10-05 从 tools/product_egress.py 收编)。
===================================================================================
回答 kcat 叙事里"产物释放"这一半: 聚酯降解的芳香/二羧酸产物(TPA/MHET 类)能不能从
催化腔出来。已知**底物** governing diameter 3.3 Å > 通道瓶颈(⇒ 必须写"对接不
要求物理通路"的局限), 但**产物**更小 —— 这一条此前从未验过, 而它直接决定
"突变缓解产物滞留"的叙事有没有几何支撑。

判定口径(与 pocket_tunnel 的膨胀扫描语义严格对齐, 2026-10-05 首版曾把 capped 方向
写反、快机实测后修正 —— **capped=True 是下界**: 真实瓶颈 ∈ [r, r+0.2)):
- ``bottleneck_radius_A = r`` 是**能挤过去的球的最大半径**(扫描从 1.4 逐档加大,
  第一个断开档的前一档)。
- ``capped=True`` ⇒ 真实半径 ∈ [r, r+0.2) ⇒ 通道直径 ∈ [2r, 2r+0.4)。产物直径
  > 2r+0.4 ⇒ 确定过不去; < 2r ⇒ 可能通过; 区间内 ⇒ 无法断言。
- ``capped=False`` 且 r < 8 ⇒ 连 1.4 Å 球都过不去 ⇒ r 是**上界**。
- 本量是**球形/截面口径**: 刚性长分子沿轴向滑行可比截面判据宽松, 结论只对
  "垂直通道轴的截面"成立。

产物 3D 由 SMILES 生成(prepare_ligand, pH 7.4 ⇒ 羧酸根, 与审计 ★C 项一致),
截面用 pocket_tunnel.ligand_cross_section(同一实现, 消除双口径漂移)。

收编说明: 命令面在 `dockbuddy product-egress`(cli/analysis_cmd.py), 本模块
只住几何/判词逻辑 —— 与 pocket_tunnel/pocket_decomp/deacylation 同款
"core 单一事实源 + CLI 薄壳"分层。
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from dockbuddy_dock.ligand import prepare_ligand
from .pocket_tunnel import (BOTTLENECK_STEP, analyze,  # noqa: F401
                            ligand_cross_section, resolve_receptor)


# ----------------------------------------------------------------------------
# 产物类型注册表(2026-10-07, #281): 把 MHET 这类**单羧酸中间体**从 TPA 二酸 /
# EG 二醇中区分出来 —— 这是 F205G 机理 3(芳香产物滞留解除)叙事的真实产物口径。
# 此前产物集只是扁平 (name, smiles) 元组, 报告无法断言"受阻的是哪种产物"。
# ----------------------------------------------------------------------------
# 产物类型常量
PRODUCT_CLASS_MONOCARBOXYLIC = "monocarboxylic_acid"   # 单羧酸(如 MHET)
PRODUCT_CLASS_DICARBOXYLIC = "dicarboxylic_acid"       # 二酸(如 TPA)
PRODUCT_CLASS_ALIPHATIC_DIACID = "aliphatic_diacid"    # 脂肪族二酸(如 ADIPIC)
PRODUCT_CLASS_DIOL = "diol"                            # 二醇(如 EG)


@dataclass(frozen=True)
class ProductSpec:
    """产物分类规格(单一事实源)。

    - ``product_class``: 单羧酸 / 二酸 / 脂肪族二酸 / 二醇 —— 决定报告里
      能讲"哪种产物被卡住"。F205G 机理 3 真正关心的是**单羧酸芳香中间体**(MHET)
      与**二酸终产物**(TPA)这两类芳香产物, 而非极小、恒通过的二醇(EG)。
    - ``aromatic``: 是否含芳香环(关联机理 3 芳香滞留; 自定义配体为 None)。
    - ``role``: 产物角色(monocarboxylate_intermediate / terminal_diacid / ...)。
    - ``informative``: 是否对叙事有信息量。EG 等极小产物截面恒 < 瓶颈上界、
      永远"可能通过", 无区分力, 设为 False 以避免报告据此支撑结论。
    """
    name: str
    smiles: str
    product_class: Optional[str]
    aromatic: Optional[bool]
    role: Optional[str]
    informative: bool = True


# 内置产物集(顺序即历史 DEFAULT_PRODUCTS 顺序, 避免 CLI/测试依赖漂移)
PRODUCT_REGISTRY: List[ProductSpec] = [
    ProductSpec("TPA", "O=C(O)c1ccc(C(=O)O)cc1",
                PRODUCT_CLASS_DICARBOXYLIC, True, "terminal_diacid"),
    ProductSpec("MHET", "O=C(OCCO)c1ccc(C(=O)O)cc1",
                PRODUCT_CLASS_MONOCARBOXYLIC, True, "monocarboxylate_intermediate"),
    ProductSpec("ADIPIC", "O=C(O)CCCCC(=O)O",
                PRODUCT_CLASS_ALIPHATIC_DIACID, False, "aliphatic_diacid"),
    ProductSpec("EG", "OCCO",
                PRODUCT_CLASS_DIOL, False, "diol", informative=False),
]

# 兼容旧 CLI: 仍是 (name, smiles) 元组列表, 顺序不变
DEFAULT_PRODUCTS: List[Tuple[str, str]] = [(p.name, p.smiles) for p in PRODUCT_REGISTRY]


def default_product_specs() -> List[ProductSpec]:
    """内置产物(带类型)规格列表; CLI 不传 --smiles 时走这里, 类型元信息随 JSON 透传。"""
    return list(PRODUCT_REGISTRY)


def _as_spec(item) -> ProductSpec:
    """归一化单个产物为 ProductSpec。

    已是 ProductSpec 直接返回; 元组 (name, smiles) 若在注册表里命中则用注册类型,
    否则返回 product_class/aromatic/role 均为 None 的匿名规格(自定义配体)。
    """
    if isinstance(item, ProductSpec):
        return item
    name, smi = item[0], item[1]
    for p in PRODUCT_REGISTRY:
        if p.name == name and p.smiles == smi:
            return p
    return ProductSpec(name=name, smiles=smi,
                       product_class=None, aromatic=None, role=None,
                       informative=True)


_EGRESS_NOTE = ("瓶颈为 capped 上界口径时, '产物 ≤ 上界' 不等于 '能通过'; "
                "结论三态: 过不去/可能通过/无法排除。产物为单构象(gen3d), "
                "柔性产物的截面是构象依赖的, 只作量级判断。")


def tunnel_side(rec: str, chain: str = "") -> Dict:
    """单侧通道口径: resolve_receptor 收敛后跑 pocket_tunnel.analyze 取瓶颈。"""
    rec_c = resolve_receptor(rec)
    r = analyze(rec_c, chain=chain)
    return {"rec": str(rec_c),
            "bottleneck_radius_A": r.get("bottleneck_radius_A"),
            "bottleneck_capped": r.get("bottleneck_capped"),
            "bottleneck_interval_A": r.get("bottleneck_interval_A"),
            "connected_to_bulk": r.get("connected_to_bulk"),
            "cavity_volume_A3": r.get("cavity_volume_A3")}


def egress_verdict(d_prod: float, r_bn, capped: bool) -> Dict:
    """产物直径 d vs 瓶颈半径 r → 判词(三态: 过不去/可能通过/无法排除)。

    语义(与 pocket_tunnel 膨胀扫描严格对齐, 2026-10-05 首版曾写反、快机实测抓到):
    - capped=True ⇒ 断开档 r' > 扫描起点, bottleneck = 能通过的最大球 = r' 的前一档,
      **真值 ∈ [r, r+档距)** ⇒ 报告值是**下界**, 通道直径 ∈ [2r, 2r+0.4)。
    - capped=False 且 r < 8 ⇒ 连 1.4 Å 的球都过不去(断开档=扫描起点) ⇒ 报告值
      是**上界**, 真实更小。
    - capped=False 且 r >= 8 ⇒ 全档通过, 无瓶颈。
    """
    if r_bn is None:
        return {"passable": None, "why": "无瓶颈数据(死端腔或未算)"}
    r = float(r_bn)
    if capped:
        d_lo, d_hi = 2.0 * r, 2.0 * (r + BOTTLENECK_STEP)
        if d_prod > d_hi:
            return {"passable": False,
                    "why": f"产物 {d_prod} Å > 通道直径上界 {d_hi:.1f} Å"
                           f"(瓶颈真值 ∈ [{r:.1f}, {r + BOTTLENECK_STEP:.1f})) ⇒ 确定过不去"}
        if d_prod < d_lo:
            return {"passable": True,
                    "why": f"产物 {d_prod} Å < 通道直径下界 {d_lo:.1f} Å ⇒ 可能通过"}
        return {"passable": None,
                "why": f"产物 {d_prod} Å 落在通道直径区间 [{d_lo:.1f}, {d_hi:.1f}) 内"
                       f" ⇒ 边界情形, 无法断言"}
    if r >= 8.0 - 1e-9:
        return {"passable": True,
                "why": "无瓶颈(8 Å 球仍能通过), 产物不受几何限制"}
    d_cap = 2.0 * r
    if d_prod > d_cap:
        return {"passable": False,
                "why": f"产物 {d_prod} Å > 通道直径上界 {d_cap:.1f} Å"
                       f"(断开档=扫描起点, 真实瓶颈只会更小) ⇒ 确定过不去"}
    return {"passable": None,
            "why": f"产物 {d_prod} Å ≤ 直径上界 {d_cap:.1f} Å, 但真实通道可能更窄"
                   f" ⇒ 无法排除, 禁写'能通过'"}


def sides_from_tunnel_json(tunnel_json: str) -> List[Dict]:
    """从已有 pocket-tunnel JSON 读两侧瓶颈(免重算; 字段名同 pocket_tunnel 输出)。"""
    import json
    with open(tunnel_json, encoding="utf-8") as f:
        tj = json.load(f)
    sides = []
    for tag in ("wt", "mut"):
        r = tj.get(tag) or {}
        if r.get("bottleneck_radius_A") is not None:
            sides.append({"rec": tj.get(f"{tag}_pdb") or tag,
                          "bottleneck_radius_A": r.get("bottleneck_radius_A"),
                          "bottleneck_capped": r.get("bottleneck_capped"),
                          "bottleneck_interval_A": r.get("bottleneck_interval_A"),
                          "connected_to_bulk": r.get("connected_to_bulk"),
                          "cavity_volume_A3": r.get("cavity_volume_A3")})
    return sides


def analyze_products(sides: Sequence[Dict], products: Sequence,
                     ph: float = 7.4) -> List[Dict]:
    """逐产物: 3D 构建临时目录 + ligand_cross_section + 对每侧出三态判词。

    入参 ``products`` 可为 ``ProductSpec`` 或 ``(name, smiles)`` 元组, 内部统一
    经 ``_as_spec`` 归一化; 每条产物附 ``product_class`` / ``aromatic`` / ``role``
    / ``informative`` 类型元信息(自定义配体这几项为 null), 供报告区分"哪种
    产物被卡住"。3D 构建失败(如 SMILES 非法)的产物直接跳过, 与旧版一致。
    """
    results: List[Dict] = []
    with tempfile.TemporaryDirectory() as td:
        for raw in products:
            spec = _as_spec(raw)
            name, smi = spec.name, spec.smiles
            pqt = os.path.join(td, f"{name}.pdbqt")
            ok, _ = prepare_ligand(smi, pqt, flexible=False, ph=ph)
            if not ok:
                continue
            cs = ligand_cross_section(pqt)
            entry = {"product": name, "smiles": smi,
                     "product_class": spec.product_class,
                     "aromatic": spec.aromatic,
                     "role": spec.role,
                     "informative": spec.informative,
                     "governing_diameter_A": cs["governing_diameter_A"],
                     "effective_radius_A": cs["effective_radius_A"],
                     "n_heavy": cs["n_heavy"], "verdicts": {}}
            for s in sides:
                v = egress_verdict(cs["governing_diameter_A"],
                                   s["bottleneck_radius_A"],
                                   s["bottleneck_capped"])
                entry["verdicts"][s.get("rec") or str(len(entry["verdicts"]))] = v
            results.append(entry)
    return results


def analyze_egress(receptors: Sequence[str] = (), tunnel_json: str = "",
                   chain: str = "", products=None, ph: float = 7.4,
                   sides: Sequence[Dict] = None) -> Dict:
    """主流程: 通道侧(现算/读 JSON/调用方直传) + 产物侧 → 统一 JSON 结构。

    返回 {"params", "tunnel_sides", "products", "note"}; 没有通道数据时
    {"error": ...}(调用方负责报错退出, 不静默落盘空报告)。
    sides: 调用方已算好的通道侧(CLI 逐 --receptor 打印进度后直传, 免二次
    resolve/analyze —— 首版收编时 CLI 传空 receptors 走到这里, 被自己
    的"没有通道数据"守卫误杀, exit 2)。
    """
    if sides is None:
        sides = []
        if tunnel_json:
            sides = sides_from_tunnel_json(tunnel_json)
        for rec in receptors:
            sides.append(tunnel_side(rec, chain))
    if not sides:
        return {"error": "没有通道数据: 给 --receptor 或 --tunnel_json"}
    # products 显式给空列表 = 不算任何产物(None 才回退 DEFAULT_PRODUCTS;
    # 空/非空都如实透传, 与 CLI --smiles 空值语义一致)
    prods = [ _as_spec(p) for p in (products if products is not None else DEFAULT_PRODUCTS) ]
    results = analyze_products(sides, prods, ph=ph)
    return {"params": {"ph": ph},
            "tunnel_sides": sides, "products": results, "note": _EGRESS_NOTE}


__all__ = ["DEFAULT_PRODUCTS", "PRODUCT_REGISTRY", "default_product_specs",
           "ProductSpec", "PRODUCT_CLASS_MONOCARBOXYLIC", "PRODUCT_CLASS_DICARBOXYLIC",
           "PRODUCT_CLASS_ALIPHATIC_DIACID", "PRODUCT_CLASS_DIOL",
           "analyze_egress", "analyze_products", "_as_spec",
           "egress_verdict", "sides_from_tunnel_json", "tunnel_side"]
