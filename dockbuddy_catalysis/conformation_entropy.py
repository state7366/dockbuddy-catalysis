# -*- coding: utf-8 -*-
"""conformation_entropy.py —— 构象熵差定量(NMA / 准谐振, 非 MD)。

为什么需要它
-------------
能量(ΔE)与几何(空腔 / 滞留)都答不了"突变是否改变了活性中心的**柔性**"。
构象熵差(ΔS_conf = S_mut − S_wt)是这条独立证据线: 若突变让催化三联体邻域
更"僵"(ΔS<0)或更"松"(ΔS>0), 会影响底物进入 / 产物释放的动力学。

实现口径(诚实优先)
-----------------
- 用 OpenMM 在 amber14-all + 隐式溶剂 OBC(GBneck2)下, 对选定原子集合做**局部
  受限最小化**(非选定原子加谐波约束固定), 在极小结构处用**有限差分力**构造
  **质量加权 Hessian**(3n×3n), 对角化得本征频率 → 谐振子振动熵
  S = R·Σ_i [x_i/tanh(x_i) − ln(2·sinh x_i)],  x_i = ħω_i / (2·k_B·T)。
- **只算选定原子子集的局部 Hessian**(典型 n≈20~50 残基侧链), 全蛋白 3N×3N 有限
  差分在本机 / 快机都不现实; 局部 Hessian 量化的是"该残基集合的局部柔性差",
  不是全蛋白, 且**冻结环境近似**会给局部集体低频模引入固有偏差(见 verdict)。
- 绝对熵无意义, 只比**同协议**下的 ΔS(WT vs 突变体)。

自否证(entropy_sides, 唯一叙事出口)
-----------------------------------
NMA 的有限差分 Hessian 数值误差大、极小化对初值 / 约束敏感, ΔS 噪声地板很高:
- 两侧任一侧 S 计算失败 / 缺失 → 不可用;
- |ΔS| < 噪声地板(默认 8 J/mol·K, 可由 DOCK_CONF_ENTROPY_NOISE_FLOOR_J 覆盖)
  → 禁写方向性结论;
- 即便超地板, 也**禁止**把 ΔS 读成 kcat / 活性变化(柔性 ≠ 速率常数), 且**必须**
  ≥2 次独立极小化重复且符号一致才可进叙事(replicate_required 恒 True)。
"""
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dockbuddy_dock._core import LOG, env_float
from dockbuddy_dock.triad import detect_catalytic_triad
from dockbuddy_dock.solvents import attach_obc_force

# —— 物理常数(SI) ——
_H_PLANCK = 6.62607015e-34          # J·s
_C_LIGHT_CM = 2.99792458e10         # cm/s
_KB = 1.380649e-23                  # J/K
_HBAR = 1.054571817e-34             # J·s
_R = 8.314462618                    # J/(mol·K)
_AMU_KG = 1.66053906660e-27         # kg
# OpenMM 单位 → SI 换算(能量 kJ/mol, 长度 nm)
_NA = 6.02214076e23
_OPENMM_FORCE_TO_SI = (1000.0 / _NA) * 1e9    # kJ/(mol·nm) → J/m
_OPENMM_HESS_TO_SI = (1000.0 / _NA) * 1e18    # kJ/(mol·nm²) → J/m²

# —— 模块可调参数(env 覆盖) ——
CONF_ENTROPY_T_K = env_float("DOCK_CONF_ENTROPY_T_K", 300.0)
CONF_ENTROPY_NOISE_FLOOR_J = env_float("DOCK_CONF_ENTROPY_NOISE_FLOOR_J", 8.0)
CONF_ENTROPY_RESTRAINT_K = env_float("DOCK_CONF_ENTROPY_RESTRAINT_K", 5000.0)
CONF_ENTROPY_FD_STEP_NM = env_float("DOCK_CONF_ENTROPY_FD_STEP_NM", 1e-4)
CONF_ENTROPY_MIN_ITER = int(env_float("DOCK_CONF_ENTROPY_MIN_ITER", 2000.0))
# 质量加权 Hessian 本征值(1/s²)零模阈值: 只剔除数值零模, 保留真实低频柔性模
CONF_ENTROPY_ZERO_TOL = env_float("DOCK_CONF_ENTROPY_ZERO_TOL", 1.0)


# ============================================================================
# 纯函数核心(单位无关, 可脱离 OpenMM 单测)
# ============================================================================
def _numerical_hessian_si(force_fn_si, q0_m: np.ndarray, h_m: float) -> np.ndarray:
    """有限差分质量加权前的**未质量加权** Hessian(单位 J/m²)。

    force_fn_si(x): 输入位移坐标 x(长度 n 的 flat 向量, 单位 m), 返回力
    (同样长度 flat 向量, 单位 J/m)。H_ij = −∂F_i/∂x_j。
    """
    q0 = np.asarray(q0_m, float)
    n = q0.size
    H = np.zeros((n, n))
    for j in range(n):
        dq = np.zeros(n)
        dq[j] = h_m
        fp = np.asarray(force_fn_si(q0 + dq), float)
        fm = np.asarray(force_fn_si(q0 - dq), float)
        H[:, j] = -(fp - fm) / (2.0 * h_m)
    # 有限差分引入非对称噪声 → 对称化
    return 0.5 * (H + H.T)


def _vibrational_entropy_from_omega(omega: np.ndarray, T: float) -> Tuple[float, np.ndarray]:
    """由角频率 ω(1/s)算谐振振动熵(单位 J/(mol·K))。

    S = R·Σ_i [x_i/tanh(x_i) − ln(2·sinh x_i)],  x_i = ħω_i/(2 k_B T)。
    仅对 ω>0 的模式求和。返回 (S, freq_cm1)。
    """
    om = np.asarray(omega, float)
    om = om[om > 0.0]
    if om.size == 0:
        return 0.0, np.array([])
    x = _HBAR * om / (2.0 * _KB * T)
    term = x / np.tanh(x) - np.log(2.0 * np.sinh(x))
    S = _R * float(np.sum(term))
    freq_cm1 = om / (2.0 * np.pi * _C_LIGHT_CM)
    return S, freq_cm1


def local_vibrational_entropy_si(force_fn_si, masses_kg: Sequence[float],
                                 q0_m: Sequence[float], T: float,
                                 h_m: float,
                                 zero_tol: float = CONF_ENTROPY_ZERO_TOL
                                 ) -> Dict[str, Any]:
    """局部振动熵主入口(纯函数, SI 单位)。

    给定选定原子集合的力函数 force_fn_si(坐标[m] → 力[J/m])、各原子质量[kg]、
    极小点坐标 q0_m[m]、温度 T[K], 算质量加权 Hessian → 本征值 → 振动熵。

    返回 {S_J_per_mol_K, n_modes, freqs_cm1, omega_s, hessian_ok}。
    """
    q0 = np.asarray(q0_m, float)
    masses = np.asarray(masses_kg, float)
    if q0.size != 3 * masses.size:
        return {"error": f"坐标数 {q0.size} 与 3×原子数 {3*masses.size} 不匹配"}
    # 每个原子质量作用于其 3 个笛卡尔坐标
    m3 = np.repeat(masses, 3)                              # 长度 3n
    H = _numerical_hessian_si(force_fn_si, q0, h_m)        # J/m²
    M_half_inv = 1.0 / np.sqrt(m3)                        # kg^{-1/2}
    Hmw = M_half_inv[:, None] * H * M_half_inv[None, :]    # 1/s²
    eig = np.linalg.eigvalsh(Hmw)
    pos = eig[eig > zero_tol]                              # 正定模(1/s²)
    omega = np.sqrt(pos)
    S, freq_cm1 = _vibrational_entropy_from_omega(omega, T)
    return {
        "S_J_per_mol_K": S,
        "n_modes": int(pos.size),
        "freqs_cm1": [float(v) for v in freq_cm1],
        "omega_s": [float(v) for v in omega],
        "hessian_ok": True,
        "n_zero_or_neg_modes": int((eig <= zero_tol).sum()),
    }


# ============================================================================
# OpenMM 集成(选定原子子集的局部 Hessian)
# ============================================================================
def _parse_residue_tokens(spec: str, chain: str) -> List[Tuple[str, int]]:
    """'126,174,A:205,B205' → [(chain,resnum), ...]; 无链前缀用给定 chain。"""
    out: List[Tuple[str, int]] = []
    for tok in (spec or "").replace(" ", "").split(","):
        if not tok:
            continue
        ch = chain
        num = tok
        if ":" in tok:
            ch, num = tok.split(":", 1)
        elif tok[0].isalpha():
            # A205 / B205 形式
            ch = tok[0]
            num = tok[1:]
        try:
            out.append((ch, int(num)))
        except ValueError:
            LOG.warning("[conf-ent] 无法解析残基 token %r, 跳过", tok)
    return out


def _resolve_selection(pdb_path: str, select_spec: Optional[str], chain: str,
                       select_radius: float = 6.0
                       ) -> Tuple[Optional[List[int]], Optional[str]]:
    """把 --select 解析为原子索引列表(基于 wt 结构)。

    - 未给 / 'triad' / 'site' → **以催化三联体 Cα 质心为球心、select_radius 为半径
      的球形邻域**(坐标法, 对 AF 模型"规范编号 vs 文件编号"不一致免疫, 且 WT vs
      突变体两套结构用同一球心自然对齐);
    - 否则按 '126,A:205' 形式解析(匹配该 PDB 文件**实际**残基编号)。
    返回 (indices, error): error 非空表示选择失败。
    """
    import openmm.app as app
    from openmm import unit as _u
    pdb = app.PDBFile(pdb_path)
    if not select_spec or select_spec.lower() in ("triad", "site"):
        triad = detect_catalytic_triad(pdb_path, chain_filter=chain)
        if not triad:
            return None, ("未检出催化三联体, 无法按位点球形邻域默认选择(请用 --select "
                          "显式指定该 PDB 实际残基编号, 如 --select 50,60,80)")
        c = triad["ca_centroid"]                       # (x,y,z) Å
        pos = pdb.positions                            # nm
        idx = []
        for atom in pdb.topology.atoms():
            p = pos[atom.index].value_in_unit(_u.nanometer) * 10.0  # Å
            if math.dist(p, c) <= select_radius:
                idx.append(atom.index)
        if not idx:
            return None, (f"三联体球心 {tuple(round(v, 1) for v in c)} Å 半径 "
                          f"{select_radius} Å 内无原子(结构可能缺坐标)")
        return sorted(set(idx)), None
    # 显式残基编号(匹配文件实际编号)
    toks = _parse_residue_tokens(select_spec, chain)
    if not toks:
        return None, f"--select 解析为空: {select_spec!r}"
    res_atoms: Dict[Tuple[str, int], List[int]] = {}
    for atom in pdb.topology.atoms():
        key = (atom.residue.chain.id, atom.residue.id)
        res_atoms.setdefault(key, []).append(atom.index)
    idx: List[int] = []
    for ch, rn in toks:
        key = (ch, rn)
        if key in res_atoms:
            idx += res_atoms[key]
        else:
            for (c2, r2), a in res_atoms.items():
                if r2 == rn:
                    idx += a
                    break
    if not idx:
        return None, f"选择 token {toks} 在 {pdb_path} 中无匹配原子"
    return sorted(set(idx)), None


def _side_entropy(pdb_path: str, sel_indices: Sequence[int],
                  solvent: str = "obc", T: float = CONF_ENTROPY_T_K,
                  restraint_k: float = CONF_ENTROPY_RESTRAINT_K,
                  fd_step_nm: float = CONF_ENTROPY_FD_STEP_NM,
                  min_iter: int = CONF_ENTROPY_MIN_ITER,
                  zero_tol: float = CONF_ENTROPY_ZERO_TOL
                  ) -> Dict[str, Any]:
    """单侧(单结构)局部振动熵。失败返回 {ok:False, error}。"""
    import openmm as mm
    import openmm.app as app
    from openmm import unit

    try:
        pdb = app.PDBFile(pdb_path)
        ff = app.ForceField("amber14-all.xml")
        mod = app.Modeller(pdb.topology, pdb.positions)
        mod.addHydrogens(ff, pH=7.0)
        system = ff.createSystem(mod.topology, nonbondedMethod=app.NoCutoff,
                                 constraints=None)
        if solvent == "obc":
            if attach_obc_force(system, LOG) is None:
                return {"ok": False, "error": f"{pdb_path}: OBC 隐式溶剂装配失败"}
        elif solvent not in ("vacuum", "none", ""):
            LOG.warning("[conf-ent] 未知溶剂 %r, 退回真空", solvent)

        sel = list(sel_indices)
        all_idx = list(range(system.getNumParticles()))
        restrained = [i for i in all_idx if i not in sel]
        if restrained:
            ef = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
            ef.addGlobalParameter("k", restraint_k)
            ef.addPerParticleParameter("x0")
            ef.addPerParticleParameter("y0")
            ef.addPerParticleParameter("z0")
            for i in restrained:
                p = mod.positions[i]
                ef.addParticle(i, [p.x, p.y, p.z])
            system.addForce(ef)

        integrator = mm.VerletIntegrator(0.001 * unit.picosecond)
        ctx = mm.Context(system, integrator,
                         mm.Platform.getPlatformByName("CPU"))
        ctx.setPositions(mod.positions)
        mm.LocalEnergyMinimizer.minimize(
            ctx, tolerance=10.0 * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=max(0, int(min_iter)))

        st = ctx.getState(getPositions=True)
        pos_nm = np.array([[v.x, v.y, v.z] for v in st.getPositions()],
                          float)  # nm
        q0_m = pos_nm[sel].reshape(-1) * 1e-9            # m
        masses_kg = [system.getParticleMass(i).value_in_unit(unit.dalton) * _AMU_KG
                     for i in sel]

        def force_fn_si(x_m):
            x_nm = (np.asarray(x_m, float) * 1e9).reshape(-1, 3)
            new_nm = pos_nm.copy()
            new_nm[sel] = x_nm
            ctx.setPositions(unit.Quantity(new_nm, unit.nanometer))
            fs = ctx.getState(getForces=True).getForces()
            f_sel = np.array([[f.x, f.y, f.z] for f in fs])[sel].reshape(-1)
            return f_sel * _OPENMM_FORCE_TO_SI      # J/m

        res = local_vibrational_entropy_si(
            force_fn_si, masses_kg, q0_m, T,
            h_m=fd_step_nm * 1e-9, zero_tol=zero_tol)
        if res.get("error"):
            return {"ok": False, "error": res["error"]}
        res.update({
            "ok": True, "n_selected": len(sel), "n_total": len(all_idx),
            "solvent": solvent, "ff": "amber14-all.xml",
            "T_K": T, "fd_step_nm": fd_step_nm,
        })
        return res
    except Exception as e:                       # 局部极小化 / 力场 / 截断等失败
        LOG.warning("[conf-ent] %s 局部熵计算失败: %s: %s",
                    pdb_path, type(e).__name__, e)
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ============================================================================
# 判词(唯一叙事出口, 自否证)
# ============================================================================
def entropy_sides(wt: Dict[str, Any], mut: Dict[str, Any]) -> Dict[str, Any]:
    """WT / 突变体两侧局部构象熵 → 可引用对比 + 自否证判词。

    入参 wt/mut 各为 _side_entropy 结果(含 S_J_per_mol_K / ok / n_modes)。

    返回齐 {"ok", "applicable", "noise_floor", "S_wt", "S_mut", "delta_S",
    "banned_phrases", "allowed_phrases", "replicate_required", "narrative"}。

    禁写规则(命中即止):
      1. 任一侧失败 / S 缺失 → 不可用, 不得写任何结论;
      2. |ΔS| < 噪声地板 → 禁写方向性结论(与数值抖动不可分辨);
      3. 超地板 → 允许写"局部柔性变化"的带限定表述, 但:
         - 禁止读成 kcat / 活性变化(柔性 ≠ 速率常数);
         - 禁止仅凭单次极小化下结论(replicate_required 恒 True);
         - 禁止把绝对熵值当物理量(仅同协议 ΔS 可解释)。
    """
    out: Dict[str, Any] = {
        "ok": True, "applicable": True,
        "noise_floor": CONF_ENTROPY_NOISE_FLOOR_J,
        "S_wt": None, "S_mut": None, "delta_S": None,
        "banned_phrases": [], "allowed_phrases": [],
        "replicate_required": True, "narrative": "",
    }
    sw, sm = wt.get("S_J_per_mol_K"), mut.get("S_J_per_mol_K")
    if sw is None or sm is None or not wt.get("ok") or not mut.get("ok"):
        out["ok"] = False
        out["banned_phrases"].append(
            "两侧构象熵未全部算成功, 不得写任何构象熵差结论")
        out["narrative"] = "构象熵差不可用(存在侧计算失败或 S 缺失)"
        return out

    dS = sm - sw
    out["S_wt"], out["S_mut"], out["delta_S"] = sw, sm, dS
    floor = CONF_ENTROPY_NOISE_FLOOR_J
    _d = f"{dS:+.3f}"
    if abs(dS) < floor:
        out["banned_phrases"].append(
            f"禁写『突变改变/未改变构象熵』: |ΔS|={abs(dS):.3f} J/mol·K < 噪声地板 "
            f"{floor} J/mol·K(有限差分 Hessian 数值误差与极小化初值敏感性量级) "
            f"—— 两侧构象熵差与数值抖动不可分辨")
        out["allowed_phrases"].append(
            f"可写: 两侧局部构象熵无实质差异(ΔS={_d} J/mol·K, 在地板内)")
        out["narrative"] = (f"局部构象熵差 ΔS={_d} J/mol·K, 低于噪声地板 {floor}, "
                            f"不能归因于突变对活性的柔性效应")
        return out

    direction = "更高(更松)" if dS > 0 else "更低(更僵)"
    out["allowed_phrases"].append(
        f"可写(带限定): 突变体局部构象熵相对 WT {direction}(ΔS={_d} J/mol·K, 超地板 "
        f"{floor}) —— 提示活性中心局部柔性变化; 须同句带 ΔS 数字与地板、注明"
        f"『冻结环境局部 NMA』口径")
    out["banned_phrases"].append(
        "禁止把 ΔS 读成 kcat / 活性变化(柔性 ≠ 速率常数; 构象熵是几何可用性, "
        "不含催化速率)")
    out["banned_phrases"].append(
        "禁止仅凭单次极小化下结论: 有限差分 Hessian 数值误差大、极小化对初值/约束敏感, "
        "必须 ≥2 次独立极小化重复且 ΔS 符号一致才可进叙事")
    out["banned_phrases"].append(
        "禁止把绝对熵值本身当物理量(仅同协议 ΔS 可解释; 不同溶剂/约束下的绝对 S 不可比)")
    out["narrative"] = (
        f"局部构象熵差 ΔS={_d} J/mol·K(超地板 {floor}), 突变体局部柔性"
        f"{direction}; 仍需 ≥2 次独立极小化重复确认符号, 且不得与 ΔE/滞留互相印证")
    return out


# ============================================================================
# 主入口: WT vs 突变体 局部构象熵差
# ============================================================================
def configurational_entropy_difference(
        wt_pdb: str, mut_pdb: str, select: Optional[str] = None, chain: str = "",
        solvent: str = "obc", T: float = CONF_ENTROPY_T_K,
        restraint_k: float = CONF_ENTROPY_RESTRAINT_K,
        fd_step_nm: float = CONF_ENTROPY_FD_STEP_NM,
        min_iter: int = CONF_ENTROPY_MIN_ITER,
        zero_tol: float = CONF_ENTROPY_ZERO_TOL,
        select_radius: float = 6.0) -> Optional[Dict[str, Any]]:
    """WT vs 突变体 局部构象熵差(非 MD: 局部受限极小化 + NMA)。

    选择基于 wt_pdb 的残基集合(默认催化三联体; --select 覆盖), 假定 mut_pdb 与之
    同残基编号(relaxed mutant 保持编号, 如 F205G 改造)。返回 None 表示无法评估。
    """
    if not (os.path.exists(wt_pdb) and os.path.exists(mut_pdb)):
        LOG.warning("[conf-ent] WT 或 突变体 PDB 不存在")
        return None
    sel, err = _resolve_selection(wt_pdb, select, chain, select_radius)
    if sel is None:
        LOG.warning("[conf-ent] 选择失败: %s", err)
        return None
    # mut 必须含同样的残基索引(编号一致前提)
    import openmm.app as app
    mut_pdb_obj = app.PDBFile(mut_pdb)
    mut_n = sum(1 for _ in mut_pdb_obj.topology.atoms())
    if max(sel) >= mut_n:
        LOG.warning("[conf-ent] mut_pdb 原子数 %d < 选择最大索引 %d(编号不一致?)",
                    mut_n, max(sel))
        return None

    wt_r = _side_entropy(wt_pdb, sel, solvent, T, restraint_k, fd_step_nm,
                         min_iter, zero_tol)
    mut_r = _side_entropy(mut_pdb, sel, solvent, T, restraint_k, fd_step_nm,
                          min_iter, zero_tol)
    sel_desc = select if select else f"triad_sphere_r{select_radius}"
    if not wt_r.get("ok") or not mut_r.get("ok"):
        return {
            "available": False,
            "error": f"WT: {wt_r.get('error')} | MUT: {mut_r.get('error')}",
            "selection": sel_desc, "n_selected": len(sel),
            "solvent": solvent, "ff": "amber14-all.xml", "T_K": T,
        }
    verdict = entropy_sides(wt_r, mut_r)
    return {
        "available": True, "mode": "local_nma",
        "selection": sel_desc, "n_selected": len(sel),
        "select_radius_A": select_radius,
        "solvent": solvent, "ff": "amber14-all.xml", "T_K": T,
        "fd_step_nm": fd_step_nm, "restraint_k": restraint_k,
        "noise_floor_J_per_mol_K": CONF_ENTROPY_NOISE_FLOOR_J,
        "wt": wt_r, "mut": mut_r,
        "delta_S_J_per_mol_K": verdict["delta_S"],
        "verdict": verdict,
        "note": ("局部 NMA(冻结环境近似): 仅量化选定残基集合的局部柔性差, "
                 "非全蛋白; 绝对熵不可比, 仅同协议 ΔS 可解释; ΔS 超地板仍需 "
                 "≥2 次独立极小化重复且符号一致才可进叙事"),
    }


__all__ = [
    "CONF_ENTROPY_T_K", "CONF_ENTROPY_NOISE_FLOOR_J",
    "CONF_ENTROPY_RESTRAINT_K", "CONF_ENTROPY_FD_STEP_NM",
    "CONF_ENTROPY_MIN_ITER", "CONF_ENTROPY_ZERO_TOL",
    "local_vibrational_entropy_si", "entropy_sides",
    "configurational_entropy_difference",
]
