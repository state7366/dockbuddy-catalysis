# -*- coding: utf-8 -*-
"""pocket_tunnel.py —— 「口袋 + 入口通道」几何分析(libdock core 层)。

为什么 libdock 需要它(pocketgeom 补不上的部分):
  pocketgeom.pocket_volume_qc 给的是**催化核心球内的自由体积**(单点球近似), 它
  回答不了解释活性必须回答的三个问题:
    (a) 变大的体积是不是**同一条空腔**(还是冒出个新的死端腔);
    (b) 这条空腔是否经**入口通道**与体相连 —— 与体不连通的死端腔对聚合物降解
        毫无贡献, 却会让「口袋体积 +X%」这种数字看起来很好看;
    (c) 通道**瓶颈半径** vs 底物横截面 —— 聚合物底物能不能进得去,
        才是活性提升的结构解释候选, 而不是空腔总体积。

本模块 = 单一事实源; `dockbuddy pocket-tunnel`(cli/pocket_tunnel_cmd.py)是它的
**薄 CLI 壳**(只做参数校验/打印对比表/JSON 落盘, 不含任何几何口径)。这样 mutate
的 QC 将来可直接 import 本模块, 不必(也不该)让 pipeline 去 import tools/ ——
pipeline 从不依赖 tools, 方向必须是 tools → pipeline, 不是反过来。
(原 tools/pocket_tunnel.py 独立脚本已于 2026-10-05 随这次迁入删除。)

口径(与 pocketgeom 同源可互相核对; 任何缺省改动都要换口径标记):
  - probe = 1.4 Å(水半径), vdW 表同 pocketgeom._VDW
  - 分析域 = **口袋球**: 球心 = **催化亲核原子**(Ser OG / Cys SG), 半径
    POCKET_RADIUS_A = 8.0 Å; 自由体积/空腔体积与 pocket_volume_qc 同球心同半径
    → 可与历史 mutate QC 的 57→117 Å³ 对照。
    ⚠ **球心不能用"催化三联体 Cα 质心"**: 实测它落在 His204 ND1 原子内部
    1.22 Å(clearance −0.33), R=8 内只能 flood 出 3 个格点的假腔。回退链见
    analyze() 的 seed_source 字段。
  - 格距 STEP = 0.6 Å(通道/瓶颈用, 半径分辨率 ~0.3 Å)。
  - **连体相判据(严格)**: 在**全蛋白网格**(域 = 全蛋白包围盒 ±10 Å, 与 radius
    **解耦**)上, 从盒边界体相 6-连通 flood; 腔的判据 = **粗网格上整个腔的足迹**。
    分 1.6 Å → 0.8 Å **两档**(实测某酶入口窄缝在 1.4~1.6 Å, 单档 1.6 会把
    "连着"误判死端)。两条必须遵守的约束见 analyze() 内 89c23b9 的注释:
      ① 网格原点/尺寸取自**未预筛的全蛋白原子**(否则换 radius 时原点漂移,
         死端判定跨半径翻号);
      ② 判据用整片腔足迹, 不用单个种子格点。
  - **曾经用"全局埋深 >=3 Å"判埋, 已废弃**(输出里 88568 Å³ 就是它的产物)。
    顺带 pocketgeom.detect_pockets 同病, 故本模块不再调用它。
  - 瓶颈半径 = **能走通"腔→外壳"通路的球的最大半径**(原子按 vdW+r 逐档膨胀做
    连通扫描)。不是"最小 clearance" —— 后者恒≈探针半径, 是纯离散化噪声。
    死端腔不报瓶颈。真实值落在 [报告值, 报告值+档距), 见 bottleneck_interval_A。
  - 底物横截面: PCA 主轴 → effective_radius = mid/2, vdW 下限 1.7 Å, 并标
    cross_section_limited_by(防棒状底物截面塌成 0)。

注意: 本模块**故意不**写进 libdock/__init__.py 的全量再导出 —— 它会把 pipeline
顶层模块 fetch 拖进核心 import 链, 而这是 opt-in 分析能力(默认不该被流水线
import 触发)。调用方显式 `from libdock.pocket_tunnel import analyze` 即可。
"""
from __future__ import annotations

import math
import os
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dockbuddy_dock._core import LOG
from dockbuddy_dock.pocketgeom import POCKET_RADIUS_A, _parse_pdb_heavy, _VDW
from dockbuddy_dock.triad import detect_catalytic_triad

# catalysis 不依赖 pipeline/stages; 调用方需先确保输入为 PDB(否则显式报错)
fetch = None  # type: ignore[assignment]


def resolve_receptor(path: str) -> str:
    """把受体路径收敛成下游真正吃的 PDB 路径。

    AlphaFold Server 下发的是 mmCIF(fold_*_model_N.cif), 而 _parse_pdb_heavy /
    detect_catalytic_triad 只认 PDB 固定列 —— 直接喂 .cif 会静默解析到 0 个重原子,
    报"未解析到重原子"把人引到"文件有问题"的错误方向。这里统一走 pipeline 的
    fetch.ensure_pdb_input(), 与 mutate/triad dock 同缓存键
    (cache/raw_pdb/cif_converted/<stem>_<源内容 sha256[:16]>.pdb), 源一改键就变,
    不会复用陈旧产物。非 .cif 输入原样返回。
    """
    if not str(path).lower().endswith(".cif"):
        return path
    if fetch is None:
        raise RuntimeError(
            f"[pocket_tunnel] 输入是 mmCIF({path}) 但需要 pipeline.fetch.ensure_pdb_input "
            f"做 CIF→PDB 转换; 当前环境未注入 pipeline 根(sys.path 无 common)。")
    got = fetch.ensure_pdb_input(path)
    # 绝不能回退到"CIF 当 PDB 解析": 那会返回 0 重原子, 再把人往"文件坏了"误导。
    if not got:
        raise RuntimeError(
            f"[pocket_tunnel] {path} 是 mmCIF 但 ensure_pdb_input 返回空(转换失败)。\n"
            f"  修法: 1) 确认 obabel 可用(`obabel -L formats | findstr cif`);\n"
            f"        2) 或直接用已转换产物:\n"
            f"           output\\cache\\raw_pdb\\cif_converted\\<stem>_<指纹>.pdb")
    return got


PROBE = 1.4
STEP = 0.6                 # 通道分析格距(比体积口径细)
EXIT_CRITERION = "口袋球内 6-连通至外壳层(dist >= R-1.5*step)"
# 连体相扫描的域(盒): 从 __ 全蛋白原子包围盒 外扩 BULK_DOMAIN_PAD。
# 必须与 radius 解耦 —— 早期版本把连通域取成"预筛后原子 ±(radius+11)+10",
# 于是小半径的盒把"腔→体相"的出口切在盒外, flood 起点(盒边界)到不了腔,
# 死端判定随 --radius 翻转(实测某 WT/突变体对: R=8 判 False, R=12 判 True)。
BULK_DOMAIN_PAD = 10.0
BOTTLENECK_STEP = 0.2      # 膨胀扫描档距(瓶颈真实值落在 [r, r+step) 内)
BOTTLENECK_NOISE_A = 0.5   # 通道叙事噪声线: 瓶颈 Δ < 此值一律判"没变宽",
                           # 与"同档"判据共同门控 compare_sides 的自否证

# **喉部位置**这个量的重复噪声地板(Å) —— 由**零假设对照**实测, 即"没有任何突变
# 效应发生时, 这个量自己能飘多少":
#   NULL(序列不变 / 仅松弛差异): 3.11 Å(对称对照#2 vs raw WT)、7.09 Å(对称对照#1
#     vs raw WT, 而空腔只差 -0.8%)、3.88 Å(同一对照两次独立松弛互比)
#     ⇒ null 观测范围 3.11 ~ 7.09 Å
#   OBS(真带突变效应的对比): 1.33 / 7.02(对称对照 vs 突变体, 两次)、
#     5.80 / 7.28(mutate 内建口径, 两次) ⇒ 观测范围 1.33 ~ 7.28 Å
# 两组**完全重叠**: 没有效应时这个量就能给出 7.09 Å 的"搬家", 而真效应时也只测到
# 1.33~7.28 Å。⇒ 地板取 null 上界 + 余量 = 8.0 Å; 低于它一律判"不可分辨"。
# (第一版定 4.0 只用了"两次对照互比 3.88 Å"这一个 null 样本, 被对称对照#1 的
#  7.09 Å 直接证伪 —— 教训: 噪声地板要按 null 分布的**上界**取, 不能按单个样本。)
BOTTLENECK_REPLICATE_NOISE_A = 8.0
VDW = _VDW                 # {"C":1.70,"N":1.55,"O":1.52,"S":1.80}
SOLVENT = {"HOH", "DOD", "WAT", "H2O", "SO4", "CL", "NA", "K", "MG", "ZN", "CA"}


def _parse_xyz(path: str) -> Tuple[np.ndarray, List[str]]:
    """读 .pdb / .pdbqt 重原子坐标, 返回 ((N,3) float 数组, 元素列表)。

    - PDB : x=30:38 y=38:46 z=46:54; PDBQT 同偏移(AutoDock 输出 31-54 列)
    - 跳过 H/D 与溶剂/离子(理由同 pocketgeom._parse_pdb_heavy)
    """
    coords, elements = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            name = line[12:16].strip()
            resn = line[17:20].strip().upper()
            if name.startswith(("H", "D")) and name not in ("HD",):
                continue
            if line.startswith("HETATM") and resn in SOLVENT:
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            coords.append((x, y, z))
            elements.append(line[76:78].strip().upper() or name[0])
    if not coords:
        raise RuntimeError(f"[pocket_tunnel] {path} 未解析到坐标(重原子)")
    return np.asarray(coords, dtype=float), elements


def _mark_occupied(shape: Sequence[int], origin: np.ndarray, step: float,
                   atoms, probe: Optional[float] = None) -> np.ndarray:
    """把 (vdW + probe) 内不可达的格点标 occupied(按原子打球)。

    probe 可调(瓶颈扫描时把探针逐档放大 = 把原子按 vdW+probe 膨胀)。
    """
    nx, ny, nz = shape
    _probe = float(PROBE if probe is None else probe)
    occ = np.zeros(shape, dtype=bool)
    for (_ch, _rn, _at, _res, x, y, z, vdw) in atoms:
        r = float(vdw) + _probe
        i0 = max(0, int((x - r - origin[0]) / step)); i1 = min(nx - 1, int((x + r - origin[0]) / step) + 1)
        j0 = max(0, int((y - r - origin[1]) / step)); j1 = min(ny - 1, int((y + r - origin[1]) / step) + 1)
        k0 = max(0, int((z - r - origin[2]) / step)); k1 = min(nz - 1, int((z + r - origin[2]) / step) + 1)
        if i0 > i1 or j0 > j1 or k0 > k1:
            continue
        ii, jj, kk = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1),
                                 np.arange(k0, k1 + 1), indexing="ij")
        d2 = ((ii * step + origin[0] - x) ** 2 + (jj * step + origin[1] - y) ** 2
              + (kk * step + origin[2] - z) ** 2)
        occ[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1] |= (d2 <= r * r)
    return occ


def _channel_profile(shape: Sequence[int], origin: np.ndarray, step: float, atoms,
                     seed_xyz: np.ndarray, goal_mask: np.ndarray,
                     probe: float = PROBE) -> Optional[Dict[str, Any]]:
    """沿"腔心 → 体相"的一条自由路径, 逐点量**到最近原子表面的余量** r。

    做法: 在 0.6 Å 细网格上对 free 格点做**多源 BFS**(源 = 域边界层上的游离格点),
    得到"离酶口几步"的距离场 D; 然后从腔心沿 D 单调下降走到 D==0 —— 得到一条
    腔心→体相的单调路径(比 parent 回溯稳, 且天然是最短通路)。沿途每点算
    r = min_over_atoms(dist - vdW), 即该点能塞下的球半径。

    goal_mask: 细网格上"算作体相出口"的游离格点(由调用方从粗网格边界采样而来)。
    """
    _free = ~_mark_occupied(shape, origin, step, atoms, probe=probe)
    _gi, _gj, _gk = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
                                indexing="ij")
    _d2 = ((_gi * step + origin[0] - seed_xyz[0]) ** 2
           + (_gj * step + origin[1] - seed_xyz[1]) ** 2
           + (_gk * step + origin[2] - seed_xyz[2]) ** 2)
    cand = np.argwhere(_free)
    if len(cand) == 0:
        return None
    dv = _d2[cand[:, 0], cand[:, 1], cand[:, 2]]
    start = tuple(int(v) for v in cand[int(np.argmin(dv))])   # 腔心(与 analyze 同一定义)
    D = -np.ones(shape, dtype=np.int32)
    src = [tuple(int(v) for v in c) for c in np.argwhere(goal_mask & _free)][:60000]
    q = deque()
    for s in src:
        if D[s] < 0:
            D[s] = 0
            q.append(s)
    while q:
        i, j, k = q.popleft()
        nd = D[i, j, k] + 1
        for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:
                if _free[ni, nj, nk] and D[ni, nj, nk] < 0:
                    D[ni, nj, nk] = nd
                    q.append((ni, nj, nk))
    if D[start] <= 0:
        return None

    path = [start]
    cur = start
    while D[cur] > 0:
        i, j, k = cur
        best, bv = None, D[cur]
        for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2] and D[ni, nj, nk] >= 0:
                if D[ni, nj, nk] < bv:
                    best, bv = (ni, nj, nk), D[ni, nj, nk]
        if best is None:
            return None
        cur = best
        path.append(cur)

    _xyz = np.array([[p[0] * step + origin[0], p[1] * step + origin[1], p[2] * step + origin[2]]
                     for p in path], dtype=float)
    _keep = np.array([a[4:7] for a in atoms], dtype=float)
    _vd = np.array([a[7] for a in atoms], dtype=float)
    _dd = np.linalg.norm(_xyz[:, None, :] - _keep[None, :, :], axis=2) - _vd[None, :]
    r_arr = _dd.min(axis=1)
    return {"n_step": len(path), "path_len_A": round(len(path) * step, 2),
            "r_min_A": round(float(r_arr.min()), 2),
            "r_min_at_A": [round(float(v), 2) for v in _xyz[int(np.argmin(r_arr))]],
            "r_mean_A": round(float(r_arr.mean()), 2),
            "buckets_A": {f"<{PROBE}": int((r_arr < PROBE).sum()),
                          ">=probe,<2.0": int(((r_arr >= PROBE) & (r_arr < 2.0)).sum()),
                          ">=2.0,<3.0": int(((r_arr >= 2.0) & (r_arr < 3.0)).sum()),
                          ">=3.0": int((r_arr >= 3.0).sum())},
            "r_at_cavity_end_A": round(float(r_arr[0]), 2),
            "r_at_exit_end_A": round(float(r_arr[-1]), 2)}


def _flood(seeds, mask, shape):
    """从 seeds(可多个)出发在 mask 内做 6 邻域连通生长, 返回 (visited, order)。"""
    visited = np.zeros(shape, dtype=bool)
    order = []
    stack = [s for s in seeds]
    for s in seeds:
        visited[s] = True
    while stack:
        i, j, k = stack.pop()
        order.append((i, j, k))
        for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:
                if mask[ni, nj, nk] and not visited[ni, nj, nk]:
                    visited[ni, nj, nk] = True
                    stack.append((ni, nj, nk))
    return visited, order


def _radius_at(points_idx, shape, origin, step, atoms_xyz, vdw_arr):
    """给定格点索引, 算它们到最近重原子表面的距离(min(dist - vdw))。"""
    if len(points_idx) == 0:
        return np.zeros(0)
    out = np.empty(len(points_idx), dtype=float)
    for s in range(0, len(points_idx), 4000):
        chunk = points_idx[s:s + 4000]
        pts = np.array([[i, j, k] for (i, j, k) in chunk], dtype=float) * step + origin
        d = np.sqrt(((pts[:, None, :] - atoms_xyz[None, :, :]) ** 2).sum(-1)) - vdw_arr[None, :]
        out[s:s + 4000] = d.min(axis=1)
    return out


def nucleophile_seed(pdb_path: str, atoms: Sequence[Tuple],
                     fallback_centroid=None, chain: str = "") -> Tuple[np.ndarray, str]:
    """催化亲核原子(Ser OG / Cys SG)坐标 + 来源串 —— **全仓唯一**球心口径。

    2026-10-05 抽出: 新增的 ``pocket_decomp``(空腔增益逐残基分解)必须与本模块的
    空腔体积**同球心同半径同探针**才能把两个数字并列。若各自实现一遍球心推导,
    漂移只会表现成"体积对不上"且极难查 —— 故收敛成这一个函数, 两边共用。

    atoms = ``_parse_pdb_heavy`` 结果(已按 chain 过滤, 未去溶剂; 亲核残基不可能是
    溶剂, 去不去溶剂不影响取值)。fallback 用**去溶剂后**的质心(与 analyze 原行为
    一致: 溶剂/离子会拉偏质心)。
    """
    tri = detect_catalytic_triad(pdb_path, chain_filter=chain)
    if not tri:
        cen = atoms and np.array([[a[4], a[5], a[6]] for a in atoms], float).mean(axis=0)
        if fallback_centroid is not None:
            cen = np.asarray(fallback_centroid, dtype=float)
        return cen, "全蛋白质心(未检出催化三联体)"
    nuc = tri["nuc"]
    _nm = (nuc.get("atom") or "").upper()
    src = f"{nuc['resname']}{nuc['resnum']}{nuc['chain']}:{_nm}"
    for a in atoms:
        if a[0] == nuc["chain"] and a[1] == nuc["resnum"] and a[2].upper() == _nm:
            return np.array([a[4], a[5], a[6]], dtype=float), src
    return (np.asarray(tri["ca_centroid"], dtype=float),
            src + "(亲核原子缺失, 回退 Cα 质心)")


def analyze(pdb_path: str, chain: str = "", seed_xyz=None,
            radius: Optional[float] = None) -> Dict[str, Any]:
    """单结构口袋-通道分析。

    seed_xyz 缺省取催化亲核原子(Ser OG / Cys SG); radius 缺省 POCKET_RADIUS_A。
    分析域 = 该球内"水探针进不去"的游离格点。返回字段见模块 docstring 与下方 return。

    坑(2026-10-05, 89c23b9): 见模块 docstring 的"两条必须遵守的约束"。
    """
    pdb_path = resolve_receptor(pdb_path)
    atoms = _parse_pdb_heavy(pdb_path)
    if not atoms:
        raise RuntimeError(
            f"[pocket_tunnel] {pdb_path} 解析到 0 个重原子(不是\"文件坏了\", 是解析口径不匹配)。\n"
            f"  最可能: ① --chain 与文件内链名不符; ② 该文件是 mmCIF/其他格式却漏过\n"
            f"          resolve_receptor(请重新 pull)或文件被截断(检查末尾 END / 行数)。")
    if chain:
        atoms = [a for a in atoms if a[0] == chain]
    sel = [a for a in atoms if a[3].upper() not in SOLVENT]
    atoms_xyz = np.array([[a[4], a[5], a[6]] for a in sel], dtype=float)
    vdw_arr = np.array([float(a[7]) for a in sel], dtype=float)

    # ---- 口袋心: **催化亲核原子**(Ser OG / Cys SG), 不是三联体 Cα 质心 --------
    # 2026-10-05 实测: 某酶的 Cα 质心 (-1.31,-8.00,12.28) 落在催化 His ND1
    # 原子**内部 1.22 Å**(clearance -0.33), 口袋球内取到的"最近游离格点"离它 4.9 Å,
    #  flood 出来只有 3 个格点(0.6 Å³)的假腔 → 后面"死端/无瓶颈"全是错种子造成的。
    seed_src = "explicit"
    if seed_xyz is None:
        # 球心口径的唯一来源(与 pocket_decomp 共用, 勿在此另写一遍)
        seed_xyz, seed_src = nucleophile_seed(pdb_path, atoms,
                                              fallback_centroid=atoms_xyz.mean(axis=0),
                                              chain=chain)
    seed_xyz = np.asarray(seed_xyz, dtype=float)

    # 原子预筛: 只有离口袋球"够近"的原子才可能占据域内格点(域 = 半径 radius 的
    # 球, 加上瓶颈扫描会把探针放大到 8 Å, 故上界 radius + vdW_max + 8)。
    # 预筛前的完整原子: 连体相扫描必须覆盖**全蛋白**(否则盒边界不在真实溶剂里),
    # 故把未预筛的副本留一份。
    sel_full = list(sel)
    atoms_xyz_full = np.array(atoms_xyz, dtype=float, copy=True)

    radius_hint = float(radius or POCKET_RADIUS_A)
    _keep = np.linalg.norm(atoms_xyz - np.asarray(seed_xyz), axis=1) <= radius_hint + 11.0
    sel = [a for a, k in zip(sel, _keep) if k]
    atoms_xyz = atoms_xyz[_keep]
    vdw_arr = vdw_arr[_keep]

    # 🔴 网格原点/尺寸必须取自**未预筛的**全蛋白原子。早期版本取自"按 radius 预筛
    # 后的原子", 于是 radius 8 → 12 时原点漂移 ~3 Å, 腔种子格点在物理空间上换了一
    # 个位置, 跨半径的 connected_to_bulk 判据会翻号(实测某 WT: R=8 连体相、
    # R=12 判死端, 而两次粗网格 flood 的到达格数完全相同 —— 纯属坐标错位)。
    lo = atoms_xyz_full.min(axis=0) - 4.0
    hi = np.maximum(atoms_xyz_full.max(axis=0) + 4.0, lo + 2.0 * (radius_hint + 14.0))
    shape = tuple(int(math.ceil((hi[d] - lo[d]) / STEP)) + 1 for d in range(3))
    origin = lo
    occ = _mark_occupied(shape, origin, STEP, sel)
    free = ~occ

    # ---- 分析域 = 口袋球: 游离且落在 seed 周围 radius 内 ----------------------
    # 为什么不用"全局埋深"判据(2026-10-05 连踩两坑, 最终弃用):
    #   坑1 从 occupied 起算膨胀 → 量的其实是"离最近原子多远", 蛋白外表面的溶剂
    #       (离原子 1~3 Å) 被 depth>=3 判成"埋在内部": 实测跑出 88568 Å³ 的
    #       "空腔"(蛋白本体才 ~2×10⁴ Å³), Δ 随网格抖动翻号。
    #   坑2 改从"外溶剂/体相"起算 → 通道接体相的酶口本该算埋深, 一并用连通性判
    #       又会把真通道丢掉, 二者不可调和。
    #   ⇒ 改用**空间域**: 直接把分析圈在口袋球内(球心=催化亲核原子, 半径 r)。
    radius = float(radius or POCKET_RADIUS_A)
    gi, gj, gk = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
                             indexing="ij")
    px = (gi * STEP + origin[0] - seed_xyz[0]) ** 2
    py = (gj * STEP + origin[1] - seed_xyz[1]) ** 2
    pz = (gk * STEP + origin[2] - seed_xyz[2]) ** 2
    dist2 = px + py + pz
    inside = dist2 <= radius * radius
    region = free & inside

    # 种子: 域内离球心最近的游离格点(亲核原子本身通常被原子占据)
    cand = np.argwhere(region)
    if len(cand) == 0:
        return {"error": f"口袋球内(R={radius}A)没有游离格点(探针可放置的空间)"
                         f" —— 该处被原子填实, 不是口袋"}
    dvals = dist2[cand[:, 0], cand[:, 1], cand[:, 2]]
    idx = tuple(int(v) for v in cand[int(np.argmin(dvals))])
    vis, order = _flood([idx], region, shape)

    # 出口判据: 域内靠外壳层(r >= R - 1.5*step)的游离格点, 能否与种子腔 6-连通。
    shell_free = free & (dist2 >= (radius - 1.5 * STEP) ** 2) & inside
    shell_idx = [tuple(int(v) for v in c) for c in np.argwhere(shell_free)]
    if shell_idx:
        ex_vis, _ = _flood(shell_idx[:2000], region, shape)
    else:
        ex_vis = np.zeros(shape, dtype=bool)
    entrance_mask = vis & ex_vis
    n_entrance = int(entrance_mask.sum())

    # ---- 严格"是否连体相": 全局(粗网格)连通性 --------------------------------
    lo_c = atoms_xyz_full.min(axis=0) - BULK_DOMAIN_PAD
    hi_c = atoms_xyz_full.max(axis=0) + BULK_DOMAIN_PAD
    connected_to_bulk, bulk_scope = False, ""
    n_boundary_free = 0
    CSTEP = 1.6
    for CSTEP in (1.6, 0.8):
        shape_c = tuple(int(math.ceil((hi_c[d] - lo_c[d]) / CSTEP)) + 1 for d in range(3))
        free_c = ~_mark_occupied(shape_c, lo_c, CSTEP, sel_full)
        _bd = np.zeros(shape_c, dtype=bool)
        _bd[0, :, :] = _bd[-1, :, :] = True
        _bd[:, 0, :] = _bd[:, -1, :] = True
        _bd[:, :, 0] = _bd[:, :, -1] = True
        _bnd = free_c & _bd
        n_boundary_free = int(_bnd.sum())
        if n_boundary_free == 0:
            bulk_scope = f"⚠ {CSTEP} Å 网格盒边界无游离起点(域未伸到溶剂), 该档不可判"
            continue
        ext_idx = [tuple(int(v) for v in c) for c in np.argwhere(_bnd)][:60000]
        ext_vis, _ = _flood(ext_idx, free_c, shape_c)
        # 判定点用**整个腔的足迹**(seed 周围 radius 内的所有粗格点), 不用单个种子格点。
        _gc = np.meshgrid(np.arange(shape_c[0]), np.arange(shape_c[1]),
                          np.arange(shape_c[2]), indexing="ij")
        _dc2 = ((_gc[0] * CSTEP + lo_c[0] - seed_xyz[0]) ** 2
                + (_gc[1] * CSTEP + lo_c[1] - seed_xyz[1]) ** 2
                + (_gc[2] * CSTEP + lo_c[2] - seed_xyz[2]) ** 2)
        cavity_coarse = free_c & (_dc2 <= radius * radius)
        if cavity_coarse.any() and bool((ext_vis & cavity_coarse).any()):
            connected_to_bulk, bulk_scope = True, f"全蛋白 {CSTEP} Å 网格:盒边界体相 → 腔"
            break
        if CSTEP == 1.6:
            bulk_scope = "1.6 Å 网格判不出连通(窄缝在分辨率以下), 已用 0.8 Å 复判"
    if not connected_to_bulk:
        bulk_scope += " | 域=全蛋白±10 Å, 若仍 False 则是真死端(不是域截断)"

    # ---- 通道剖面: 沿"腔心 → 体相"的一条自由路径逐点量余量 ------------------
    channel_profile = None
    if connected_to_bulk and ext_vis is not None:
        _goal = np.zeros(shape, dtype=bool)
        _half = CSTEP / 2.0
        for (ci_, cj_, ck_) in np.argwhere(ext_vis & _bnd):
            cx = lo_c[0] + (ci_ + 0.5) * CSTEP
            cy = lo_c[1] + (cj_ + 0.5) * CSTEP
            cz = lo_c[2] + (ck_ + 0.5) * CSTEP
            i0 = max(0, int((cx - _half - origin[0]) / STEP))
            i1 = min(shape[0] - 1, int((cx + _half - origin[0]) / STEP))
            j0 = max(0, int((cy - _half - origin[1]) / STEP))
            j1 = min(shape[1] - 1, int((cy + _half - origin[1]) / STEP))
            k0 = max(0, int((cz - _half - origin[2]) / STEP))
            k1 = min(shape[2] - 1, int((cz + _half - origin[2]) / STEP))
            _goal[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1] = True
        channel_profile = _channel_profile(shape, origin, STEP, sel, seed_xyz, _goal)

    # ---- 瓶颈: 能走通"腔→外壳"这条路的**球的最大半径**(膨胀扫描) ------------
    # 旧口径 = entrance 上 min(格点到最近原子表面)。这个数是废的: free 的定义
    # 就是"到原子表面 > probe", 所以任何通道的最小 clearance 恒等于 probe 附近,
    # 纯离散化噪声。物理正确的定义: 把探针半径逐档加大再判连通, 第一个断开的档位
    # 之前的最大值 = 能挤过去的球的最大半径。
    def _connects(probe_r: float) -> bool:
        free_r = ~_mark_occupied(shape, origin, STEP, sel, probe=probe_r)
        region_r = free_r & inside
        if not region_r[idx]:
            return False
        vis_r, _ = _flood([idx], region_r, shape)
        shell_r = inside & (dist2 >= (radius - 1.5 * STEP) ** 2) & free_r
        sh = [(int(c[0]), int(c[1]), int(c[2])) for c in np.argwhere(shell_r)]
        if not sh:
            return False
        ex_r, _ = _flood(sh[:2000], region_r, shape)
        return bool((vis_r & ex_r).any())

    # 膨胀扫描的语义(**别读反**): r 越大 = 原子按 vdW+r 膨胀得越大 = 越**难**走通。
    bottleneck = None
    bottleneck_capped = False
    if connected_to_bulk:      # 死端腔的"域内最窄处"与活性无关, 不报
        _scan = [round(float(v), 2)
                 for v in np.arange(PROBE, 8.0 + 1e-9, BOTTLENECK_STEP)]
        for _r in _scan:
            if not _connects(_r):
                _ok = [x for x in _scan if x < _r]
                bottleneck = _ok[-1] if _ok else _scan[0]
                bottleneck_capped = bool(_ok)
                break
        else:                                          # 全档都通 → 没有瓶颈
            bottleneck, bottleneck_capped = _scan[-1], False

    if n_entrance > 0:
        pts = np.argwhere(entrance_mask)
        rad = _radius_at(pts, shape, origin, STEP, atoms_xyz, vdw_arr)
        bpt = pts[int(np.argmin(rad))]
    else:
        bpt = None

    # 瓶颈残基(5.5 A 内)
    res = []
    if bpt is not None:
        bp_xyz = np.array(bpt, dtype=float) * STEP + np.array(origin)
        dmin = np.linalg.norm(atoms_xyz - bp_xyz, axis=1)
        near = [sel[i] for i in np.argsort(dmin)[:12] if dmin[i] <= 5.5]
        res = sorted({f"{a[3]}{a[1]}{a[0]}" for a in near})

    return {
        # 必须是**上面那个** connected_to_bulk(粗网格全局 flood), 不能写回成
        # n_entrance>0(域内触壳) —— 两者语义不同, 同名会把门控和结论搞反。
        "connected_to_bulk": bool(connected_to_bulk),
        "domain_reaches_shell": bool(n_entrance > 0),
        "n_entrance_grid": n_entrance,
        "n_shell_grid": int(shell_free.sum()),
        "exit_criterion": EXIT_CRITERION,
        "connected_to_bulk_scope": bulk_scope,
        "n_bulk_boundary_free_grid": n_boundary_free,
        "bulk_domain_scope": f"全蛋白包围盒 ±{BULK_DOMAIN_PAD} Å(与 radius 解耦)",
        "dead_end_frac": round(1.0 - n_entrance / max(int(vis.sum()), 1), 3),
        "pdb": os.path.basename(pdb_path),
        "chain": chain or "(auto)",
        "seed_A": [round(float(v), 2) for v in seed_xyz],
        "seed_source": seed_src,
        "step_A": STEP,
        "probe_A": PROBE,
        "sphere_radius_A": radius,
        "n_cavity_grid": int(vis.sum()),
        "cavity_volume_A3": round(float(vis.sum()) * STEP ** 3, 1),
        "free_volume_in_sphere_A3": round(float(region.sum()) * STEP ** 3, 1),
        "max_radius_in_cavity_A": round(float(np.sqrt(dist2[vis].max())), 2) if vis.any() else None,
        "bottleneck_radius_A": (round(bottleneck, 2) if bottleneck is not None else None),
        "bottleneck_capped": bottleneck_capped,
        "bottleneck_interval_A": (f"[{bottleneck}, {round(bottleneck + BOTTLENECK_STEP, 2)})"
                                  if bottleneck_capped and bottleneck is not None
                                  else ("无瓶颈(8 Å 球仍能通过)" if bottleneck is not None
                                        and bottleneck >= 8.0 - 1e-9
                                        else f"< {BOTTLENECK_STEP}(低于扫描下限 {PROBE})")),
        "bottleneck_xyz_A": ([round(float(v), 2) for v in bp_xyz] if bpt is not None else None),
        "bottleneck_residues": res,
        "channel_profile": channel_profile,
        "n_gridpoints": int(len(order)),
    }


def mutation_sites_from_pdbs(wt_pdb: str, mut_pdb: str,
                             chain: str = "") -> List[str]:
    """由两份 PDB 反推"突变位点"标签(如 `PHE101A`, 取 **WT 侧**的残基名)。

    为什么需要(2026-10-05, 某 Phe→Gly 突变体系实测): 喉部残基列表 `bottleneck_residues`
    **经常就含突变残基自身** —— WT 喉部 = HIS203A/ILE176A/**PHE101A**/PHE56A/SER102A,
    而突变体喉部 = 同一批去掉 PHE101A(该位已被改成 Gly, 不再是 Phe)。于是两侧
    列表必然不同, `compare_sides` 会据此外推"喉部位置搬家"; 可那只是"被突变的残基没了"
    的机械后果 —— 用序列不变的对称对照(Phe→Phe ctrl)实测, 两侧喉部坐标只差 **1.19 Å**,
    通道剖面更是逐项一致(47 步 / 28.2 Å / r_min 1.43 / 出口 6.43 vs 6.44)。
    ⇒ 调用方把突变位点传进 `compare_sides(..., mutated_residues=...)`, 让这类差异
    不再算作"搬家"的证据。
    """
    def _map(path: str) -> Dict[Tuple[str, int], str]:
        m: Dict[Tuple[str, int], str] = {}
        for a in _parse_pdb_heavy(path):
            if chain and a[0] != chain:
                continue
            m.setdefault((a[0], int(a[1])), str(a[3]).upper())
        return m

    wm, mm = _map(wt_pdb), _map(mut_pdb)
    out = []
    for key, wname in sorted(wm.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        mname = mm.get(key)
        if mname and mname != wname:
            out.append(f"{wname}{key[1]}{key[0]}")
    return out


def compare_sides(wt_res: Dict[str, Any], mut_res: Dict[str, Any],
                  mutated_residues: Sequence[str] = ()) -> Dict[str, Any]:
    """WT / MUT 两侧 analyze() 结果 → 可引用对比 + **自否证判词**。

    为什么这里必须是唯一判词出口(2026-10-05): "入口变宽"这句话在某个 Phe→Gly 突变
    体系上被推导过两次而两次都踩坑 —— ① 旧 min-clearance 口径两侧给同一个 ~2.8 Å(纯噪声);
    ② 修好后两侧瓶颈**同在 [1.4,1.6) 档**, 体积 +21% 却怎么也读不出"通道变宽"。
    体积变大 ≠ 入口变宽: 腔可以在催化中心附近"变胖"而喉部一格不动。所以把
    「什么话能写 / 什么话禁止写」写成一个函数, mutate QC 与 report 都只引用它,
    不许各自编判词。

    判据(任一成立即**禁止**"入口变宽/通道加宽"):
      * 两侧 bottleneck_interval_A 同档(真实瓶颈分辨率只有 BOTTLENECK_STEP=0.2 Å,
        跨档才叫变宽);
      * 或瓶颈半径 Δ < BOTTLENECK_NOISE_A(0.5 Å 噪声线)。
    此时唯一允许的通道叙事是「喉部位置搬家」(bottleneck_residues 变了)或
    「空腔扩大/缩小」(cavity Δ), 二者可同时成立。
    """
    def _side(tag):
        r = (wt_res if tag == "wt" else mut_res) or {}
        if "error" in r:
            return {"ok": False, "error": f"{tag}: {r['error']}"}
        return {"ok": True, "r": r}

    w, m = _side("wt"), _side("mut")
    if not w["ok"] or not m["ok"]:
        # 失败也必须带齐下游会读的键(报告侧一律按同一套字段取), 只是全部为否定。
        _e = " / ".join([x for x in (w.get("error"), m.get("error")) if x])
        return {"ok": False, "error": _e, "entry_wider": False,
                "same_cavity": False, "both_connected_to_bulk": False,
                "cavity_delta_A3": None, "bottleneck_delta_A": None,
                "banned_phrases": ["任何通道/入口结论(通道分析未产出结果)"],
                "allowed_phrases": [],
                "narrative": f"禁止写任何通道/入口结论(分析失败: {_e})"}

    wr, mr = w["r"], m["r"]
    cw, cm = wr.get("cavity_volume_A3"), mr.get("cavity_volume_A3")
    bw, bm = wr.get("bottleneck_radius_A"), mr.get("bottleneck_radius_A")
    iw, im = wr.get("bottleneck_interval_A"), mr.get("bottleneck_interval_A")
    delta_bn = (round(float(bm) - float(bw), 2)
                if isinstance(bw, (int, float)) and isinstance(bm, (int, float)) else None)
    delta_cav = (round(float(cm) - float(cw), 1)
                 if isinstance(cw, (int, float)) and isinstance(cm, (int, float)) else None)
    delta_pct = (round(100.0 * delta_cav / float(cw), 1)
                 if delta_cav is not None and isinstance(cw, (int, float)) and cw else None)

    both_open = bool(wr.get("connected_to_bulk")) and bool(mr.get("connected_to_bulk"))
    # 同一条腔: 两侧都连体相 + 腔心同源且几乎同位(同一口袋的口径判据)
    sw, sm = wr.get("seed_A"), mr.get("seed_A")
    seed_gap = (round(float(np.linalg.norm(np.array(sw, float) - np.array(sm, float))), 2)
                if sw and sm else None)
    same_cavity = bool(both_open and seed_gap is not None and seed_gap <= 1.0)

    moved = (set(wr.get("bottleneck_residues") or []) != set(mr.get("bottleneck_residues") or []))

    # ---- 突变位点污染守卫(2026-10-05) ------------------------------------
    # 两侧喉部残基列表的**对称差**若全都是"被突变的残基自身", 那这个差值是替换的
    # 机械后果(芳香残基换成 Gly 后该条目当然没了), 不是喉部搬了家的证据。
    # 标签匹配允许省略链 ID(调用方常只给 PHE101): 喉部标签形如 PHE101A, 去掉末位链
    # 名后应等于传入标签。
    _mutated = {str(x).strip().upper() for x in (mutated_residues or ()) if str(x).strip()}

    def _is_mut_site(label: str) -> bool:
        lu = str(label).strip().upper()
        if lu in _mutated:
            return True
        return bool(lu) and lu[-1].isalnum() and lu[:-1] in _mutated

    _wneck = set(wr.get("bottleneck_residues") or [])
    _mneck = set(mr.get("bottleneck_residues") or [])
    neck_diff = sorted((_wneck | _mneck) - (_wneck & _mneck))
    neck_artifact = bool(neck_diff) and all(_is_mut_site(r) for r in neck_diff)

    # ---- 喉部迁移距离 vs 重复噪声地板(2026-10-05 零假设对照实测) ------------
    # 同序列两次独立松弛的喉部就能差 3.88 Å, 所以"迁移了 X Å"必须先超过这个地板。
    _bw, _bm = wr.get("bottleneck_xyz_A"), mr.get("bottleneck_xyz_A")
    neck_move_A = None
    if _bw and _bm and len(_bw) == 3 and len(_bm) == 3:
        try:
            neck_move_A = round(float(np.linalg.norm(
                np.array(_bw, float) - np.array(_bm, float))), 2)
        except Exception:  # pragma: no cover - 坐标异常时退化为"不判断"
            neck_move_A = None
    neck_move_below_floor = (neck_move_A is not None
                             and neck_move_A < BOTTLENECK_REPLICATE_NOISE_A)
    moved_effective = bool(moved) and not neck_artifact and not neck_move_below_floor
    same_interval = (iw is not None and im is not None and iw == im)
    noise_band = (same_interval or (delta_bn is not None and delta_bn < BOTTLENECK_NOISE_A))
    entry_wider = (not noise_band) and delta_bn is not None and delta_bn >= BOTTLENECK_NOISE_A

    banned: List[str] = []
    allowed: List[str] = []
    # 是否需要"独立重复 ≥2 次(不同松弛路径)"才能把 allowed 里的描述性结论写进论文
    replicate_required = False
    if not both_open:
        banned.append("两侧必须都 connected_to_bulk 才算「同一条开口通路」;"
                      " 任一侧为死端腔时不得对通道宽度下任何结论")
    if noise_band:
        banned.append("入口变宽 / 通道加宽(两侧瓶颈同档 %s, Δ=%s Å < %s Å 噪声线)"
                      % (iw if same_interval else "或差 < 阈值", delta_bn, BOTTLENECK_NOISE_A))
    if entry_wider:
        allowed.append(f"入口变宽(瓶颈 {bw} → {bm} Å, Δ={delta_bn} Å ≥ {BOTTLENECK_NOISE_A} Å)")
    if moved_effective:
        allowed.append("喉部位置搬家(%s → %s)"
                       % (",".join(wr.get("bottleneck_residues") or []) or "无",
                          ",".join(mr.get("bottleneck_residues") or []) or "无"))
    if neck_move_below_floor:
        # 措辞里同样不复述"喉部位置搬家"四字(下游子串扫描会把"不构成…"误判成可写)
        # 用 f-string: 这句里有字面 %("0.8%"), 走 % 格式化会撞出 TypeError。
        banned.append(
            f"喉部位移类结论(两侧喉部坐标相距 {neck_move_A} Å < 重复噪声地板 "
            f"{BOTTLENECK_REPLICATE_NOISE_A} Å —— 零假设实测: 序列不变、空腔只差 0.8% "
            f"时喉部就能'搬' 7.09 Å, 与真突变效应测到的 1.33~7.28 Å 完全重叠, 不可分辨)")
    if neck_artifact:
        # 差集 = 突变位点自身: 明确写清"这不算搬家证据", 免得被读成"喉部仍在动"
        # 措辞里刻意**不复述**"喉部位置搬家"这四个字(连引号里也不写): 下游有用子串
        # 扫判词的检查, 一句"不构成'喉部位置搬家'"会被误判成"可写"。
        allowed.append("喉部残基列表的差异**仅来自突变位点自身**(%s)被替换 —— 属替换的"
                       "机械后果, 不得据此写喉部位移类结论" % ",".join(neck_diff))
    if moved_effective and same_interval:
        # 2026-10-05 二次 discover: 两侧瓶颈被 capped 在同一档时, "最窄处"其实是一条
        # ~26 Å 通路上的**近平台 argmin** —— 谁当 argmin 对坐标级扰动极敏感。同一对
        # 同一对 WT/突变体在两个独立批次上的实测: 迁移距离 8.90 Å vs 7.45 Å, 而
        # **同一个 WT 的喉部坐标批次间就差 3.89 Å、同一个突变体差 6.04 Å**; 甚至嗓子
        # 搬到哪儿都变了(一批落到 55–58/125/208 氧阴离子穴, 另一批只剩 ILE176A)。
        # 唯独 differentiate 掉的量(delta_cm/档位)是可复现的. ⇒ 此时只许说"构成最窄处
        # 的残基组合变了", 不许报坐标、迁移距离、也不许把终点当成定点结论。
        banned.append("喉部坐标 / 迁移距离 / \"搬到某结构域\"的定点结论(两侧瓶颈 capped 在"
                      "同一档 %s, 该档内条~26 Å 通路上近平台 argmin 对坐标级扰动极敏感;"
                      " 实测同一批对比的两次数值差可达数 Å 且终点不一致)" % iw)
        replicate_required = True
    if delta_cav is not None and abs(delta_cav) >= 10.0:
        allowed.append(f"空腔{'扩大' if delta_cav > 0 else '缩小'}({cw} → {cm} Å³, "
                       f"Δ={delta_cav:+} Å³" +
                       (f", {delta_pct:+.1f}%" if delta_pct is not None else "") + ")")
    if not both_open:
        allowed = ["死端腔: 只能报空腔体积变化, 不得对通路/入口宽度下任何结论"]
    elif not allowed:
        allowed = ["通道几何无实质变化(不得据此解释活性差异)"]

    def _short(s: str) -> str:
        # 括号里是证据; 塞进一句"禁止写"会堆成一串长句, 取主句即可
        return s.split("(")[0].rstrip()

    narrative = ("禁止写: " + "；".join(_short(b) for b in banned) + "。可写: "
                 + "；".join(allowed) if banned else "可写: " + "；".join(allowed))

    return {
        "ok": True,
        "radius_A": wr.get("sphere_radius_A"),
        "center_source": wr.get("seed_source"),
        "same_cavity": same_cavity,
        "seed_gap_A": seed_gap,
        "both_connected_to_bulk": both_open,
        "cavity_wt_A3": cw, "cavity_mut_A3": cm,
        "cavity_delta_A3": delta_cav, "cavity_delta_pct": delta_pct,
        "bottleneck_wt_A": bw, "bottleneck_mut_A": bm,
        "bottleneck_delta_A": delta_bn,
        "bottleneck_interval_wt": iw, "bottleneck_interval_mut": im,
        # bottleneck_moved = **有效**的搬家判定(已剔除"突变位点自身被替换"的机械差异);
        # _raw 是未剔除前的原始集合比较, 供排查用。
        "bottleneck_moved": moved_effective,
        "bottleneck_moved_raw": bool(moved),
        "bottleneck_move_artifact": neck_artifact,
        "bottleneck_diff_residues": neck_diff,
        "bottleneck_move_A": neck_move_A,
        "bottleneck_replicate_noise_A": BOTTLENECK_REPLICATE_NOISE_A,
        # True = 判词里已把"喉部坐标/迁移距离"列为不可引用, 必须先拿到 ≥2 个独立
        # 松弛模型重复、且两次的迁移终点一致, 才允许把搬家写成定点结论。
        "replicate_required": replicate_required,
        "bottleneck_residues_wt": wr.get("bottleneck_residues"),
        "bottleneck_residues_mut": mr.get("bottleneck_residues"),
        "entry_wider": bool(entry_wider),
        "banned_phrases": banned,
        "allowed_phrases": allowed,
        "narrative": narrative,
    }


def ligand_cross_section(path: str) -> Dict[str, Any]:
    """底物横截面: 主成分分析三个主轴尺度。

    为什么不用"最小轴厚度"当横截面: 聚合物/酯类底物是**长棒状**, 最小主轴尺度
    恒 ≈0(线形分子), 拿它算横截面会得到 0 半径、比值无穷 —— 方向性错误。
    对棒状底物, 真正决定能不能过瓶颈的是**垂直于管轴的那两个尺度**, 取其中
    较小的那个(最保守)即 effective_radius。紧凑配体上 t2≈t3, 该定义退化为
    "governing 半径", 依然正确。
    """
    xyz, els = _parse_xyz(path)
    if len(xyz) < 3:
        return {"error": "少于 3 个原子, 无法做 PCA"}
    cen = xyz.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov((xyz - cen).T))
    order = np.argsort(vals)[::-1]
    extents = []
    for oi in order:
        ext = (xyz - cen) @ vecs[:, int(oi)]
        extents.append(float(ext.max() - ext.min()))
    t1, t2, t3 = max(extents[0], 0.0), max(extents[1], 0.0), max(extents[2], 0.0)
    # 物理下限: 哪怕原子共线/共面, 真实分子截面仍由原子 vdW 尺寸撑开
    # (12 个碳排成直线, 截面半径仍是 ~1.7 A, 不是 0)。
    vdw_floor = max(float(VDW.get(e.upper(), 1.7)) for e in els)
    if t2 / 2.0 < vdw_floor:
        eff, src = vdw_floor, "vdW限厚"
    else:
        eff, src = t2 / 2.0, "几何截面"
    # governing 直径也必须带 vdW 下限: 退化直链(棒状底物)t2→0, 但真实横截面直径是
    # 2×vdW(12 个碳排成直线, 截面仍是 3.4 Å)。不加下限会出现"有效半径 1.7 而
    # governing 直径 0"的自相矛盾输出(2026-10-05 test_unit_pocket_tunnel 抓到)。
    return {
        "ligand": os.path.basename(path),
        "n_heavy": int(len(xyz)),
        "max_pair_dist_A": round(float(np.max(np.linalg.norm(xyz[:, None] - xyz[None], axis=-1))), 2),
        "extent_long_axis_A": round(t1, 2),
        "extent_mid_axis_A": round(t2, 2),
        "extent_short_axis_A": round(t3, 2),
        "vdw_floor_A": round(vdw_floor, 2),
        "cross_section_limited_by": src,
        "effective_radius_A": round(float(eff), 2),
        "governing_diameter_A": round(float(max(t2, t3, 2.0 * vdw_floor)), 2),
    }
