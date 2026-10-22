# -*- coding: utf-8 -*-
"""deacylation.py —— 去酰化水分子可及性(**非 MD**的 kcat 机理计算, 2026-10-05)
====================================================================================
动机
----
设计源头案例(某单残基突变)的叙事把活性来源归到 **kcat**(酰基酶水解 / 产物释放), 而不是 KM。但在此之前
这两步**一次都没有被算过** —— 已有的证据只有"空腔变大"和"取向占比", 讲的其实都还是
"底物怎么摆", 不属于"酰基酶形成之后"的事。要支撑 kcat 归因, 至少需要回答:

> 酰基酶形成后, **水解它的那个水分子有没有地方待?**

丝氨酸水解酶的去酰化要求一个水分子同时满足三件事: 处在羰基碳的亲核攻击距离上、
以 Bürgi–Dunitz 角(~107°)逼近、并且被 His NE2 活化(作为广义碱夺质子)。本模块用
**纯几何网格搜索**回答"这样的位置在 WT / 突变体里各有几个", 不需要跑 MD。

判据(全部可调, 写进产物的 params 里)
------------------------------------
- 亲核距离: ``WATER_NUC_MIN``~``WATER_NUC_MAX``(缺省 2.6~3.4 Å) —— 短于 2.6 已是
  共价/半键, 长于 3.4 不构成攻击构型。
- 攻击角: Bürgi–Dunitz, 顶点在羰基碳, 目标 107°±``BD_ANGLE_TOL``(缺省 20°)。
- 活化: 水分子氧到 His NE2 ≤ ``HIS_WATER_MAX``(缺省 3.5 Å, 通用氢键上限)。
- 可容纳: 水分子氧中心到**蛋白重原子 vdW 表面**的余隙 ≥ ``WATER_CLEARANCE``
  (缺省 1.4 Å = 水探针半径)。

⚠️ 两条容易写反 / 静默出错的口径
--------------------------------
1. **障碍集合只算蛋白, 不算配体**: 水要贴到配体的羰基碳上(2.6~3.4 Å), 若把配体
   自身算成障碍, 任何候选点都会 clash ⇒ 恒为 0 个位点, 且看起来"很有道理"。
2. **亲核 Ser OG 保留在障碍集合里**: 酰基酶状态下 OG 与羰基碳是共价相连的, 水只能
   从**另一侧**进攻; 把 OG 排除会让搜索跑到 OG 那一侧, 得到物理上不存在的位点。
   用非共价对接 pose 近似时, OG 仍在原处 ⇒ 保留它正好起"挡住错误一侧"的作用。

局限(写进产物, 防止下游误用)
----------------------------
- 这是**几何可及性**, 不是水分子占据率/停留时间 —— 不能写"水分子更稳定地结合"。
- 起点是**非共价对接 pose**, 而真实底物在酰基酶状态下位置略有不同(共价连接会
  把羰基拉向 OG ~1.35 Å)。故本量只做**两侧相对比较**, 绝对值不引用。
- 需 ≥2 个独立松弛模型 / ≥2 个独立对接批次重复才可写进结论(与喉部、残基分解同律)。
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from dockbuddy_dock.orientation import (NUC_TO_SCISSILE_C_MAX, find_scissile_sites,
                          pose_orientation)
from dockbuddy_dock.pocketgeom import _parse_pdb_heavy
from dockbuddy_dock.triad import detect_catalytic_triad
# 阈值单一事实源(2026-10-07 证据门槛注册表)
from dockbuddy_dock.evidence import THRESHOLDS as _EV, vol_delta_verdict

WATER_CLEARANCE_A = _EV["water_clearance_A"]
WATER_NUC_MIN_A = _EV["water_nuc_min_A"]
WATER_NUC_MAX_A = _EV["water_nuc_max_A"]
BD_ANGLE_TARGET = _EV["bd_angle_target"]
BD_ANGLE_TOL = _EV["bd_angle_tol"]
HIS_WATER_MAX_A = _EV["his_water_max_A"]
SEARCH_RADIUS_A = _EV["search_radius_A"]
SEARCH_STEP_A = _EV["search_step_A"]

# 噪声地板按**一个水分子的体积**取: 4/3·π·1.4³ ≈ 11.5 Å³。合格区域体积差小于
# 一个水分子的体积, 就没有"多/少一个水位点"的物理意义。取值见
# libdock.evidence.THRESHOLDS["deacyl_noise_vol_A3"]。
# (不用簇数当地板: 簇数对拓扑碎裂极敏感 —— 合成测试里把配体混进障碍集合, 簇数
#  反而从 1 涨到 20, 是同一片区域被切开, 不是位点变多。)
DEACYL_NOISE_VOL_A3 = _EV["deacyl_noise_vol_A3"]


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """两向量夹角(度)。零向量退化为 180°(不通过判据, 不会误放行)。"""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 180.0
    c = float(np.dot(a, b) / (na * nb))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def water_sites_for_site(pose_atoms: Sequence[Sequence[Any]],
                         site: Dict[str, Any],
                         rec_xyz: np.ndarray, rec_vdw: np.ndarray,
                         his_xyz: Optional[np.ndarray],
                         clearance: float = WATER_CLEARANCE_A,
                         step: float = SEARCH_STEP_A,
                         radius: float = SEARCH_RADIUS_A) -> Dict[str, Any]:
    """单个 scissile 位点周围的"合格催化水位点"搜索 → 连通簇。

    pose_atoms = (x, y, z, elem, name, resn, resi) 序列(parse_pdbqt_poses 的单 pose)。
    site = find_scissile_sites 的元素(需 index / carbonyl_O)。
    """
    c_xyz = np.asarray(pose_atoms[site["index"]][:3], dtype=float)
    o_xyz = np.asarray(pose_atoms[site.get("carbonyl_O", site["index"])][:3], dtype=float)

    # 预筛: 只有离攻击中心够近的蛋白原子才可能挡住候选点
    keep = np.linalg.norm(rec_xyz - c_xyz, axis=1) <= radius + 3.0
    rx, rv = rec_xyz[keep], rec_vdw[keep]
    # 预筛可能把**全部**原子筛掉(蛋白离攻击中心很远), 此时 gap 恒 +inf(不挡水),
    # 不能让 d_rec.min(axis=1) 撞 "zero-size array to reduction" 而崩。
    no_obstacle = (len(rx) == 0)

    n = int(radius / step)
    axes = np.arange(-n, n + 1, dtype=float) * step
    gx, gy, gz = np.meshgrid(axes, axes, axes, indexing="ij")
    m = (gx ** 2 + gy ** 2 + gz ** 2) <= radius * radius
    idxs = np.argwhere(m)
    pts = np.stack([gx[m], gy[m], gz[m]], axis=1) + c_xyz

    ok = np.zeros(len(pts), dtype=bool)
    best_gap = np.full(len(pts), -1e9, dtype=float)
    block = 512
    for i in range(0, len(pts), block):
        blk = pts[i:i + block]
        if no_obstacle:
            gap = np.full(len(blk), np.inf)
        else:
            d_rec = np.linalg.norm(blk[:, None, :] - rx[None, :, :], axis=2) - rv[None, :]
            gap = d_rec.min(axis=1)
        dc = np.linalg.norm(blk - c_xyz, axis=1)
        v1 = blk - c_xyz
        v2 = (o_xyz - c_xyz)[None, :]
        cosang = (v1 * v2).sum(axis=1) / np.maximum(dc * np.linalg.norm(v2), 1e-9)
        ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        his_ok = np.ones(len(blk), dtype=bool)
        if his_xyz is not None:
            his_ok = np.linalg.norm(blk - np.asarray(his_xyz, float), axis=1) <= HIS_WATER_MAX_A
        good = ((dc >= WATER_NUC_MIN_A) & (dc <= WATER_NUC_MAX_A)
                & (np.abs(ang - BD_ANGLE_TARGET) <= BD_ANGLE_TOL)
                & (gap >= clearance) & his_ok)
        ok[i:i + block] = good
        best_gap[i:i + block] = gap

    # 连通聚类(6-邻域 BFS): 相邻的合格格点属于同一"可放置区域", 不能重复计数
    shape = tuple(int(v) + 1 for v in idxs.max(axis=0))
    grid = np.zeros(shape, dtype=bool)
    grid[tuple(idxs.T)] = ok
    # best_gap 是**按 pts 一维顺序**排列的, 必须先摊回 3D 网格才能用网格索引取
    # (直接 best_gap[tuple(idx.T)] 会撞 "too many indices for array")
    gap_grid = np.zeros(shape, dtype=float)
    gap_grid[tuple(idxs.T)] = best_gap
    seen = np.zeros(shape, dtype=bool)
    clusters: List[Dict[str, Any]] = []
    for p in idxs[ok]:
        p = tuple(int(v) for v in p)
        if seen[p]:
            continue
        q = deque([p])
        seen[p] = True
        comp = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                nb = (cur[0] + d[0], cur[1] + d[1], cur[2] + d[2])
                if (0 <= nb[0] < shape[0] and 0 <= nb[1] < shape[1] and 0 <= nb[2] < shape[2]
                        and grid[nb] and not seen[nb]):
                    seen[nb] = True
                    q.append(nb)
        comp_arr = np.array(comp, dtype=int)
        origin = c_xyz - np.array([n, n, n], dtype=float) * step
        arr = comp_arr * step + origin
        # 簇代表点 = 余隙最大者(最不可能 clash)
        gaps = gap_grid[tuple(comp_arr.T)]
        j = int(np.argmax(gaps))
        rep = arr[j]
        clusters.append({
            "n_grid": int(len(comp)),
            "volume_A3": round(float(len(comp)) * step ** 3, 2),
            "rep_xyz_A": [round(float(v), 2) for v in rep],
            "nuc_dist_A": round(float(np.linalg.norm(rep - c_xyz)), 2),
            "attack_angle_deg": round(_angle_deg(rep - c_xyz, o_xyz - c_xyz), 1),
            "his_dist_A": (round(float(np.linalg.norm(rep - np.asarray(his_xyz, float))), 2)
                           if his_xyz is not None else None),
            "clearance_A": round(float(gaps[j]), 2),
        })
    clusters.sort(key=lambda d: (-d["volume_A3"], -d["clearance_A"]))
    return {"n_clusters": len(clusters), "clusters": clusters,
            "n_qualified_grid": int(ok.sum()),
            "qualified_volume_A3": round(float(ok.sum()) * step ** 3, 2),
            "scissile_index": int(site["index"]),
            "carbonyl_xyz_A": [round(float(v), 2) for v in c_xyz]}


def analyze_side(rec_pdb: str, poses_pdbqt: str, chain: str = "",
                 max_poses: int = 9,
                 nuc_max: float = NUC_TO_SCISSILE_C_MAX) -> Dict[str, Any]:
    """单侧(WT 或突变体)的去酰化水可及性统计。

    poses_pdbqt = Vina 输出(parse_pdbqt_poses 可读)。只统计**酯型** scissile 位点
    (羧酸端不算 —— 产物端不形成酰基酶, 见 orientation.find_scissile_sites)。
    """
    from dockbuddy_dock.poses import parse_pdbqt_poses

    atoms = _parse_pdb_heavy(rec_pdb)
    if chain:
        atoms = [a for a in atoms if a[0] == chain]
    if not atoms:
        return {"error": f"{rec_pdb} 解析到 0 重原子(检查 --chain / 格式收敛)"}
    rec_xyz = np.array([[a[4], a[5], a[6]] for a in atoms], dtype=float)
    rec_vdw = np.array([float(a[7]) for a in atoms], dtype=float)

    tri = detect_catalytic_triad(rec_pdb, chain_filter=chain)
    his_xyz, nuc_xyz = None, None
    if tri:
        for a in atoms:
            if (a[1] == tri["his"]["resnum"] and a[0] == tri["his"]["chain"]
                    and a[2].upper() == "NE2"):
                his_xyz = np.array([a[4], a[5], a[6]], dtype=float)
            if (a[1] == tri["nuc"]["resnum"] and a[0] == tri["nuc"]["chain"]
                    and a[2].upper() == (tri["nuc"].get("atom") or "OG").upper()):
                nuc_xyz = np.array([a[4], a[5], a[6]], dtype=float)

    poses = parse_pdbqt_poses(poses_pdbqt)
    if not poses:
        return {"error": f"{poses_pdbqt} 未解析到 pose"}
    poses = poses[:int(max_poses)]

    per_pose = []
    for pi, (energy, p_atoms) in enumerate(poses):
        sites = find_scissile_sites(p_atoms)
        ester = [s for s in sites if not s["acid"]]
        oriented, nearest = False, None
        if ester and nuc_xyz is not None:
            od = pose_orientation(p_atoms, sites, [tuple(nuc_xyz)],
                                  [tuple(his_xyz)] if his_xyz is not None else [])
            oriented = bool(od.get("oriented"))
            nearest = od.get("nearest_A")
        n_cl = 0
        vol = 0.0
        best = None
        for s in ester:
            r = water_sites_for_site(p_atoms, s, rec_xyz, rec_vdw, his_xyz)
            n_cl += int(r["n_clusters"])
            vol += float(r["qualified_volume_A3"])
            if r["clusters"] and (best is None
                                  or r["clusters"][0]["clearance_A"] > best["clearance_A"]):
                best = dict(r["clusters"][0], scissile_index=r["scissile_index"])
        # 缺 REMARK VINA RESULT 的 model 能量是 None(poses.py 的哨兵), 不能 float()
        per_pose.append({"pose_index": pi,
                         "energy": (round(float(energy), 3) if energy is not None else None),
                         "oriented": oriented, "nearest_A": nearest,
                         "n_ester_sites": len(ester), "n_clusters": int(n_cl),
                         "qualified_volume_A3": round(vol, 2), "best": best})
    n_with = sum(1 for p in per_pose if p["qualified_volume_A3"] > 0)
    n_oriented = sum(1 for p in per_pose if p["oriented"])
    total = sum(p["n_clusters"] for p in per_pose)
    # ⚠️ 口径(2026-10-05 实测后修正): "每 oriented pose 的可及空间"的分子必须
    # **只对 oriented pose 求和** —— 首版拿全部 pose 的体积除以 oriented 数, 分子
    # 分母不匹配。实测某双酯底物: 全 pose 3.08 A3 里 oriented 贡献为 0(水空间全在
    # 非 oriented pose 上), 旧口径会把它算成 0.77 A3/oriented 的假阳性。
    total_vol_oriented = round(sum(p["qualified_volume_A3"] for p in per_pose
                                   if p["oriented"]), 2)
    vol_per_oriented = (round(total_vol_oriented / n_oriented, 2) if n_oriented else None)
    return {
        "ok": True,
        "rec": str(rec_pdb), "poses": str(poses_pdbqt),
        "n_poses": len(per_pose), "n_oriented": n_oriented,
        "n_poses_with_water": n_with,
        "frac_poses_with_water": round(n_with / max(1, len(per_pose)), 3),
        "total_clusters": int(total),
        # 主指标 = 合格区域**体积**(簇数对拓扑碎裂敏感, 只作辅助)。
        # qualified_volume_A3 = 全部 pose 的合计(参考); oriented 版才是判词用的主口径。
        "qualified_volume_A3": round(sum(p["qualified_volume_A3"] for p in per_pose), 2),
        "qualified_volume_oriented_A3": total_vol_oriented,
        "volume_per_oriented_pose_A3": vol_per_oriented,
        "clusters_per_oriented_pose": (round(total / n_oriented, 2) if n_oriented else None),
        "per_pose": per_pose,
        "params": {"water_clearance_A": WATER_CLEARANCE_A,
                   "nuc_range_A": [WATER_NUC_MIN_A, WATER_NUC_MAX_A],
                   "bd_angle_deg": [BD_ANGLE_TARGET - BD_ANGLE_TOL,
                                    BD_ANGLE_TARGET + BD_ANGLE_TOL],
                   "his_water_max_A": HIS_WATER_MAX_A,
                   "search_radius_A": SEARCH_RADIUS_A, "step_A": SEARCH_STEP_A},
    }


def compare_sides_deacylation(wt: Dict[str, Any], mut: Dict[str, Any]) -> Dict[str, Any]:
    """两侧对比 + 自否证判词(与 pocket_tunnel.compare_sides 同律: 判词只出一处)。"""
    if wt.get("error") or mut.get("error"):
        return {"error": wt.get("error") or mut.get("error")}
    fw = float(wt.get("frac_poses_with_water", 0.0))
    fm = float(mut.get("frac_poses_with_water", 0.0))
    vw = float(wt.get("volume_per_oriented_pose_A3") or 0.0)
    vm = float(mut.get("volume_per_oriented_pose_A3") or 0.0)
    cw = int(wt.get("total_clusters", 0))
    cm = int(mut.get("total_clusters", 0))
    d_frac = round(fm - fw, 3)
    d_vol = round(vm - vw, 2)
    d_cl = int(cm - cw)
    banned: List[str] = []
    allowed: List[str] = ["两侧**相对**比较下的可及空间体积差(同协议同 pose 数)"]
    allowed.append("'是否存在满足去酰化几何的水位点'这一**定性**结论")
    if not vol_delta_verdict(vw, vm, floor=DEACYL_NOISE_VOL_A3)["passes"]:
        banned.append(f"可及空间的定量差(Δ体积 {d_vol:+.1f} Å³ 小于一个水分子的体积 "
                      f"{DEACYL_NOISE_VOL_A3} Å³ —— 没有'多/少一个水位点'的物理意义; "
                      f"真值需 ≥2 个独立对接批次/松弛模型重复)")
    banned.append("水分子的结合强度 / 停留时间 / 占据率(纯几何判据测不到动力学量)")
    banned.append("绝对位点坐标当结论(起点是非共价对接 pose, 酰基酶状态会拉近距离)")
    return {
        "ok": True,
        "frac_wt": fw, "frac_mut": fm, "frac_delta": d_frac,
        "vol_per_oriented_wt_A3": vw, "vol_per_oriented_mut_A3": vm, "vol_delta_A3": d_vol,
        "clusters_wt": cw, "clusters_mut": cm, "clusters_delta": d_cl,
        "n_poses_wt": wt.get("n_poses"), "n_poses_mut": mut.get("n_poses"),
        "n_oriented_wt": wt.get("n_oriented"), "n_oriented_mut": mut.get("n_oriented"),
        "replicate_required": True,
        "banned_phrases": banned, "allowed_phrases": allowed,
        "narrative": (f"可及去酰化水空间: 每个 oriented pose WT {vw:.1f} Å³ → 突变体 "
                      f"{vm:.1f} Å³ (Δ {d_vol:+.1f} Å³); 有位点 pose 占比 {fw:.0%} → "
                      f"{fm:.0%} (Δ {d_frac:+.3f}); 簇数 {cw}→{cm} 仅作辅助"),
    }
