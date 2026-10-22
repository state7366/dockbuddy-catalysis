# -*- coding: utf-8 -*-
"""dockbuddy_catalysis —— 酶催化机理分析(从 DockBuddy libdock 抽出的解耦版本)。

纯下游分析(依赖 dockbuddy_dock 的对接结果): 去酰化水 / 口袋通道 / 空腔分解 /
产物截面 / 芳香滞留 / 构象熵 / 突变综合报告。
"""
from .deacylation import analyze_side
from .pocket_tunnel import analyze, resolve_receptor
from .pocket_decomp import cavity_residue_decomposition
from .product_egress import analyze_egress
from .retention import retention_dock_set
from .conformation_entropy import configurational_entropy_difference
from .mutation_report import assemble_mutation_report

__version__ = "0.1.0"

__all__ = ["analyze_side", "analyze", "resolve_receptor",
           "cavity_residue_decomposition", "analyze_egress",
           "retention_dock_set", "configurational_entropy_difference",
           "assemble_mutation_report", "__version__"]
