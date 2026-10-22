# dockbuddy-catalysis

酶催化机理分析，从 DockBuddy `libdock` 抽出的解耦版本。

纯下游分析（依赖 `dockbuddy-dock` 的对接结果）：

| 模块 | 分析 |
|---|---|
| `deacylation` | 去酰化水分子可及性（kcat 机理） |
| `pocket_tunnel` | 口袋 + 入口通道几何 |
| `pocket_decomp` | 空腔增益逐残基分解 |
| `product_egress` | 产物释放可及性 |
| `retention` | 芳香产物滞留 |
| `conformation_entropy` | 构象熵差（准谐振） |
| `mutation_report` | 突变综合报告聚合 |

## 安装

```bash
pip install dockbuddy-catalysis \
  --index-url https://pypi.cnb.cool/vcinex/bioinfo/-/packages/simple
```

## 使用

```python
from dockbuddy_catalysis import analyze_side, analyze, cavity_residue_decomposition
```
