# 论文图表 — 生成与使用说明

为 `docs/ACL_PAPER_ZH.md`（绑定而非匹配：多跳检索中的查询槽因果性）绘制的全部论文图。
每个图提供 **矢量 PDF**（LaTeX 嵌入用）与 **300 DPI PNG**（Markdown/预览用）。

## 绘图规范依据

三个参考仓库已下载到 `../figure_skills/`，本套图遵循其核心规范：

| 仓库 | 采纳的规范 |
|---|---|
| `0xE1337/thesis-figure-skill` | 语义化配色（用户指定 12 色盘 α=C7，按图轮换主色：#8074C8 紫罗兰=核心方法/绑定(正向锚)、#A8CBDF 粉蓝=基线/整句、#7895C1 长春花蓝=dense/Self-Ask/输入、#EF8B67 鲑橙=BM25、#B54764 玫红=IRCoT、#F0C284 杏黄=规划/HyDE、#E3625D 珊瑚红=缺失/负向、#992224 深红=whole-query 反事实线、#D6EFF4 冰蓝=上限柱填充/恢复区、#F2FAFC→#8074C8 紫罗兰色带=图5(c)、#F5EBAE/#F7FBC9 奶油=规划填充/参照框；系列对比辅以明度差+线型冗余）；浅色底、整图 ≤6 色+1 accent；灰度打印安全（相邻系列明度有差）；数字必须有出处、视觉编码 ∝ 数值；嵌入真实数据小图作为"证据面板"（Figure 1 右下） |
| `K-Dense-AI/scientific-agent-skills` (scientific-visualization) | 出版级 matplotlib 样式：无衬线正文 + 每图统一字号阶梯、去顶右刺、浅网格、矢量 PDF、300 DPI、色盲可辨的主色对 |
| `BinSpa/awesome-scientific-figures` | "paper-figure" 工作流：300 DPI、矢量、图内不放总标题（标题由 LaTeX caption 承担，面板标签 (a)(b)(c) 保留）；缺失数据不伪造（图 6 以"—"标示） |

## 文件清单

| 文件 | 内容 | 论文位置 | 数据出处 |
|---|---|---|---|
| `fig1_method_overview.(pdf\|png)` | 主图：CovBind 方法总览 + 查询槽反事实 | §4 | §6.1/6.2 |
| `fig1_method_overview.tex` | 主图 TikZ 源码（LaTeX 嵌入等价版） | §4 | 同上 |
| `fig2_queryslot_counterfactual.(pdf\|png)` | (a) 总体 EM 反事实 (b) 跳数分层 dumbbell | §6.1 | §6.1 与 6.7 跳数分层表 |
| `fig3_hop_reachability.(pdf\|png)` | 逐跳 Recall@50 恢复曲线（dense/BM25 × 整句/绑定） | §6.2 | §6.2 表 |
| `fig4_reach_efficiency.(pdf\|png)` | 候选可达性(上限) vs 覆盖@3(实际) + η 标注 | §6.3 | §6.3 表 |
| `fig5_coverage_bottleneck.(pdf\|png)` | (a) 覆盖选证器 (b) 证据完整度→EM (c) 训练×推理 query 矩阵 | §6.4/6.5 | §6.4/6.5 表 |
| `fig6_end_to_end.(pdf\|png)` | 跨数据集端到端方法比较 + whole-query 反事实 | §6.7 | §6.6/6.7 表 |
| `fig7_applicability.(pdf\|png)` | 适用性判据：绑定缺口 B × 预算 → ΔEM | §6.8 | §6.8 表 |
| `fig8_method_comparison_full.(pdf\|png)` | 多方法端到端比较(表 6.6 全方法 + whole-query 反事实) | §6.6 | §6.6 表 |
| `fig9_mechanism_crossreach.(pdf\|png)` | (a) 2Wiki 50 题机制复验 (b) whole-query BM25 可达性曲线 | §6.7 | §6.7 文本 + final_manifest.json |
| `fig10_binding_quality_singlehop.(pdf\|png)` | (a) 金标/预测绑定质量 (b) 单跳控制 | §6.9/7.1 | §6.9/7.1 文本 |
| `fig11_negative_results.(pdf\|png)` | (a) 改写消除率 9.5% (b) 答案似然边际 | §6.10 | §6.10 列表 |
| `figures.tex` | 全部图的 LaTeX `figure/figure*` 环境与中文图注，可直接 `\input` | — | — |
| `figstyle.py` | 统一样式模块（配色/字体/尺寸/导出） | — | — |
| `fig1_method.py`, `figs_data.py`, `figs_more.py` | 生成脚本（`python3 fig1_method.py`; `python3 figs_data.py [2-7]`） | — | — |
| `check_pdf.py`, `check_edge.py` | 几何质检：文本重叠 / 出页边 / 文字跨框边界（pymupdf） | — | — |
| `fonts/NotoSansSC-*.ttf` | 中文字体（自动注册到 matplotlib） | — | — |

## 重新生成

```bash
cd figures
python3 fig1_method.py        # Figure 1
python3 figs_data.py          # Figure 2-7（可传参 2..7 单独重绘）
python3 check_pdf.py fig1_method_overview.pdf   # 质检
```

## LaTeX 使用

```latex
\usepackage{graphicx}
\input{figures/figures.tex}   % 或逐个复制 figure 环境
```

主图另有 TikZ 源码 `fig1_method_overview.tex`（`xelatex` 编译，需 `ctex`），
便于在正文中按投稿模板微调字号与配色。

## 质量闸门状态（如实声明）

- ✅ 数字语义核对：全部数值逐一取自 `docs/ACL_PAPER_ZH.md` 与 `results/final_manifest.json`，未使用估计值；
- ✅ 几何自检：`check_pdf.py`（跨文本块重叠、出页边）与 `check_edge.py`（文字跨越容器边界）全部通过（0 blocker；"OUT-OF-PAGE" 均为 tight-bbox 裁剪前坐标，最终导出已包含）；
- ⚠️ 独立多镜头视觉 gate 未能运行：当前环境模型不具备图像输入，无法对渲染 PNG 做对抗式目检（子代理同样无图像输入）。**请人工过目 7 张 PNG**——重点看：Figure 1 各框文字是否拥挤、Figure 2(b) 右侧 CI 标注、Figure 6 红色反事实标注是否遮挡。
