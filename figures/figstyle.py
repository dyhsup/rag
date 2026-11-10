# -*- coding: utf-8 -*-
"""
统一学术绘图样式 —— 遵循 thesis-figure-skill 配色语义 + scientific-visualization 规范
- 语义配色: 绿=核心方法(CovBind/绑定), 灰=基线(整句问题), 蓝=dense系, 橙=BM25/数据流,
  紫=决策/规划, 红=负向/退化;  全图 <=6 色, 1 个 accent
- 灰度打印安全: 相邻系列明度有差
- 300 DPI, 矢量 PDF + PNG 双输出; 图内不放总标题(标题由 LaTeX caption 承担)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ 字体
_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
for _f in ("NotoSansSC-Regular.ttf", "NotoSansSC-Bold.ttf"):
    _p = os.path.join(_FONT_DIR, _f)
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)

def apply_style():
    plt.rcParams.update({
        "font.family": ["Noto Sans SC", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.45,
        "grid.alpha": 0.85,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # TrueType 嵌入, 避免位图字体
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })

# ------------------------------------------------------------------ 学术配色 (源自 tikz-colors.md 学术方案, 降饱和微调)
C = {
    # —— 用户 12 色盘 (α=C7): 紫罗兰=方法锚, 粉蓝=基线, 深红=反事实/负向 ——
    "green":     "#8074C8",   # 核心方法 / 绑定 (紫罗兰 violet)
    "greenFill": "#E6E1F5",
    "greenDeep": "#5D53A3",
    "grey":      "#A8CBDF",   # 基线 / 整句问题 (粉蓝 powder, 安静)
    "greyFill":  "#EDF3F7",
    "greyDark":  "#6E8CA8",
    "blue":      "#7895C1",   # dense / Self-Ask / 输入 (长春花蓝 periwinkle)
    "blueFill":  "#E2EAF5",
    "orange":    "#EF8B67",   # BM25 / 数据流 (鲑橙 salmon)
    "orangeFill":"#FAE8DE",
    "purple":    "#F0C284",   # 规划 / 决策 (杏黄 apricot)
    "purpleFill":"#F5EBAE",
    "red":       "#E3625D",   # 负向 / 缺失 (珊瑚红 coral)
    "redFill":   "#F9DCDA",
    "rose":      "#B54764",   # IRCoT / 玫红系列
    "crimson":   "#992224",   # 反事实线 / 强负向 (深红)
    "gold":      "#F5EBAE",   # 奶油黄 (高亮填充)
    "ice":       "#D6EFF4",   # 冰蓝 (上限柱填充/恢复区)
    "mist":      "#F2FAFC",   # 近白冰 (色带起点)
    "cream":     "#F7FBC9",   # 极浅奶油 (参照框填充)
    "ink":       "#2B2830",
}

# ACL 版面: 单栏 \columnwidth≈3.3in, 双栏 \textwidth≈6.8in
COL1 = 3.35
COL2 = 6.9

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"[saved] {name}.pdf / {name}.png")
