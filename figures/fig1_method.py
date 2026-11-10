# -*- coding: utf-8 -*-
"""Figure 1 —— CovBind 方法总览(主图, 双栏):
左列: 问题→规划→需求程序; 中部虚线框=查询槽单变量反事实(灰=整句问题对照, 绿=绑定需求);
右下嵌入真实 EM 小图作为证据面板。数字来源: docs/ACL_PAPER_ZH.md §6.1/6.2 与 final_manifest.json
图契约:
  核心结论: 只把检索查询槽从整句问题换成绑定需求, EM 即从 0.170 升至 0.425
  图原型:   左编译 + 双分支反事实 + 右证据面板
  证据层级: hero=双分支候选池对比(证据缺失 vs 命中); 支撑=EM 小图与图例协议
  能不能更少: 去掉图例协议行则读者不懂 β; 去掉小图则结论无数字 —— 都保留
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from figstyle import apply_style, C, save, OUT

apply_style()

fig = plt.figure(figsize=(6.9, 3.9))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ax.grid(False)

SER = dict(fontfamily="DejaVu Serif")


def box(x, y, w, h, fc, ec, lw=0.9, r=1.6, ls="-", z=2):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    ax.add_patch(b)
    return b


def arrow(p, q, color, lw=1.1, ls="-", z=3, sh=4):
    a = FancyArrowPatch(p, q, arrowstyle="-|>,head_length=5,head_width=3.2",
                        shrinkA=sh, shrinkB=sh, color=color, lw=lw, ls=ls, zorder=z)
    ax.add_patch(a)


def txt(x, y, s, size=7.0, color=C["ink"], weight="normal", ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=5, **kw)


def chip(x, y, fc, ec, mark=None, mk_color="white", dashed=False):
    ls = (0, (2, 1.4)) if dashed else "-"
    ax.add_patch(Rectangle((x, y), 2.4, 2.8, fc=fc, ec=ec, lw=0.8, ls=ls, zorder=4))
    if mark:
        txt(x + 1.2, y + 1.4, mark, size=5.4, color=mk_color, weight="bold")


# ================================================== 顶部阶段标签
ST = dict(size=6.6, color="#6E8CA8", weight="bold")
txt(10.75, 96.5, "① 编译需求程序", **ST)
txt(45, 96.5, "② 查询槽替换(单变量干预)", size=6.6, color=C["greenDeep"], weight="bold")
txt(72, 96.5, "③ 检索与候选池", **ST)
txt(88, 96.5, "④ 覆盖选证 · ⑤ 阅读", **ST)

# ================================================== 左列
qb = box(2.0, 66, 17.5, 17, C["blueFill"], C["blue"], lw=1.0)
txt(10.75, 79.6, "多跳问题 q", size=7.4, weight="bold", color=C["blue"])
txt(10.75, 72.3, "Who is the spouse of\nthe actor of Ethan in\nA Dog's Purpose?",
    size=6.0, **SER)

arrow((10.75, 65.4), (10.75, 59.0), C["purple"])

box(2.0, 43.5, 17.5, 15, C["purpleFill"], C["purple"], lw=1.0)
txt(10.75, 54.8, "规划器(固定)", size=7.4, weight="bold", color=C["purple"])
txt(10.75, 48.2, "编译为需求程序 P=(R,E)\n答案类型 · 依赖位置", size=6.0)

arrow((10.75, 43.0), (10.75, 36.6), C["purple"])

box(2.0, 17.5, 17.5, 18.5, "#FFFFFF", C["purple"], lw=1.0)
txt(10.75, 32.2, "需求程序", size=7.4, weight="bold", color=C["purple"])
txt(10.75, 24.9, "$r_1$: who plays old Ethan\n        in A Dog's Purpose\n$r_2$: #1 spouse", size=6.3)
txt(10.75, 19.9, "依赖边: $r_2$ 引用 $r_1$ 的答案(#1)", size=5.6, color="#B58438")

# ================================================== 干预区外框 + 固定协议
box(21.0, 30, 78, 61, "none", "#C5D3DE", lw=0.8, ls=(0, (4, 2.5)), r=2.0, z=1)
txt(60, 87.8, "固定: 语料 · 金标 · 计划 · 检索器 · 重排器 · reader · 每跳预算 k=3    "
              "干预: 仅查询槽", size=6.3, color="#6E8CA8")

# ================================================== 上分支(灰): 整句问题
Y1 = 74
txt(22.3, Y1 + 10.8, "查询槽 A(对照)", size=6.6, weight="bold", color=C["greyDark"], ha="left")
box(23.0, Y1 - 7, 16.5, 14, C["greyFill"], C["grey"], lw=1.0)
txt(31.25, Y1 + 3.6, "整句问题 q", size=7.2, weight="bold", color=C["greyDark"])
txt(31.25, Y1 - 1.8, "查询不含中间实体", size=5.8, color="#6E8CA8")

arrow((19.7, Y1), (22.4, Y1), C["grey"])
arrow((39.9, Y1), (42.6, Y1), C["grey"])

box(43.0, Y1 - 7, 15.5, 14, "#FFFFFF", C["grey"], lw=1.0)
txt(50.75, Y1 + 3.6, "检索 + 重排", size=7.2, weight="bold", color=C["greyDark"])
txt(50.75, Y1 - 1.8, "top-50 候选 / 跳", size=5.8, color="#6E8CA8")

arrow((59.0, Y1), (61.6, Y1), C["grey"])

box(62.0, Y1 - 8, 14.5, 16, "#FFFFFF", C["grey"], lw=1.0)
txt(69.25, Y1 + 5.2, "候选池", size=7.2, weight="bold", color=C["greyDark"])
cx0 = 63.25
for row, cy in enumerate((Y1 + 0.4, Y1 - 3.6)):
    for i in range(4):
        chip(cx0 + i * 3.2, cy, "#EDF3F7", "#C5D3DE")
chip(cx0 + 3 * 3.2, Y1 - 3.6, "none", C["red"], mark="×", mk_color=C["red"], dashed=True)
txt(69.25, Y1 - 6.2, "$r_2$ 证据不可达", size=5.8, color=C["red"])

arrow((77.0, Y1), (79.4, Y1), C["grey"])

box(79.8, Y1 - 7, 14.0, 14, C["greyFill"], C["grey"], lw=1.0)
txt(86.8, Y1 + 3.2, "逐需求选证 + reader", size=6.8, weight="bold", color=C["greyDark"])
txt(86.8, Y1 - 1.6, "(全部固定)", size=5.8, color="#6E8CA8")

arrow((94.2, Y1), (95.2, Y1), C["grey"], sh=2)
txt(96.0, Y1 + 4.0, "EM", size=6.2, color=C["greyDark"], weight="bold")
txt(96.0, Y1 - 0.6, "0.170", size=7.8, color=C["greyDark"], weight="bold")

# ================================================== 两分支之间
txt(49.5, 55.5, "第 2 跳 dense Recall@50:  0.615 → 0.935", size=6.4, color="#6E8CA8")

# ================================================== 下分支(绿): 绑定需求
Y2 = 43
txt(22.3, Y2 + 10.8, "查询槽 B(CovBind)", size=6.6, weight="bold", color=C["greenDeep"], ha="left")

arrow((19.7, 27), (22.4, Y2 + 2), C["green"], lw=1.2, sh=3)

box(23.0, Y2 - 7, 16.5, 14, C["greenFill"], C["green"], lw=1.2)
txt(31.25, Y2 + 3.6, "绑定 β + 替换查询槽", size=6.8, weight="bold", color=C["greenDeep"])
txt(31.25, Y2 - 1.9, "$\\hat{r}_2$ = “Dennis Quaid\nspouse”", size=5.8, color=C["greenDeep"])

arrow((39.9, Y2), (42.6, Y2), C["green"])

box(43.0, Y2 - 7, 15.5, 14, "#FFFFFF", C["green"], lw=1.0)
txt(50.75, Y2 + 3.6, "检索 + 重排", size=7.2, weight="bold", color=C["greenDeep"])
txt(50.75, Y2 - 1.8, "同一检索器 / 重排器", size=5.8, color="#6E8CA8")

arrow((59.0, Y2), (61.6, Y2), C["green"])

box(62.0, Y2 - 8, 14.5, 16, "#FFFFFF", C["green"], lw=1.0)
txt(69.25, Y2 + 5.2, "候选池", size=7.2, weight="bold", color=C["greenDeep"])
for cy in (Y2 + 0.4, Y2 - 3.6):
    for i in range(4):
        chip(cx0 + i * 3.2, cy, "#EDF3F7", "#C5D3DE")
chip(cx0 + 3 * 3.2, Y2 - 3.6, C["green"], C["greenDeep"], mark="✓")
txt(69.25, Y2 - 6.2, "$r_2$ 证据进入候选", size=5.8, color=C["greenDeep"])

arrow((77.0, Y2), (79.4, Y2), C["green"])

box(79.8, Y2 - 7, 14.0, 14, C["greenFill"], C["green"], lw=1.0)
txt(86.8, Y2 + 3.2, "逐需求选证 + reader", size=6.8, weight="bold", color=C["greenDeep"])
txt(86.8, Y2 - 1.6, "覆盖停止条件", size=5.8, color=C["greenDeep"])

arrow((94.2, Y2), (95.2, Y2), C["green"], sh=2)
txt(96.0, Y2 + 4.0, "EM", size=6.2, color=C["greenDeep"], weight="bold")
txt(96.0, Y2 - 0.6, "0.425", size=7.8, color=C["greenDeep"], weight="bold")

# ================================================== 底部: 图例 + 绑定说明
ly = 17.5
chip(23.0, ly, "#EDF3F7", "#C5D3DE")
txt(26.6, ly + 1.4, "候选段落", size=6.0, ha="left", color="#6E8CA8")
chip(38.0, ly, "none", C["red"], mark="×", mk_color=C["red"], dashed=True)
txt(41.6, ly + 1.4, "金标证据缺失", size=6.0, ha="left", color=C["red"])
chip(56.0, ly, C["green"], C["greenDeep"], mark="✓")
txt(59.6, ly + 1.4, "金标证据进入候选", size=6.0, ha="left", color=C["greenDeep"])

txt(23.0, 8.0, "绑定 β: 将第 1 跳答案 “Dennis Quaid” 填入 $r_2$ 的依赖槽 #1, 得到绑定需求 "
              "$\\hat{r}_2$;检索与重排均使用 $\\hat{r}_2$, reader 与停止条件不变",
    size=6.0, ha="left", color="#6E8CA8")

# ================================================== 右下: EM 小图(真实数据)
ax_in = fig.add_axes([0.795, 0.045, 0.185, 0.215])
vals = [0.170, 0.425]
bars = ax_in.bar([0, 1], vals, width=0.62, color=[C["grey"], C["green"]],
                 edgecolor=[C["greyDark"], C["greenDeep"]], linewidth=0.7)
ax_in.set_ylim(0, 0.60)
ax_in.set_xlim(-0.55, 1.55)
ax_in.set_xticks([0, 1])
ax_in.set_xticklabels(["整句 q", "绑定 $\\hat{r}$"], fontsize=6.2)
ax_in.set_yticks([0, 0.25, 0.5])
ax_in.tick_params(labelsize=5.8, length=1.6, pad=1.2)
ax_in.grid(axis="y", linewidth=0.4, color="#DDDDDD")
ax_in.set_axisbelow(True)
for s in ("top", "right"):
    ax_in.spines[s].set_visible(False)
for sp in ("left", "bottom"):
    ax_in.spines[sp].set_linewidth(0.6)
for r_, v in zip(bars, vals):
    ax_in.text(r_.get_x() + r_.get_width() / 2, v + 0.02, f"{v:.3f}",
               ha="center", fontsize=6.0, fontweight="bold",
               color=C["greyDark"] if v < 0.3 else C["greenDeep"])
ax_in.annotate("", xy=(1, 0.545), xytext=(0, 0.545),
               arrowprops=dict(arrowstyle="-", lw=0.8, color=C["ink"]))
ax_in.plot([0, 0], [0.515, 0.545], lw=0.8, color=C["ink"])
ax_in.plot([1, 1], [0.515, 0.545], lw=0.8, color=C["ink"])
ax_in.text(0.5, 0.575, "+0.255", ha="center", fontsize=6.2, fontweight="bold", color=C["ink"])
ax_in.text(0.5, 0.50, "[+0.210, +0.303]", ha="center", fontsize=5.2, color="#6E8CA8")
ax_in.set_xlabel("MuSiQue-400 EM", fontsize=6.2, labelpad=1.5)

save(fig, "fig1_method_overview")
