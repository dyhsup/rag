# -*- coding: utf-8 -*-
"""Figure 2-7 —— 实验结果图表。所有数字取自 docs/ACL_PAPER_ZH.md §6 与 results/final_manifest.json。
图内不放总标题(LaTeX caption 承担); (a)(b) 面板标签保留。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from figstyle import apply_style, C, save, COL1, COL2

apply_style()
R = np.random.RandomState(7)  # 未用, 占位保持确定性


# ============================================================ Figure 2
def fig2():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL1, 2.55),
                                 gridspec_kw=dict(width_ratios=[1, 1.12], wspace=0.62))
    # ---- (a) 总体 EM
    groups = ["200 题", "400 题"]
    whole = [0.170, 0.170]
    bound = [0.440, 0.425]
    deltas = [("+0.270", "[+0.210,+0.335]"), ("+0.255", "[+0.210,+0.303]")]
    x = np.arange(2)
    w = 0.34
    b1 = a1.bar(x - w / 2, whole, w, color=C["grey"], ec=C["greyDark"], lw=0.6, label="整句问题")
    b2 = a1.bar(x + w / 2, bound, w, color=C["green"], ec=C["greenDeep"], lw=0.6, label="绑定需求")
    for r, v in list(zip(b1, whole)) + list(zip(b2, bound)):
        a1.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.3f}".lstrip("0"),
                ha="center", fontsize=6.0,
                color=C["greyDark"] if v < 0.3 else C["greenDeep"], fontweight="bold")
    for i, (d, ci) in enumerate(deltas):
        y = max(whole[i], bound[i]) + 0.085
        a1.annotate("", xy=(i + w / 2, y - 0.012), xytext=(i - w / 2, y - 0.012),
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=C["ink"]))
        a1.plot([i - w / 2] * 2, [y - 0.012, y - 0.036], lw=0.7, color=C["ink"])
        a1.plot([i + w / 2] * 2, [y - 0.012, y - 0.036], lw=0.7, color=C["ink"])
        a1.text(i, y + 0.004, d, ha="center", fontsize=6.4, fontweight="bold")
        a1.text(i, y + 0.048, ci, ha="center", fontsize=5.0, color="#6E6B6B")
    a1.set_xticks(x)
    a1.set_xticklabels(groups)
    a1.set_xlabel("MuSiQue", fontsize=7.5)
    a1.set_ylabel("EM")
    a1.set_ylim(0, 0.80)
    a1.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    a1.legend(loc="upper left", handlelength=1.1, handleheight=0.9,
              borderaxespad=0.1, labelspacing=0.25, fontsize=6.2)
    a1.text(0, 1.045, "(a) 查询槽反事实(配对 bootstrap)", transform=a1.transAxes,
            fontsize=7.0, fontweight="bold")

    # ---- (b) 跳数分层 dumbbell
    rows = [("2 跳", 0.197, 0.520, "+0.323", "[+0.258,+0.394]", 198),
            ("3 跳", 0.189, 0.377, "+0.189", "[+0.107,+0.270]", 122),
            ("4 跳", 0.079, 0.276, "+0.197", "[+0.105,+0.289]", 76),
            ("全部", 0.170, 0.425, "+0.255", "[+0.210,+0.300]", 400)]
    ys = np.arange(len(rows))[::-1]
    for y, (name, wq, bq, d, ci, n) in zip(ys, rows):
        a2.plot([wq, bq], [y, y], color="#B9B9B9", lw=2.2, zorder=2, solid_capstyle="round")
        a2.scatter([wq], [y], s=26, color=C["grey"], ec=C["greyDark"], lw=0.6, zorder=3)
        a2.scatter([bq], [y], s=30, color=C["green"], ec=C["greenDeep"], lw=0.6, zorder=3)
        a2.text(bq + 0.028, y + 0.16, d, fontsize=6.2, fontweight="bold", color=C["greenDeep"])
        a2.text(bq + 0.028, y - 0.30, ci, fontsize=4.8, color="#7A828A")
    a2.set_yticks(ys)
    a2.set_yticklabels([f"{n}  (n={m})" for n, *_, m in rows], fontsize=6.8)
    a2.set_xlim(0, 0.80)
    a2.set_xticks([0, 0.2, 0.4, 0.6])
    a2.set_xlabel("EM")
    a2.set_ylim(-0.55, len(rows) - 0.4)
    a2.text(0, 1.045, "(b) 按跳数分层(MuSiQue-400)", transform=a2.transAxes,
            fontsize=7.0, fontweight="bold")
    h1 = a2.scatter([], [], s=26, color=C["grey"], ec=C["greyDark"], lw=0.6, label="整句问题")
    h2 = a2.scatter([], [], s=30, color=C["green"], ec=C["greenDeep"], lw=0.6, label="绑定需求")
    fig.legend([h1, h2], ["整句问题", "绑定需求"], loc="lower center", ncols=2,
               bbox_to_anchor=(0.52, 0.005), handlelength=1.0, columnspacing=1.2,
               handletextpad=0.4, fontsize=6.4)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.235, top=0.90)
    save(fig, "fig2_queryslot_counterfactual")


# ============================================================ Figure 3
def fig3():
    fig, ax = plt.subplots(figsize=(COL1, 2.35))
    hops = [1, 2, 3, 4]
    dense_u = [0.915, 0.615, 0.673, 0.538]
    dense_b = [0.970, 0.935, 0.921, 0.923]
    bm_u = [0.795, 0.425, 0.554, 0.256]
    bm_b = [0.945, 0.940, 0.891, 0.897]
    ax.axvspan(1.55, 4.45, color=C["greenFill"], alpha=0.55, zorder=0)
    ax.text(2.95, 0.74, "中间实体绑定 → 后续跳恢复", ha="center", fontsize=6.4,
            color=C["greenDeep"], fontweight="bold")
    ax.plot(hops, dense_u, ls=(0, (4, 2.2)), lw=1.2, color="#A8CBDF", marker="o", ms=3.2,
            mfc="white", mec="#A8CBDF", mew=1.0, label="dense · 整句")
    ax.plot(hops, dense_b, ls="-", lw=1.5, color=C["blue"], marker="o", ms=3.4,
            label="dense · 绑定")
    ax.plot(hops, bm_u, ls=(0, (4, 2.2)), lw=1.2, color="#F0C284", marker="s", ms=3.0,
            mfc="white", mec="#F0C284", mew=1.0, label="BM25 · 整句")
    ax.plot(hops, bm_b, ls="-", lw=1.5, color=C["orange"], marker="s", ms=3.2,
            label="BM25 · 绑定")
    for xx, yy in zip(hops, dense_b):
        ax.annotate(f"{yy:.2f}".lstrip("0"), (xx, yy), textcoords="offset points",
                    xytext=(0, 4.5), ha="center", fontsize=5.6, color=C["blue"])
    for xx, yy in zip(hops, bm_b):
        ax.annotate(f"{yy:.2f}".lstrip("0"), (xx, yy), textcoords="offset points",
                    xytext=(0, -10.5), ha="center", fontsize=5.6, color=C["orange"])
    ax.set_xticks(hops)
    ax.set_xlabel("跳")
    ax.set_ylabel("逐跳 Recall@50")
    ax.set_ylim(0.18, 1.03)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="lower left", ncols=2, columnspacing=0.9, handlelength=1.7,
              borderaxespad=0.15, fontsize=6.2)
    fig.subplots_adjust(left=0.135, right=0.98, bottom=0.155, top=0.97)
    save(fig, "fig3_hop_reachability")


# ============================================================ Figure 4
def fig4():
    fig, ax = plt.subplots(figsize=(COL1, 2.35))
    names = ["BM25", "MiniLM", "BGE-small", "HyDE\n+MiniLM"]
    reach = [0.610, 0.760, 0.765, 0.770]
    cov3 = [0.288, 0.424, 0.431, 0.417]
    eff = [0.472, 0.558, 0.563, 0.542]
    x = np.arange(4)
    ax.bar(x, reach, 0.56, color="#D6EFF4", ec="#A8CBDF", lw=0.7,
           label="候选可达性(上限)")
    ax.bar(x, cov3, 0.56, color=C["green"], ec=C["green"], lw=0.7,
           label="覆盖@3(实际达到)")
    for i, (rc, cv, ef) in enumerate(zip(reach, cov3, eff)):
        ax.text(i, rc + 0.015, f"{rc:.3f}".lstrip("0"), ha="center", fontsize=6.0,
                color="#54719E", fontweight="bold")
        ax.text(i, cv / 2, f"{cv:.3f}".lstrip("0"), ha="center", va="center",
                fontsize=6.0, color="white", fontweight="bold")
        ax.text(i, rc - 0.055, f"$\\eta$={ef:.2f}".replace("0.", "."), ha="center",
                fontsize=6.0, color="#54719E")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.6)
    ax.set_ylim(0, 0.92)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel("需求覆盖率")
    ax.legend(loc="upper left", borderaxespad=0.15,
              handlelength=1.1, handleheight=0.9, labelspacing=0.25, fontsize=6.2)
    ax.text(0.99, 0.965, "更强编码器抬高上限, $\\eta$ 不变",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.2, color="#6E6B6B")
    fig.subplots_adjust(left=0.135, right=0.98, bottom=0.17, top=0.97)
    save(fig, "fig4_reach_efficiency")


# ============================================================ Figure 5
def fig5():
    fig, (a1, a2, a3) = plt.subplots(
        1, 3, figsize=(COL2, 2.25),
        gridspec_kw=dict(width_ratios=[1.15, 0.82, 0.95], wspace=0.34))
    # ---- (a) 选择器对比
    mets = ["需求覆盖", "全证据率", "EM"]
    whole = [0.598, 0.304, 0.268]
    bound = [0.866, 0.750, 0.571]
    x = np.arange(3)
    w = 0.34
    b1 = a1.bar(x - w / 2, whole, w, color=C["grey"], ec=C["greyDark"], lw=0.6, label="整句问题")
    b2 = a1.bar(x + w / 2, bound, w, color=C["green"], ec=C["greenDeep"], lw=0.6, label="绑定需求")
    for r, v in list(zip(b1, whole)) + list(zip(b2, bound)):
        a1.text(r.get_x() + r.get_width() / 2, v + 0.014, f"{v:.3f}".lstrip("0"),
                ha="center", fontsize=5.8,
                color=C["greyDark"] if v < 0.4 else C["greenDeep"], fontweight="bold")
    a1.set_xticks(x)
    a1.set_xticklabels(mets, fontsize=6.8)
    a1.set_ylim(0, 1.14)
    a1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a1.set_ylabel("取值")
    a1.legend(loc="upper left", borderaxespad=0.1, handlelength=1.1,
              handleheight=0.9, labelspacing=0.25, fontsize=6.0)
    a1.text(0, 1.06, "(a) 覆盖选证器(2 跳, gold 计划, k=2)", transform=a1.transAxes,
            fontsize=7.0, fontweight="bold")

    # ---- (b) 证据完整度 → EM
    cats = ["全部需求证据\n同时出现", "证据\n不完整"]
    ems = [0.635, 0.162]
    lo = [0.62, 0.155]
    hi = [0.65, 0.169]
    bars = a2.bar([0, 1], ems, 0.52, color=[C["green"], C["crimson"]],
                  ec=[C["greenDeep"], "#6E1517"], lw=0.6)
    a2.errorbar([0, 1], ems, yerr=[[e - l for e, l in zip(ems, lo)],
                                   [h - e for e, h in zip(ems, hi)]],
                fmt="none", ecolor=C["ink"], elinewidth=0.8, capsize=2.4, capthick=0.8)
    for r_, v in zip(bars, ems):
        a2.text(r_.get_x() + r_.get_width() / 2, v + 0.045, f"≈{v:.2f}",
                ha="center", fontsize=6.4, fontweight="bold")
    a2.text(0.5, 0.735, "4 倍差距", ha="center", fontsize=6.4, color=C["ink"],
            fontweight="bold")
    a2.annotate("", xy=(1, 0.70), xytext=(0, 0.70),
                arrowprops=dict(arrowstyle="-", lw=0.7, color=C["ink"]))
    a2.set_xticks([0, 1])
    a2.set_xticklabels(cats, fontsize=6.4)
    a2.set_ylim(0, 0.82)
    a2.set_ylabel("reader EM")
    a2.text(0, 1.06, "(b) 覆盖是 reader 的瓶颈", transform=a2.transAxes,
            fontsize=7.0, fontweight="bold")

    # ---- (c) 训练×推理 query 矩阵
    mat = np.array([[0.608, 0.684], [0.509, 0.738]])
    ems_m = [["EM .23", "EM .35"], ["EM .25", "EM .35"]]
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list("tblues", ["#F2FAFC", "#8074C8"])
    im = a3.imshow(mat, cmap=cmap, vmin=0.45, vmax=0.78, aspect="auto")
    for i in range(2):
        for j in range(2):
            a3.text(j, i - 0.10, f"{mat[i, j]:.3f}".lstrip("0"), ha="center", va="center",
                    fontsize=7.4, fontweight="bold",
                    color="white" if mat[i, j] > 0.70 else C["ink"])
            a3.text(j, i + 0.26, ems_m[i][j], ha="center", va="center", fontsize=5.8,
                    color="white" if mat[i, j] > 0.70 else "#6E6B6B")
    a3.set_xticks([0, 1])
    a3.set_xticklabels(["完整问题", "绑定需求"], fontsize=6.6)
    a3.set_yticks([0, 1])
    a3.set_yticklabels(["问题 CE", "需求 CE"], fontsize=6.6)
    a3.set_xlabel("推理 query", fontsize=6.8, labelpad=1.5)
    a3.set_ylabel("训练 query", fontsize=6.8, labelpad=1.5)
    a3.tick_params(length=0)
    for s in a3.spines.values():
        s.set_visible(False)
    a3.text(0, 1.06, "(c) 覆盖@2: 训练×推理 query", transform=a3.transAxes,
            fontsize=7.0, fontweight="bold")
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.20, top=0.86)
    save(fig, "fig5_coverage_bottleneck")


# ============================================================ Figure 6
def fig6():
    fig, ax = plt.subplots(figsize=(COL2, 2.6))
    dss = ["MuSiQue\n200", "MuSiQue\n400", "2Wiki\n200", "HotpotQA\n200", "MultiHopRAG\n200"]
    vanilla = [0.195, 0.190, 0.475, 0.625, None]
    ircot = [0.315, 0.302, 0.615, 0.665, 0.465]
    covb = [0.440, 0.425, 0.620, 0.680, 0.530]
    rel = ["+0.125", "+0.123", "+0.005", "+0.015", "+0.065"]
    wq = [0.170, 0.170, None, None, None]  # CovBind-whole-query 反事实
    x = np.arange(5)
    w = 0.26
    for i, v in enumerate(vanilla):
        if v is None:
            ax.text(i - w, 0.03, "—", ha="center", fontsize=8, color="#A6ADB4")
            continue
        ax.bar(i - w, v, w, color=C["grey"], ec=C["greyDark"], lw=0.6, label="Vanilla+CE")
    for i, v in enumerate(ircot):
        ax.bar(i, v, w, color=C["rose"], ec="#8F3550", lw=0.6, label="IRCoT+CE")
    for i, v in enumerate(covb):
        ax.bar(i + w, v, w, color=C["green"], ec=C["greenDeep"], lw=0.6, label="CovBind")
        ax.text(i + w, v + 0.016, rel[i], ha="center", fontsize=6.4,
                fontweight="bold", color=C["greenDeep"])
    for i, v in enumerate(wq):
        if v is None:
            continue
        ax.plot([i + w - 0.055, i + w + 0.055], [v, v], color=C["red"], lw=1.5, zorder=5)
        ax.annotate("查询槽换回整句: .170", (i + w + 0.075, v), fontsize=5.8,
                    color=C["red"], va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(dss, fontsize=6.8)
    ax.set_ylim(0, 0.80)
    ax.set_ylabel("EM")
    ax.legend(loc="upper left", ncols=3, borderaxespad=0.15, handlelength=1.1,
              columnspacing=1.0, handleheight=0.9, labelspacing=0.25, fontsize=6.4)
    ax.text(0.995, 0.97, "柱上数字 = CovBind 相对 IRCoT+CE 的增益;  缺失项无缓存结果, 不填数",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.0, color="#6E6B6B")
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.155, top=0.97)
    save(fig, "fig6_end_to_end")


# ============================================================ Figure 7
def fig7():
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    rows = [("MuSiQue 2 跳\nk = m", 0.304, "+0.268", "足"),
            ("2Wiki\nk = m", 0.005, "+0.040", "足"),
            ("MuSiQue 4 跳\nk = 2 < m", -0.105, "+0.184", "不足"),
            ("MuSiQue 3 跳\nk = 2 < m", -0.200, "+0.173", "不足")]
    ys = np.arange(len(rows))[::-1]
    for y, (name, d, b, bud) in zip(ys, rows):
        color = C["green"] if d > 0.02 else (C["grey"] if d > 0 else C["red"])
        ecolor = C["greenDeep"] if d > 0.02 else (C["greyDark"] if d > 0 else "#B03A35")
        ax.barh(y, d, 0.52, color=color, ec=ecolor, lw=0.6, zorder=3)
        if d > 0.15:
            ax.text(d - 0.012, y, f"{d:+.3f}", va="center", ha="right",
                    fontsize=6.6, fontweight="bold", color="white")
        else:
            ax.text(d + (0.012 if d >= 0 else -0.012), y, f"{d:+.3f}",
                    va="center", ha="left" if d >= 0 else "right",
                    fontsize=6.6, fontweight="bold", color=ecolor)
        ax.text(1.03, y, f"B = {b}\n预算 {bud}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=5.8, color="#6E6B6B",
                linespacing=1.35)
    ax.axvline(0, color=C["ink"], lw=0.9, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.set_xlim(-0.30, 0.36)
    ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2, 0.3])
    ax.set_xlabel("端到端 ΔEM(绑定 − 整句)")
    ax.set_ylim(-0.6, 3.6)
    ax.text(0.03, 0.03, "收益需要: 绑定缺口 B 大 且 预算 k ≥ m", transform=ax.transAxes,
            fontsize=6.4, color=C["ink"],
            bbox=dict(boxstyle="round,pad=0.32", fc="#F6F6F6", ec="#CCCCCC", lw=0.6))
    fig.subplots_adjust(left=0.185, right=0.72, bottom=0.175, top=0.97)
    save(fig, "fig7_applicability")


if __name__ == "__main__":
    import sys
    only = sys.argv[1:] or ["2", "3", "4", "5", "6", "7"]
    fns = {"2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6, "7": fig7}
    for k in only:
        try:
            fns[k]()
        except Exception as e:
            import traceback
            print(f"[FAIL] fig{k}: {e}")
            traceback.print_exc()
