# -*- coding: utf-8 -*-
"""Figure 8-11 —— 论文其余结果小节的图表(与 fig1-7 同一色盘)。
数字来源: docs/ACL_PAPER_ZH.md §6.6/6.7/6.9/6.10/7.1 与 results/final_manifest.json
(cross_dataset_extended.whole_bm25_reachability / wiki2_slot_50)。
"""
import numpy as np
import matplotlib.pyplot as plt
from figstyle import apply_style, C, save, COL1, COL2

apply_style()


# ============================================================ Figure 8
def fig8():
    fig, ax = plt.subplots(figsize=(COL2, 2.7))
    dss = ["MuSiQue\n200", "MuSiQue\n400", "2Wiki\n200", "HotpotQA\n200", "MultiHopRAG\n200"]
    # 表 6.6 原始数字; None = 论文中为"—"(无缓存结果, 不填数)
    methods = [
        ("Vanilla",   [0.175, 0.128, 0.375, 0.530, 0.480], C["grey"],   "#6E8CA8"),
        ("Self-Ask",  [0.075, 0.060, 0.265, 0.295, 0.565], C["blue"],   "#54719E"),
        ("HyDE",      [0.205, 0.223, 0.390, 0.530, 0.540], C["purple"], "#C08B3F"),
        ("IRCoT",     [0.330, 0.248, 0.510, 0.650, 0.485], C["rose"],   "#8F3550"),
        ("CovBind",   [0.440, 0.425, 0.620, 0.680, 0.530], C["green"],  C["greenDeep"]),
    ]
    wq = [0.170, 0.170, None, None, None]  # CovBind-whole-query 反事实
    x = np.arange(5)
    w = 0.155
    for k, (name, vals, fc, ec) in enumerate(methods):
        for i, v in enumerate(vals):
            if v is None:
                ax.text(i + (k - 2) * w, 0.022, "—", ha="center", fontsize=7, color="#9FB4C6")
                continue
            ax.bar(i + (k - 2) * w, v, w, color=fc, ec=ec, lw=0.6, label=name)
            if name == "CovBind":
                ax.text(i + (k - 2) * w, v + 0.014, f"{v:.3f}".lstrip("0"), ha="center",
                        fontsize=6.0, fontweight="bold", color=C["greenDeep"])
    for i, v in enumerate(wq):
        if v is None:
            continue
        ax.plot([i + (4 - 2) * w - 0.045, i + (4 - 2) * w + 0.045], [v, v],
                color=C["red"], lw=1.5, zorder=5)
        ax.annotate("查询槽换回整句: .170", (i + (4 - 2) * w + 0.06, v),
                    fontsize=5.8, color=C["red"], va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(dss, fontsize=6.8)
    ax.set_ylim(0, 0.80)
    ax.set_ylabel("EM")
    ax.legend(loc="upper left", ncols=5, borderaxespad=0.15, handlelength=1.0,
              columnspacing=0.9, handletextpad=0.4, handleheight=0.9,
              labelspacing=0.25, fontsize=6.2)
    ax.text(0.995, 0.965, "CovBind 列 * 号端点使用 dataset-domain CE(协议差异见表 6.6 说明)",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.0, color="#6E6B6B")
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.155, top=0.97)
    save(fig, "fig8_method_comparison_full")


# ============================================================ Figure 9
def fig9():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.45),
                                 gridspec_kw=dict(width_ratios=[0.9, 1.25], wspace=0.30))
    # ---- (a) 2Wiki 50 题机制复验 (final_manifest.wiki2_slot_50)
    mets = ["需求\ncoverage", "逐跳\nrecall", "全证据率"]
    whole = [0.475, 0.480, 0.100]
    bound = [0.800, 0.795, 0.600]
    x = np.arange(3)
    w = 0.34
    b1 = a1.bar(x - w / 2, whole, w, color=C["grey"], ec=C["greyDark"], lw=0.6, label="整句问题")
    b2 = a1.bar(x + w / 2, bound, w, color=C["green"], ec=C["greenDeep"], lw=0.6, label="绑定需求")
    for r_, v in list(zip(b1, whole)) + list(zip(b2, bound)):
        a1.text(r_.get_x() + r_.get_width() / 2, v + 0.016, f"{v:.3f}".lstrip("0"),
                ha="center", fontsize=5.8,
                color=C["greyDark"] if v < 0.5 else C["greenDeep"], fontweight="bold")
    a1.set_xticks(x)
    a1.set_xticklabels(mets, fontsize=6.6)
    a1.set_ylim(0, 1.0)
    a1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a1.set_ylabel("取值")
    a1.legend(loc="upper right", borderaxespad=0.1, handlelength=1.1,
              handleheight=0.9, labelspacing=0.25, fontsize=6.0)
    a1.text(0, 1.055, "(a) 2Wiki 50 题机制复验(top-20 候选)", transform=a1.transAxes,
            fontsize=7.0, fontweight="bold")

    # ---- (b) whole-query BM25 可达性曲线
    ks = [1, 3, 5, 10]
    reach = {
        "2Wiki":    ([0.800, 0.970, 0.985, 1.000], [0.340, 0.6225, 0.74375, 1.000], C["green"]),
        "HotpotQA": ([0.710, 0.905, 0.970, 1.000], [0.355, 0.6625, 0.7725, 1.000], C["orange"]),
        "MultiHopRAG": ([0.805, 0.840, 0.850, 0.860], [0.3246, 0.6658, 0.7663, 0.8442], C["blue"]),
    }
    xk = np.arange(4)
    for name, (any_k, cov_k, col) in reach.items():
        a2.plot(xk, any_k, "-", marker="o", ms=3.4, lw=1.5, color=col, label=f"{name} · any@k")
        a2.plot(xk, cov_k, ls=(0, (4, 2.2)), marker="s", ms=3.0, lw=1.2, color=col,
                alpha=0.55, label=f"{name} · coverage@k")
    a2.axhline(1.0, color="#B9BEC5", lw=0.7, ls=":")
    a2.text(3.02, 0.965, "饱和", fontsize=5.8, color="#8FA6B8", ha="right")
    a2.annotate("top-10 仍存在\n不可达证据", xy=(3, 0.860), xytext=(1.85, 0.60),
                fontsize=6.0, color=C["blue"],
                arrowprops=dict(arrowstyle="-|>,head_length=4,head_width=2.6",
                                lw=0.8, color=C["blue"]))
    a2.set_xticks(xk)
    a2.set_xticklabels(["1", "3", "5", "10"])
    a2.set_xlabel("候选预算 k")
    a2.set_ylabel("whole-query BM25 可达性")
    a2.set_ylim(0.28, 1.04)
    a2.set_yticks([0.4, 0.6, 0.8, 1.0])
    a2.legend(loc="lower right", ncols=2, columnspacing=0.8, handlelength=1.6,
              borderaxespad=0.15, fontsize=5.6, labelspacing=0.22)
    a2.text(0, 1.055, "(b) 跨数据集 whole-query 可达性", transform=a2.transAxes,
            fontsize=7.0, fontweight="bold")
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.165, top=0.875)
    save(fig, "fig9_mechanism_crossreach")


# ============================================================ Figure 10
def fig10():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL1, 2.5),
                                 gridspec_kw=dict(width_ratios=[1, 1.12], wspace=0.5))
    # ---- (a) 金标绑定 vs 预测绑定 (§7.1)
    bars = a1.bar([0, 1], [0.915, 0.359], 0.52, color=[C["green"], C["grey"]],
                  ec=[C["greenDeep"], C["greyDark"]], lw=0.6)
    a1.errorbar([0], [0.915], yerr=[[0.025], [0.025]], fmt="none",
                ecolor=C["ink"], elinewidth=0.8, capsize=2.4, capthick=0.8)
    a1.text(0, 0.955, "0.89–0.94", ha="center", fontsize=6.0, fontweight="bold",
            color=C["greenDeep"])
    a1.text(1, 0.385, ".359", ha="center", fontsize=6.4, fontweight="bold",
            color=C["greyDark"])
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["金标绑定\n(第 2–4 跳)", "预测绑定\n(第 4 跳)"], fontsize=6.4)
    a1.set_ylim(0, 1.06)
    a1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a1.set_ylabel("Recall@50")
    a1.text(0, 1.10, "(a) 绑定质量影响传播", transform=a1.transAxes,
            fontsize=7.0, fontweight="bold")
    a1.text(0.5, -0.46, "自动绑定约消耗 65%\n的 oracle 绑定收益", transform=a1.transAxes,
            ha="center", fontsize=5.8, color="#6E6B6B")

    # ---- (b) 单跳控制 (§6.9)
    dss = ["SQuAD", "TriviaQA"]
    vanilla = [0.665, 0.755]
    cot = [0.715, 0.755]
    x = np.arange(2)
    w = 0.32
    b1 = a2.bar(x - w / 2, vanilla, w, color=C["grey"], ec=C["greyDark"], lw=0.6, label="Vanilla")
    b2 = a2.bar(x + w / 2, cot, w, color=C["green"], ec=C["greenDeep"], lw=0.6, label="CoT reader")
    for r_, v in list(zip(b1, vanilla)) + list(zip(b2, cot)):
        a2.text(r_.get_x() + r_.get_width() / 2, v + 0.012, f"{v:.3f}".lstrip("0"),
                ha="center", fontsize=5.8,
                color=C["greyDark"] if v < 0.72 else C["greenDeep"], fontweight="bold")
    a2.set_xticks(x)
    a2.set_xticklabels(dss, fontsize=6.8)
    a2.set_ylim(0, 0.92)
    a2.set_yticks([0, 0.25, 0.5, 0.75])
    a2.set_ylabel("EM")
    a2.legend(loc="lower right", borderaxespad=0.15, handlelength=1.1,
              handleheight=0.9, labelspacing=0.25, fontsize=6.0)
    a2.text(0, 1.10, "(b) 单跳控制: reader 增益有限", transform=a2.transAxes,
            fontsize=7.0, fontweight="bold")
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.33, top=0.87)
    save(fig, "fig10_binding_quality_singlehop")


# ============================================================ Figure 11
def fig11():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL1, 2.0),
                                 gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.55))
    # ---- (a) 查询改写仅消除 9.5% gap case
    a1.barh([0], [0.095], 0.5, color=C["red"], ec="#B03A35", lw=0.6)
    a1.barh([0], [1.0], 0.5, color=C["cream"], ec="#A8CBDF", lw=0.7)
    a1.text(0.115, 0, "9.5%", va="center", fontsize=7.0, fontweight="bold",
            color="#B03A35")
    a1.text(1.0, 0.62, "全部 gap case", ha="right", fontsize=5.8, color="#8FA6B8")
    a1.set_yticks([0])
    a1.set_yticklabels(["三路查询改写"], fontsize=6.6)
    a1.set_xlim(0, 1.02)
    a1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    a1.set_xticklabels(["0", ".25", ".50", ".75", "1.0"])
    a1.set_xlabel("gap case 消除率")
    a1.text(0, 1.10, "(a) 负结果: 查询改写", transform=a1.transAxes,
            fontsize=7.0, fontweight="bold")

    # ---- (b) 答案似然边际效用
    rows = [("gold 段", -0.113), ("干扰段", 0.096)]
    ys = [1, 0]
    for y, (name, v) in zip(ys, rows):
        col = C["crimson"] if v < 0 else C["purple"]
        ec = "#B03A35" if v < 0 else "#8A9E33"
        a2.barh(y, v, 0.5, color=col, ec=ec, lw=0.6, zorder=3)
        if v < 0:
            a2.text(v + 0.010, y, f"{v:+.3f}", va="center", ha="left",
                    fontsize=6.6, fontweight="bold", color="white", zorder=5)
        else:
            a2.text(v + 0.008, y, f"{v:+.3f}", va="center", ha="left",
                    fontsize=6.6, fontweight="bold", color=ec)
    a2.axvline(0, color=C["ink"], lw=0.9, zorder=4)
    a2.set_yticks(ys)
    a2.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    a2.set_xlim(-0.16, 0.16)
    a2.set_xticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15])
    a2.set_xticklabels(["−.15", "−.10", "−.05", "0", ".05", ".10", ".15"])
    a2.set_xlabel("平均边际(首跳有效 → 后续跳失效)")
    a2.set_ylim(-0.55, 1.55)
    a2.text(0, 1.10, "(b) 负结果: 答案似然", transform=a2.transAxes,
            fontsize=7.0, fontweight="bold")
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.215, top=0.87)
    save(fig, "fig11_negative_results")


if __name__ == "__main__":
    import sys
    only = sys.argv[1:] or ["8", "9", "10", "11"]
    fns = {"8": fig8, "9": fig9, "10": fig10, "11": fig11}
    for k in only:
        try:
            fns[k]()
        except Exception:
            import traceback
            print(f"[FAIL] fig{k}")
            traceback.print_exc()
