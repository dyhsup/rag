# -*- coding: utf-8 -*-
"""检查文字是否跨越矩形框边界(文字溢出框体的经典 bug)。用法: python3 check_edge.py x.pdf"""
import sys
import fitz


def main(path):
    doc = fitz.open(path)
    page = doc[0]
    # 收集"容器候选"矩形: 点数多的闭合路径(roundrect), 排除箭头线段
    rects = []
    for d in page.get_drawings():
        r = d["rect"]
        if d.get("fill") is None:      # 只把"有填充"的路径视为容器(排除纯描边线条)
            continue
        npts = sum(len(it) for it in d["items"] if it[0] in ("l", "c", "re"))
        if npts >= 4 and r.width > 8 and r.height > 5:
            rects.append(r)
    words = page.get_text("words")
    bad = 0
    for w in words:
        wr = fitz.Rect(w[:4])
        # 找包含词中心的最小矩形
        cx, cy = (wr.x0 + wr.x1) / 2, (wr.y0 + wr.y1) / 2
        containers = [r for r in rects if r.contains(fitz.Point(cx, cy))]
        if not containers:
            continue
        smallest = min(containers, key=lambda r: r.get_area())
        # 词 bbox 必须完全在最小容器内(留 0.5pt 容差); 否则=跨边界
        if not (smallest.x0 - 0.5 <= wr.x0 and wr.x1 <= smallest.x1 + 0.5 and
                smallest.y0 - 0.5 <= wr.y0 and wr.y1 <= smallest.y1 + 0.5):
            # 位于容器边缘的词: 计算溢出量
            ov = max(smallest.x0 - wr.x0, wr.x1 - smallest.x1,
                     smallest.y0 - wr.y0, wr.y1 - smallest.y1)
            print(f"  !! CROSS-EDGE '{w[4]}' out={ov:.1f}pt  word={wr}  box={smallest}")
            bad += 1
    print(f"== {path}: {len(rects)} container rects, {len(words)} words, {bad} cross-edge issues")


if __name__ == "__main__":
    main(sys.argv[1])
