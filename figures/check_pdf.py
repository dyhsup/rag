# -*- coding: utf-8 -*-
"""基于 pymupdf 的文本几何质检(仿 thesis-figure-skill 的 pdf-overlap-checker):
1) 文本-文本 bbox 重叠  2) 文本出页边  3) 文本-文本极近(<1pt 间隙)
用法: python3 check_pdf.py fig1_method_overview.pdf
"""
import sys
import fitz


def rects_overlap(a, b, pad=0.0):
    return not (a.x1 - pad <= b.x0 or b.x1 - pad <= a.x0 or
                a.y1 - pad <= b.y0 or b.y1 - pad <= a.y0)


def main(path):
    doc = fitz.open(path)
    page = doc[0]
    words = page.get_text("words")  # x0,y0,x1,y1,word,block,line,word_no
    issues = []
    pw, ph = page.rect.width, page.rect.height
    # 1) 文本重叠: 只报告"不同文本块"之间的重叠(同一文本块的相邻行 bbox 贴合是行距误报)
    n = len(words)
    for i in range(n):
        r1 = fitz.Rect(words[i][:4])
        for j in range(i + 1, n):
            if words[i][5:7] == words[j][5:7]:
                continue  # 同一 block+line
            r2 = fitz.Rect(words[j][:4])
            if rects_overlap(r1, r2, pad=-0.2):  # 允许 0.2pt 贴合
                inter = fitz.Rect(r1) & r2
                area = inter.get_area()
                small = min(r1.get_area(), r2.get_area())
                if small > 0 and area / small > 0.25:
                    issues.append(("TEXT-OVERLAP",
                                   words[i][4], words[j][4],
                                   f"({r1.x0:.0f},{r1.y0:.0f})-({r2.x0:.0f},{r2.y0:.0f})",
                                   f"iov={area/small:.0%}"))
    # 2) 出页边
    for w in words:
        r = fitz.Rect(w[:4])
        if r.x0 < -0.5 or r.y0 < -0.5 or r.x1 > pw + 0.5 or r.y1 > ph + 0.5:
            issues.append(("OUT-OF-PAGE", w[4], "", f"page={pw:.0f}x{ph:.0f}", str(r)))
    # 3) 紧邻提示(间隙 0~0.5pt, 可能视觉拥挤)
    near = 0
    for i in range(n):
        r1 = fitz.Rect(words[i][:4])
        for j in range(i + 1, n):
            r2 = fitz.Rect(words[j][:4])
            if rects_overlap(r1, r2, pad=-0.2):
                continue
            gap_x = max(r2.x0 - r1.x1, r1.x0 - r2.x1)
            gap_y = max(r2.y0 - r1.y1, r1.y0 - r2.y1)
            if (gap_x <= 0 and abs(gap_y) < 0.7) or (gap_y <= 0 and abs(gap_x) < 0.7):
                near += 1
    print(f"== {path}  ({pw:.0f}x{ph:.0f}pt, {n} words)")
    if not issues:
        print("   no text-overlap / out-of-page issues")
    for it in issues:
        print("  !!", *it)
    print(f"   near-touch pairs (0~0.7pt gap): {near}")


if __name__ == "__main__":
    main(sys.argv[1])
