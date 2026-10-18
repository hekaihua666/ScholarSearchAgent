"""Final Nature-style system architecture for the V3.0 project document.

Core conclusion: the system separates broad candidate discovery from
evidence-controlled ranking and final-set selection, while EBP closes a
recoverable one-hop citation gap.
Backend: Python/Matplotlib only. Output: 183-mm editable publication figure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/submission_2026/figures_nature/final_v30"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
})

C = {
    "ink": "#262833", "muted": "#6A6E78", "line": "#B9BDC6",
    "blue": "#536792", "blue_soft": "#E1E7F2",
    "teal": "#4C9798", "teal_soft": "#DDEEEE",
    "rose": "#B56580", "rose_soft": "#F2DFE5",
    "amber": "#CF9849", "amber_soft": "#F5E9D6",
    "purple": "#78669B", "purple_soft": "#EAE5F2",
    "green": "#3B8B5A", "white": "#FFFFFF",
}


def mm(v: float) -> float:
    return v / 25.4


def arrow(ax, a, b, color=C["line"], rad=0, lw=1.1):
    ax.add_patch(FancyArrowPatch(
        a, b, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=8,
        linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=2, zorder=2,
    ))


def box(ax, x, y, w, h, title, lines, face, edge, tracked, *, title_size=7.3, body_size=5.7):
    patch = FancyBboxPatch(
        (x, y), w, h, transform=ax.transAxes,
        boxstyle="round,pad=.006,rounding_size=.015",
        facecolor=face, edgecolor=edge, linewidth=1.05, zorder=3,
    )
    ax.add_patch(patch)
    t = ax.text(x+.06*w, y+h-.16*h, title, transform=ax.transAxes,
                ha="left", va="top", fontsize=title_size, fontweight="bold",
                color=C["ink"], zorder=4)
    b = ax.text(x+.06*w, y+h-.43*h, lines, transform=ax.transAxes,
                ha="left", va="top", fontsize=body_size, color=C["muted"],
                linespacing=1.35, zorder=4)
    tracked.extend([(t, patch), (b, patch)])
    return patch


def validate(fig, tracked):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, patch in tracked:
        a = artist.get_window_extent(renderer); p = patch.get_window_extent(renderer)
        if a.x0 < p.x0-2 or a.x1 > p.x1+2 or a.y0 < p.y0-2 or a.y1 > p.y1+2:
            raise RuntimeError(f"text overflow: {artist.get_text()!r}")


def build() -> dict[str, str]:
    tracked: list[tuple[Any, Any]] = []
    fig, ax = plt.subplots(figsize=(mm(183), mm(102)))
    fig.subplots_adjust(left=.025, right=.985, top=.96, bottom=.055)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    box(ax, .36, .82, .28, .13, "复杂学术查询",
        "主题 · 方法 · 任务 · 时间 · 关系约束",
        C["white"], C["ink"], tracked, title_size=8.2, body_size=6.0)
    arrow(ax, (.50, .82), (.10, .70), C["ink"], rad=.04, lw=1.25)

    specs = [
        (.035, "查询认知", "Query Contract\n意图路由 · 子查询", C["blue_soft"], C["blue"]),
        (.225, "自主检索", "多源词项—语义检索\nSCGQ / QCF 引文前沿", C["teal_soft"], C["teal"]),
        (.415, "全局排序", "身份合并 · TRACE-Rank\n相关性 · 权威性 · 多样性", C["purple_soft"], C["purple"]),
        (.605, "证据决策", "REG 条件合取\nMEQ 多视图一致性", C["rose_soft"], C["rose"]),
        (.795, "结构化交付", "精选集合 · 补充候选\n矩阵 · 图谱 · 导出", C["amber_soft"], C["amber"]),
    ]
    for x, title, body, face, edge in specs:
        box(ax, x, .48, .17, .24, title, body, face, edge, tracked)
    for x in (.205, .395, .585, .775):
        arrow(ax, (x, .60), (x+.02, .60), C["ink"])

    ebp = box(ax, .26, .19, .48, .16, "EBP v21 · 证据加权引文反投影",
              "五路证据聚合 → 未解析引文 ID 投票 → Top100 一次批量物化 → 扩展补充候选边界",
              "#F7F7FA", C["teal"], tracked, title_size=7.5, body_size=5.8)
    arrow(ax, (.69, .48), (.64, .35), C["rose"], rad=.08)
    arrow(ax, (.36, .35), (.30, .48), C["teal"], rad=.08)
    ax.text(.72, .38, "证据回传", transform=ax.transAxes, fontsize=5.6,
            color=C["rose"], ha="left", va="center")

    base = FancyBboxPatch((.035, .035), .93, .085, transform=ax.transAxes,
                          boxstyle="round,pad=.005,rounding_size=.012",
                          facecolor="#F4F5F7", edgecolor="#D4D7DD", linewidth=.9)
    ax.add_patch(base)
    t1 = ax.text(.055, .078, "数据与运行底座", transform=ax.transAxes, fontsize=6.8,
                 fontweight="bold", color=C["ink"], ha="left", va="center")
    t2 = ax.text(.20, .078,
                 "OpenAlex · arXiv · Crossref · Semantic Scholar   |   MiniLM   |   缓存 · 冻结回放 · 离线演示",
                 transform=ax.transAxes, fontsize=5.9, color=C["muted"], ha="left", va="center")
    tracked.extend([(t1, base), (t2, base)])
    for x in (.12, .31, .50, .69, .88):
        arrow(ax, (x, .12), (x, .48), "#C7CAD1", lw=.85)

    validate(fig, tracked)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "fig1_final_system_architecture"
    paths = {
        "svg": stem.with_suffix(".svg"), "pdf": stem.with_suffix(".pdf"),
        "tiff": stem.with_suffix(".tiff"), "png": stem.with_suffix(".png"),
    }
    fig.savefig(paths["svg"], facecolor="white")
    fig.savefig(paths["pdf"], facecolor="white")
    fig.savefig(paths["tiff"], dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(paths["png"], dpi=300, facecolor="white")
    plt.close(fig)
    if "<text" not in paths["svg"].read_text(encoding="utf-8"):
        raise RuntimeError("SVG text was converted to paths")
    qa = {"backend": "Python/Matplotlib", "width_mm": 183,
          "editable_svg_text": True, "overflow": "passed",
          "exports": {k: str(v) for k, v in paths.items()}}
    (OUT / "qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    return qa


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
