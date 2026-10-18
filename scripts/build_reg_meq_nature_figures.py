"""Nature-style architecture and evidence figures for REG-MEQ v20.

Figure contract
---------------
Core conclusion: A dual-path relational and multi-view evidence router converts
additional blind candidate-pool hits into Top20 gains without extra online cost.
Archetypes: schematic-led composite (Fig. 1); quantitative grid (Fig. 2).
Target/output: 183-mm double-column, editable SVG/PDF plus 600-dpi TIFF and PNG.
"""

from __future__ import annotations

import csv
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "reports/evidence_quorum_v20/state.json"
ABLATION_PATH = ROOT / "reports/reg_harddata_v20/08_blind50_fixed_ablation.csv"
OUT = ROOT / "artifacts/submission_2026/figures_nature/reg_meq_v20"

# Mandatory editable-SVG and font settings from the Nature Figure workflow.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "Noto Sans SC", "Arial", "DejaVu Sans", "Liberation Sans"
]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8


COLORS = {
    "ink": "#27272E",
    "muted": "#6B6F76",
    "line": "#A8ACB5",
    "baseline": "#484878",
    "baseline_mid": "#7884B4",
    "baseline_soft": "#B4C0E4",
    "hero": "#D996AE",
    "hero_dark": "#9D526F",
    "lilac": "#E9E5F4",
    "aqua": "#E0F0F0",
    "teal": "#4B9CA5",
    "peach": "#F0E0D0",
    "green": "#2E9E44",
    "red": "#C74B50",
    "white": "#FFFFFF",
}


def mm(value: float) -> float:
    return value / 25.4


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.06, y: float = 1.02) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, fontweight="bold", color=COLORS["ink"])


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
          *, color: str = COLORS["line"], rad: float = 0.0, lw: float = 1.25) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2,
        zorder=2,
    ))


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    *,
    face: str,
    edge: str,
    title_color: str = COLORS["ink"],
    body_color: str = COLORS["muted"],
    title_size: float = 8.0,
    body_size: float = 6.5,
    tracked: list[tuple[Any, Any]] | None = None,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.018",
        transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=1.1,
        zorder=3,
    )
    ax.add_patch(patch)
    title_artist = ax.text(
        x + 0.06 * w, y + h - 0.16 * h, title,
        transform=ax.transAxes, ha="left", va="top", fontsize=title_size,
        fontweight="bold", color=title_color, zorder=4,
    )
    body_artist = ax.text(
        x + 0.06 * w, y + h - 0.42 * h, body,
        transform=ax.transAxes, ha="left", va="top", fontsize=body_size,
        color=body_color, linespacing=1.35, zorder=4,
    )
    if tracked is not None:
        tracked.extend([(title_artist, patch), (body_artist, patch)])
    return patch


def assert_figure_bounds(fig: plt.Figure, tracked: list[tuple[Any, Any]], name: str) -> list[str]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    issues: list[str] = []
    tolerance = 2.0
    for artist in fig.findobj(match=lambda item: hasattr(item, "get_window_extent")):
        if not isinstance(artist, mpl.text.Text) or not artist.get_text().strip() or not artist.get_visible():
            continue
        box = artist.get_window_extent(renderer=renderer)
        if (box.x0 < figure_box.x0 - tolerance or box.y0 < figure_box.y0 - tolerance or
                box.x1 > figure_box.x1 + tolerance or box.y1 > figure_box.y1 + tolerance):
            issues.append(f"figure overflow: {artist.get_text()!r}")
    for artist, patch in tracked:
        text_box = artist.get_window_extent(renderer=renderer)
        patch_box = patch.get_window_extent(renderer=renderer)
        pad = 1.5
        if (text_box.x0 < patch_box.x0 - pad or text_box.y0 < patch_box.y0 - pad or
                text_box.x1 > patch_box.x1 + pad or text_box.y1 > patch_box.y1 + pad):
            issues.append(f"box overflow: {artist.get_text()!r}")
    if issues:
        raise RuntimeError(f"{name} layout QA failed:\n" + "\n".join(issues))
    return issues


def export(fig: plt.Figure, stem: str, tracked: list[tuple[Any, Any]]) -> dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    assert_figure_bounds(fig, tracked, stem)
    paths = {
        "svg": OUT / f"{stem}.svg",
        "pdf": OUT / f"{stem}.pdf",
        "tiff": OUT / f"{stem}.tiff",
        "png": OUT / f"{stem}.png",
    }
    fig.savefig(paths["svg"], facecolor="white")
    fig.savefig(paths["pdf"], facecolor="white")
    fig.savefig(paths["tiff"], dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(paths["png"], dpi=300, facecolor="white")
    plt.close(fig)
    svg_text = paths["svg"].read_text(encoding="utf-8")
    if "<text" not in svg_text:
        raise RuntimeError(f"{stem}: SVG text was converted to paths")
    return {key: str(value) for key, value in paths.items()}


def build_architecture() -> dict[str, str]:
    tracked: list[tuple[Any, Any]] = []
    fig, ax = plt.subplots(figsize=(mm(183), mm(112)))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.965, bottom=0.055)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    add_panel_label(ax, "a", x=-0.012, y=0.985)

    rounded_box(
        ax, 0.025, 0.59, 0.17, 0.25,
        "复杂科研查询", "主题 · 方法 · 数据\n时间 · 关系 · 排除项",
        face=COLORS["lilac"], edge=COLORS["baseline_mid"], tracked=tracked,
    )
    rounded_box(
        ax, 0.245, 0.59, 0.18, 0.25,
        "受控多路检索", "词法 / 语义\n引文正向 / 反向\n缺口触发扩展",
        face=COLORS["aqua"], edge=COLORS["teal"], tracked=tracked,
    )
    arrow(ax, (0.195, 0.715), (0.245, 0.715), color=COLORS["baseline_mid"])

    graph_patch = FancyBboxPatch(
        (0.47, 0.30), 0.20, 0.58, boxstyle="round,pad=0.008,rounding_size=0.018",
        transform=ax.transAxes, facecolor="#FAFAFC", edgecolor=COLORS["line"],
        linewidth=1.1, zorder=1,
    )
    ax.add_patch(graph_patch)
    graph_title = ax.text(0.485, 0.84, "关系证据图", transform=ax.transAxes,
                          fontsize=8.2, fontweight="bold", color=COLORS["ink"], va="top")
    graph_sub = ax.text(0.485, 0.785, "candidate–seed evidence", transform=ax.transAxes,
                        fontsize=5.8, color=COLORS["muted"], va="top")
    tracked.extend([(graph_title, graph_patch), (graph_sub, graph_patch)])
    nodes = {
        "C": (0.515, 0.67, COLORS["baseline_soft"]),
        "R": (0.575, 0.72, COLORS["hero"]),
        "D": (0.63, 0.66, COLORS["peach"]),
        "B": (0.525, 0.53, "#CFE6E8"),
        "G": (0.62, 0.52, "#B9DADD"),
        "T": (0.54, 0.40, "#D9D2EA"),
        "X": (0.61, 0.39, "#E8CBD6"),
    }
    links = [("C", "R"), ("R", "D"), ("C", "B"), ("R", "G"),
             ("B", "G"), ("B", "T"), ("G", "X"), ("T", "X")]
    for a, b in links:
        xa, ya, _ = nodes[a]; xb, yb, _ = nodes[b]
        ax.plot([xa, xb], [ya, yb], transform=ax.transAxes, color="#C4C6CD", lw=1.0, zorder=2)
    for label, (x, y, color) in nodes.items():
        circle = Circle((x, y), 0.026, transform=ax.transAxes, facecolor=color,
                        edgecolor=COLORS["white"], linewidth=1.2, zorder=4)
        ax.add_patch(circle)
        ax.text(x, y, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=7.2, fontweight="bold", color=COLORS["ink"], zorder=5)
    legend = ax.text(
        0.485, 0.325,
        "C 条件  R 关系  D 方向\nB 桥接  G 图  T TRACE  X Cross",
        transform=ax.transAxes, fontsize=5.6, color=COLORS["muted"],
        ha="left", va="bottom", linespacing=1.35, zorder=4,
    )
    tracked.append((legend, graph_patch))
    arrow(ax, (0.425, 0.715), (0.47, 0.67), color=COLORS["teal"])

    rounded_box(
        ax, 0.725, 0.67, 0.245, 0.20,
        "严格路径 · REG", "组合条件 × 关系谓词\n方向与桥接证据\n绝对阈值 + 相对间隔",
        face="#F5F3FA", edge=COLORS["baseline"], tracked=tracked,
    )
    rounded_box(
        ax, 0.725, 0.37, 0.245, 0.23,
        "仲裁路径 · MEQ", "Cross · TRACE · Graph\nLexical · Fusion\n3-of-5 仲裁 + Top18 保护",
        face="#FBF2F5", edge=COLORS["hero_dark"], tracked=tracked,
    )
    arrow(ax, (0.67, 0.69), (0.725, 0.77), color=COLORS["baseline"])
    arrow(ax, (0.67, 0.50), (0.725, 0.49), color=COLORS["hero_dark"])

    rounded_box(
        ax, 0.725, 0.09, 0.245, 0.18,
        "结构化结果", "Top20 · 精选集合\n证据说明 · 引文关系图",
        face="#F7F7F8", edge=COLORS["line"], tracked=tracked,
    )
    merge = Circle((0.847, 0.315), 0.012, transform=ax.transAxes,
                   facecolor=COLORS["white"], edgecolor=COLORS["line"], linewidth=1.0, zorder=5)
    ax.add_patch(merge)
    arrow(ax, (0.97, 0.75), (0.86, 0.325), color=COLORS["baseline"], rad=0.25)
    arrow(ax, (0.847, 0.37), (0.847, 0.327), color=COLORS["hero_dark"])
    arrow(ax, (0.847, 0.303), (0.847, 0.27), color=COLORS["line"])

    strip = FancyBboxPatch(
        (0.025, 0.09), 0.645, 0.12, boxstyle="round,pad=0.006,rounding_size=0.015",
        transform=ax.transAxes, facecolor="#F7F8FA", edgecolor="#D8DAE0", linewidth=0.9,
    )
    ax.add_patch(strip)
    cost_title = ax.text(0.045, 0.165, "计算复用", transform=ax.transAxes,
                         fontsize=7.4, fontweight="bold", color=COLORS["ink"], va="center")
    cost_body = ax.text(0.16, 0.165, "已有候选与本地评分", transform=ax.transAxes,
                        fontsize=6.3, color=COLORS["muted"], va="center")
    cost_delta = ax.text(0.405, 0.165, "+0 API   +0 LLM   +0 Token", transform=ax.transAxes,
                         fontsize=7.0, fontweight="bold", color=COLORS["green"], va="center")
    tracked.extend([(cost_title, strip), (cost_body, strip), (cost_delta, strip)])

    return export(fig, "fig1_reg_meq_architecture", tracked)


def metric_rows(state: dict[str, Any], split: str) -> dict[str, dict[str, Any]]:
    return {row["configuration"]: row for row in state[split]}


def build_hard_data(state: dict[str, Any]) -> dict[str, str]:
    tracked: list[tuple[Any, Any]] = []
    fig = plt.figure(figsize=(mm(183), mm(122)))
    gs = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.95), width_ratios=(1.10, 1.25, 1.0),
                          left=0.075, right=0.985, top=0.955, bottom=0.12,
                          hspace=0.48, wspace=0.48)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])
    for label, ax in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        add_panel_label(ax, label)

    blind = metric_rows(state, "blind50")
    recall_labels = ["候选池", "原 Top20", "REG", "REG–MEQ"]
    recall_values = [
        float(state["candidate"]["recall"]),
        float(blind["Baseline Top20"]["recall"]),
        float(blind["REG strict"]["recall"]),
        float(blind["REG + evidence quorum"]["recall"]),
    ]
    colors = [COLORS["baseline_soft"], COLORS["baseline"], COLORS["baseline_mid"], COLORS["hero"]]
    y = np.arange(len(recall_labels))[::-1]
    bars = ax_a.barh(y, recall_values, height=0.58, color=colors, edgecolor="white", linewidth=0.8)
    ax_a.set_yticks(y, recall_labels, fontsize=6.7)
    ax_a.set_xlim(0.36, 0.56)
    ax_a.set_xticks([0.40, 0.45, 0.50, 0.55])
    ax_a.set_xlabel("Recall", fontsize=7)
    ax_a.set_title("盲测检索漏斗  (n=50)", fontsize=7.8, loc="left", pad=7)
    ax_a.tick_params(labelsize=6.2, length=3)
    ax_a.spines["left"].set_visible(False)
    for bar, value in zip(bars, recall_values):
        ax_a.text(value + 0.004, bar.get_y() + bar.get_height()/2, f"{value:.3f}",
                  va="center", ha="left", fontsize=6.3, color=COLORS["ink"])
    ax_a.annotate("+3.0 pp", xy=(recall_values[-1], y[-1]), xytext=(0.515, y[-1] + 0.7),
                  fontsize=6.4, color=COLORS["green"], fontweight="bold",
                  arrowprops=dict(arrowstyle="->", lw=0.9, color=COLORS["green"]))

    splits = ["calibration40", "dev20", "blind50"]
    split_names = ["Calibration\nn=40", "Dev\nn=20", "Blind\nn=50"]
    methods = ["Baseline Top20", "REG strict", "REG + evidence quorum"]
    method_labels = ["Baseline", "REG", "REG–MEQ"]
    method_colors = [COLORS["baseline"], COLORS["baseline_mid"], COLORS["hero"]]
    x = np.arange(len(splits))
    width = 0.22
    for index, (method, label, color) in enumerate(zip(methods, method_labels, method_colors)):
        values = [metric_rows(state, split)[method]["TP"] for split in splits]
        bars = ax_b.bar(x + (index-1)*width, values, width, color=color, edgecolor="white", label=label)
        for bar, value in zip(bars, values):
            ax_b.text(bar.get_x()+bar.get_width()/2, value+0.45, f"{int(value)}",
                      ha="center", va="bottom", fontsize=6.0, color=COLORS["ink"])
    ax_b.set_xticks(x, split_names, fontsize=6.4)
    ax_b.set_ylabel("Top20 命中论文数", fontsize=7)
    ax_b.set_ylim(0, 43)
    ax_b.set_yticks([0, 10, 20, 30, 40])
    ax_b.tick_params(axis="y", labelsize=6.2, length=3)
    ax_b.set_title("跨切分一致性", fontsize=7.8, loc="left", pad=7)
    ax_b.legend(frameon=False, fontsize=6.2, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.0))

    base_recall = float(blind["Baseline Top20"]["recall"])
    base_f1 = float(blind["Baseline Top20"]["f1"])
    rel = {
        "Recall": [100.0, 100*float(blind["REG strict"]["recall"])/base_recall,
                    100*float(blind["REG + evidence quorum"]["recall"])/base_recall],
        "F1": [100.0, 100*float(blind["REG strict"]["f1"])/base_f1,
                100*float(blind["REG + evidence quorum"]["f1"])/base_f1],
    }
    for name, values, color, marker in (
        ("Recall", rel["Recall"], COLORS["teal"], "o"),
        ("F1", rel["F1"], COLORS["hero_dark"], "s"),
    ):
        ax_c.plot(range(3), values, color=color, marker=marker, lw=1.7, ms=4.2, label=name)
        ax_c.text(2.05, values[-1], f"{values[-1]-100:+.1f}%", color=color,
                  fontsize=6.3, fontweight="bold", va="center")
    ax_c.axhline(100, color="#C9CBD1", lw=0.9, ls="--")
    ax_c.set_xticks(range(3), ["Baseline", "REG", "REG–MEQ"], fontsize=6.0)
    ax_c.set_ylabel("相对基线 (%)", fontsize=7)
    ax_c.set_ylim(98, 110)
    ax_c.set_yticks([100, 104, 108])
    ax_c.tick_params(axis="y", labelsize=6.1, length=3)
    ax_c.set_title("盲测相对提升", fontsize=7.8, loc="left", pad=7)
    ax_c.legend(frameon=False, fontsize=6.2, loc="upper left")

    with ABLATION_PATH.open(encoding="utf-8-sig", newline="") as handle:
        ablations = list(csv.DictReader(handle))
    chosen_names = ["Full REG", "w/o condition conjunction", "w/o seed bridge and graph", "w/o local cross-encoder"]
    label_map = {
        "Full REG": "完整 REG",
        "w/o condition conjunction": "– 条件合取",
        "w/o seed bridge and graph": "– 桥接与图",
        "w/o local cross-encoder": "– Cross",
    }
    by_name = {row["configuration"]: row for row in ablations}
    values = [float(by_name[name]["recall"]) for name in chosen_names]
    yy = np.arange(len(values))[::-1]
    alpha_colors = [COLORS["baseline"], "#9FA8CB", "#BAC1DB", "#CDD2E5"]
    bars = ax_d.barh(yy, values, height=0.58, color=alpha_colors, edgecolor="white")
    ax_d.set_yticks(yy, [label_map[name] for name in chosen_names], fontsize=5.9)
    ax_d.set_xlim(0.401, 0.421)
    ax_d.set_xticks([0.405, 0.415])
    ax_d.tick_params(axis="x", labelsize=6.0, length=3)
    ax_d.set_xlabel("Recall", fontsize=7)
    ax_d.set_title("冻结路由消融  (n=50)", fontsize=7.8, loc="left", pad=7)
    ax_d.spines["left"].set_visible(False)
    for bar, value in zip(bars, values):
        ax_d.text(value + 0.0004, bar.get_y()+bar.get_height()/2, f"{value:.3f}",
                  fontsize=5.9, va="center", ha="left")

    fig.text(0.075, 0.035,
             "成本：6.96 OpenAlex calls/query  |  本地重排 1.46 s/query  |  MEQ 新增 0 API · 0 LLM · 0 Token",
             fontsize=6.5, color=COLORS["muted"], ha="left", va="center")
    fig.text(0.985, 0.035, "50/50 LIVE–REPLAY 一致", fontsize=6.5,
             color=COLORS["green"], fontweight="bold", ha="right", va="center")
    return export(fig, "fig2_reg_meq_hard_data", tracked)


def write_sources(state: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("calibration40", "dev20", "blind50"):
        for row in state[split]:
            rows.append({"split": split, **row})
    with (OUT / "figure_source_data.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    contract = """# Figure contract

Core conclusion: REG-MEQ converts additional candidate-pool hits into Top20 gains on 50 blind queries without extra online cost.
Figure archetype: schematic-led architecture + quantitative grid.
Target/output: 183-mm double-column; SVG/PDF/TIFF/PNG.
Backend: Python (Matplotlib only).
Panel map: Fig.1 end-to-end mechanism; Fig.2a recall funnel; 2b split consistency; 2c relative gain; 2d frozen-route ablation.
Evidence hierarchy: blind50 is hero evidence; calibration/dev are validation; fixed-route ablation is mechanism support.
Statistics: fixed disjoint splits; paired 10,000-draw query bootstrap available in the experiment report.
Source data: evidence_quorum_v20/state.json and reg_harddata_v20 ablation CSV.
Image integrity: vector-native schematic and plots; no raster source manipulation.
Reviewer risk: modest number of improved blind queries; report exact TP and split sizes alongside macro metrics.
"""
    (OUT / "figure_contract.md").write_text(contract, encoding="utf-8")


def main() -> None:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    write_sources(state)
    outputs = {
        "architecture": build_architecture(),
        "hard_data": build_hard_data(state),
    }
    qa = {
        "backend": "Python/Matplotlib",
        "final_width_mm": 183,
        "editable_svg_text": True,
        "box_overflow_checks": "passed",
        "figure_boundary_checks": "passed",
        "exports": outputs,
    }
    (OUT / "qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
