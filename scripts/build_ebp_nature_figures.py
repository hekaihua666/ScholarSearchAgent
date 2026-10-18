"""Nature-style figures for EBP v21 using Python/Matplotlib exclusively.

Figure contract
---------------
Core conclusion: multi-view evidence backprojection recovers papers that are
implicitly reachable through the existing citation graph and raises candidate
recall on all three frozen splits with one batched API call per query.
Archetypes: schematic-led composite (Fig. 3); quantitative grid (Fig. 4).
Target/output: 183-mm double-column, editable SVG/PDF plus 600-dpi TIFF and PNG.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/evidence_backprojection_v21"
OUT = ROOT / "artifacts/submission_2026/figures_nature/evidence_backprojection_v21"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": .8,
    "legend.frameon": False,
})

P = {
    "ink": "#252631",
    "muted": "#686C75",
    "grid": "#E4E6EA",
    "navy": "#485178",
    "blue": "#7B8EBC",
    "blue_soft": "#DCE3F2",
    "teal": "#4D9A98",
    "teal_soft": "#DCEFED",
    "rose": "#B76582",
    "rose_soft": "#F2DEE5",
    "amber": "#D39A4B",
    "amber_soft": "#F5E8D4",
    "green": "#3A8D5C",
    "white": "#FFFFFF",
}


def mm(value: float) -> float:
    return value / 25.4


def panel_label(ax: plt.Axes, label: str, x: float = -.035, y: float = 1.01) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            color=P["ink"], ha="left", va="bottom")


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
          *, color: str = P["navy"], rad: float = 0, lw: float = 1.15) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=8,
        linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}",
        shrinkA=3, shrinkB=3, zorder=2,
    ))


def box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, body: str,
        *, face: str, edge: str, title_size: float = 7.3, body_size: float = 5.8,
        tracked: list[tuple[Any, Any]]) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y), w, h, transform=ax.transAxes,
        boxstyle="round,pad=0.006,rounding_size=0.016",
        facecolor=face, edgecolor=edge, linewidth=1.0, zorder=3,
    )
    ax.add_patch(patch)
    title_artist = ax.text(
        x + .06*w, y + h - .16*h, title, transform=ax.transAxes,
        fontsize=title_size, fontweight="bold", color=P["ink"], ha="left", va="top", zorder=4,
    )
    body_artist = ax.text(
        x + .06*w, y + h - .45*h, body, transform=ax.transAxes,
        fontsize=body_size, color=P["muted"], ha="left", va="top",
        linespacing=1.35, zorder=4,
    )
    tracked.extend([(title_artist, patch), (body_artist, patch)])
    return patch


def validate(fig: plt.Figure, tracked: list[tuple[Any, Any]], stem: str) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tolerance = 2
    for artist, patch in tracked:
        text_box = artist.get_window_extent(renderer)
        patch_box = patch.get_window_extent(renderer)
        if (text_box.x0 < patch_box.x0 - tolerance or text_box.x1 > patch_box.x1 + tolerance or
                text_box.y0 < patch_box.y0 - tolerance or text_box.y1 > patch_box.y1 + tolerance):
            raise RuntimeError(f"{stem}: text exceeds box: {artist.get_text()!r}")
    figure_box = fig.bbox
    for artist in fig.findobj(match=lambda item: isinstance(item, mpl.text.Text)):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        text_box = artist.get_window_extent(renderer)
        if (text_box.x0 < figure_box.x0 - tolerance or text_box.x1 > figure_box.x1 + tolerance or
                text_box.y0 < figure_box.y0 - tolerance or text_box.y1 > figure_box.y1 + tolerance):
            raise RuntimeError(f"{stem}: text exceeds figure: {artist.get_text()!r}")


def export(fig: plt.Figure, stem: str, tracked: list[tuple[Any, Any]]) -> dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    validate(fig, tracked, stem)
    paths = {suffix: OUT / f"{stem}.{suffix}" for suffix in ("svg", "pdf", "tiff", "png")}
    fig.savefig(paths["svg"], facecolor="white")
    fig.savefig(paths["pdf"], facecolor="white")
    fig.savefig(paths["tiff"], dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(paths["png"], dpi=300, facecolor="white")
    plt.close(fig)
    if "<text" not in paths["svg"].read_text(encoding="utf-8"):
        raise RuntimeError(f"{stem}: SVG text is not editable")
    return {key: str(value) for key, value in paths.items()}


def build_architecture() -> dict[str, str]:
    tracked: list[tuple[Any, Any]] = []
    fig, ax = plt.subplots(figsize=(mm(183), mm(108)))
    fig.subplots_adjust(left=.025, right=.985, top=.965, bottom=.06)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel_label(ax, "a", x=-.012, y=.985)

    box(ax, .025, .62, .17, .23, "已有候选证据场",
        "Top20 与扩展候选\n保留标题、摘要与引文边",
        face=P["blue_soft"], edge=P["blue"], tracked=tracked)
    arrow(ax, (.195, .735), (.235, .735), color=P["blue"])

    evidence = [("Cross", P["rose_soft"], P["rose"]), ("TRACE", P["blue_soft"], P["navy"]),
                ("Graph", P["teal_soft"], P["teal"]), ("Lexical", P["amber_soft"], P["amber"]),
                ("Fusion", "#EAE6F2", "#7E6A9F")]
    y_positions = [.82, .705, .59, .475, .36]
    for (label, face, edge), y in zip(evidence, y_positions):
        patch = FancyBboxPatch((.245, y), .115, .075, transform=ax.transAxes,
                               boxstyle="round,pad=.005,rounding_size=.012",
                               facecolor=face, edgecolor=edge, linewidth=.9, zorder=3)
        ax.add_patch(patch)
        artist = ax.text(.3025, y+.0375, label, transform=ax.transAxes, ha="center", va="center",
                         fontsize=6.2, fontweight="bold", color=P["ink"], zorder=4)
        tracked.append((artist, patch))
        arrow(ax, (.36, y+.0375), (.405, .64), color=edge, lw=.85)

    quorum = box(ax, .405, .545, .18, .20, "稳健多视图证据",
                 "第三强视图控制单点偏差\n前 3 个视图形成一致性加成",
                 face="#F7F5FA", edge=P["navy"], tracked=tracked)
    formula = ax.text(.495, .575, "e = s3 + 0.05 · avg(top3)", transform=ax.transAxes,
                      ha="center", va="center", fontsize=5.3, color=P["navy"], fontweight="bold", zorder=4)
    tracked.append((formula, quorum))
    arrow(ax, (.585, .64), (.625, .64), color=P["navy"])

    vote = box(ax, .625, .545, .16, .20, "引文反投影",
               "证据沿参考文献边回传\n长参考列表按 √|R| 归一化",
               face=P["teal_soft"], edge=P["teal"], tracked=tracked)
    formula2 = ax.text(.705, .575, "vote = e / sqrt(|R|)", transform=ax.transAxes,
                       ha="center", va="center", fontsize=5.5, color=P["teal"], fontweight="bold", zorder=4)
    tracked.append((formula2, vote))
    arrow(ax, (.785, .64), (.825, .64), color=P["teal"])

    box(ax, .825, .545, .15, .20, "一次批量物化",
        "聚合相同 OpenAlex ID\nTop100 目标 / 1 次请求",
        face=P["amber_soft"], edge=P["amber"], tracked=tracked)

    # Citation evidence graph: several source papers converge on unresolved targets.
    graph = FancyBboxPatch((.07, .09), .64, .22, transform=ax.transAxes,
                           boxstyle="round,pad=.006,rounding_size=.015",
                           facecolor="#FAFAFB", edgecolor=P["grid"], linewidth=.9)
    ax.add_patch(graph)
    label = ax.text(.09, .275, "全候选共同决定下一跳", transform=ax.transAxes,
                    fontsize=7.1, fontweight="bold", color=P["ink"], va="top")
    tracked.append((label, graph))
    sources = [(.13, .16, "P1"), (.25, .22, "P2"), (.34, .14, "P3"), (.46, .22, "P4")]
    targets = [(.56, .20, "Wa", P["rose_soft"], P["rose"]),
               (.64, .135, "Wb", P["teal_soft"], P["teal"])]
    edges = [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1)]
    for si, ti in edges:
        sx, sy, _ = sources[si]; tx, ty, *_ = targets[ti]
        ax.plot([sx, tx], [sy, ty], transform=ax.transAxes, color="#BFC3CA", lw=.9, zorder=1)
    for x, y, text in sources:
        circle = Circle((x, y), .024, transform=ax.transAxes, facecolor=P["blue_soft"],
                        edgecolor=P["blue"], linewidth=.9, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, text, transform=ax.transAxes, ha="center", va="center", fontsize=5.8,
                color=P["ink"], fontweight="bold", zorder=3)
    for x, y, text, face, edge in targets:
        circle = Circle((x, y), .03, transform=ax.transAxes, facecolor=face,
                        edgecolor=edge, linewidth=1.1, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, text, transform=ax.transAxes, ha="center", va="center", fontsize=6,
                color=P["ink"], fontweight="bold", zorder=3)
    ax.text(.555, .255, "3 个证据源", transform=ax.transAxes, fontsize=5.6,
            color=P["rose"], ha="center", va="bottom")
    ax.text(.65, .09, "2 个证据源", transform=ax.transAxes, fontsize=5.6,
            color=P["teal"], ha="center", va="top")

    box(ax, .755, .09, .22, .22, "可解释候选边界",
        "新增论文 + 反投影得分\n支持论文数 + 支持标题\n进入补充候选，不扰动冻结 Top20",
        face="#F5F6F8", edge="#A7ABB4", tracked=tracked)
    arrow(ax, (.71, .20), (.755, .20), color=P["muted"])
    return export(fig, "fig3_ebp_architecture", tracked)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_hard_data() -> dict[str, str]:
    summary = read_csv(REPORT / "06_summary_metrics.csv")
    curve = read_csv(REPORT / "04_target_budget_curve.csv")
    state = json.loads((REPORT / "state.json").read_text(encoding="utf-8"))
    tracked: list[tuple[Any, Any]] = []

    fig = plt.figure(figsize=(mm(183), mm(116)))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.05, 1.35), height_ratios=(1, 1),
                          left=.075, right=.98, top=.945, bottom=.13,
                          wspace=.40, hspace=.48)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    for label, ax in zip("abc", (ax_a, ax_b, ax_c)):
        panel_label(ax, label)

    splits = ["calibration", "dev", "blind50"]
    labels = ["Calibration\n(n=40)", "Dev\n(n=20)", "Blind\n(n=50)"]
    base = {row["split"]: float(row["recall"]) for row in summary if row["method"] == "Existing candidate pool"}
    ebp = {row["split"]: float(row["recall"]) for row in summary if row["method"] == "Existing pool + EBP"}
    x = np.arange(3)
    width = .34
    bars0 = ax_a.bar(x-width/2, [base[s] for s in splits], width, color=P["navy"], label="原候选池")
    bars1 = ax_a.bar(x+width/2, [ebp[s] for s in splits], width, color=P["rose"], label="+ EBP")
    ax_a.set_xticks(x, labels, fontsize=6.3)
    ax_a.set_ylim(.36, .61)
    ax_a.set_yticks([.40, .45, .50, .55, .60])
    ax_a.set_ylabel("候选 Recall", fontsize=7)
    ax_a.set_title("三个隔离数据集均提高", fontsize=7.8, loc="left", pad=8)
    ax_a.grid(axis="y", color=P["grid"], lw=.65, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.legend(fontsize=6.2, loc="upper left", ncol=2)
    for bars, values in ((bars0, [base[s] for s in splits]), (bars1, [ebp[s] for s in splits])):
        for bar, value in zip(bars, values):
            ax_a.text(bar.get_x()+bar.get_width()/2, value+.006, f"{value:.3f}",
                      ha="center", va="bottom", fontsize=5.8, color=P["ink"])
    for i, split in enumerate(splits):
        delta = (ebp[split]-base[split])*100
        ax_a.text(i, .382, f"+{delta:.1f} pp", ha="center", va="center",
                  fontsize=6.1, color=P["green"], fontweight="bold")

    colors = {"calibration": P["blue"], "dev": P["teal"], "blind50": P["rose"]}
    for split, label in zip(splits, ["Calibration", "Dev", "Blind"]):
        rows = sorted((row for row in curve if row["split"] == split), key=lambda row: int(row["target_limit"]))
        budgets = [int(row["target_limit"]) for row in rows]
        recalls = [float(row["candidate_recall"]) for row in rows]
        ax_b.plot(budgets, recalls, color=colors[split], marker="o", ms=3.5,
                  lw=1.55, label=label)
    ax_b.set_xlim(-3, 103)
    ax_b.set_xticks([0, 25, 50, 75, 100])
    ax_b.set_ylim(.40, .59)
    ax_b.set_yticks([.40, .45, .50, .55])
    ax_b.set_xlabel("反投影目标预算", fontsize=7)
    ax_b.set_ylabel("候选 Recall", fontsize=7)
    ax_b.set_title("预算—召回响应", fontsize=7.8, loc="left", pad=7)
    ax_b.grid(color=P["grid"], lw=.65)
    ax_b.legend(fontsize=6.1, ncol=3, loc="lower right")

    blind = state["splits"]["blind50"]
    base_tp = int(blind["baseline_candidate"]["TP"])
    ebp_tp = int(blind["ebp_candidate"]["TP"])
    top20_tp = int(blind["reg_meq_top20"]["TP"])
    stages = ["原候选池", "+ EBP", "冻结 Top20"]
    values = [base_tp, ebp_tp, top20_tp]
    stage_colors = [P["navy"], P["rose"], P["blue"]]
    bars = ax_c.barh(np.arange(3)[::-1], values, height=.55, color=stage_colors, edgecolor="white")
    ax_c.set_yticks(np.arange(3)[::-1], stages, fontsize=6.3)
    ax_c.set_xlim(0, 60)
    ax_c.set_xticks([0, 20, 40, 60])
    ax_c.set_xlabel("Blind50 命中论文数", fontsize=7)
    ax_c.set_title("召回扩展与集合决策解耦", fontsize=7.8, loc="left", pad=7)
    ax_c.grid(axis="x", color=P["grid"], lw=.65)
    ax_c.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax_c.text(value+1, bar.get_y()+bar.get_height()/2, str(value), ha="left", va="center",
                  fontsize=6.3, color=P["ink"], fontweight="bold")
    ax_c.annotate("新增 3 篇候选命中", xy=(ebp_tp, 1), xytext=(42, .35),
                  fontsize=6.2, color=P["green"], fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=P["green"], lw=.9))

    cost = blind["cost"]
    fig.text(.075, .045,
             f"Blind50：+1 OpenAlex 批量请求/查询  |  检索 {cost['mean_retrieval_seconds']:.2f} s/查询  |  本地评分 {cost['mean_local_scoring_seconds']:.2f} s/查询  |  +0 LLM / +0 Token",
             fontsize=6.2, color=P["muted"], ha="left", va="center")
    fig.text(.98, .045, "冻结回放 · n=50", fontsize=6.2, color=P["green"],
             fontweight="bold", ha="right", va="center")
    return export(fig, "fig4_ebp_hard_data", tracked)


def write_source_bundle() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_rows = []
    for path in (REPORT / "06_summary_metrics.csv", REPORT / "04_target_budget_curve.csv"):
        for row in read_csv(path):
            source_rows.append({"source_table": path.name, **row})
    fields = sorted({key for row in source_rows for key in row})
    with (OUT / "figure_source_data.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(source_rows)
    contract = """# Figure contract

Core conclusion: evidence-weighted citation backprojection recovers implicitly reachable papers and raises candidate recall on all three frozen splits with one batched API call per query.
Evidence chain: Fig.3 explains whole-pool evidence aggregation and citation backprojection; Fig.4a verifies cross-split recall gains; Fig.4b shows budget response; Fig.4c separates candidate discovery from frozen Top20 selection.
Archetypes: schematic-led composite and quantitative grid.
Backend: Python/Matplotlib exclusively.
Target: 183-mm double-column; editable SVG/PDF; 600-dpi TIFF; 300-dpi PNG preview.
Split definition: calibration40, dev20 and blind50 are fixed, disjoint query sets.
Metric: macro candidate Recall; exact TP counts are shown for Blind50.
Source data: frozen EBP v21 experiment tables generated by the evaluation runner.
Review risk: EBP expands the candidate frontier; final Top20 improvement requires a separately validated selector.
"""
    (OUT / "figure_contract.md").write_text(contract, encoding="utf-8")


def main() -> None:
    write_source_bundle()
    outputs = {"architecture": build_architecture(), "hard_data": build_hard_data()}
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
