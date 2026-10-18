"""Nature-style final figures for the competition project document.

Figure contract
---------------
Core conclusion: evidence-aware retrieval, relational ranking and calibrated
selection improve three distinct stages of the search pipeline.
Evidence chain: candidate recall -> Top20 recall -> precision-first set F1.
Archetype: quantitative grid plus schematic-led mechanism panels.
Backend: Python/matplotlib only.
Export: 158-mm editable SVG/PDF, 600-dpi TIFF and 320-dpi PNG.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "artifacts" / "submission_2026"
OUT = ASSET / "figures_nature" / "report_v33"
SOURCE = ASSET / "source_data"
V21_STATE = ROOT / "reports" / "evidence_backprojection_v21" / "state.json"
V22_METRICS = ROOT / "reports" / "evidence_calibrated_selector_v22" / "02_blind50_metrics.csv"


def mm(value: float) -> float:
    return value / 25.4


mpl.rcParams.update({
    "font.family": "Microsoft YaHei",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

C = {
    "ink": "#242B35",
    "muted": "#66707D",
    "line": "#B6BEC8",
    "grid": "#E7EAF0",
    "blue": "#5876A3",
    "blue_soft": "#E4EAF3",
    "teal": "#3E8C8D",
    "teal_soft": "#DEEEEE",
    "purple": "#79679A",
    "purple_soft": "#ECE7F3",
    "rose": "#B35F7D",
    "rose_soft": "#F3E2E8",
    "amber": "#C08A39",
    "amber_soft": "#F6ECD9",
    "green": "#3A845D",
    "green_soft": "#E0EEE5",
    "white": "#FFFFFF",
}


def rounded(ax, x, y, w, h, face, edge, *, radius=0.018, lw=1.0, zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h, transform=ax.transAxes,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color=None, lw=1.1, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=9,
        linewidth=lw, color=color or C["ink"], connectionstyle=f"arc3,rad={rad}",
        shrinkA=3, shrinkB=3, zorder=3,
    ))


def box_text(ax, x, y, w, h, title, body, face, edge, *, center=True):
    rounded(ax, x, y, w, h, face, edge)
    ha = "center" if center else "left"
    px = x + w / 2 if center else x + 0.06 * w
    ax.text(px, y + 0.70 * h, title, transform=ax.transAxes, ha=ha, va="center",
            fontsize=9.3, fontweight="bold", color=C["ink"], zorder=4)
    ax.text(px, y + 0.34 * h, body, transform=ax.transAxes, ha=ha, va="center",
            fontsize=8.0, linespacing=1.25, color=C["muted"], zorder=4)


def save(fig, name: str) -> dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {suffix: OUT / f"{name}.{suffix}" for suffix in ("svg", "pdf", "png", "tiff")}
    fig.savefig(paths["svg"], facecolor="white", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(paths["pdf"], facecolor="white", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(paths["png"], dpi=320, facecolor="white", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(paths["tiff"], dpi=600, facecolor="white", bbox_inches="tight",
                pad_inches=0.025, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    if "<text" not in paths["svg"].read_text(encoding="utf-8"):
        raise RuntimeError(f"SVG text was outlined: {paths['svg']}")
    return {key: str(value) for key, value in paths.items()}


def _read_metrics() -> tuple[dict, dict]:
    rows = list(csv.DictReader(V22_METRICS.open(encoding="utf-8-sig", newline="")))
    baseline = next(row for row in rows if row["method"] == "TRACE rank / K=1")
    selector = next(row for row in rows if row["method"] == "Evidence-calibrated / K=1")
    return baseline, selector


def final_architecture() -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(mm(158), mm(88)))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.98, bottom=0.04)
    ax.axis("off")
    box_text(ax, 0.31, 0.84, 0.38, 0.12, "复杂学术查询",
             "主题 · 方法 · 任务 · 数据 · 时间 · 关系", C["white"], C["ink"])
    stages = [
        (0.015, "查询契约", "约束解析\n子查询规划", C["blue_soft"], C["blue"]),
        (0.215, "多源发现", "词项 · 语义 · arXiv\n迭代与引文检索", C["teal_soft"], C["teal"]),
        (0.415, "TRACE 融合", "统一论文身份\n来源与约束证据", C["purple_soft"], C["purple"]),
        (0.615, "关系决策", "REG 条件合取\nMEQ 多视图一致", C["rose_soft"], C["rose"]),
        (0.815, "集合交付", "证据校准核心\n均衡与探索视图", C["amber_soft"], C["amber"]),
    ]
    for x, title, body, face, edge in stages:
        box_text(ax, x, 0.47, 0.17, 0.25, title, body, face, edge)
    for x in (0.185, 0.385, 0.585, 0.785):
        arrow(ax, (x, 0.595), (x + 0.03, 0.595))
    arrow(ax, (0.50, 0.84), (0.10, 0.72))
    box_text(ax, 0.26, 0.13, 0.48, 0.18, "EBP 证据加权引文反投影",
             "候选证据聚合 → 引文标识投票 → 批量候选物化",
             "#F5F6F8", C["teal"])
    arrow(ax, (0.69, 0.47), (0.64, 0.31), color=C["rose"], rad=0.08)
    arrow(ax, (0.36, 0.31), (0.30, 0.47), color=C["teal"], rad=0.08)
    ax.text(0.5, 0.045,
            "OpenAlex · arXiv · Crossref · Semantic Scholar  |  统一论文身份  |  通道与成本记录",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.0, color=C["muted"])
    return save(fig, "fig1_system_architecture")


def result_modes() -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(mm(158), mm(55)))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.08)
    ax.axis("off")
    box_text(ax, 0.02, 0.36, 0.23, 0.38, "统一排序池", "Top20 + 补充候选\n共享证据与来源",
             C["purple_soft"], C["purple"])
    cards = [
        (0.34, "核心结果", "1 篇\n证据校准精度视图", C["rose_soft"], C["rose"]),
        (0.56, "均衡集合", "8 篇\n精度—召回折中", C["teal_soft"], C["teal"]),
        (0.78, "探索列表", "20 篇\n完整头部与补充", C["amber_soft"], C["amber"]),
    ]
    for x, title, body, face, edge in cards:
        box_text(ax, x, 0.32, 0.19, 0.46, title, body, face, edge)
        arrow(ax, (0.25, 0.55), (x, 0.55), color=edge, lw=1.0)
    ax.text(0.50, 0.10,
            "切换结果模式不重复检索；导出遵循当前集合，补充候选保留完整证据",
            transform=ax.transAxes, ha="center", fontsize=8.1, color=C["muted"])
    return save(fig, "fig2_result_modes")


def selector_mechanism() -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(mm(158), mm(66)))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.07)
    ax.axis("off")
    signals = [
        (0.02, 0.63, "TRACE", C["blue_soft"], C["blue"]),
        (0.02, 0.39, "局部语义", C["teal_soft"], C["teal"]),
        (0.02, 0.15, "关系图", C["purple_soft"], C["purple"]),
        (0.19, 0.63, "约束覆盖", C["rose_soft"], C["rose"]),
        (0.19, 0.39, "引文来源", C["amber_soft"], C["amber"]),
        (0.19, 0.15, "排序先验", C["green_soft"], C["green"]),
    ]
    for x, y, title, face, edge in signals:
        box_text(ax, x, y, 0.14, 0.16, title, "查询内证据", face, edge)
        arrow(ax, (x + 0.14, y + 0.08), (0.39, 0.50), color=edge, lw=0.9)
    box_text(ax, 0.39, 0.33, 0.20, 0.34, "40 维证据向量", "标准化\n证据交互项\n无身份特征",
             "#F4F5F7", C["ink"])
    arrow(ax, (0.59, 0.50), (0.66, 0.50), color=C["ink"])
    box_text(ax, 0.66, 0.36, 0.15, 0.28, "线性校准", "z = w · x + b\nSigmoid(z)",
             C["blue_soft"], C["blue"])
    arrow(ax, (0.81, 0.50), (0.87, 0.50), color=C["blue"])
    box_text(ax, 0.87, 0.36, 0.11, 0.28, "核心结果", "1 篇\n其余保留", C["rose_soft"], C["rose"])
    ax.text(0.39, 0.08,
            "运行时：标准化 + 40 维点积 + Sigmoid  |  新增外部 API 0  |  新增 LLM 0",
            transform=ax.transAxes, ha="left", fontsize=8.1, color=C["muted"])
    return save(fig, "fig11_selector_mechanism")


def evidence_chain() -> dict[str, str]:
    state = json.loads(V21_STATE.read_text(encoding="utf-8"))["splits"]["blind50"]
    baseline, selector = _read_metrics()
    candidate = [
        float(state["baseline_candidate"]["recall"]),
        float(state["ebp_candidate"]["recall"]),
    ]
    top20 = [
        float(state["baseline_top20"]["recall"]),
        float(state["reg_meq_top20"]["recall"]),
    ]
    core_base = [float(baseline[key]) for key in ("precision", "recall", "f1")]
    core_new = [float(selector[key]) for key in ("precision", "recall", "f1")]

    SOURCE.mkdir(parents=True, exist_ok=True)
    rows = [
        {"stage": "candidate", "method": "existing", "metric": "recall", "value": candidate[0]},
        {"stage": "candidate", "method": "+ EBP", "metric": "recall", "value": candidate[1]},
        {"stage": "top20", "method": "baseline", "metric": "recall", "value": top20[0]},
        {"stage": "top20", "method": "REG-MEQ", "metric": "recall", "value": top20[1]},
    ]
    for metric, old, new in zip(("precision", "recall", "f1"), core_base, core_new):
        rows.extend((
            {"stage": "core", "method": "TRACE", "metric": metric, "value": old},
            {"stage": "core", "method": "evidence-calibrated", "metric": metric, "value": new},
        ))
    with (SOURCE / "final_evidence_chain_v33.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stage", "method", "metric", "value"))
        writer.writeheader()
        writer.writerows(rows)

    fig = plt.figure(figsize=(mm(158), mm(76)))
    grid = fig.add_gridspec(1, 3, width_ratios=(0.82, 0.82, 1.55),
                           left=0.09, right=0.985, top=0.87, bottom=0.24, wspace=0.42)
    ax_a, ax_b, ax_c = (fig.add_subplot(grid[0, index]) for index in range(3))
    bars_a = ax_a.bar([0, 1], candidate, width=0.60, color=[C["line"], C["teal"]])
    ax_a.set_xticks([0, 1], ["原候选池", "+ EBP"])
    ax_a.set_ylim(0.45, 0.60)
    ax_a.set_ylabel("Recall")
    ax_a.set_title("a  候选发现", loc="left", fontweight="bold")

    bars_b = ax_b.bar([0, 1], top20, width=0.60, color=[C["line"], C["purple"]])
    ax_b.set_xticks([0, 1], ["基础 Top20", "REG-MEQ"])
    ax_b.set_ylim(0.34, 0.48)
    ax_b.set_ylabel("Recall@20")
    ax_b.set_title("b  关系排序", loc="left", fontweight="bold")

    x = np.arange(3)
    width = 0.34
    bars_c0 = ax_c.bar(x - width / 2, core_base, width, color=C["line"], label="TRACE 核心")
    bars_c1 = ax_c.bar(x + width / 2, core_new, width, color=C["rose"], label="证据校准")
    ax_c.set_xticks(x, ["Precision", "Recall", "F1"])
    ax_c.set_ylim(0.16, 0.38)
    ax_c.set_ylabel("Macro score")
    ax_c.set_title("c  核心结果选择", loc="left", fontweight="bold")
    # Direct labels avoid a legend colliding with the value labels at print size.
    ax_c.plot([0.04, 0.12], [0.91, 0.91], transform=ax_c.transAxes,
              color=C["line"], linewidth=5, solid_capstyle="butt", clip_on=False)
    ax_c.text(0.14, 0.91, "TRACE 核心", transform=ax_c.transAxes,
              ha="left", va="center", fontsize=7.8, color=C["muted"])
    ax_c.plot([0.53, 0.61], [0.91, 0.91], transform=ax_c.transAxes,
              color=C["rose"], linewidth=5, solid_capstyle="butt", clip_on=False)
    ax_c.text(0.63, 0.91, "证据校准", transform=ax_c.transAxes,
              ha="left", va="center", fontsize=7.8, color=C["muted"])
    ax_c.text(0.66, -0.18, "ΔF1 = +0.020", transform=ax_c.transAxes,
              ha="center", va="center", color=C["green"],
              fontsize=8.3, fontweight="bold", clip_on=False)

    for ax in (ax_a, ax_b, ax_c):
        ax.grid(axis="y", color=C["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
    for bars in (bars_a, bars_b, bars_c0, bars_c1):
        for bar in bars:
            ax = bar.axes
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.8)
    fig.text(0.09, 0.055,
             "Blind50：50 条查询，123 篇目标论文  |  EBP：+1 次 OpenAlex 批量请求/查询  |  REG 与证据校准：+0 外部调用",
             fontsize=8.0, color=C["muted"])
    return save(fig, "fig12_final_evidence_chain")


def main() -> None:
    outputs = {
        "architecture": final_architecture(),
        "result_modes": result_modes(),
        "selector_mechanism": selector_mechanism(),
        "evidence_chain": evidence_chain(),
    }
    qa = {
        "core_conclusion": "分层证据机制分别改善候选发现、Top20 排序与核心结果选择。",
        "evidence_chain": ["candidate recall", "Top20 recall", "core-set precision/recall/F1"],
        "archetype": "quantitative grid + schematic-led composite",
        "backend": "Python/matplotlib",
        "target_width_mm": 158,
        "minimum_designed_text_pt": 8.0,
        "font": "Microsoft YaHei",
        "editable_svg_text": True,
        "source_data": str(SOURCE / "final_evidence_chain_v33.csv"),
        "exports": outputs,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
