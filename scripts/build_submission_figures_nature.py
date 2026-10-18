from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "artifacts" / "submission_2026"
DATA_DIR = ASSET_ROOT / "source_data"
OUT_DIR = ASSET_ROOT / "figures_nature"

# Restrained Nature/NMI family: neutral baseline, muted violet support, rose hero.
INK = "#292B33"
MUTED = "#747780"
AXIS = "#B9BCC4"
GRID = "#ECECF1"
BASE = "#9EA3AD"
BASE_DARK = "#565A73"
SUPPORT = "#7B86B6"
SUPPORT_SOFT = "#C7CCE4"
HERO = "#C86F86"
HERO_DARK = "#9F4F66"
HERO_SOFT = "#F0CED8"
GAIN = "#2E8B57"
TEAL = "#4D9BA3"
GOLD = "#D2A13C"
WHITE = "#FFFFFF"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "font.size": 7.3,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 7.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def read_rows(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def style_ax(ax: plt.Axes, *, grid: str | None = None) -> None:
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(color=AXIS, labelcolor=INK, width=0.65, length=2.7)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8.6, fontweight="bold", color=INK, va="bottom")


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=420, bbox_inches="tight")
    fig.savefig(
        OUT_DIR / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def dumbbell(
    ax: plt.Axes,
    labels: list[str],
    before: np.ndarray,
    after: np.ndarray,
    *,
    xlim: tuple[float, float],
    xlabel: str,
    fmt: str,
    delta_scale: float = 100.0,
) -> None:
    y = np.arange(len(labels))[::-1]
    for yi, b, a in zip(y, before, after):
        ax.plot([b, a], [yi, yi], color=SUPPORT_SOFT, lw=2.2, solid_capstyle="round", zorder=1)
        ax.scatter(b, yi, s=27, marker="s", color=BASE, edgecolor=WHITE, lw=0.5, zorder=3)
        ax.scatter(a, yi, s=35, marker="o", color=HERO, edgecolor=WHITE, lw=0.6, zorder=4)
        ax.text(b, yi + 0.19, fmt.format(b), ha="center", color=MUTED, fontsize=6.4)
        ax.text(a, yi + 0.19, fmt.format(a), ha="center", color=HERO_DARK, fontsize=6.6, fontweight="bold")
        ax.text((a + b) / 2, yi - 0.26, f"+{(a-b)*delta_scale:.1f}", ha="center", color=GAIN, fontsize=6.2)
    ax.set_yticks(y, labels)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    style_ax(ax, grid="x")


def fig1_reg_generalization() -> None:
    rows = read_rows("reg_across_splits.csv")
    labels = [r["split"] for r in rows]
    br = np.array([float(r["baseline_recall"]) for r in rows])
    rr = np.array([float(r["reg_recall"]) for r in rows])
    bf = np.array([float(r["baseline_f1"]) for r in rows])
    rf = np.array([float(r["reg_f1"]) for r in rows])
    gains = np.array([int(r["reg_tp"]) - int(r["baseline_tp"]) for r in rows])

    fig = plt.figure(figsize=(7.20, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 0.78], wspace=0.55)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    dumbbell(ax1, labels, br, rr, xlim=(0.27, 0.55), xlabel="Macro Recall", fmt="{:.3f}")
    dumbbell(ax2, labels, bf, rf, xlim=(0.045, 0.115), xlabel="Macro F1", fmt="{:.3f}", delta_scale=1000)
    panel_label(ax1, "a")
    panel_label(ax2, "b")

    y = np.arange(len(labels))[::-1]
    ax3.hlines(y, 0, gains, color=HERO_SOFT, lw=5.2, zorder=1)
    ax3.scatter(gains, y, s=42, color=HERO, edgecolor=WHITE, lw=0.6, zorder=3)
    for yi, value in zip(y, gains):
        ax3.text(value + 0.18, yi, f"+{value}", va="center", color=HERO_DARK, fontsize=7.2, fontweight="bold")
    ax3.set_yticks(y, labels)
    ax3.set_xlim(0, 6.2)
    ax3.set_xlabel("新增Top20命中")
    style_ax(ax3, grid="x")
    panel_label(ax3, "c")

    fig.text(0.02, 0.015, "■ 基础排序", color=BASE_DARK, fontsize=6.5)
    fig.text(0.125, 0.015, "● REG", color=HERO_DARK, fontsize=6.5)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.23)
    save(fig, "fig1_reg_generalization")


def fig2_final_set_trajectory() -> None:
    rows = read_rows("final_set_progress_dev20.csv")
    stages = [r["method"] for r in rows]
    f1 = np.array([float(r["macro_f1"]) for r in rows])
    recall = np.array([float(r["macro_recall"]) for r in rows])
    output = np.array([float(r["mean_output"]) for r in rows])
    x = np.arange(len(stages))

    fig = plt.figure(figsize=(7.20, 2.65))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0], wspace=0.43)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.plot(x, recall, color=SUPPORT, lw=1.8, marker="o", ms=5.2, mfc=WHITE, mec=SUPPORT, mew=1.4)
    ax1.plot(x, f1, color=HERO, lw=2.1, marker="o", ms=5.6, mfc=HERO, mec=WHITE, mew=0.6)
    for i, (v1, v2) in enumerate(zip(recall, f1)):
        ax1.text(i, v1 + 0.011, f"{v1:.3f}", ha="center", color=SUPPORT, fontsize=6.4)
        ax1.text(i, v2 - 0.017, f"{v2:.3f}", ha="center", color=HERO_DARK, fontsize=6.4)
    ax1.text(3.10, recall[-1], "Recall", color=SUPPORT, va="center", fontweight="bold")
    ax1.text(3.10, f1[-1], "F1", color=HERO_DARK, va="center", fontweight="bold")
    ax1.set_xticks(x, stages)
    ax1.set_ylim(0.19, 0.38)
    ax1.set_xlim(-0.18, 3.48)
    ax1.set_ylabel("指标值")
    style_ax(ax1, grid="y")
    panel_label(ax1, "a", x=-0.09)

    point_colors = [BASE, SUPPORT_SOFT, SUPPORT, HERO]
    sizes = 105 * (output / output.max()) ** 2
    ax2.plot(f1, recall, color=AXIS, lw=1.2, zorder=1)
    ax2.scatter(f1, recall, s=sizes, color=point_colors, edgecolor=WHITE, lw=0.8, zorder=3)
    offsets = [(-0.002, -0.012), (-0.006, 0.009), (-0.006, 0.009), (0.004, 0.007)]
    for name, xx, yy, (dx, dy) in zip(stages, f1, recall, offsets):
        ax2.text(xx + dx, yy + dy, name, fontsize=6.2, color=INK)
    ax2.annotate("", xy=(f1[-1], recall[-1]), xytext=(f1[0], recall[0]), arrowprops={"arrowstyle": "->", "color": HERO_DARK, "lw": 1.0})
    ax2.set_xlabel("Macro F1")
    ax2.set_ylabel("Macro Recall")
    ax2.set_xlim(0.215, 0.292)
    ax2.set_ylim(0.26, 0.365)
    style_ax(ax2, grid=None)
    panel_label(ax2, "b")
    ax2.text(0.98, 0.04, "圆点面积：平均返回规模", transform=ax2.transAxes, ha="right", color=MUTED, fontsize=6.1)

    fig.subplots_adjust(left=0.08, right=0.965, top=0.90, bottom=0.22)
    save(fig, "fig2_final_set_trajectory")


def draw_funnel(ax: plt.Axes, values: list[int], title: str) -> None:
    colors = [SUPPORT_SOFT, SUPPORT, HERO]
    labels = ["候选", "Top20", "精选"]
    maxv = max(values)
    y_top = 0.98
    h = 0.23
    for i, (value, color, label) in enumerate(zip(values, colors, labels)):
        next_value = values[i + 1] if i < len(values) - 1 else value * 0.78
        w1 = 0.82 * value / maxv
        w2 = 0.82 * next_value / maxv
        yc = y_top - i * h
        poly = Polygon(
            [[0.5 - w1 / 2, yc], [0.5 + w1 / 2, yc], [0.5 + w2 / 2, yc - h * 0.78], [0.5 - w2 / 2, yc - h * 0.78]],
            closed=True,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor=WHITE,
            lw=0.7,
        )
        ax.add_patch(poly)
        txt_color = WHITE if i > 0 else INK
        ax.text(0.5, yc - h * 0.38, f"{label}  {value}", transform=ax.transAxes, ha="center", va="center", color=txt_color, fontsize=6.8, fontweight="bold")
    ax.text(0.5, 1.04, title, transform=ax.transAxes, ha="center", color=INK, fontsize=7.2, fontweight="bold")
    ax.set_axis_off()


def fig3_retrieval_evidence() -> None:
    channel_rows = read_rows("retrieval_channel_contribution.csv")
    funnel_rows = read_rows("evidence_funnel.csv")
    names = [r["channel"] for r in channel_rows]
    gains = np.array([int(r["incremental_gold"]) for r in channel_rows])
    total = gains.sum()

    fig = plt.figure(figsize=(7.20, 3.15))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.7, 0.72, 0.72], height_ratios=[1, 1], wspace=0.42, hspace=0.40)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:, 1])
    ax3 = fig.add_subplot(gs[:, 2])

    seg_colors = [BASE_DARK, TEAL, SUPPORT, GOLD, HERO, GAIN]
    left = 0
    small_index = 0
    for name, gain, color in zip(names, gains, seg_colors):
        ax1.barh([0], [gain], left=left, height=0.36, color=color, edgecolor=WHITE, lw=0.7)
        if gain >= 6:
            ax1.text(left + gain / 2, 0, f"{name}\n+{gain}", ha="center", va="center", color=WHITE if color != GOLD else INK, fontsize=6.3, fontweight="bold")
        else:
            y_levels = [0.50, -0.43, 0.68, -0.70]
            y_text = y_levels[min(small_index, len(y_levels) - 1)]
            y_anchor = 0.18 if y_text > 0 else -0.18
            ax1.annotate(
                f"{name} +{gain}",
                xy=(left + gain / 2, y_anchor),
                xytext=(left + gain / 2, y_text),
                ha="center",
                va="center",
                fontsize=5.8,
                color=INK,
                arrowprops={"arrowstyle": "-", "lw": 0.65, "color": color},
            )
            small_index += 1
        left += gain
    ax1.text(total, -0.93, f"累计命中 {total}", ha="right", color=INK, fontsize=7.0, fontweight="bold")
    ax1.set_xlim(0, total)
    ax1.set_ylim(-1.03, 0.88)
    ax1.set_xlabel("各通道新增相关论文")
    ax1.set_yticks([])
    style_ax(ax1, grid=None)
    panel_label(ax1, "a", x=-0.06, y=1.00)

    draw_funnel(ax2, [int(funnel_rows[0]["candidate_tp"]), int(funnel_rows[0]["top20_tp"]), int(funnel_rows[0]["final_tp"])], "开发集")
    draw_funnel(ax3, [int(funnel_rows[1]["candidate_tp"]), int(funnel_rows[1]["top20_tp"]), int(funnel_rows[1]["final_tp"])], "盲测集")
    panel_label(ax2, "b", x=-0.10, y=1.02)
    panel_label(ax3, "c", x=-0.10, y=1.02)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.22)
    save(fig, "fig3_retrieval_evidence")


def dot_ablation(ax: plt.Axes, methods: list[str], values: list[float], *, xlim: tuple[float, float]) -> None:
    y = np.arange(len(methods))[::-1]
    colors = [BASE, SUPPORT, HERO, TEAL][: len(methods)]
    ax.hlines(y, xlim[0], values, color=GRID, lw=4.5, zorder=1)
    for yi, value, color, method in zip(y, values, colors, methods):
        ax.scatter(value, yi, s=42 if method == "REG" else 32, color=color, edgecolor=WHITE, lw=0.6, zorder=3)
        ax.text(value + 0.28, yi, f"{int(value)}", va="center", color=HERO_DARK if method == "REG" else INK, fontweight="bold" if method == "REG" else "normal")
    ax.set_yticks(y, methods)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Top20相关论文命中数")
    style_ax(ax, grid="x")


def fig4_structured_evidence_ablation() -> None:
    rows = read_rows("structured_evidence_ablation.csv")
    methods_cal = [r["method"] for r in rows]
    values_cal = [float(r["calibration_top20_tp"]) for r in rows]
    methods_dev = [r["method"] for r in rows if r["dev_top20_tp"]]
    values_dev = [float(r["dev_top20_tp"]) for r in rows if r["dev_top20_tp"]]

    fig, axes = plt.subplots(1, 2, figsize=(7.20, 2.55), gridspec_kw={"wspace": 0.48})
    dot_ablation(axes[0], methods_cal, values_cal, xlim=(19.5, 30.0))
    dot_ablation(axes[1], methods_dev, values_dev, xlim=(18.5, 27.0))
    axes[0].text(0.02, 1.04, "校准集", transform=axes[0].transAxes, color=INK, fontweight="bold")
    axes[1].text(0.02, 1.04, "开发集", transform=axes[1].transAxes, color=INK, fontweight="bold")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    axes[1].annotate("+3", xy=(25, 0), xytext=(23.6, 0.55), color=GAIN, fontsize=7.2, fontweight="bold", arrowprops={"arrowstyle": "->", "color": GAIN, "lw": 0.9})
    fig.subplots_adjust(left=0.16, right=0.975, top=0.84, bottom=0.24)
    save(fig, "fig4_structured_evidence_ablation")


def fig5_efficiency_pareto() -> None:
    rows = read_rows("cost_quality_modes.csv")
    names = [r["mode"] for r in rows]
    f1 = np.array([float(r["macro_f1"]) for r in rows])
    calls = np.array([float(r["api_calls_per_query"]) for r in rows])
    latency = np.array([float(r["latency_seconds"]) for r in rows])

    fig = plt.figure(figsize=(7.20, 2.65))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.48)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    sizes = 3.2 * latency
    ax1.plot(calls, f1, color=AXIS, lw=1.1, zorder=1)
    ax1.scatter(calls, f1, s=sizes, color=[BASE_DARK, TEAL], edgecolor=WHITE, lw=0.8, zorder=3)
    ax1.text(calls[0] + 0.25, f1[0] + 0.0015, "质量模式", color=BASE_DARK, fontweight="bold")
    ax1.text(calls[1] - 0.15, f1[1] - 0.0022, "低成本模式", color=TEAL, ha="right", fontweight="bold")
    ax1.annotate("", xy=(calls[1], f1[1]), xytext=(calls[0], f1[0]), arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.1})
    ax1.set_xlabel("API调用 / 查询")
    ax1.set_ylabel("Macro F1")
    ax1.set_xlim(7.5, 16.5)
    ax1.set_ylim(0.210, 0.227)
    style_ax(ax1, grid=None)
    panel_label(ax1, "a")
    ax1.text(0.02, 0.04, "圆点面积：端到端延迟", transform=ax1.transAxes, color=MUTED, fontsize=6.0)

    metrics = ["F1保留", "API调用", "端到端延迟"]
    ratios = np.array([f1[1] / f1[0], calls[1] / calls[0], latency[1] / latency[0]]) * 100
    y = np.arange(3)[::-1]
    ax2.hlines(y, 0, 100, color=GRID, lw=5.5)
    ax2.scatter([100] * 3, y, s=26, color=BASE, marker="s", label="质量模式")
    ax2.scatter(ratios, y, s=38, color=TEAL, label="低成本模式", zorder=3)
    for yi, ratio in zip(y, ratios):
        ax2.text(ratio - 2.1, yi + 0.20, f"{ratio:.1f}%", color=TEAL, ha="center", fontsize=6.5, fontweight="bold")
    ax2.set_yticks(y, metrics)
    ax2.set_xlim(48, 104)
    ax2.set_xlabel("相对质量模式（%）")
    style_ax(ax2, grid="x")
    panel_label(ax2, "b")
    ax2.legend(loc="lower right", ncol=2, fontsize=6.1)

    fig.subplots_adjust(left=0.09, right=0.975, top=0.88, bottom=0.23)
    save(fig, "fig5_efficiency_pareto")


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig1_reg_generalization()
    fig2_final_set_trajectory()
    fig3_retrieval_evidence()
    fig4_structured_evidence_ablation()
    fig5_efficiency_pareto()
    print(f"Generated Nature-style figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
