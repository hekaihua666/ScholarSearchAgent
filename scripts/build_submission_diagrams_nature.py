from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "submission_2026" / "figures_nature"

INK = "#292B33"
MUTED = "#70747E"
LINE = "#AEB4C0"
PAPER = "#FFFFFF"
BLUE = "#6674A8"
BLUE_SOFT = "#E6E9F4"
TEAL = "#4D9BA3"
TEAL_SOFT = "#DDEDEF"
ROSE = "#C86F86"
ROSE_SOFT = "#F4E0E6"
GOLD = "#C99A3D"
GOLD_SOFT = "#F5ECD8"
GREEN = "#2E8B57"
GREEN_SOFT = "#DFEFE6"


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
            "font.size": 7.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
        }
    )


def box(ax, xy, wh, title, body="", *, face=BLUE_SOFT, edge=BLUE, title_color=INK, lw=0.9, radius=0.025):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", color=title_color, fontsize=7.2, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h * 0.30, body, ha="center", va="center", color=MUTED, fontsize=5.9, linespacing=1.35)
    return patch


def arrow(ax, start, end, *, color=LINE, lw=1.05, style="-|>", rad=0.0, mutation=8):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        color=color,
        linewidth=lw,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=1,
        shrinkB=1,
    )
    ax.add_patch(patch)
    return patch


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=420, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    columns = [
        (0.035, "查询认知", "意图路由\n约束抽取 · 子查询生成", BLUE_SOFT, BLUE),
        (0.275, "自主检索", "多源双通道\n二轮扩展 · 引文探索", TEAL_SOFT, TEAL),
        (0.515, "证据决策", "TRACE-Rank\nREG 关系证据 · 风险路由", ROSE_SOFT, ROSE),
        (0.755, "结构化交付", "相关论文集合\n矩阵 · 图谱 · 摘要 · 导出", GOLD_SOFT, GOLD),
    ]
    for x, title, body, face, edge in columns:
        box(ax, (x, 0.47), (0.205, 0.25), title, body, face=face, edge=edge)
    for x in (0.24, 0.48, 0.72):
        arrow(ax, (x, 0.595), (x + 0.035, 0.595), color=INK, lw=1.0)

    box(ax, (0.365, 0.82), (0.27, 0.105), "复杂学术查询", "主题 · 方法 · 数据 · 时间 · 关系约束", face=PAPER, edge=INK)
    arrow(ax, (0.50, 0.82), (0.14, 0.735), color=INK, rad=0.03)

    rail = FancyBboxPatch((0.035, 0.10), 0.925, 0.20, boxstyle="round,pad=0.012,rounding_size=0.025", facecolor="#F6F7F9", edgecolor="#D5D8DF", linewidth=0.85)
    ax.add_patch(rail)
    ax.text(0.058, 0.265, "数据与运行底座", color=INK, fontsize=7.0, fontweight="bold")
    sources = ["OpenAlex", "arXiv", "Crossref", "Semantic Scholar", "向量模型", "响应缓存", "离线演示"]
    source_widths = [0.10, 0.09, 0.10, 0.14, 0.105, 0.105, 0.105]
    sx = 0.06
    for i, (source, width) in enumerate(zip(sources, source_widths)):
        face = TEAL_SOFT if i < 4 else BLUE_SOFT
        edge = TEAL if i < 4 else BLUE
        box(ax, (sx, 0.145), (width, 0.07), source, face=face, edge=edge, radius=0.018)
        sx += width + 0.018

    for x in (0.135, 0.375, 0.615, 0.855):
        arrow(ax, (x, 0.30), (x, 0.47), color=LINE, style="-|>", mutation=7)

    ax.text(0.035, 0.035, "从查询理解到可核验结果交付的端到端学术检索架构", color=MUTED, fontsize=6.3)
    save(fig, "fig6_system_architecture")


def pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    xs = [0.025, 0.155, 0.285, 0.415, 0.545, 0.675, 0.805]
    nodes = [
        ("Query Contract", "解析多维约束", BLUE_SOFT, BLUE),
        ("检索规划", "子查询与通道分配", BLUE_SOFT, BLUE),
        ("混合召回", "词项 · 语义 · 引文", TEAL_SOFT, TEAL),
        ("TRACE-Rank", "融合 · 覆盖 · 多样性", TEAL_SOFT, TEAL),
        ("缺口诊断", "未满足条件与关系", GOLD_SOFT, GOLD),
        ("REG", "关系证据受控救援", ROSE_SOFT, ROSE),
        ("集合决策", "F1 · 均衡 · 探索", GREEN_SOFT, GREEN),
    ]
    for x, (title, body, face, edge) in zip(xs, nodes):
        box(ax, (x, 0.50), (0.115, 0.20), title, body, face=face, edge=edge)
    for x in xs[:-1]:
        arrow(ax, (x + 0.115, 0.60), (x + 0.13, 0.60), color=INK, mutation=7)

    # Iterative retrieval loop: the evidence gap controls whether another action is worthwhile.
    arrow(ax, (0.602, 0.49), (0.342, 0.34), color=GOLD, rad=0.14, lw=1.15)
    arrow(ax, (0.342, 0.34), (0.212, 0.49), color=GOLD, rad=0.08, lw=1.15)
    ax.text(0.415, 0.305, "证据不足 → 生成受控补充检索动作", ha="center", color=GOLD, fontsize=6.3, fontweight="bold")

    # Evidence preservation rail.
    box(ax, (0.155, 0.12), (0.68, 0.105), "候选池与证据账本", "统一论文身份 · 通道贡献 · 摘要证据 · 约束覆盖 · 引文路径 · 成本记录", face="#F6F7F9", edge=LINE)
    for x in (0.215, 0.345, 0.475, 0.605, 0.735):
        arrow(ax, (x, 0.50), (x, 0.225), color=LINE, lw=0.7, style="-")

    ax.text(0.025, 0.89, "a", fontsize=8.8, fontweight="bold", color=INK)
    ax.text(0.055, 0.89, "检索—诊断—救援—选择闭环", fontsize=8.0, fontweight="bold", color=INK)
    ax.text(0.025, 0.035, "仅当新动作能够补足约束或关系证据时进入下一轮，避免无效扩展污染候选集合。", color=MUTED, fontsize=6.2)
    save(fig, "fig7_algorithm_pipeline")


def reg_mechanism() -> None:
    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.20)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax1.text(0.00, 0.96, "a", fontsize=8.8, fontweight="bold", color=INK)
    ax1.text(0.09, 0.96, "查询约束图", fontsize=7.8, fontweight="bold", color=INK)
    box(ax1, (0.08, 0.72), (0.80, 0.12), "目标论文", "理论分析 MAML 收敛性，并面向 few-shot learning", face=PAPER, edge=INK)
    criteria = [
        (0.10, 0.48, "C1", "主题\nMAML"),
        (0.32, 0.48, "C2", "关系\n理论分析"),
        (0.54, 0.48, "C3", "属性\n收敛性"),
        (0.76, 0.48, "C4", "场景\nfew-shot"),
    ]
    for x, y, code, label in criteria:
        circle = Circle((x + 0.07, y + 0.07), 0.067, facecolor=BLUE_SOFT, edgecolor=BLUE, linewidth=0.9)
        ax1.add_patch(circle)
        ax1.text(x + 0.07, y + 0.082, code, ha="center", va="center", color=BLUE, fontsize=6.7, fontweight="bold")
        ax1.text(x + 0.07, y - 0.015, label, ha="center", va="top", color=MUTED, fontsize=5.8, linespacing=1.25)
        arrow(ax1, (0.48, 0.72), (x + 0.07, y + 0.14), color=LINE, style="-", lw=0.65)
    box(ax1, (0.16, 0.13), (0.64, 0.13), "合取约束", "以最弱条件约束整体证据强度，而非仅累计关键词相似度", face=GOLD_SOFT, edge=GOLD)
    for x, *_ in criteria:
        arrow(ax1, (x + 0.07, 0.47), (0.48, 0.26), color=LINE, style="-", lw=0.55)

    ax2.text(0.00, 0.96, "b", fontsize=8.8, fontweight="bold", color=INK)
    ax2.text(0.08, 0.96, "关系证据评分与受控替换", fontsize=7.8, fontweight="bold", color=INK)
    # Seed/candidate evidence graph.
    nodes = {
        "Q": (0.14, 0.57, BLUE),
        "S1": (0.33, 0.74, TEAL),
        "S2": (0.34, 0.40, TEAL),
        "P": (0.56, 0.58, ROSE),
    }
    for label, (x, y, color) in nodes.items():
        r = 0.055 if label != "P" else 0.067
        ax2.add_patch(Circle((x, y), r, facecolor=PAPER, edgecolor=color, linewidth=1.5))
        ax2.text(x, y, label, ha="center", va="center", color=color, fontweight="bold")
    arrow(ax2, (0.195, 0.59), (0.275, 0.70), color=LINE, style="-", lw=0.8)
    arrow(ax2, (0.195, 0.55), (0.285, 0.43), color=LINE, style="-", lw=0.8)
    arrow(ax2, (0.385, 0.72), (0.495, 0.61), color=ROSE, lw=1.15)
    arrow(ax2, (0.392, 0.43), (0.495, 0.55), color=ROSE, lw=1.15)
    ax2.text(0.35, 0.86, "种子桥接与引文方向", ha="center", color=MUTED, fontsize=6.0)

    metrics = [
        ("条件合取", 1.00, BLUE),
        ("关系谓词", 1.00, TEAL),
        ("引文方向", 1.00, GOLD),
        ("种子桥接", 0.38, ROSE),
    ]
    y0 = 0.78
    for i, (name, value, color) in enumerate(metrics):
        y = y0 - i * 0.115
        ax2.text(0.68, y, name, va="center", color=INK, fontsize=6.1)
        ax2.plot([0.80, 0.91], [y, y], color="#E7E8EC", lw=4.2, solid_capstyle="round")
        ax2.plot([0.80, 0.80 + 0.11 * value], [y, y], color=color, lw=4.2, solid_capstyle="round")
        ax2.text(0.99, y, f"{value:.2f}", ha="right", va="center", color=color, fontsize=6.0, fontweight="bold")

    box(ax2, (0.08, 0.13), (0.40, 0.12), "Top20 尾部候选", "相关性较弱 · 关系证据不足", face="#F3F4F6", edge=LINE)
    box(ax2, (0.57, 0.13), (0.35, 0.12), "REG 晋升候选", "多条件同时满足", face=ROSE_SOFT, edge=ROSE)
    arrow(ax2, (0.48, 0.19), (0.57, 0.19), color=ROSE, lw=1.4)
    ax2.text(0.525, 0.255, "证据间隔门控", ha="center", color=ROSE, fontsize=6.0, fontweight="bold")
    ax2.text(0.07, 0.035, "REG 将“语义相似”细化为可解释的条件、关系与引文证据，只在证据优势明确时改变排序。", color=MUTED, fontsize=6.0)
    save(fig, "fig8_reg_mechanism")


def main() -> None:
    setup()
    architecture()
    pipeline()
    reg_mechanism()
    print(f"Generated diagrams in {OUT}")


if __name__ == "__main__":
    main()
