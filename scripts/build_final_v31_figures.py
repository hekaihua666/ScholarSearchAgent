"""Nature-style schematic package for the V3.1 competition project document.

Figure contracts
----------------
Fig. 1 conclusion: a complex query is converted into an evidence-controlled
retrieval loop whose discovery, ranking and delivery layers remain auditable.
Fig. 2 conclusion: Query Contract, stable paper identity and multi-view evidence
jointly prevent a single similarity score from dominating relation-aware ranking.
Fig. 3 conclusion: LIVE/FROZEN/REPLAY separates external retrieval variance from
algorithm changes and links candidate, ranking and final-set metrics.

Backend: Python/Matplotlib only.  Exports: editable SVG/PDF plus TIFF/PNG preview.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/submission_2026/figures_nature/final_v31"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.8,
})

C = {
    "ink": "#252832", "muted": "#686D78", "line": "#B9BEC8",
    "blue": "#516993", "blue2": "#7286AE", "blue_soft": "#E4E9F3",
    "teal": "#438F91", "teal2": "#75AFB0", "teal_soft": "#DDEEEE",
    "purple": "#746495", "purple2": "#9A8BB9", "purple_soft": "#EBE6F2",
    "rose": "#B45F7D", "rose2": "#CC8DA4", "rose_soft": "#F3E0E7",
    "amber": "#C88F3E", "amber2": "#DFB36F", "amber_soft": "#F6EAD8",
    "green": "#43835A", "green_soft": "#E1EFE5",
    "gray_soft": "#F3F4F6", "gray_mid": "#D9DCE2", "white": "#FFFFFF",
}


def mm(value: float) -> float:
    return value / 25.4


def arrow(ax, start, end, *, color=None, lw=1.0, rad=0.0, style="-|>", z=2):
    ax.add_patch(FancyArrowPatch(
        start, end, transform=ax.transAxes, arrowstyle=style, mutation_scale=8,
        linewidth=lw, color=color or C["line"],
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=z,
    ))


def rounded(ax, x, y, w, h, face, edge, *, radius=.015, lw=1.0, z=3):
    patch = FancyBboxPatch(
        (x, y), w, h, transform=ax.transAxes,
        boxstyle=f"round,pad=.006,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label_box(ax, x, y, w, h, title, lines, face, edge, tracked,
              *, title_size=7.0, body_size=5.5, align="left"):
    patch = rounded(ax, x, y, w, h, face, edge)
    px = x + (.055 * w if align == "left" else .5 * w)
    ha = align
    title_artist = ax.text(px, y + h - .17 * h, title, transform=ax.transAxes,
                           ha=ha, va="top", fontsize=title_size, fontweight="bold",
                           color=C["ink"], zorder=4)
    body_artist = ax.text(px, y + h - .43 * h, lines, transform=ax.transAxes,
                          ha=ha, va="top", fontsize=body_size, color=C["muted"],
                          linespacing=1.35, zorder=4)
    tracked.extend([(title_artist, patch), (body_artist, patch)])
    return patch


def panel_label(ax, letter, x=.012, y=.985):
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, fontweight="bold", color=C["ink"], zorder=10)


def validate(fig, tracked):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    errors = []
    for artist, patch in tracked:
        a = artist.get_window_extent(renderer)
        p = patch.get_window_extent(renderer)
        if a.x0 < p.x0 - 3 or a.x1 > p.x1 + 3 or a.y0 < p.y0 - 3 or a.y1 > p.y1 + 3:
            errors.append(artist.get_text())
    if errors:
        raise RuntimeError("text overflow: " + " | ".join(errors))


def save(fig, name: str, tracked) -> dict[str, str]:
    validate(fig, tracked)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / name
    paths = {
        "svg": stem.with_suffix(".svg"), "pdf": stem.with_suffix(".pdf"),
        "tiff": stem.with_suffix(".tiff"), "png": stem.with_suffix(".png"),
    }
    fig.savefig(paths["svg"], facecolor="white", bbox_inches="tight", pad_inches=.03)
    fig.savefig(paths["pdf"], facecolor="white", bbox_inches="tight", pad_inches=.03)
    fig.savefig(paths["tiff"], dpi=600, facecolor="white", bbox_inches="tight",
                pad_inches=.03, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(paths["png"], dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.03)
    plt.close(fig)
    if "<text" not in paths["svg"].read_text(encoding="utf-8"):
        raise RuntimeError(f"SVG text was outlined: {paths['svg']}")
    return {k: str(v) for k, v in paths.items()}


def figure_system_loop():
    tracked: list[tuple[Any, Any]] = []
    fig = plt.figure(figsize=(mm(183), mm(107)))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.0], hspace=.11,
                          left=.025, right=.985, top=.97, bottom=.055)
    ax = fig.add_subplot(gs[0]); ax.axis("off"); panel_label(ax, "a")
    ax2 = fig.add_subplot(gs[1]); ax2.axis("off"); panel_label(ax2, "b")

    # Hero pipeline.
    label_box(ax, .35, .835, .30, .13, "复杂学术查询",
              "主题 · 方法 · 任务 · 数据 · 时间 · 关系", C["white"], C["ink"], tracked,
              title_size=8.2, body_size=5.9)
    arrow(ax, (.50, .835), (.09, .71), color=C["ink"], lw=1.25, rad=.03)
    stages = [
        (.025, "查询认知", "Query Contract\n约束图 · 意图路由", C["blue_soft"], C["blue"]),
        (.220, "多源发现", "词项 + 语义 + arXiv\n标题探测 · 第二轮检索", C["teal_soft"], C["teal"]),
        (.415, "统一候选池", "P(q) · 来源轨迹\nRRF · TRACE-Rank", C["purple_soft"], C["purple"]),
        (.610, "证据决策", "REG 条件合取\nMEQ 多视图仲裁", C["rose_soft"], C["rose"]),
        (.805, "结果交付", "F1 / 均衡 / 探索\n矩阵 · 图谱 · 导出", C["amber_soft"], C["amber"]),
    ]
    for x, title, lines, face, edge in stages:
        label_box(ax, x, .42, .17, .275, title, lines, face, edge, tracked)
    for x in (.195, .390, .585, .780):
        arrow(ax, (x, .557), (x + .025, .557), color=C["ink"], lw=1.1)

    # EBP recovery loop.
    ebp = label_box(ax, .25, .105, .50, .185, "EBP · 证据加权引文反投影",
                    "全候选证据聚合 → 未解析引文 ID 投票 → 一次批量物化 → 补充候选边界",
                    C["gray_soft"], C["teal"], tracked, title_size=7.4, body_size=5.7)
    arrow(ax, (.69, .42), (.64, .29), color=C["rose"], rad=.10, lw=1.2)
    arrow(ax, (.36, .29), (.29, .42), color=C["teal"], rad=.10, lw=1.2)
    ax.text(.705, .335, "证据回传", transform=ax.transAxes, fontsize=5.4,
            color=C["rose"], ha="left", va="center")

    # Supporting infrastructure strip.
    strip = rounded(ax2, .025, .16, .95, .68, C["gray_soft"], C["gray_mid"], radius=.02, lw=.9)
    ax2.text(.05, .66, "可复现运行底座", transform=ax2.transAxes, fontsize=7.2,
             fontweight="bold", color=C["ink"], va="center")
    chips = [
        (.21, "OpenAlex", C["blue_soft"], C["blue"]), (.35, "arXiv", C["teal_soft"], C["teal"]),
        (.47, "Crossref", C["purple_soft"], C["purple"]), (.61, "S2", C["rose_soft"], C["rose"]),
        (.71, "本地 MiniLM", C["amber_soft"], C["amber"]),
    ]
    for x, txt, face, edge in chips:
        p = rounded(ax2, x, .50, .105, .29, face, edge, radius=.035, lw=.8)
        t = ax2.text(x+.0525, .645, txt, transform=ax2.transAxes, ha="center", va="center",
                     fontsize=5.5, fontweight="bold", color=C["ink"])
        tracked.append((t, p))
    ax2.text(.21, .33, "稳定论文身份  |  分层缓存  |  LIVE—FROZEN—REPLAY  |  离线演示  |  全链路审计",
             transform=ax2.transAxes, fontsize=5.7, color=C["muted"], ha="left", va="center")
    return fig, tracked


def figure_evidence_kernel():
    tracked: list[tuple[Any, Any]] = []
    fig = plt.figure(figsize=(mm(183), mm(111)))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.08, .92], width_ratios=[1.0, 1.15, 1.0],
                          hspace=.18, wspace=.12, left=.025, right=.985, top=.96, bottom=.055)
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1:]);
    ax_c = fig.add_subplot(gs[1, :]);
    for ax in (ax_a, ax_b, ax_c): ax.axis("off")
    panel_label(ax_a, "a"); panel_label(ax_b, "b"); panel_label(ax_c, "c")

    # Query Contract hypergraph.
    center = Circle((.50, .50), .135, transform=ax_a.transAxes, facecolor=C["white"],
                    edgecolor=C["ink"], linewidth=1.1, zorder=4)
    ax_a.add_patch(center)
    tc = ax_a.text(.50, .50, "Query\nContract", transform=ax_a.transAxes, ha="center", va="center",
                   fontsize=7.2, fontweight="bold", color=C["ink"], linespacing=1.15, zorder=5)
    tracked.append((tc, center))
    nodes = [
        (.50, .86, "主题", C["blue"]), (.82, .70, "方法", C["teal"]),
        (.84, .31, "任务", C["purple"]), (.50, .14, "数据/时间", C["amber"]),
        (.16, .31, "关系谓词", C["rose"]), (.18, .70, "排除条件", C["blue2"]),
    ]
    for x, y, txt, color in nodes:
        arrow(ax_a, (.50, .50), (x, y), color=color, lw=.9)
        p = rounded(ax_a, x-.11, y-.055, .22, .11, C["white"], color, radius=.05, lw=.9)
        t = ax_a.text(x, y, txt, transform=ax_a.transAxes, ha="center", va="center",
                     fontsize=5.6, color=C["ink"], fontweight="bold")
        tracked.append((t, p))
    ax_a.text(.50, .03, "约束不再被压缩为单一关键词串", transform=ax_a.transAxes,
              ha="center", va="bottom", fontsize=5.3, color=C["muted"])

    # Identity and evidence vector.
    title = rounded(ax_b, .03, .82, .94, .14, C["gray_soft"], C["gray_mid"], radius=.02, lw=.8)
    t = ax_b.text(.06, .89, "候选论文 p：先统一身份，再形成多视图证据向量",
                  transform=ax_b.transAxes, fontsize=6.6, fontweight="bold", color=C["ink"],
                  ha="left", va="center")
    tracked.append((t, title))
    ids = [("arXiv ID", C["blue"]), ("DOI", C["teal"]), ("OpenAlex ID", C["purple"]),
           ("S2 ID", C["rose"]), ("标题 fallback", C["amber"])]
    for idx, (txt, color) in enumerate(ids):
        x = .035 + idx*.19
        p = rounded(ax_b, x, .61, .165, .12, C["white"], color, radius=.025, lw=.85)
        tt = ax_b.text(x+.0825, .67, txt, transform=ax_b.transAxes, ha="center", va="center",
                       fontsize=5.2, color=C["ink"], fontweight="bold")
        tracked.append((tt, p))
    arrow(ax_b, (.50, .60), (.50, .48), color=C["ink"], lw=1.0)
    evidence = [
        ("Lexical", "L", C["blue"]), ("Semantic", "S", C["teal"]),
        ("Constraint", "C", C["purple"]), ("Graph", "G", C["rose"]),
        ("Cross", "X", C["amber"]),
    ]
    for idx, (txt, value, color) in enumerate(evidence):
        x = .04 + idx*.19
        ax_b.add_patch(Circle((x+.075, .35), .067, transform=ax_b.transAxes,
                              facecolor=color, edgecolor="none", alpha=.92, zorder=3))
        ax_b.text(x+.075, .35, value, transform=ax_b.transAxes, ha="center", va="center",
                  fontsize=6.2, color="white", fontweight="bold", zorder=4)
        ax_b.text(x+.075, .24, txt, transform=ax_b.transAxes, ha="center", va="top",
                  fontsize=5.0, color=C["muted"])
    bracket = Polygon([[.11,.13],[.11,.09],[.89,.09],[.89,.13]], closed=False,
                      transform=ax_b.transAxes, fill=False, edgecolor=C["ink"], linewidth=.9)
    ax_b.add_patch(bracket)
    ax_b.text(.50, .025, "同一查询内归一化，保留来源与解释字段", transform=ax_b.transAxes,
              ha="center", va="bottom", fontsize=5.3, color=C["muted"])

    # Decision kernel.
    label_box(ax_c, .035, .20, .21, .62, "条件合取 C",
              "平均覆盖 × 最弱条件\n避免“多数条件高分”掩盖\n必要条件缺失",
              C["blue_soft"], C["blue"], tracked, title_size=6.8, body_size=5.2)
    label_box(ax_c, .275, .20, .21, .62, "REG 关系证据",
              "关系谓词 · 引文方向\n种子桥接 · 图支持\n形成可解释证据间隔",
              C["rose_soft"], C["rose"], tracked, title_size=6.8, body_size=5.2)
    label_box(ax_c, .515, .20, .21, .62, "MEQ 多视图仲裁",
              "Cross / TRACE / Graph\nLexical / Fusion\n以第三强视图抑制单点高分",
              C["purple_soft"], C["purple"], tracked, title_size=6.8, body_size=5.2)
    label_box(ax_c, .755, .20, .21, .62, "受控排序动作",
              "头部保护 · 最大晋升数\n绝对下限 · 相对间隔\n输出排序池与补充候选",
              C["amber_soft"], C["amber"], tracked, title_size=6.8, body_size=5.2)
    for x in (.245, .485, .725):
        arrow(ax_c, (x, .51), (x+.03, .51), color=C["ink"], lw=1.0)
    return fig, tracked


def figure_evaluation_protocol():
    tracked: list[tuple[Any, Any]] = []
    fig, ax = plt.subplots(figsize=(mm(183), mm(95)))
    fig.subplots_adjust(left=.025, right=.985, top=.96, bottom=.07)
    ax.axis("off"); panel_label(ax, "a")

    # Data split stream.
    label_box(ax, .035, .72, .19, .20, "评测数据隔离",
              "calibration40\ndev20 · Blind50\nindependent validation30",
              C["blue_soft"], C["blue"], tracked, title_size=6.8, body_size=5.1)
    arrow(ax, (.225, .82), (.285, .82), color=C["ink"], lw=1.1)
    modes = [
        (.285, "LIVE", "真实外部检索\n记录 API 与时延", C["teal_soft"], C["teal"]),
        (.495, "FROZEN", "冻结候选、元数据\n与查询理解结果", C["purple_soft"], C["purple"]),
        (.705, "REPLAY", "相同候选重放\n排序与集合决策", C["rose_soft"], C["rose"]),
    ]
    for x, title, lines, face, edge in modes:
        label_box(ax, x, .72, .17, .20, title, lines, face, edge, tracked,
                  title_size=7.0, body_size=5.1)
    arrow(ax, (.455, .82), (.495, .82), color=C["ink"], lw=1.1)
    arrow(ax, (.665, .82), (.705, .82), color=C["ink"], lw=1.1)
    label_box(ax, .895, .72, .07, .20, "比较",
              "版本\n消融", C["amber_soft"], C["amber"], tracked,
              title_size=6.1, body_size=4.8, align="center")
    arrow(ax, (.875, .82), (.895, .82), color=C["ink"], lw=1.0)

    # Three diagnostic layers.
    ax.text(.035, .61, "同一论文身份规则与日期约束下的三层指标", transform=ax.transAxes,
            fontsize=6.6, fontweight="bold", color=C["ink"], ha="left", va="center")
    layers = [
        (.035, "候选层", "Candidate Recall\n通道贡献 · 新增命中", C["teal_soft"], C["teal"],
         "判断是否“找到”目标论文"),
        (.355, "排序层", "Recall@20 · NDCG@20\nGold 平均排名 · Top20 TP", C["purple_soft"], C["purple"],
         "判断是否“排到”头部"),
        (.675, "集合层", "Macro P / R / F1\n结果规模 · 有效密度", C["amber_soft"], C["amber"],
         "判断是否“交付”有效结果"),
    ]
    for x, title, lines, face, edge, note in layers:
        p = label_box(ax, x, .25, .29, .27, title, lines, face, edge, tracked,
                      title_size=7.1, body_size=5.3)
        ax.text(x+.145, .19, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=5.3, color=C["muted"])
    arrow(ax, (.325, .385), (.355, .385), color=C["ink"], lw=1.05)
    arrow(ax, (.645, .385), (.675, .385), color=C["ink"], lw=1.05)
    ax.text(.50, .065,
            "外部变化与算法变化被分离：每项提升均可回溯到候选、排序或集合决策的具体环节",
            transform=ax.transAxes, ha="center", va="center", fontsize=5.8,
            color=C["muted"])
    return fig, tracked


def build():
    outputs = {}
    for name, fn in [
        ("fig1_system_evidence_loop", figure_system_loop),
        ("fig2_query_contract_evidence_kernel", figure_evidence_kernel),
        ("fig3_reproducible_evaluation_protocol", figure_evaluation_protocol),
    ]:
        fig, tracked = fn()
        outputs[name] = save(fig, name, tracked)
    qa = {
        "backend": "Python/Matplotlib",
        "figure_width_mm": 183,
        "editable_svg_text": True,
        "overflow_check": "passed",
        "palette": "restrained colorblind-aware family palette",
        "exports": outputs,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qa_report.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    return qa


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
