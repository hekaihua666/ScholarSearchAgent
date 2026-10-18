"""Readable report figures for the A4 competition document.

The figures are generated at the document's actual 158-mm text width.  All
labels use Microsoft YaHei so Chinese, English and numerals share one clean
sans-serif family without unreliable glyph-level fallback.  The minimum
designed text size is 8 pt at final placement, avoiding the 183-mm-to-158-mm
down-scaling that made the previous figure package difficult to read in print.
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
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "artifacts" / "submission_2026"
OUT = ASSET / "figures_nature" / "report_v32"
SHOTS = ASSET / "screenshots"
EBP_REPORT = ROOT / "reports" / "evidence_backprojection_v21"
SOURCE = ASSET / "source_data"


def mm(x: float) -> float:
    return x / 25.4


mpl.rcParams.update({
    # Matplotlib does not reliably perform glyph-level fallback on Windows.
    # Microsoft YaHei contains a clean sans-serif Latin set as well as complete
    # Chinese glyphs, so using it directly prevents missing-glyph boxes.
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
    "axes.linewidth": .8,
})

C = {
    "ink": "#222832", "muted": "#606975", "line": "#AEB7C3", "grid": "#E6E9EE",
    "blue": "#4F6D9B", "blue_soft": "#E3EAF4", "teal": "#3D8F91",
    "teal_soft": "#DDEEEE", "purple": "#75639A", "purple_soft": "#EBE6F3",
    "rose": "#B45F7D", "rose_soft": "#F3E1E8", "amber": "#C58C35",
    "amber_soft": "#F6EBD8", "green": "#34845A", "green_soft": "#E0EFE5",
    "navy_bg": "#09162E", "navy_card": "#14244A", "white": "#FFFFFF",
}


def rounded(ax, x, y, w, h, face, edge, radius=.018, lw=1.0, z=2):
    p = FancyBboxPatch((x, y), w, h, transform=ax.transAxes,
                       boxstyle=f"round,pad=.006,rounding_size={radius}",
                       facecolor=face, edgecolor=edge, linewidth=lw, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, a, b, color=None, lw=1.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, transform=ax.transAxes, arrowstyle="-|>",
                                mutation_scale=9, linewidth=lw, color=color or C["ink"],
                                connectionstyle=f"arc3,rad={rad}", shrinkA=3, shrinkB=3, zorder=3))


def panel(ax, letter, x=.01, y=.99):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            ha="left", va="top", color=C["ink"], zorder=8)


def box_text(ax, x, y, w, h, title, body, face, edge, *, title_size=9.5,
             body_size=8.0, align="left"):
    p = rounded(ax, x, y, w, h, face, edge)
    px = x + (.06*w if align == "left" else .5*w)
    ha = align
    ax.text(px, y+h*.78, title, transform=ax.transAxes, fontsize=title_size,
            fontweight="bold", ha=ha, va="center", color=C["ink"], zorder=4)
    ax.text(px, y+h*.42, body, transform=ax.transAxes, fontsize=body_size,
            ha=ha, va="center", color=C["muted"], linespacing=1.28, zorder=4)
    return p


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {s: OUT / f"{name}.{s}" for s in ("svg", "pdf", "png", "tiff")}
    fig.savefig(paths["svg"], facecolor="white", bbox_inches="tight", pad_inches=.025)
    fig.savefig(paths["pdf"], facecolor="white", bbox_inches="tight", pad_inches=.025)
    fig.savefig(paths["png"], dpi=320, facecolor="white", bbox_inches="tight", pad_inches=.025)
    fig.savefig(paths["tiff"], dpi=600, facecolor="white", bbox_inches="tight",
                pad_inches=.025, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    if "<text" not in paths["svg"].read_text(encoding="utf-8"):
        raise RuntimeError(f"SVG text was outlined: {paths['svg']}")
    return {k: str(v) for k, v in paths.items()}


def system_architecture():
    fig, ax = plt.subplots(figsize=(mm(158), mm(82)))
    fig.subplots_adjust(left=.015, right=.985, top=.97, bottom=.05); ax.axis("off")
    box_text(ax, .31, .83, .38, .13, "复杂学术查询", "主题 · 方法 · 任务 · 数据 · 时间 · 关系",
             C["white"], C["ink"], title_size=10.5, body_size=8.2)
    stages = [
        (.015, "查询认知", "Query Contract\n意图与约束", C["blue_soft"], C["blue"]),
        (.215, "多源发现", "词项 · 语义 · arXiv\n补充检索", C["teal_soft"], C["teal"]),
        (.415, "统一候选池", "稳定 ID · 来源轨迹\nRRF · TRACE", C["purple_soft"], C["purple"]),
        (.615, "证据决策", "REG 条件合取\nMEQ 多视图", C["rose_soft"], C["rose"]),
        (.815, "结果交付", "F1 · 均衡 · 探索\n矩阵 · 图谱 · 导出", C["amber_soft"], C["amber"]),
    ]
    for x,t,b,f,e in stages: box_text(ax,x,.45,.17,.25,t,b,f,e,title_size=9.5,body_size=8.0)
    for x in (.185,.385,.585,.785): arrow(ax,(x,.575),(x+.03,.575),lw=1.15)
    arrow(ax,(.50,.83),(.10,.70),lw=1.15)
    box_text(ax,.285,.13,.43,.17,"EBP 证据加权引文反投影",
             "全候选证据聚合 → 引文 ID 投票 → 批量物化",
             "#F5F6F8",C["teal"],title_size=9.2,body_size=8.0)
    arrow(ax,(.69,.45),(.64,.30),color=C["rose"],rad=.08,lw=1.1)
    arrow(ax,(.36,.30),(.30,.45),color=C["teal"],rad=.08,lw=1.1)
    ax.text(.5,.045,"OpenAlex · arXiv · Crossref · Semantic Scholar  |  统一论文身份  |  通道审计与成本记录",
            transform=ax.transAxes,ha="center",va="center",fontsize=8.0,color=C["muted"])
    return save(fig,"fig1_system_architecture")


def result_modes():
    fig, ax = plt.subplots(figsize=(mm(158), mm(55)))
    fig.subplots_adjust(left=.02,right=.98,top=.94,bottom=.08); ax.axis("off")
    box_text(ax,.02,.37,.22,.35,"统一排序池","Top20 + 补充候选\n共享证据与来源",C["purple_soft"],C["purple"],title_size=9.6,body_size=8.0)
    cards=[(.34,"F1 优先","2 篇","高置信交付",C["rose_soft"],C["rose"]),
           (.56,"均衡集合","8 篇","精度—召回平衡",C["teal_soft"],C["teal"]),
           (.78,"探索列表","20 篇","完整头部与补充",C["amber_soft"],C["amber"])]
    for x,title,k,note,face,edge in cards:
        box_text(ax,x,.33,.19,.43,title,f"{k}\n{note}",face,edge,title_size=9.6,body_size=8.2,align="center")
        arrow(ax,(.24,.545),(x,.545),color=edge,lw=1.0)
    ax.text(.50,.12,"切换结果模式不重新检索；CSV/BibTeX 默认导出当前集合，JSON 保留完整证据对象",
            transform=ax.transAxes,ha="center",fontsize=8.1,color=C["muted"])
    return save(fig,"fig2_result_modes")


def query_evidence_kernel():
    fig = plt.figure(figsize=(mm(158),mm(90)))
    gs=fig.add_gridspec(2,2,height_ratios=[1.02,.78],width_ratios=[1,1.25],
                        left=.03,right=.985,top=.96,bottom=.06,wspace=.12,hspace=.16)
    a=fig.add_subplot(gs[0,0]); b=fig.add_subplot(gs[0,1]); c=fig.add_subplot(gs[1,:])
    for ax in (a,b,c): ax.axis("off")
    panel(a,"a");panel(b,"b");panel(c,"c")
    center=Circle((.50,.50),.145,transform=a.transAxes,facecolor="white",edgecolor=C["ink"],lw=1.1)
    a.add_patch(center);a.text(.50,.50,"Query\nContract",transform=a.transAxes,ha="center",va="center",fontsize=9.2,fontweight="bold")
    nodes=[(.50,.86,"主题",C["blue"]),(.83,.68,"方法",C["teal"]),(.82,.30,"任务",C["purple"]),
           (.50,.14,"数据/时间",C["amber"]),(.17,.30,"关系谓词",C["rose"]),(.17,.68,"排除条件",C["blue"])]
    for x,y,t,col in nodes:
        arrow(a,(.50,.50),(x,y),color=col,lw=.9);rounded(a,x-.12,y-.055,.24,.11,"white",col,radius=.045,lw=.9)
        a.text(x,y,t,transform=a.transAxes,ha="center",va="center",fontsize=8.0,fontweight="bold")
    a.text(.50,.02,"约束在改写、过滤与排序中持续保留",transform=a.transAxes,ha="center",fontsize=8.0,color=C["muted"])

    b.text(.04,.90,"稳定论文身份",transform=b.transAxes,fontsize=9.5,fontweight="bold")
    ids=[("arXiv",C["blue"]),("DOI",C["teal"]),("OpenAlex",C["purple"]),("S2",C["rose"]),("标题",C["amber"])]
    for i,(t,col) in enumerate(ids):
        x=.04+i*.19;rounded(b,x,.69,.16,.12,"white",col,radius=.03,lw=.9);b.text(x+.08,.75,t,transform=b.transAxes,ha="center",va="center",fontsize=8.0,fontweight="bold")
    arrow(b,(.50,.67),(.50,.55),lw=1.0)
    b.text(.04,.49,"五路证据向量",transform=b.transAxes,fontsize=9.5,fontweight="bold")
    vals=[("L","词项",C["blue"]),("S","语义",C["teal"]),("C","约束",C["purple"]),("G","图结构",C["rose"]),("X","交叉编码",C["amber"])]
    for i,(v,t,col) in enumerate(vals):
        x=.12+i*.19;b.add_patch(Circle((x,.30),.065,transform=b.transAxes,facecolor=col,edgecolor="none"));
        b.text(x,.30,v,transform=b.transAxes,ha="center",va="center",fontsize=9,color="white",fontweight="bold")
        b.text(x,.16,t,transform=b.transAxes,ha="center",fontsize=8.0,color=C["muted"])

    blocks=[(.03,"条件合取 C","平均覆盖 × 最弱条件",C["blue_soft"],C["blue"]),
            (.355,"REG 关系证据","谓词 · 引文方向 · 桥接",C["rose_soft"],C["rose"]),
            (.68,"MEQ 多视图","第三强视图抑制单点高分",C["purple_soft"],C["purple"])]
    for x,t,body,face,edge in blocks: box_text(c,x,.18,.29,.60,t,body,face,edge,title_size=9.4,body_size=8.0,align="center")
    arrow(c,(.32,.48),(.355,.48),lw=1);arrow(c,(.645,.48),(.68,.48),lw=1)
    return save(fig,"fig3_query_evidence_kernel")


def ebp_mechanism():
    fig,ax=plt.subplots(figsize=(mm(158),mm(74)));fig.subplots_adjust(left=.02,right=.98,top=.95,bottom=.06);ax.axis("off")
    stages=[(.02,"候选证据","Cross · TRACE\nGraph · Lexical",C["blue_soft"],C["blue"]),
            (.22,"稳健聚合","第三强视图\n形成一致性",C["purple_soft"],C["purple"]),
            (.42,"引文回传","按 √|R|\n归一化投票",C["teal_soft"],C["teal"]),
            (.62,"目标合并","相同 OpenAlex ID\n累积支持",C["rose_soft"],C["rose"]),
            (.82,"批量物化","Top100 目标\n一次 API 请求",C["amber_soft"],C["amber"])]
    for x,t,b,f,e in stages: box_text(ax,x,.58,.16,.28,t,b,f,e,title_size=9.2,body_size=8.0,align="center")
    for x in (.18,.38,.58,.78):arrow(ax,(x,.72),(x+.04,.72),lw=1.0)
    ax.text(.03,.43,"多篇已评分论文",transform=ax.transAxes,fontsize=9.2,fontweight="bold")
    src=[(.10,.25,"P1"),(.22,.34,"P2"),(.32,.22,"P3"),(.43,.34,"P4")]
    # Avoid Unicode subscript forms: their glyph coverage is inconsistent in
    # Windows PDF/SVG export.  Greek suffixes keep the node identities clear.
    tar=[(.62,.31,"Wα",C["rose"]),(.72,.18,"Wβ",C["teal"])]
    for si,ti in [(0,0),(1,0),(2,0),(2,1),(3,1)]:
        sx,sy,_=src[si];tx,ty,_,_=tar[ti];ax.plot([sx,tx],[sy,ty],transform=ax.transAxes,color=C["line"],lw=1)
    for x,y,t in src:
        ax.add_patch(Circle((x,y),.028,transform=ax.transAxes,facecolor=C["blue_soft"],edgecolor=C["blue"]));ax.text(x,y,t,transform=ax.transAxes,ha="center",va="center",fontsize=8,fontweight="bold")
    for x,y,t,col in tar:
        ax.add_patch(Circle((x,y),.034,transform=ax.transAxes,facecolor="white",edgecolor=col,lw=1.5));ax.text(x,y,t,transform=ax.transAxes,ha="center",va="center",fontsize=8.5,fontweight="bold")
    box_text(ax,.79,.11,.18,.28,"补充候选层","支持论文数\n投票分与引文路径",C["green_soft"],C["green"],title_size=9.2,body_size=8.0,align="center")
    arrow(ax,(.75,.23),(.79,.23),color=C["green"])
    return save(fig,"fig4_ebp_mechanism")


def evaluation_protocol():
    fig,ax=plt.subplots(figsize=(mm(158),mm(70)));fig.subplots_adjust(left=.02,right=.98,top=.95,bottom=.06);ax.axis("off")
    top=[(.05,"固定数据划分","Calibration · Dev\nBlind · Validation",C["teal_soft"],C["teal"]),
         (.40,"统一论文身份","arXiv · DOI\nOpenAlex · S2",C["purple_soft"],C["purple"]),
         (.75,"一致约束规则","日期范围 · 版本号\n跨来源去重",C["rose_soft"],C["rose"])]
    for x,t,b,f,e in top:box_text(ax,x,.62,.20,.25,t,b,f,e,title_size=9.8,body_size=8.0,align="center")
    arrow(ax,(.25,.745),(.40,.745),lw=1.1);arrow(ax,(.60,.745),(.75,.745),lw=1.1)
    layers=[(.05,"候选层","Candidate Recall\n通道贡献 · 新增命中",C["teal_soft"],C["teal"]),
            (.40,"排序层","Recall@20 · NDCG@20\nGold 排名 · Top20 TP",C["purple_soft"],C["purple"]),
            (.75,"集合层","Macro P / R / F1\n结果规模 · 有效密度",C["amber_soft"],C["amber"])]
    for x,t,b,f,e in layers:box_text(ax,x,.17,.20,.27,t,b,f,e,title_size=9.5,body_size=8.0,align="center")
    arrow(ax,(.25,.305),(.40,.305),lw=1);arrow(ax,(.60,.305),(.75,.305),lw=1)
    ax.text(.50,.04,"同一查询、身份和日期规则下，将性能变化定位到候选、排序或集合决策",
            transform=ax.transAxes,ha="center",fontsize=8.1,color=C["muted"])
    return save(fig,"fig5_evaluation_protocol")


def evidence_matrix():
    fig,ax=plt.subplots(figsize=(mm(158),mm(72)));fig.subplots_adjust(left=.015,right=.985,top=.90,bottom=.08);ax.axis("off")
    rows=[
        ["Global Convergence of MAML…","桥接工作","100","100","74"],
        ["Improving Generalization of MAML…","桥接工作","100","100","74"],
        ["How to train your MAML","奠基工作","100","100","74"],
        ["Rapid Learning or Feature Reuse?","奠基工作","100","100","68"],
        ["Alpha MAML","桥接工作","100","100","74"],
    ]
    cols=["论文","研究角色","主题","方法","证据"]
    tbl=ax.table(cellText=rows,colLabels=cols,cellLoc="center",colLoc="center",loc="center",colWidths=[.47,.17,.12,.12,.12])
    tbl.auto_set_font_size(False);tbl.set_fontsize(8.2);tbl.scale(1,1.55)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_edgecolor(C["grid"]);cell.set_linewidth(.7)
        if r==0:
            cell.set_facecolor(C["blue_soft"]);cell.set_text_props(weight="bold",color=C["ink"])
        else:
            cell.set_facecolor("white");
            if c==0: cell.set_text_props(ha="left")
            if c in (2,3): cell.set_text_props(color=C["green"],weight="bold")
            if c==4: cell.set_text_props(color=C["blue"],weight="bold")
    ax.set_title("可核验证据覆盖矩阵（界面字段重绘）",fontsize=10.0,fontweight="bold",pad=8)
    return save(fig,"fig6_evidence_matrix")


def relation_graph():
    fig,ax=plt.subplots(figsize=(mm(158),mm(76)));fig.subplots_adjust(left=.02,right=.98,top=.94,bottom=.06);ax.axis("off")
    ax.add_patch(Circle((.50,.48),.055,transform=ax.transAxes,facecolor=C["blue_soft"],edgecolor=C["blue"],lw=1.6));ax.text(.50,.48,"查询 Q",transform=ax.transAxes,ha="center",va="center",fontsize=9,fontweight="bold")
    nodes=[(.18,.76,"Global Convergence\nof MAML",C["teal"]),(.50,.83,"Bi-Level Constraint",C["teal"]),(.82,.72,"How to train\nyour MAML",C["amber"]),(.82,.26,"Rapid Learning or\nFeature Reuse?",C["amber"]),(.50,.13,"Alpha MAML",C["teal"]),(.18,.24,"MAML Convergence\nTheory",C["rose"])]
    for i,(x,y,t,col) in enumerate(nodes):
        ax.plot([.50,x],[.48,y],transform=ax.transAxes,color=C["line"],lw=1.1,zorder=1)
        rounded(ax,x-.12,y-.07,.24,.14,"white",col,radius=.03,lw=1.2,z=2)
        ax.text(x,y,t,transform=ax.transAxes,ha="center",va="center",fontsize=8.0,fontweight="bold",zorder=3)
    ax.text(.02,.03,"蓝绿：桥接/改进工作    金色：奠基工作    玫红：理论关系目标",
            transform=ax.transAxes,fontsize=8.0,color=C["muted"])
    return save(fig,"fig7_relation_graph")


def paper_card():
    fig,ax=plt.subplots(figsize=(mm(158),mm(80)));fig.subplots_adjust(left=.015,right=.985,top=.97,bottom=.03);ax.axis("off")
    ax.set_facecolor(C["navy_bg"]);fig.patch.set_facecolor(C["navy_bg"])
    rounded(ax,.02,.04,.96,.91,C["navy_card"],"#2E4D86",radius=.02,lw=1.2)
    ax.text(.05,.86,"1",transform=ax.transAxes,color="#62C9FF",fontsize=10,fontweight="bold")
    ax.text(.10,.86,"Global Convergence of MAML and Theory-Inspired NAS",
            transform=ax.transAxes,color="white",fontsize=9.0,fontweight="bold",va="center")
    ax.text(.92,.86,"0.8038",transform=ax.transAxes,color="#64BFFF",fontsize=12,fontweight="bold",ha="right",va="center")
    ax.text(.05,.75,"桥接工作    第一轮    OpenAlex    LLM 判定相关    2022    被引 37 次",
            transform=ax.transAxes,color="#AFC3DF",fontsize=8.0)
    metrics=[("RRF",1.00,C["blue"]),("语义匹配",.86,C["teal"]),("约束覆盖",1.00,C["purple"]),("引文影响",.46,C["amber"]),("新近性",.67,C["rose"]),("REG 证据",.80,"#8A6CD1")]
    for i,(name,val,col) in enumerate(metrics):
        row=i//2;colidx=i%2;x=.05+colidx*.47;y=.62-row*.13
        ax.text(x,y+0.035,name,transform=ax.transAxes,color="#D3DEEF",fontsize=8.0)
        ax.plot([x+.12,x+.41],[y+0.04,y+0.04],transform=ax.transAxes,color="#30466D",lw=5,solid_capstyle="round")
        ax.plot([x+.12,x+.12+.29*val],[y+0.04,y+0.04],transform=ax.transAxes,color=col,lw=5,solid_capstyle="round")
        ax.text(x+.43,y+0.035,f"{val:.2f}",transform=ax.transAxes,color="white",fontsize=8.0,ha="right")
    rounded(ax,.05,.11,.90,.13,"#10213F","#244576",radius=.012,lw=.8)
    ax.text(.07,.185,"证据摘要",transform=ax.transAxes,color="#67CFFF",fontsize=8.3,fontweight="bold")
    ax.text(.07,.135,"理论分析、收敛性与少样本学习约束均被摘要和引文路径共同支持。",
            transform=ax.transAxes,color="#C6D4E8",fontsize=8.0)
    return save(fig,"fig8_paper_card")


def _csv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def ebp_results():
    summary=_csv(EBP_REPORT/"06_summary_metrics.csv");curve=_csv(EBP_REPORT/"04_target_budget_curve.csv")
    state=json.loads((EBP_REPORT/"state.json").read_text(encoding="utf-8"))
    fig=plt.figure(figsize=(mm(158),mm(92)));gs=fig.add_gridspec(2,2,width_ratios=[1.0,1.28],left=.09,right=.98,top=.92,bottom=.20,wspace=.36,hspace=.52)
    a=fig.add_subplot(gs[:,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,1])
    splits=["calibration","dev","blind50"];labels=["Calibration\n(n=40)","Dev\n(n=20)","Blind\n(n=50)"]
    base={r["split"]:float(r["recall"]) for r in summary if r["method"]=="Existing candidate pool"};ebp={r["split"]:float(r["recall"]) for r in summary if r["method"]=="Existing pool + EBP"}
    x=np.arange(3);w=.34
    z0=a.bar(x-w/2,[base[s] for s in splits],w,color=C["blue"],label="原候选池");z1=a.bar(x+w/2,[ebp[s] for s in splits],w,color=C["rose"],label="+ EBP")
    a.set_xticks(x,labels);a.set_ylim(.36,.61);a.set_ylabel("Candidate Recall");a.set_title("a  跨划分候选召回",loc="left",fontweight="bold");a.grid(axis="y",color=C["grid"]);a.set_axisbelow(True);a.legend(loc="upper left",ncol=2)
    for bars in (z0,z1):
        for bar in bars:a.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.006,f"{bar.get_height():.3f}",ha="center",fontsize=8.0)
    cols={"calibration":C["blue"],"dev":C["teal"],"blind50":C["rose"]}
    for s,l in zip(splits,["Calibration","Dev","Blind"]):
        rr=sorted((r for r in curve if r["split"]==s),key=lambda r:int(r["target_limit"]));b.plot([int(r["target_limit"]) for r in rr],[float(r["candidate_recall"]) for r in rr],marker="o",ms=4,lw=1.6,color=cols[s],label=l)
    # The panel title already states the budget variable; omitting a repeated
    # x-axis label leaves a clean inter-panel gutter at final A4 size.
    b.set_title("b  预算—召回响应",loc="left",fontweight="bold");b.set_ylabel("Recall");b.grid(color=C["grid"]);b.legend(ncol=3,loc="lower right")
    blind=state["splits"]["blind50"];vals=[int(blind["baseline_candidate"]["TP"]),int(blind["ebp_candidate"]["TP"]),int(blind["reg_meq_top20"]["TP"])];names=["原候选池","+ EBP","REG-MEQ Top20"]
    bars=c.barh(np.arange(3)[::-1],vals,height=.55,color=[C["blue"],C["rose"],"#8396BE"]);c.set_yticks(np.arange(3)[::-1],names);c.set_xlim(0,60);c.set_xlabel("Blind50 命中论文数");c.set_title("c  召回扩展与集合边界",loc="left",fontweight="bold");c.grid(axis="x",color=C["grid"]);c.set_axisbelow(True)
    for bar,v in zip(bars,vals):c.text(v+1,bar.get_y()+bar.get_height()/2,str(v),va="center",fontsize=8.2,fontweight="bold")
    fig.text(.09,.025,"Blind50：+1 次 OpenAlex 批量请求/查询；+0 LLM；+0 Token",fontsize=8.0,color=C["muted"])
    return save(fig,"fig9_ebp_results")


def ablation():
    rows=_csv(SOURCE/"structured_evidence_ablation.csv")
    fig,axes=plt.subplots(1,2,figsize=(mm(158),mm(58)),gridspec_kw={"wspace":.48})
    specs=[("校准集",[r["method"] for r in rows],[float(r["calibration_top20_tp"]) for r in rows],(19.5,30)),
           ("开发集",[r["method"] for r in rows if r["dev_top20_tp"]],[float(r["dev_top20_tp"]) for r in rows if r["dev_top20_tp"]],(18.5,27))]
    for idx,(title,names,vals,xlim) in enumerate(specs):
        ax=axes[idx];y=np.arange(len(names))[::-1];ax.hlines(y,xlim[0],vals,color=C["grid"],lw=4.5);ax.scatter(vals,y,s=42,color=[C["line"],C["blue"],C["rose"],C["teal"]][:len(vals)],edgecolor="white",zorder=3)
        for yy,v,n in zip(y,vals,names):ax.text(v+.25,yy,f"{int(v)}",va="center",fontsize=8.3,fontweight="bold" if n=="REG" else "normal")
        ax.set_yticks(y,names);ax.set_xlim(*xlim);ax.set_xlabel("Top20 相关论文命中数");ax.set_title(f"{'ab'[idx]}  {title}",loc="left",fontweight="bold");ax.grid(axis="x",color=C["grid"]);ax.set_axisbelow(True)
    axes[1].annotate("+3",xy=(25,0),xytext=(23.7,.55),color=C["green"],fontsize=9,fontweight="bold",arrowprops={"arrowstyle":"->","color":C["green"]})
    fig.subplots_adjust(left=.17,right=.97,top=.87,bottom=.23)
    return save(fig,"fig10_ablation")


def readable_crops():
    OUT.mkdir(parents=True,exist_ok=True)
    specs=[("03_results_overview_viewport.png",(20,55,835,505),"ui_query_controls.png"),
           ("06_reg_action.png",(35,70,845,365),"ui_reg_action.png")]
    for src,box,name in specs:
        im=Image.open(SHOTS/src).convert("RGB").crop(box)
        im.save(OUT/name,quality=95)
    # Two enlarged paper-card details are stacked so both title/metadata and
    # scoring evidence remain readable at full text width.
    src=Image.open(SHOTS/"11_paper_cards_view.png").convert("RGB")
    top=src.crop((55,105,820,355));bottom=src.crop((430,245,1220,520))
    target=1200
    def fit(im):
        h=round(im.height*target/im.width);return im.resize((target,h),Image.Resampling.LANCZOS)
    top=fit(top);bottom=fit(bottom)
    canvas=Image.new("RGB",(target,top.height+bottom.height+16),C["navy_bg"])
    canvas.paste(top,(0,0));canvas.paste(bottom,(0,top.height+16));canvas.save(OUT/"ui_paper_card_detail.png",quality=95)


def main():
    readable_crops()
    outputs={
        "system":system_architecture(),"modes":result_modes(),"kernel":query_evidence_kernel(),
        "ebp":ebp_mechanism(),"evaluation":evaluation_protocol(),"matrix":evidence_matrix(),
        "graph":relation_graph(),"card":paper_card(),"ebp_results":ebp_results(),"ablation":ablation(),
    }
    qa={"target_width_mm":158,"minimum_designed_text_pt":8.0,
        "figure_font":"Microsoft YaHei (CJK and sans-serif Latin glyphs)","exports":outputs}
    (OUT/"qa_report.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(qa,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
