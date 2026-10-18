from __future__ import annotations

import hashlib
import re
import subprocess
from copy import deepcopy
from pathlib import Path

import build_project_document_v30 as b
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "artifacts" / "submission_2026"
OUT = ASSETS / "draft"
FIGS = ASSETS / "figures_nature"
V31_FIGS = FIGS / "final_v31"
EBP_FIGS = FIGS / "evidence_backprojection_v21"
V32_FIGS = FIGS / "report_v32"
V33_FIGS = FIGS / "report_v33"
SHOTS = ASSETS / "screenshots"
FORMULA_CACHE = OUT / "formula_cache_v31"
CONTENT_WIDTH_CM = 15.8
_bookmark_id = 100


def configure_section(section) -> None:
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.65)
    section.bottom_margin = Cm(2.55)
    section.left_margin = Cm(2.6)
    section.right_margin = Cm(2.6)
    section.header_distance = Cm(1.8)
    section.footer_distance = Cm(1.8)


def add_styles(doc: Document) -> None:
    b.add_styles(doc)
    for style_name in ("BodyTextCN", "BulletCN"):
        s = doc.styles[style_name]
        s.paragraph_format.line_spacing = Pt(19)
        s.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        s.paragraph_format.widow_control = True
    for style_name in ("FigureCaptionCN", "TableCaptionCN"):
        s = doc.styles[style_name]
        s.font.size = Pt(10.5)
        s.paragraph_format.keep_with_next = style_name == "TableCaptionCN"
    for level in (1, 2, 3):
        doc.styles[f"Heading {level}"].paragraph_format.keep_with_next = True


def set_table_borders_none(table) -> None:
    tbl_pr = table._tbl.tblPr
    old = tbl_pr.find(qn("w:tblBorders"))
    if old is not None:
        tbl_pr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = OxmlElement(f"w:{edge}")
        node.set(qn("w:val"), "nil")
        node.set(qn("w:sz"), "0")
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), "FFFFFF")
        borders.append(node)
    tbl_pr.append(borders)


def set_table_grid_widths(table, widths_cm: list[float]) -> None:
    """Set both the table grid and preferred table width.

    Cell ``tcW`` values alone are only hints in Word/WPS and can collapse back
    to equal columns.  Writing the grid as well keeps wide text columns wide
    and prevents narrow columns from creating unnecessary wrapped lines.
    """
    total_twips = sum(Cm(width).twips for width in widths_cm)
    for grid_col, width in zip(table._tbl.tblGrid.gridCol_lst, widths_cm):
        grid_col.set(qn("w:w"), str(Cm(width).twips))
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(total_twips))
    tbl_w.set(qn("w:type"), "dxa")


def add_bookmark_to_run(run, name: str) -> None:
    global _bookmark_id
    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), str(_bookmark_id))
    start.set(qn("w:name"), name)
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), str(_bookmark_id))
    run._r.addprevious(start)
    run._r.addnext(end)
    _bookmark_id += 1


def caption(doc: Document, prefix: str, title: str, *, kind="figure", key: str | None = None):
    style = "FigureCaptionCN" if kind == "figure" else "TableCaptionCN"
    p = doc.add_paragraph(style=style)
    p.paragraph_format.keep_with_next = kind == "table"
    r = b.add_run_text(p, prefix, size=10.5)
    if key:
        add_bookmark_to_run(r, key)
    b.add_run_text(p, "  " + title, size=10.5)
    return p


def figure(doc: Document, path: Path, prefix: str, title: str, *, key=None, width_in=6.20):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width_in))
    caption(doc, prefix, title, kind="figure", key=key)
    return p


def table(doc: Document, headers: list[str], rows: list[list[str]], widths_cm: list[float],
          prefix: str | None = None, title: str | None = None, *, key=None,
          font_size=9.6, repeat_header: bool | None = None):
    if prefix and title:
        caption(doc, prefix, title, kind="table", key=key)
    total = sum(widths_cm)
    if total > CONTENT_WIDTH_CM:
        scale = CONTENT_WIDTH_CM / total
        widths_cm = [x * scale for x in widths_cm]
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    set_table_grid_widths(t, widths_cm)
    b.set_table_three_line_borders(t)
    hdr = t.rows[0]
    b.prevent_row_split(hdr)
    if repeat_header if repeat_header is not None else len(rows) >= 8:
        b.set_repeat_table_header(hdr)
    for cell, text, width in zip(hdr.cells, headers, widths_cm):
        b.set_cell_width(cell, width)
        b.set_cell_shading(cell)
        b.set_cell_margins(cell, top=35, start=55, bottom=35, end=55)
        b.set_header_rule(cell)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.keep_with_next = True
        b.add_run_text(p, text, bold=True, size=font_size)
    for row in rows:
        new_row = t.add_row()
        b.prevent_row_split(new_row)
        for j, (cell, text, width) in enumerate(zip(new_row.cells, row, widths_cm)):
            b.set_cell_width(cell, width)
            b.set_cell_margins(cell, top=35, start=55, bottom=35, end=55)
            b.set_cell_shading(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j in (0, len(headers)-1) else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            b.add_run_text(p, text, size=font_size)
    return t


def grouped_metric_table(doc: Document, *, stub_header: str,
                         groups: list[tuple[str, list[str]]], rows: list[list[str]],
                         widths_cm: list[float], prefix: str, title: str, key: str,
                         font_size: float = 8.9, tail_header: str | None = None):
    """Compact two-level header used by SPAR/LitSearch-style results tables.

    ``groups`` contains a group title and its leaf metric labels.  The first
    column is a vertically merged dataset/mode stub; ``tail_header`` optionally
    adds a second vertically merged summary column (for example delta TP).
    """
    caption(doc, prefix, title, kind="table", key=key)
    total = sum(widths_cm)
    if total > CONTENT_WIDTH_CM:
        scale = CONTENT_WIDTH_CM / total
        widths_cm = [x * scale for x in widths_cm]

    leaf_count = sum(len(labels) for _, labels in groups)
    ncols = 1 + leaf_count + (1 if tail_header else 0)
    t = doc.add_table(rows=2, cols=ncols)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    set_table_grid_widths(t, widths_cm)
    b.set_table_three_line_borders(t)
    for row in t.rows:
        b.prevent_row_split(row)
        b.set_repeat_table_header(row)

    # Establish widths and restrained cell padding before merges.
    for row in t.rows:
        for cell, width in zip(row.cells, widths_cm):
            b.set_cell_width(cell, width)
            b.set_cell_margins(cell, top=28, start=35, bottom=28, end=35)
            b.set_cell_shading(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

    stub = t.cell(0, 0).merge(t.cell(1, 0))
    top_col = 1
    for group_title, labels in groups:
        start = top_col
        end = start + len(labels) - 1
        group_cell = t.cell(0, start)
        if end > start:
            group_cell = group_cell.merge(t.cell(0, end))
        p = group_cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        b.add_run_text(p, group_title, bold=True, size=font_size)
        for offset, leaf in enumerate(labels):
            cell = t.cell(1, start + offset)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            b.add_run_text(p, leaf, bold=True, size=font_size)
            b.set_header_rule(cell)
        top_col = end + 1

    def _fill_merged_header(cell, text):
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        b.add_run_text(p, text, bold=True, size=font_size)
        b.set_header_rule(cell)

    _fill_merged_header(stub, stub_header)
    if tail_header:
        tail = t.cell(0, ncols - 1).merge(t.cell(1, ncols - 1))
        _fill_merged_header(tail, tail_header)

    for values in rows:
        row = t.add_row()
        b.prevent_row_split(row)
        for j, (cell, value, width) in enumerate(zip(row.cells, values, widths_cm)):
            b.set_cell_width(cell, width)
            b.set_cell_margins(cell, top=30, start=35, bottom=30, end=35)
            b.set_cell_shading(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            b.add_run_text(p, value, size=font_size)
    return t


def compact_algorithm(doc: Document, prefix: str, title: str, rows: list[tuple[str, str]], *, key=None):
    caption(doc, prefix, title, kind="table", key=key)
    t = doc.add_table(rows=0, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    set_table_grid_widths(t, [CONTENT_WIDTH_CM])
    b.set_table_three_line_borders(t)
    for idx, text in rows:
        row = t.add_row()
        b.prevent_row_split(row)
        cell = row.cells[0]
        b.set_cell_width(cell, CONTENT_WIDTH_CM)
        b.set_cell_margins(cell, top=18, start=45, bottom=18, end=45)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        b.add_run_text(p, f"{idx}  ", bold=True, size=9.8)
        b.add_run_text(p, text, size=9.8)
    return t


def _formula_paragraph(latex: str):
    FORMULA_CACHE.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(latex.encode("utf-8")).hexdigest()[:12]
    md = FORMULA_CACHE / f"{digest}.md"
    docx = FORMULA_CACHE / f"{digest}.docx"
    if not docx.exists():
        md.write_text(f"$$\n{latex}\n$$\n", encoding="utf-8")
        subprocess.run(["pandoc", str(md), "-o", str(docx)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    source = Document(docx)
    para = next(p for p in source.paragraphs if p._p.xpath(".//m:oMath"))
    return deepcopy(para._p)


def equation(doc: Document, latex: str, number: int, *, key: str):
    # Three equal-purpose zones keep the mathematical expression centered on
    # the text block while the equation number remains flush right.  A two-
    # column layout centers the expression inside the left cell and therefore
    # visibly shifts every formula to the left.
    t = doc.add_table(rows=1, cols=3)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    set_table_borders_none(t)
    row = t.rows[0]
    b.prevent_row_split(row)
    widths = (1.5, 12.8, 1.5)
    for grid_col, width in zip(t._tbl.tblGrid.gridCol_lst, widths):
        grid_col.set(qn("w:w"), str(Cm(width).twips))
    for cell, width in zip(row.cells, widths):
        b.set_cell_width(cell, width)
        b.set_cell_margins(cell, top=45, start=8, bottom=45, end=8)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    # The empty left cell balances the right-hand number column.
    row.cells[0].paragraphs[0].clear()
    formula_cell = row.cells[1]
    formula_cell._tc.remove(formula_cell.paragraphs[0]._p)
    p_xml = _formula_paragraph(latex)
    ppr = p_xml.get_or_add_pPr()
    jc = ppr.find(qn("w:jc"))
    if jc is None:
        jc = OxmlElement("w:jc")
        ppr.append(jc)
    jc.set(qn("w:val"), "center")
    formula_cell._tc.append(p_xml)
    p = row.cells[2].paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    r = b.add_run_text(p, f"({number})", size=10.5, font=b.BODY_LATIN, latin_font=b.BODY_LATIN)
    add_bookmark_to_run(r, key)
    return t


def add_reference(doc: Document, text: str):
    p = doc.add_paragraph(style="BodyTextCN")
    p.paragraph_format.left_indent = Cm(0.85)
    p.paragraph_format.first_line_indent = Cm(-0.85)
    p.paragraph_format.line_spacing = Pt(14.2)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    p.paragraph_format.space_before = Pt(1.2)
    p.paragraph_format.space_after = Pt(0)
    b.add_run_text(p, text, size=9.2)
    return p


def build() -> Path:
    global _bookmark_id
    _bookmark_id = 100
    OUT.mkdir(parents=True, exist_ok=True)
    doc = Document()
    doc.core_properties.title = "研问智搜——科研场景下复杂学术查询的智能论文搜索与推荐系统"
    doc.core_properties.subject = "第八届中国研究生人工智能创新大赛项目文档"
    doc.core_properties.author = "虫虫特工队"
    doc.core_properties.last_modified_by = "虫虫特工队"
    doc.core_properties.keywords = "学术检索, 智能搜索, 论文推荐, 检索Agent, 证据排序"
    add_styles(doc)

    cover = doc.sections[0]
    configure_section(cover)
    b.set_page_number_format(cover, "upperRoman", 1)
    cover.header.is_linked_to_previous = False
    cover.header.paragraphs[0].clear()
    b.add_front_footer(cover)

    p = doc.add_paragraph(); p.paragraph_format.space_before = Cm(1.9); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    b.add_run_text(p, "第八届中国研究生人工智能创新大赛", size=22, font=b.HEAD_CJK, latin_font=b.HEAD_CJK)
    p = doc.add_paragraph(); p.paragraph_format.space_before = Cm(2.2); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    b.add_run_text(p, "研问智搜", size=42, font=b.HEAD_CJK, latin_font=b.HEAD_CJK)
    for text in ("科研场景下复杂学术查询的", "智能论文搜索与推荐系统"):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        b.add_run_text(p, text, size=16, font=b.HEAD_CJK, latin_font=b.HEAD_CJK)
    p = doc.add_paragraph(); p.paragraph_format.space_before = Cm(1.35); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    b.add_run_text(p, "项目文档", size=22, font=b.HEAD_CJK, latin_font=b.HEAD_CJK)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    b.add_run_text(p, "版本 ", size=14); b.add_run_text(p, "V3.3", size=14, font=b.BODY_LATIN)
    p = doc.add_paragraph(); p.paragraph_format.space_before = Cm(1.9); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    b.add_run_text(p, "日    期：2026 年 8 月", size=12)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    b.add_run_text(p, "团队名称：虫虫特工队", size=12)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    b.add_run_text(p, "参赛组别：企业赛题", size=12)

    # Revision history is intentionally the second page and uses concise staged dates.
    hist_sec = doc.add_section(WD_SECTION.NEW_PAGE)
    configure_section(hist_sec); b.set_page_number_format(hist_sec, "upperRoman", 2); b.add_body_header_footer(hist_sec)
    hist_sec.footer.paragraphs[0].clear(); field = b.add_field(hist_sec.footer.paragraphs[0], "PAGE", "II")
    b.set_run_font(field, b.BODY_CJK, b.BODY_LATIN, 10.5); hist_sec.footer.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before = Pt(18); p.paragraph_format.space_after = Pt(14)
    b.add_run_text(p, "记录更改历史", size=16, font=b.HEAD_CJK, latin_font=b.HEAD_CJK)
    table(doc, ["序号", "更改内容", "版本", "作者", "更改日期", "备注"], [
        ["1", "原型设计", "V1.0", "虫虫特工队", "2026.06.12", "完成查询理解、基础检索与结果页"],
        ["2", "多源检索", "V1.4", "虫虫特工队", "2026.06.28", "接入多学术数据源并统一论文身份"],
        ["3", "身份治理", "V1.7", "虫虫特工队", "2026.07.06", "建立 arXiv、DOI、OpenAlex 与 S2 稳定标识规则"],
        ["4", "排序重构", "V2.0", "虫虫特工队", "2026.07.16", "形成 TRACE-Rank 与证据化排序链路"],
        ["5", "评测固化", "V2.3", "虫虫特工队", "2026.07.29", "建立候选、排序、集合三层评测与指标记录"],
        ["6", "证据增强", "V2.6", "虫虫特工队", "2026.08.08", "加入 REG-MEQ 与受控引文前沿"],
        ["7", "召回扩展", "V3.0", "虫虫特工队", "2026.08.19", "完成 EBP、双层结果集合与成本审计"],
        ["8", "材料重构", "V3.1", "虫虫特工队", "2026.08.26", "统一图表、公式、引文、交叉引用和材料表述"],
        ["9", "证据校准", "V3.3", "虫虫特工队", "2026.08.30", "完成最终选择器、联合实验与提交版系统联调"],
    ], [1.0, 2.4, 1.3, 2.3, 2.3, 6.5], font_size=9.2)
    b.body(doc, "说明：更改记录按原型形成、多源检索、身份治理、证据排序、评测固化、召回扩展与提交重构等阶段归档，用于追踪系统能力与文档版本之间的对应关系。")

    toc_sec = doc.add_section(WD_SECTION.NEW_PAGE)
    configure_section(toc_sec); b.set_page_number_format(toc_sec, "upperRoman", 3); b.add_body_header_footer(toc_sec)
    toc_sec.footer.paragraphs[0].clear(); field = b.add_field(toc_sec.footer.paragraphs[0], "PAGE", "III")
    b.set_run_font(field, b.BODY_CJK, b.BODY_LATIN, 10.5); toc_sec.footer.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before = Pt(18); p.paragraph_format.space_after = Pt(14)
    b.add_run_text(p, "目录", size=16, font=b.HEAD_CJK, latin_font=b.HEAD_CJK)
    toc_p = doc.add_paragraph(); b.add_field(toc_p, 'TOC \\o "1-3" \\h \\z \\u', "目录将在打开文档时自动更新")

    main_sec = doc.add_section(WD_SECTION.NEW_PAGE)
    configure_section(main_sec); b.set_page_number_format(main_sec, "decimal", 1); b.add_body_header_footer(main_sec)

    b.heading(doc, "1 项目概况", 1)
    b.heading(doc, "1.1 背景和基础", 2)
    b.body(doc, "文献检索决定了研究者能否及时识别已有结论、比较技术路线并建立可靠的问题边界。真实科研查询往往同时包含主题、方法、任务、数据、时间和论文关系等条件，例如“哪些论文从理论上分析 MAML 在小样本学习中的收敛性”。这类问题并非简单的语义近邻搜索：系统既要找到相关领域论文，也要验证候选是否满足“理论分析”“收敛性”等必要约束，并识别候选与奠基论文、改进工作或比较对象之间的关系。")
    b.body(doc, "稀疏检索、稠密表示和交互式重排为语义匹配奠定了基础。SciBERT、Sentence-BERT、SPECTER 与 ColBERT 分别从科学语料预训练、句向量、引文监督和晚交互建模等角度改善文档表示与相关性计算[15-18]；RRF 和 MMR 则分别提供多排序融合与多样性控制的经典机制[19-20]。这些方法能够提高单轮召回或排序质量，但复杂学术查询还要求系统主动分解问题、选择检索工具并根据已获证据调整下一步动作。")
    b.body(doc, "大语言模型推动学术搜索从“输入关键词—返回列表”转向可规划的检索 Agent。PaSa 采用 Crawler—Selector 双 Agent 并通过强化学习优化完整搜索[1]；SPAR 以 RefChain 组织查询分解和查询演化[2]；PaperQA/PaperQA2 将全文检索、证据评估和可溯源生成连接起来[5-6]。DSP、ReAct 与 AGILE 等通用 Agent 研究进一步说明，检索、工具调用和环境反馈可以被组织为连续决策过程[7-10]。")
    b.body(doc, "本项目面向赛题要求构建端到端系统：以 Query Contract 显式表达查询约束，以多源检索和受控引文前沿扩大候选覆盖，以 TRACE-Rank、REG-MEQ 与 EBP 组织全局排序、关系证据和引文缺口恢复，再通过不同规模的结果集合与结构化界面完成交付。系统同时记录论文身份、来源通道、排序分项、证据动作与运行成本，使最终结果能够追溯到对应的查询约束、检索来源和排序依据。")
    figure(doc, V32_FIGS / "ui_query_controls.png", "图 1", "查询输入、检索深度、结果模式与检索策略（界面局部放大）", key="fig_home")

    b.heading(doc, "1.1.1 国内外研究现状", 3)
    b.body(doc, "现有学术检索技术可概括为四条路线。第一类是以倒排索引、BM25 和引文排序为代表的传统检索，优点是稳定、低成本和覆盖广，局限在于复杂约束需要用户自行拆解。第二类是基于科学文本表示的语义检索，SPECTER 将引文图作为论文表示学习信号[15]，SciBERT 与 Sentence-BERT 提供科学语料和句级语义基础[16-17]，ColBERT 通过晚交互保留细粒度词元匹配[18]。BEIR 的跨域评测与 BRIGHT 的推理密集型检索实验均显示，零样本迁移和多步推理查询仍然具有明显难度[13-14]。")
    b.body(doc, "第三类是面向复杂问题的检索 Agent。PaSa 将搜索动作和论文选择拆分为两个角色[1]；SPAR 通过参考链与查询演化扩大搜索范围[2]；AGILE、ReAct 和 DSP 强调规划、工具调用与环境反馈的协同[7-10]。这类方法能够减少用户手工改写检索式的负担，但多轮扩展也可能带来查询漂移、候选重复和调用成本增长，因此需要明确的动作停止条件和一致的评测协议。")
    b.body(doc, "第四类是科学证据问答与综合。PaperQA 及其后续工作从全文中检索证据片段并生成带来源的答案[5-6]，RAG 为参数化模型连接外部可更新知识提供了通用框架[9]。与证据问答不同，本赛题首先评价论文集合本身的精确率和召回率，因此系统不能只生成一段看似合理的综述，还必须保证返回论文的身份、范围和数量均可核验。")
    b.body(doc, "基于上述分析，研问智搜把相关工作中的有效机制重新组织为四个彼此独立但可传递证据的层次：发现层追求候选覆盖，排序层处理多源得分与约束，证据层判断关系和引文路径，集合层控制最终交付密度。图 2 展示总体流程，表 1 从查询、检索、证据和交付四个方面给出方法定位。")
    figure(doc, V33_FIGS / "fig1_system_architecture.png", "图 2", "研问智搜端到端检索架构与证据反馈链路", key="fig_system_loop")
    table(doc, ["系统/方法", "查询处理", "检索与排序", "关系/引文证据", "输出形态"], [
        ["传统学术搜索", "关键词与筛选条件", "词项、引用与用户迭代", "引用网络", "论文列表"],
        ["PaSa[1]", "LLM 查询理解", "Crawler + Selector + RL", "搜索过程读取论文", "筛选论文集合"],
        ["SPAR[2]", "RefChain 分解", "查询演化与多 Agent", "参考链", "论文集合"],
        ["PaperQA2[5]", "问题驱动检索", "全文检索与证据评估", "全文证据", "可溯源答案"],
        ["研问智搜", "Query Contract", "多源召回 + TRACE-Rank", "REG-MEQ + EBP", "集合、矩阵、图谱、导出"],
    ], [2.8, 3.1, 4.0, 3.2, 3.5], "表 1", "代表性学术检索路线及其能力侧重", key="tab_related", font_size=9.1)

    b.heading(doc, "1.2 场景和价值", 2)
    b.body(doc, "研问智搜服务于开题调研、综述撰写、技术路线比较、研究前沿追踪和跨领域选题探索。用户可直接描述研究问题，系统自动识别主题、方法、任务、数据和时间范围，生成互补子查询并呈现检索过程。对于“奠基工作”“理论分析”“改进某方法”“在特定数据上比较”等关系型需求，系统额外保留关系谓词和引文方向，避免只返回主题相似论文。")
    b.body(doc, "系统同时减少检索式编写与结果核验成本。候选论文卡片呈现约束覆盖、摘要证据、引文路径、来源通道和排序分项；证据矩阵用于横向比较多篇论文对查询条件的覆盖；关系图谱帮助用户识别奠基、改进、比较和应用脉络；BibTeX、CSV、JSON 与打印报告则连接阅读、引用管理和团队讨论。")
    b.body(doc, "面向不同科研阶段，系统提供三类结果集合。核心结果给出证据最完整的论文，适合快速判断；均衡集合保留更多关键证据，适合方案调研；探索列表展示完整 Top20 与补充候选，适合综述式扩展。该设计把“检索得多”与“交付得准”分开处理，用户可以在不重复检索的情况下切换结果粒度。")
    figure(doc, V33_FIGS / "fig2_result_modes.png", "图 3", "共享排序池上的三类结果集合与导出边界", key="fig_overview")

    b.heading(doc, "1.3 所需支持", 2)
    b.body(doc, "系统后端基于 FastAPI，前端为浏览器可访问的单页应用，可在普通 Windows 笔记本的 CPU 环境中运行。外部论文数据来自 OpenAlex、arXiv、Crossref 和 Semantic Scholar；其中 OpenAlex 与 Semantic Scholar 提供稳定论文标识和引文图，Crossref 补充 DOI 与出版元数据，arXiv 提供预印本版本信息[21-26]。本地 MiniLM 交叉编码器只在证据重排阶段使用，无需现场训练。")
    b.body(doc, "在线运行需要网络与可用的大模型及学术 API 配置。系统提供环境检查、一键启动、请求超时和错误降级；当某一数据源限流或暂时不可用时，其余通道仍可继续返回候选；当大模型不可用时，规则式 Query Contract、确定性子查询和模板化归纳仍能维持基础检索链路。")
    table(doc, ["资源", "来源", "系统用途", "降级方式"], [
        ["论文与引文元数据", "OpenAlex[21,25]", "候选检索、身份、引用关系", "其他数据源补充"],
        ["预印本元数据", "arXiv[24]", "版本、日期与领域检索", "OpenAlex/Crossref 补充"],
        ["DOI/出版元数据", "Crossref[23]", "身份校验与元数据补全", "标题与其他稳定 ID"],
        ["学术图谱", "Semantic Scholar[22,26]", "语义候选与引文补充", "OpenAlex 引文图"],
        ["查询与归纳模型", "可配置 LLM", "约束解析、规划与摘要", "规则解析与模板归纳"],
        ["本地计算", "CPU + MiniLM", "交叉编码与证据评分", "轻量词项/图特征"],
    ], [3.0, 3.2, 5.2, 4.4], "表 2", "数据、模型和计算资源来源", key="tab_resources", font_size=9.2)

    b.heading(doc, "2 项目规划", 1)
    b.heading(doc, "2.1 整体目标", 2)
    b.body(doc, "项目目标是形成面向复杂学术查询的端到端智能论文搜索与推荐系统，使自然语言问题能够直接转化为可执行检索计划，并完成候选发现、身份融合、证据排序、结果选择和结构化交付。系统的设计同时面向三项约束：检索质量需要兼顾 Precision 与 Recall，推理过程需要控制 API、Token 和端到端时延，最终回复需要以列表、矩阵和关系图等形式组织。")
    b.bullet(doc, "（1）查询认知：解析主题、方法、任务、数据、时间、发表范围和关系谓词，形成 Query Contract；")
    b.bullet(doc, "（2）自主检索：在词项、语义、arXiv、标题探测、第二轮检索和引文前沿之间规划动作；")
    b.bullet(doc, "（3）候选组织：使用稳定标识统一论文身份，保存每个检索通道与引文路径的来源；")
    b.bullet(doc, "（4）证据排序：以 TRACE-Rank 形成全局排序，以 REG-MEQ 处理条件合取、关系和多视图一致性；")
    b.bullet(doc, "（5）缺口恢复：由 EBP 聚合全候选的一跳引文证据，批量物化高价值未解析论文；")
    b.bullet(doc, "（6）结构化交付：输出不同规模的论文集合、证据矩阵、关系图谱、推荐理由和标准化导出文件。")

    b.heading(doc, "2.2 技术创新点", 2)
    b.body(doc, "项目围绕复杂查询中的约束丢失、候选噪声、关系证据稀疏、引文可达论文未物化和最终集合有效密度不足等问题，形成六项相互衔接的系统设计。各设计分别作用于查询、召回、排序、缺口恢复、证据校准和结果交付，并通过统一论文身份与证据对象连接为完整检索链路。")
    b.body(doc, "（1）Query Contract 驱动的约束保持。系统不把复杂查询压缩为单个语义向量或关键词串，而是显式保留主题、方法、任务、数据/时间、排除条件和关系谓词。不同约束分别参与子查询生成、候选过滤和关系证据评分，使检索链路能够回答“候选为何相关”以及“尚缺哪一项证据”。图 4 展示查询认知与证据向量的衔接。", bold_prefix="（1）Query Contract 驱动的约束保持。")
    figure(doc, V32_FIGS / "fig3_query_evidence_kernel.png", "图 4", "Query Contract、稳定论文身份与五路证据核", key="fig_evidence_kernel")
    b.body(doc, "（2）多源双通道与受控引文前沿。系统并行组织 OpenAlex 词项检索、OpenAlex 语义检索和 arXiv 检索，并保留标题探测、Crossref、Semantic Scholar 与第二轮检索作为补充。SCGQ/QCF 从高置信种子出发探索参考文献和被引论文，同时记录引文方向、共同种子数量、桥接强度与新增候选率，避免无差别扩展。", bold_prefix="（2）多源双通道与受控引文前沿。")
    b.body(doc, "（3）TRACE-Rank 与 REG-MEQ 的分层证据排序。TRACE-Rank 负责跨来源融合、约束覆盖、权威性、新近性和多样性；REG 进一步判断候选是否同时满足必要条件并具有对应关系证据；MEQ 将 Cross、TRACE、Graph、Lexical 和 Fusion 五路信号转化为查询内百分位，以第三强视图抑制单一通道偶然高分。图 5 给出一次受控晋升动作。", bold_prefix="（3）TRACE-Rank 与 REG-MEQ 的分层证据排序。")
    figure(doc, V32_FIGS / "ui_reg_action.png", "图 5", "关系证据触发的候选检索与受控晋升动作（界面局部放大）", key="fig_reg_action")
    b.body(doc, "（4）证据加权引文反投影。EBP 不按单一种子顺序截取参考文献，而是把所有已评分候选视为证据节点，将五路一致性得分沿参考文献边回传至尚未解析的 OpenAlex ID。投票按参考列表长度平方根归一化，多篇候选共同指向的论文获得更高累计支持；Top100 目标由一次批量请求物化，并保留支持论文与引文路径。", bold_prefix="（4）证据加权引文反投影。")
    figure(doc, V32_FIGS / "fig4_ebp_mechanism.png", "图 6", "EBP 稳健聚合、引文回传与批量候选物化", key="fig_ebp")
    b.body(doc, "（5）证据校准核心结果选择。Evidence-Calibrated Selector（ECS）把 TRACE 分数、局部语义、关系图、约束覆盖、引文来源、元数据与排序先验组织为 40 维查询内证据向量，经标准化与线性校准得到论文级置信度。该选择器不使用论文身份作为特征，在保持相同输出规模的比较中直接检验“证据组合”而非结果篇数带来的变化。", bold_prefix="（5）证据校准核心结果选择。")
    figure(doc, V33_FIGS / "fig11_selector_mechanism.png", "图 7", "证据校准选择器的查询内特征、校准与结果交付", key="fig_selector")
    b.body(doc, "（6）候选池—排序池—结果集合分层。系统将高召回候选、Top20 排序和最终结果集合分开保存。核心结果、均衡集合与探索列表只改变交付边界，不重复调用外部 API；未进入当前集合的论文保留在补充候选区。该设计使高精度交付与宽范围探索能够共享同一检索过程。", bold_prefix="（6）候选池—排序池—结果集合分层。")

    b.heading(doc, "2.3 关键技术指标", 2)
    b.body(doc, "项目采用固定查询划分和统一评测规则记录检索质量、排序质量、最终集合质量和运行成本。候选层用于判断目标论文是否被找到，Top20 层用于判断其是否被排入头部，最终集合层用于衡量直接交付的有效密度。当前主要结果汇总于表 3，跨划分候选召回变化见图 8。")
    table(doc, ["评测对象", "数据划分", "当前实测", "说明"], [
        ["证据校准核心结果", "Blind50", "P=0.3200，R=0.2267，F1=0.2533", "同规模 TRACE：F1=0.2333"],
        ["REG-MEQ Top20", "Blind50", "Recall：0.4070→0.4370", "TP：36→38"],
        ["EBP 候选边界", "Calibration40", "Recall：0.4226→0.5185", "TP：35→44"],
        ["EBP 候选边界", "Dev20", "Recall：0.5346→0.5721", "TP：26→28"],
        ["EBP 候选边界", "Blind50", "Recall：0.5323→0.5623", "TP：50→53"],
        ["EBP 成本", "Blind50", "+1 API；2.43 s/查询", "+0 LLM；+0 Token"],
        ["自动化测试", "全链路", "168 项通过", "算法、评测、接口与系统运行"],
    ], [3.0, 2.8, 5.7, 4.3], "表 3", "当前实测结果与运行指标", key="tab_kpi", font_size=9.0, repeat_header=True)
    figure(doc, V32_FIGS / "fig9_ebp_results.png", "图 8", "EBP 的跨划分候选召回、预算响应与集合边界", key="fig_ebp_results")

    b.heading(doc, "2.4 与赛题参考系统的对比定位", 2)
    b.body(doc, "赛题参考系统分别强调强化学习搜索、查询演化、多索引语义检索和全文证据问答。研问智搜借鉴这些方向的共同经验，但不复刻其训练路线，而是围绕比赛端到端交付构建可在普通笔记本运行的证据化检索链路。表 4 的比较聚焦系统机制与输出形态，不将不同数据集、不同评测器下的数值直接并列。")
    table(doc, ["维度", "代表性已有路线", "研问智搜实现"], [
        ["查询理解", "LLM 改写、RefChain、Agent 规划[1-2,7-10]", "Query Contract + 关系谓词 + 意图路由"],
        ["候选召回", "语义索引、网页/论文搜索、引文追踪[1-6]", "多源双通道 + SCGQ/QCF + EBP"],
        ["综合排序", "Selector、稠密相关性或采样策略[1-3,15-18]", "TRACE-Rank + REG-MEQ + ECS"],
        ["结果组织", "论文集合或带引用答案[1-6]", "三类集合 + 矩阵 + 图谱 + 多格式导出"],
        ["成本与工程", "依系统实现与工具访问而异[4]", "并发调用、预算控制与通道审计"],
    ], [3.0, 6.0, 6.8], "表 4", "本项目与参考路线的方法定位", key="tab_position", font_size=9.2)

    b.heading(doc, "2.5 阶段成果与交付形态", 2)
    b.body(doc, "当前成果由可执行系统、算法与评测链路、项目文档、演示视频和辅助材料共同组成。各类材料不再分别维护一套结论，而是统一读取同一响应协议与实验登记结果：系统界面展示查询约束、候选来源、排序证据和结果集合，文档解释相应机制与指标，视频则沿同一示例查询呈现完整运行过程。由此可保证功能、图表、数据和现场演示之间相互对应。")
    b.body(doc, "系统交付以端到端可运行性、结果可核验性和过程可复现性为统一标准。用户输入自然语言科研问题后，可依次查看 Query Contract、子查询与检索通道、候选合并、关系证据动作、最终论文集合、证据矩阵和研究关系图谱；论文条目保留稳定标识、摘要、来源链接和推荐依据，并可导出 BibTeX、CSV、JSON 与打印报告，覆盖检索、筛选、阅读和引用管理的连续工作流。")
    b.body(doc, "系统前后端采用统一的数据对象和响应结构。每篇论文均保留身份标识、检索来源、约束证据、引文路径和排序分项，用户可从最终论文列表继续查看其检索与推荐依据。调用次数、Token、通道耗时和新增候选数同步记录，用于比较不同检索策略的效果与成本。")
    b.body(doc, "交付版本同时保存环境检查脚本、依赖清单、示例查询和自动化测试记录。项目文档、系统界面和演示视频使用同一组算法名称与实验结果，并在提交前统一检查页码、目录、图表编号、引用关系和敏感信息。")

    b.heading(doc, "3 实施方案", 1)
    b.heading(doc, "3.1 技术可行性分析", 2)
    b.body(doc, "系统按照查询认知、多源发现、候选组织、证据决策和结构化交付五层实现。查询认知层将自然语言问题转换为 Query Contract 和子查询；多源发现层调用学术 API 并探索受控引文前沿；候选组织层统一论文身份并完成 TRACE-Rank；证据决策层由 REG、MEQ 与 EBP 组成；交付层输出不同规模的论文集合、证据矩阵、图谱和导出文件。各层之间传递结构化对象而非不可解释的长文本，使错误能够定位到具体环节。")
    b.body(doc, "从部署角度看，系统不依赖专用 GPU 或现场训练。外部学术检索主要消耗网络请求，本地计算集中在文本规范化、轻量向量相似度、MiniLM 交叉编码和图特征聚合。后端以 FastAPI 组织异步数据源调用和算法流水线，前端通过浏览器访问，普通 Windows 笔记本即可完成端到端运行。")

    b.heading(doc, "3.1.1 数据与知识来源可行性", 3)
    b.body(doc, "OpenAlex 是开放的全球学术知识图谱，为论文、作者、来源、主题和引文关系提供稳定 ID 与 REST API[21,25]；Semantic Scholar 的文献图谱将论文、作者、实体和引用连接为可计算结构[22,26]；Crossref 提供可复用的 DOI 与出版元数据[23]；arXiv 提供预印本元数据和版本信息[24]。这些数据源的覆盖侧重点不同，适合以多源融合而非单一依赖的方式接入。")
    b.body(doc, "系统优先按 arXiv ID、DOI、OpenAlex ID 和 Semantic Scholar ID 合并候选，仅在稳定标识缺失时使用规范化标题。标题 fallback 会去除 arXiv 版本号、统一大小写与空白并规范标点，匹配结果仍保留原始来源和置信字段。该策略既减少同一论文的重复计数，也将“真正未召回”与“已经召回但身份不一致”区分开。")

    b.heading(doc, "3.1.2 算法与工程可行性", 3)
    b.body(doc, "多源检索、RRF 融合、语义相似度和 MMR 多样性均有成熟的信息检索研究基础[15-20]。本项目在此基础上增加 Query Contract、关系证据、查询内多视图仲裁和引文反投影，核心运算由确定性特征、查询内归一化和小规模本地重排构成，不需要额外训练大型模型。")
    b.body(doc, "工程链路已经完成查询理解、数据源客户端、候选去重、TRACE-Rank、SCGQ/QCF、REG-MEQ、EBP、证据校准与前端展示的接口统一。每个候选对象保留 paper_id、source_hits、constraint_evidence、citation_path、score_breakdown 和 cost_trace 等字段，便于前端解释结果并开展分层实验。当前 168 项自动化测试覆盖算法模块、身份匹配、评测协议、API 响应和系统运行。")

    b.heading(doc, "3.1.3 运行与复现可行性", 3)
    b.body(doc, "系统提供环境检查、一键启动和示例查询，后端启动后可依次验证健康接口、学术数据源、查询解析、候选检索、排序与导出功能。请求超时和单源错误不会中断完整流程，调用记录可直接用于定位异常通道。")
    b.body(doc, "项目提供依赖清单、配置模板、运行说明和自动化测试。相同输入在统一论文身份、日期约束和评测规则下产生可比较的候选、排序与结果集合，便于复现实验结论并检查各算法模块的实际贡献。")

    b.heading(doc, "3.2 技术细节", 2)
    b.heading(doc, "3.2.1 问题定义与符号", 3)
    b.body(doc, "给定自然语言学术查询 q，系统需要从可访问论文空间中构造候选池 P(q)，并输出规模为 k 的有序论文集合 Y_k(q)。检索源集合记为 M，单个检索源 m 返回的有序列表记为 R_m(q)。系统的直接目标是在调用预算约束下提高最终集合的 Precision、Recall 与 F1，同时保留论文身份和证据来源。候选合并与最终选择可写为式（1）。")
    equation(doc, r"\begin{aligned}\mathcal{P}(q)&=\bigcup_{m\in\mathcal{M}}R_m(q),\\[2pt]\mathcal{Y}_k(q)&=\operatorname{TopK}_{p\in\mathcal{P}(q)}S(p,q)\end{aligned}", 1, key="eq_problem")
    table(doc, ["符号", "含义", "取值/来源"], [
        ["q", "用户自然语言学术查询", "文本"],
        ["C(q)", "Query Contract 约束集合", "主题、方法、任务、数据、时间、关系等"],
        ["M", "检索源与检索动作集合", "OpenAlex、arXiv、Crossref、S2、引文动作"],
        ["P(q)", "统一身份后的候选论文池", "多源候选并集"],
        ["c_j(p,q)", "候选 p 对第 j 个必要条件的覆盖度", "[0,1]"],
        ["S_TRACE", "TRACE-Rank 全局排序分数", "查询内可比较分数"],
        ["S_REG / S_MEQ", "关系证据分数 / 多视图仲裁分数", "[0,1]"],
        ["V(j|q)", "EBP 对未解析论文 j 的累计投票", "非负实数"],
        ["Y_k(q)", "最终返回的 k 篇论文", "核心、均衡或探索集合"],
    ], [2.8, 7.2, 5.8], "表 5", "主要符号与数据对象", key="tab_notation", font_size=9.0, repeat_header=True)

    b.heading(doc, "3.2.2 端到端流程", 3)
    b.body(doc, "如图 2 所示，系统把宽召回与高精度交付分成可单独诊断的阶段。算法 1 给出一次检索的执行顺序：首先构建 Query Contract，再并行召回并统一身份；随后完成全局排序、关系证据判断和引文缺口恢复；最后根据结果模式形成最终集合。所有动作均写入 trace，供界面展示和结果追溯。")
    compact_algorithm(doc, "算法 1", "研问智搜端到端证据化检索流程", [
        ("输入", "自然语言查询 q、检索源集合 M、结果模式与运行预算。"),
        ("输出", "最终论文集合 Y_k(q)、补充候选与可追溯证据路径。"),
        ("1", "解析自然语言查询，形成 Query Contract、关系谓词、排除条件和意图类型。"),
        ("2", "生成互补子查询，并行执行词项检索、语义检索、arXiv 检索及补充数据源。"),
        ("3", "依据稳定论文标识合并候选，记录来源、排名、元数据和日期约束。"),
        ("4", "使用 RRF 与 TRACE-Rank 形成全局排序池，并保存各分项得分。"),
        ("5", "由 SCGQ/QCF 探索引文前沿，REG-MEQ 对关系证据和多视图一致性执行受控晋升。"),
        ("6", "EBP 从全候选的未解析引文 ID 中聚合投票，批量物化高支持候选。"),
        ("7", "以证据校准形成核心结果，并生成均衡或探索集合、矩阵、图谱和导出文件。"),
    ], key="alg_pipeline")

    b.heading(doc, "3.2.3 Query Contract 与意图路由", 3)
    b.body(doc, "Query Contract 将查询拆解为主题 T、方法 M、任务 A、数据/领域 D、时间范围 τ、发表范围 V、关系谓词 R 和排除条件 N。结构化表示见式（2）。字段允许为空，但主题与至少一类限定信息必须被保留；当 LLM 解析失败时，规则解析器根据实体词、年份模式、关系触发词和否定词生成确定性结果。")
    equation(doc, r"\mathcal{C}(q)=\{T,M,A,D,\tau,V,\mathcal{R},\mathcal{N}\}", 2, key="eq_contract")
    b.body(doc, "意图路由把查询划分为奠基工作、最新进展、方法比较、理论分析、应用检索和综述探索等类型。路由结果用于选择子查询模板、时间偏好和引文方向。例如，“theoretically analyze”提高关系谓词与全文证据权重；“follow-up work”提高前向引文和新近性权重；“foundational paper”则提高早期年份和被后续工作共同引用的证据。图 4 展示约束图与多视图证据之间的连接。")

    b.heading(doc, "3.2.4 多源混合召回与排序融合", 3)
    b.body(doc, "第一阶段并行执行 OpenAlex lexical、OpenAlex semantic 和 arXiv 检索，必要时补充 Crossref、Semantic Scholar、标题探测与第二轮检索。词项通道保留精确术语和实体，语义通道处理改写与表述差异，预印本通道补充最新论文。HyDE、GritLM 等研究说明生成式改写和统一表示可改善零样本检索，但也可能引入查询漂移[11-12]；因此系统保留原查询，并要求扩展词能够映射回 Query Contract 字段。")
    b.body(doc, "不同检索源的原始得分不可直接比较。系统首先采用带通道权重的 RRF 合并各有序列表[19]。若论文 p 在通道 m 中的排名为 r_m(p,q)，平滑常数为 k_0，通道权重为 w_m，则融合分数如式（3）。缺失通道不贡献分数，重复命中会通过多路径累积得到更高支持。")
    equation(doc, r"S_{\mathrm{RRF}}(p,q)=\sum_{m\in\mathcal{M}_p}\frac{w_m}{k_0+r_m(p,q)}", 3, key="eq_rrf")

    b.heading(doc, "3.2.5 候选身份合并与日期约束", 3)
    b.body(doc, "候选融合采用稳定标识优先级：arXiv ID → DOI → OpenAlex ID → Semantic Scholar ID → 标题 fallback。arXiv 标识会移除 v1/v2 等版本后缀；DOI 统一小写并去除 URL 前缀；标题仅用于补充，规范化过程包括 Unicode 归一、标点清理、空白压缩和常见副标题处理。合并后保留更完整的摘要、作者、年份、引用量与原文链接。")
    b.body(doc, "所有查询中的日期要求在候选进入排序前统一处理。系统优先使用正式发表日期，其次使用在线发表或预印本日期，并记录日期来源。对于带截止时间的评测查询，晚于截止日期的论文不会进入候选评测；对于产品模式，用户可选择严格过滤或将越界论文放入补充候选。")

    b.heading(doc, "3.2.6 TRACE-Rank 综合排序", 3)
    b.body(doc, "TRACE-Rank 负责形成稳定的全局排序。分数由 RRF 融合、语义相关性、Query Contract 覆盖、权威性、新近性与多样性共同构成。式（4）给出统一形式，其中 A(p) 表示权威性，N(p,q) 表示与查询时间意图一致的新近性，D_MMR 表示与已选结果的冗余惩罚[20]。各权重在开发集确定后保持一致，不针对单条查询临时调节。")
    equation(doc, r"\begin{aligned}S_{\mathrm{TRACE}}(p,q)={}&\lambda_1S_{\mathrm{RRF}}+\lambda_2S_{\mathrm{sem}}+\lambda_3C\\&+\lambda_4A+\lambda_5N-\lambda_6D_{\mathrm{MMR}}\end{aligned}", 4, key="eq_trace")
    b.body(doc, "排序过程保存每个分项和来源排名。前端论文卡片可以说明候选是由哪些通道发现、覆盖了哪些条件、是否获得引文支持，以及其多样性惩罚为何发生。该设计避免将最终名次解释为一个不可拆分的黑盒分数。")

    b.heading(doc, "3.2.7 SCGQ 与 QCF 受控引文前沿", 3)
    b.body(doc, "SCGQ 从 Top20 中选择高置信种子，批量获取其参考文献与被引论文；QCF 根据查询关系类型、种子质量、候选重复率和新增率压缩前沿。系统不把“被某篇论文引用”直接视为相关，而是保留 citation_direction、shared_seed_count、bridge_strength 和 relation_family 等字段，供后续 REG 评分。")
    b.body(doc, "前沿动作受到查询级预算控制：当新候选的约束覆盖持续降低、与已有候选高度重复或新增稳定身份数量接近零时，停止继续扩展；当某一必要条件长期缺失且仍存在可解释引文方向时，允许执行一次定向动作。该机制把多轮搜索从固定轮数改为受当前证据状态约束的过程。")

    b.heading(doc, "3.2.8 REG 条件合取与关系证据", 3)
    b.body(doc, "复杂查询中的必要条件不能只取平均值。若候选在主题、任务和数据上得分较高，但完全缺少“理论分析”证据，简单平均仍可能给出较高相关性。REG 因此同时考虑平均覆盖和最弱必要条件。设候选 p 对 m 个必要条件的覆盖度为 c_j(p,q)，条件合取分数定义为式（5）。")
    equation(doc, r"C(p,q)=\sqrt{\left(\frac{1}{m}\sum_{j=1}^{m}c_j(p,q)\right)\min_{1\le j\le m}c_j(p,q)}", 5, key="eq_conjunction")
    b.body(doc, "在条件合取之外，REG 还计算关系谓词 R、引文方向 D、种子桥接 B、图支持 G 与本地交叉编码 X。统一关系证据分数见式（6）。不同关系族使用预设系数组合，并设置绝对证据下限、相对证据间隔、Top20 头部保护和最大晋升数量。只有同时满足证据阈值和风险约束的候选才会改变头部排序。")
    equation(doc, r"S_{\mathrm{REG}}(p,q)=\alpha C+\beta R+\gamma D+\delta B+\epsilon G+\zeta X", 6, key="eq_reg")

    b.heading(doc, "3.2.9 MEQ 多视图一致性仲裁", 3)
    b.body(doc, "关系抽取可能因摘要缺失或表达方式变化而稀疏。MEQ 将 Cross、TRACE、Graph、Lexical 和 Fusion 五路信号转化为查询内百分位，避免原始量纲差异。与取最大值不同，系统使用第三强视图作为 3-of-5 仲裁分数，如式（7）所示；候选必须至少获得三路相对支持，才能绕过单一证据缺失。")
    equation(doc, r"S_{\mathrm{MEQ}}(p,q)=Q_{3/5}\!\left(\pi_X,\pi_{\mathrm{TRACE}},\pi_G,\pi_L,\pi_F\right)", 7, key="eq_meq")
    b.body(doc, "MEQ 不替代 REG，而是对关系证据稀疏的候选提供补充判断。若候选只有单一语义通道高分，第三强视图仍然较低；若候选在词项、图结构和交叉编码上均得到支持，即使某个关系触发词未被抽取，也能形成稳定的查询内一致性。")

    b.heading(doc, "3.2.10 EBP 证据加权引文反投影", 3)
    b.body(doc, "候选池中经常存在已解析论文指向、但尚未被物化的参考文献 ID。EBP 首先把候选 i 的多视图证据聚合为 e_i，再将其均摊到未解析参考文献集合 R_i。为降低长参考列表造成的无差别扩散，单条边的贡献按参考文献数量平方根归一化；目标论文 j 的累计投票见式（8）。")
    equation(doc, r"V(j\mid q)=\sum_{i:\,j\in\mathcal{R}_i}\frac{e_i}{\sqrt{|\mathcal{R}_i|}}", 8, key="eq_ebp")
    b.body(doc, "目标按累计投票、支持论文数与来源多样性排序，Top100 通过一次 OpenAlex 批量请求物化。EBP 结果默认进入补充候选层，并保留 supporting_papers、vote_score 和 citation_direction。该路径的作用是扩大可解释候选边界，而不是直接绕过 TRACE-Rank 与 REG-MEQ。")

    b.heading(doc, "3.2.11 证据校准与集合决策", 3)
    b.body(doc, "候选扩展与证据排序完成后，系统为每篇论文构造 40 维查询内证据向量 x(p,q)。向量同时编码 TRACE、局部语义、关系图、约束覆盖、引文来源、元数据和排序先验，并加入能够表达证据协同的交互项。训练阶段仅使用开发查询估计均值、尺度和线性权重；论文标题、ID 与数据划分标签均不作为特征。")
    b.body(doc, "式（9）给出证据校准分数。核心结果选择置信度最高的论文；均衡集合保留 8 篇，探索列表展示 Top20 与补充候选。三类模式共享候选对象和证据分项，因此结果粒度改变时无需再次访问外部数据源。")
    equation(doc, r"\begin{aligned}\mathbf{z}(p,q)&=\frac{\mathbf{x}(p,q)-\boldsymbol{\mu}}{\boldsymbol{\sigma}},\\[2pt]P_{\mathrm{ECS}}(p\mid q)&=\operatorname{sigmoid}\!\left(\mathbf{w}^{\mathsf{T}}\mathbf{z}(p,q)+b\right)\end{aligned}", 9, key="eq_selection")
    b.body(doc, "最终集合用于默认结果载荷与 CSV/BibTeX 导出，补充候选保留在界面供用户继续检查。每篇论文的推荐理由由 Query Contract 覆盖、来源通道、引文关系和排序分项生成，不使用无法追溯到候选元数据的自由扩写。")

    b.heading(doc, "3.3 计划和分工", 2)
    b.body(doc, "项目按赛题分析、算法设计、后端实现、前端展示、评测验证和材料交付六条工作线推进。算法与工程采用同一响应协议：前端、评测脚本和演示材料均读取统一字段，减少同一指标在不同材料中的表述偏差。团队成员在总体方案统一的基础上分别承担算法、数据链路和展示工作，并共同完成联调与现场复现。")
    table(doc, ["成员", "主要职责", "具体工作"], [
        ["贾梓旭", "算法设计与项目统筹", "Query Contract、TRACE-Rank、REG-MEQ、EBP、集合决策；实验与项目材料统筹"],
        ["贺凯华", "后端系统与数据链路", "FastAPI、多源客户端、论文身份与调用审计；运行环境和接口联调"],
        ["刘凯", "前端展示与演示交付", "检索过程、证据矩阵、关系图谱、论文卡片与导出；视频与现场展示"],
    ], [2.5, 4.2, 9.1], "表 6", "团队成员分工", key="tab_team", font_size=9.4)
    b.body(doc, "版本管理以可运行提交为最小单元。每次算法修改同步给出自动化测试、对比结果和变更记录；界面修改验证主查询、空结果与错误降级；材料修改从实验登记表引用数字。该流程保证系统、文档与视频围绕同一版本更新。")

    b.heading(doc, "3.4 系统实现", 2)
    b.heading(doc, "3.4.1 后端与数据流", 3)
    b.body(doc, "后端以 FastAPI 组织 Query Contract、检索规划、学术 API 客户端、身份合并、TRACE-Rank、SCGQ/QCF、REG-MEQ、EBP 和集合决策。多源请求采用异步并发，单一通道失败不会中断全部流程；每个通道记录 request_count、latency、new_candidates 和 gold_diagnostic 等审计字段。")
    b.body(doc, "候选对象经过四次边界变化：raw_candidates 保存各源原始结果；candidate_pool 保存统一身份后的高召回候选；ranked_candidates 保存 TRACE 与证据分数；selected_results 保存当前模式的最终集合。supplementary_candidates 独立承载未被直接选中的排序候选与 EBP 扩展结果。该数据结构与式（1）、式（9）的候选—集合分离保持一致。")

    b.heading(doc, "3.4.2 前端与结构化交互", 3)
    b.body(doc, "前端采用科研检索驾驶舱布局。顶部区域提供自然语言查询、检索深度、检索策略和结果模式；结果区依次呈现 Query Contract、运行指标、迭代检索过程、证据晋升动作、研究全景、证据矩阵、关系图谱与论文卡片。用户可展开单篇论文查看标题、摘要、作者、年份、来源、DOI/arXiv 链接、约束覆盖和推荐依据。")
    b.body(doc, "结构化展示不等同于把所有中间变量堆叠在页面。系统默认显示核心结论和论文集合，检索路径、分项得分与图谱按需展开；三类结果模式改变集合规模，但不隐藏补充候选。图 9 和图 10 分别展示证据矩阵与关系图谱。")
    figure(doc, V32_FIGS / "fig6_evidence_matrix.png", "图 9", "多论文—多约束证据覆盖矩阵", key="fig_matrix")
    figure(doc, V32_FIGS / "fig7_relation_graph.png", "图 10", "查询意图、候选论文与研究关系图谱", key="fig_graph")

    b.heading(doc, "3.4.3 功能与导出", 3)
    table(doc, ["功能组", "已实现功能", "输出/交互"], [
        ["查询认知", "Query Contract、意图路由、子查询与排除条件", "结构化字段、检索式复制"],
        ["自主检索", "五种检索深度、两类策略、多源并发与迭代动作", "通道状态、候选数、动作时间线"],
        ["证据排序", "TRACE-Rank、REG-MEQ、EBP 与分项解释", "晋升记录、证据矩阵、补充候选"],
        ["结果组织", "证据校准核心结果、均衡集合、探索列表", "论文卡片、研究全景、关系图谱"],
        ["数据导出", "BibTeX、CSV、JSON、打印报告", "默认导出最终集合，可选择完整候选"],
        ["运行保障", "健康检查、请求超时、单源故障降级", "状态提示、备用通道与调用记录"],
    ], [3.0, 7.7, 5.1], "表 7", "系统功能与结构化输出", key="tab_functions", font_size=9.2)

    b.heading(doc, "3.4.4 成本控制与故障降级", 3)
    b.body(doc, "系统通过异步并发、请求合并、批量元数据获取和查询级动作预算控制外部调用。词项、语义、引文和第二轮检索分别记录请求次数、耗时、候选增量与重复率；某一通道超时或返回异常时，路由器保留其他来源结果并继续完成排序与展示。")
    b.body(doc, "成本控制以“新增相关论文/新增动作”为基本观察单位。每次第二轮检索、引文扩展或 EBP 批量物化均记录新增稳定身份数、重复率、耗时和调用次数。若动作只产生重复候选或证据增量不足，路由器降低其后续优先级。EBP 采用一次批量物化而非逐 ID 请求，是在候选覆盖与网络成本之间的工程折中。")

    b.heading(doc, "3.5 实验与评测", 2)
    b.heading(doc, "3.5.1 评测协议", 3)
    b.body(doc, "实验按照候选发现、Top20 排序、核心结果选择和运行成本四个层次组织。Calibration40 与 Dev20 用于算法开发，Blind50 作为保留评测集，包含 50 条复杂查询和 123 篇目标论文。所有比较使用一致的查询、目标论文、日期约束与论文身份规则；身份匹配依次采用 arXiv ID、DOI、OpenAlex/Semantic Scholar ID 和规范化标题。")
    b.body(doc, "LitSearch、BEIR、BRIGHT 与 AstaBench 分别强调真实科学检索、跨域零样本检索、推理密集检索和科研 Agent 的受控评测[3-4,13-14]。本项目据此把“是否找到”“是否排入头部”和“是否进入最终交付集合”分别度量，避免用单一结果掩盖不同环节的作用。图 11 展示三层诊断指标及其数据边界。")
    figure(doc, V32_FIGS / "fig5_evaluation_protocol.png", "图 11", "统一评测规则与候选—排序—集合三层诊断指标", key="fig_eval_protocol")
    b.body(doc, "对单条查询 q，设预测集合为 Y_q、相关论文集合为 G_q，则 Precision、Recall 与 F1 按式（10）计算；最终结果对查询取宏平均。候选 Recall 衡量检索覆盖，Recall@20 与 NDCG@20 衡量头部排序，集合 F1 衡量直接交付的有效密度。")
    equation(doc, r"\begin{aligned}P_q&=\frac{|\mathcal{Y}_q\cap\mathcal{G}_q|}{|\mathcal{Y}_q|},&R_q&=\frac{|\mathcal{Y}_q\cap\mathcal{G}_q|}{|\mathcal{G}_q|},\\[2pt]F1_q&=\frac{2P_qR_q}{P_q+R_q}.\end{aligned}", 10, key="eq_metrics")

    b.heading(doc, "3.5.2 EBP 候选召回扩展", 3)
    b.body(doc, "EBP 在三个数据划分上均扩大了候选召回边界。Calibration40 的候选命中从 35 增至 44，Recall 从 0.4226 增至 0.5185；Dev20 从 26 增至 28，Recall 从 0.5346 增至 0.5721；Blind50 从 50 增至 53，Recall 从 0.5323 增至 0.5623。图 8 展示跨划分变化和预算响应，表 8 给出完整数值。")
    grouped_metric_table(
        doc,
        stub_header="数据划分",
        groups=[("原候选池", ["TP", "Recall"]), ("加入 EBP", ["TP", "Recall"])],
        rows=[
            ["Calibration40", "35", "0.4226", "44", "0.5185", "+9"],
            ["Dev20", "26", "0.5346", "28", "0.5721", "+2"],
            ["Blind50", "50", "0.5323", "53", "0.5623", "+3"],
        ],
        widths_cm=[3.0, 1.8, 2.45, 1.8, 2.45, 1.55],
        prefix="表 8",
        title="EBP 候选召回结果。TP 表示候选池中的目标论文数；新增命中由批量引文目标物化获得。",
        key="tab_ebp",
        font_size=8.8,
        tail_header="ΔTP",
    )
    b.body(doc, "Blind50 新恢复的目标包括 CLIP、GPT-3 和最大熵探索相关代表工作。它们由多个已有候选共同引用，体现了式（8）对跨候选引文证据的聚合作用，并将一部分词项和语义通道未发现的论文重新带回候选空间。")

    b.heading(doc, "3.5.3 REG-MEQ 头部排序", 3)
    b.body(doc, "在相同候选基础上，REG-MEQ 在 Calibration40、Dev20 与 Blind50 上均增加了 Top20 命中。三个划分的 TP 分别由 23、22、36 提高到 29、25、38；Blind50 的 Recall@20 从 0.4070 提高到 0.4370。结果表明，必要条件合取、关系方向和多视图一致性能够把部分已在候选池中的目标论文推进头部。")
    grouped_metric_table(
        doc,
        stub_header="数据划分",
        groups=[("基础 Top20", ["TP", "Recall", "F1"]), ("REG-MEQ", ["TP", "Recall", "F1"])],
        rows=[
            ["Calibration40", "23", "0.3051", "0.0511", "29", "0.3753", "0.0643", "+6"],
            ["Dev20", "22", "0.4346", "0.0933", "25", "0.5180", "0.1069", "+3"],
            ["Blind50", "36", "0.4070", "0.0642", "38", "0.4370", "0.0680", "+2"],
        ],
        widths_cm=[2.75, 1.25, 1.75, 1.65, 1.25, 1.75, 1.65, 1.35],
        prefix="表 9",
        title="REG-MEQ 跨数据划分 Top20 结果。TP 表示命中论文总数，指标沿用同一评测协议。",
        key="tab_reg",
        font_size=8.4,
        tail_header="ΔTP",
    )

    b.heading(doc, "3.5.4 证据校准核心结果", 3)
    b.body(doc, "为区分证据建模与结果篇数的影响，核心结果实验固定每条查询输出 1 篇论文，并保持候选池、日期规则和身份匹配不变。Blind50 上，TRACE 排名的 Macro Precision、Recall 和 F1 分别为 0.3000、0.2067 和 0.2333；加入证据校准后分别达到 0.3200、0.2267 和 0.2533，命中论文由 15 篇增至 16 篇。")
    grouped_metric_table(
        doc,
        stub_header="核心结果方法",
        groups=[("结果规模", ["平均篇数", "TP"]), ("Blind50（macro）", ["Precision", "Recall", "F1"])],
        rows=[
            ["TRACE 排名", "1.00", "15", "0.3000", "0.2067", "0.2333"],
            ["证据校准", "1.00", "16", "0.3200", "0.2267", "0.2533"],
        ],
        widths_cm=[3.2, 2.0, 1.4, 3.0, 3.0, 3.0],
        prefix="表 10",
        title="Blind50 核心结果比较。两种方法均固定返回 1 篇，差异仅来自论文级证据排序。",
        key="tab_selector",
        font_size=8.8,
    )
    b.body(doc, "证据校准在 1 条查询上新增命中，其余 49 条保持一致；运行阶段只进行标准化、40 维点积与 Sigmoid 计算。核心结果用于快速给出最有把握的论文，均衡集合和探索列表继续保留更宽的研究视野。")

    b.heading(doc, "3.5.5 联合证据链与运行成本", 3)
    b.body(doc, "三组实验对应检索链路中的不同位置：EBP 扩大候选边界，REG-MEQ 改善 Top20 排序，证据校准提高核心结果的有效密度。Blind50 上，候选 Recall 从 0.5323 提高到 0.5623，Top20 Recall 从 0.4070 提高到 0.4370，核心结果 F1 从 0.2333 提高到 0.2533。图 12 将三项结果放在同一证据链中。")
    figure(doc, V33_FIGS / "fig12_final_evidence_chain.png", "图 12", "候选发现、关系排序与核心结果选择的分层实验结果", key="fig_final_chain")
    b.body(doc, "EBP 目标预算从 10 增至 75 时，Blind50 候选命中由 51 逐步增至 53；预算达到 75 后进入平台区。最终使用 Top100 目标预算，以保留 Calibration40 上的最后一篇新增目标论文。每条查询增加 1 次 OpenAlex 批量请求，平均检索耗时 2.43 s；REG-MEQ 与证据校准均在本地证据对象上计算，不增加外部 API、LLM 调用或 Token。")
    table(doc, ["模块", "新增 API/查询", "平均耗时", "新增 LLM/Token", "主要作用"], [
        ["EBP 批量物化", "1", "2.43 s", "0 / 0", "恢复一跳可达论文"],
        ["本地证据评分", "0", "0.61 s", "0 / 0", "关系与多视图特征"],
        ["证据校准选择", "0", "并入本地评分", "0 / 0", "形成核心结果"],
    ], [3.3, 2.5, 2.7, 3.1, 4.2], "表 11", "新增算法模块的运行成本", key="tab_cost", font_size=9.2)

    b.heading(doc, "3.5.6 结构化证据消融", 3)
    b.body(doc, "在相同候选扩展上，通用局部语义重排在 Dev20 得到 20 篇 Top20 命中，REG 得到 25 篇；Calibration40 中基础排序、通用图重排、REG 与原子条件路由分别得到 23、25、28 和 24 篇命中。图 13 将校准集与开发集结果并列，表 12 概括各组件承担的功能。")
    figure(doc, V32_FIGS / "fig10_ablation.png", "图 13", "结构化关系证据消融与跨划分头部命中变化", key="fig_ablation")
    table(doc, ["组件", "解决的问题", "可观测作用"], [
        ["Query Contract", "复杂查询字段在改写中丢失", "约束覆盖可分项检查"],
        ["TRACE-Rank", "多源得分量纲和候选冗余", "形成稳定全局排序"],
        ["REG", "必要条件与关系证据未显式建模", "在 Top20 中恢复关系型目标"],
        ["MEQ", "单一关系特征稀疏或偶然高分", "以 3-of-5 视图形成一致性"],
        ["EBP", "一跳引文 ID 尚未进入候选池", "扩大候选召回边界"],
        ["证据校准", "单一排序分数难以表达证据协同", "同规模核心结果 F1 提高"],
        ["集合决策", "单一结果规模难以覆盖不同科研阶段", "提供核心、均衡与探索视图"],
    ], [3.0, 7.2, 5.6], "表 12", "关键组件与对应证据环节", key="tab_ablation", font_size=9.2)

    b.heading(doc, "3.5.7 查询案例", 3)
    b.body(doc, "以查询“Which papers theoretically analyze MAML convergence for few-shot learning?”为例，Query Contract 识别 MAML、理论分析、收敛性和小样本学习等约束。系统生成多条子查询并获得 191 篇原始候选，统一身份后保留 118 篇。TRACE-Rank 完成全局排序，关系证据路径从引文前沿中识别《Theoretical Convergence of Multi-Step Model-Agnostic Meta-Learning》，REG-MEQ 根据理论关系、引文方向和多视图一致性执行受控晋升。")
    b.body(doc, "核心结果优先给出证据最完整的论文，均衡集合保留 8 篇关键工作；论文卡片、补充候选和关系图谱共同呈现约束证据、引文来源与研究脉络，验证查询分解、关系排序和结果组织之间的联动。")

    b.heading(doc, "3.5.8 系统测试与现场复现", 3)
    b.body(doc, "系统自动化测试共 168 项，覆盖 Query Contract、数据源客户端、论文身份与版本处理、日期约束、评测指标、证据校准、REG-MEQ、EBP、API 响应和导出接口，当前环境全部通过。交付包同时提供环境检查、一键启动、配置模板、示例查询和评测脚本，可依次验证后端健康状态、前端页面、候选检索、排序与最终集合。")

    b.heading(doc, "3.6 风险与对策", 2)
    table(doc, ["风险", "影响", "已实现对策"], [
        ["外部 API 限流或不可用", "候选减少或延迟升高", "多源并发、超时降级与备用数据源"],
        ["大模型调用失败", "查询解析或归纳不可用", "规则式 Query Contract、确定性子查询与模板化归纳"],
        ["候选扩展引入噪声", "Precision 与头部排序下降", "REG-MEQ 门控、头部保护、最大晋升数与补充层隔离"],
        ["论文身份不一致", "重复计数或命中误判", "稳定 ID 优先、arXiv 去版本号与标题 fallback"],
        ["日期与元数据缺失", "时间约束误判", "日期来源优先级、缺失标记与补充数据源"],
        ["现场环境差异", "部署或运行异常", "一键启动、健康检查与环境说明"],
    ], [4.3, 4.2, 7.3], "表 13", "主要风险与应对措施", key="tab_risk", font_size=9.1)

    b.heading(doc, "3.7 总结与后续完善", 2)
    b.body(doc, "研问智搜已经形成从复杂查询认知、多源发现、论文身份统一、关系排序、引文缺口恢复、证据校准到结构化交付的端到端系统。Blind50 实验中，EBP 将候选 Recall 从 0.5323 提高到 0.5623，REG-MEQ 将 Top20 Recall 从 0.4070 提高到 0.4370，证据校准将同规模核心结果 F1 从 0.2333 提高到 0.2533，三项结果分别对应候选、排序和交付阶段。")
    b.body(doc, "系统以一次批量引文物化扩展候选边界，关系排序与证据校准均在本地完成，并通过核心、均衡与探索三类视图连接自动评价和实际阅读。后续将继续扩大复杂关系查询覆盖，增强全文证据与研究脉络组织，并优化首次检索时延和本地化部署，使检索结果更顺畅地进入开题、综述和技术路线比较等科研工作流。")

    ref_heading = b.heading(doc, "4 参考资料", 1)
    # Keep the reference list attached to the preceding conclusion when space
    # permits; competition reports should not leave a mostly blank page merely
    # to force a chapter opening.
    ref_heading.paragraph_format.page_break_before = False
    refs = [
        "[1] HE Y, HUANG G, FENG P, et al. PaSa: An LLM Agent for Comprehensive Academic Paper Search[C]. Proceedings of ACL 2025, 2025. arXiv:2501.10120.",
        "[2] SHI X, LI Y, KOU Q, et al. SPAR: Scholar Paper Retrieval with LLM-based Agents for Enhanced Academic Search[EB/OL]. arXiv:2507.15245, 2025.",
        "[3] AJITH A, XIA M, CHEVALIER A, et al. LitSearch: A Retrieval Benchmark for Scientific Literature Search[C]. Proceedings of EMNLP 2024, 2024. arXiv:2407.18940.",
        "[4] BRAGG J, D'ARCY M, BALEPUR N, et al. AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite[EB/OL]. arXiv:2510.21652, 2025.",
        "[5] SKARLINSKI M D, COX S, LAURENT J M, et al. Language Agents Achieve Superhuman Synthesis of Scientific Knowledge[EB/OL]. arXiv:2409.13740, 2024.",
        "[6] LÁLA J, O'DONOGHUE O, SHTEDRITSKI A, et al. PaperQA: Retrieval-Augmented Generative Agent for Scientific Research[EB/OL]. arXiv:2312.07559, 2023.",
        "[7] FENG P, HE Y, HUANG G, et al. AGILE: A Novel Reinforcement Learning Framework of LLM Agents[C]//Advances in Neural Information Processing Systems. 2024, 37.",
        "[8] KHATTAB O, SANTHANAM K, LI X L, et al. Demonstrate-Search-Predict: Composing Retrieval and Language Models for Knowledge-Intensive NLP[EB/OL]. arXiv:2212.14024, 2022.",
        "[9] LEWIS P, PEREZ E, PIKTUS A, et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks[C]//Advances in Neural Information Processing Systems. 2020, 33: 9459-9474.",
        "[10] YAO S, ZHAO J, YU D, et al. ReAct: Synergizing Reasoning and Acting in Language Models[C]. International Conference on Learning Representations, 2023.",
        "[11] MUENNIGHOFF N, SU H, WANG L, et al. Generative Representational Instruction Tuning[C]. International Conference on Learning Representations, 2024.",
        "[12] GAO L, MA X, LIN J, et al. Precise Zero-Shot Dense Retrieval without Relevance Labels[C]//Proceedings of ACL 2023. 2023: 1762-1777. DOI:10.18653/v1/2023.acl-long.99.",
        "[13] SU H, YEN H, XIA M, et al. BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval[EB/OL]. arXiv:2407.12883, 2024.",
        "[14] THAKUR N, REIMERS N, RÜCKLÉ A, et al. BEIR: A Heterogeneous Benchmark for Zero-Shot Evaluation of Information Retrieval Models[C]//NeurIPS Datasets and Benchmarks. 2021.",
        "[15] COHAN A, FELDMAN S, BELTAGY I, et al. SPECTER: Document-level Representation Learning using Citation-informed Transformers[C]//Proceedings of ACL 2020. 2020: 2270-2282. DOI:10.18653/v1/2020.acl-main.207.",
        "[16] BELTAGY I, LO K, COHAN A. SciBERT: A Pretrained Language Model for Scientific Text[C]//Proceedings of EMNLP-IJCNLP 2019. 2019: 3615-3620. DOI:10.18653/v1/D19-1371.",
        "[17] REIMERS N, GUREVYCH I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks[C]//Proceedings of EMNLP-IJCNLP 2019. 2019: 3982-3992. DOI:10.18653/v1/D19-1410.",
        "[18] KHATTAB O, ZAHARIA M. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT[C]//Proceedings of SIGIR 2020. 2020: 39-48.",
        "[19] CORMACK G V, CLARKE C L A, BÜTTCHER S. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods[C]//Proceedings of SIGIR 2009. 2009: 758-759. DOI:10.1145/1571941.1572114.",
        "[20] CARBONELL J, GOLDSTEIN J. The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries[C]//Proceedings of SIGIR 1998. 1998: 335-336.",
        "[21] PRIEM J, PIWOWAR H, ORR R. OpenAlex: A Fully-Open Index of Scholarly Works, Authors, Venues, Institutions, and Concepts[EB/OL]. arXiv:2205.01833, 2022.",
        "[22] AMMAR W, GROENEVELD D, BHAGAVATULA C, et al. Construction of the Literature Graph in Semantic Scholar[C]//Proceedings of NAACL-HLT 2018. 2018: 84-91. DOI:10.18653/v1/N18-3011.",
        "[23] HENDRICKS G, TKACZYK D, LIN J, et al. Crossref: The Sustainable Source of Community-Owned Scholarly Metadata[J]. Quantitative Science Studies, 2020, 1(1): 414-427. DOI:10.1162/qss_a_00022.",
        "[24] ARXIV. arXiv API User's Manual[EB/OL]. https://info.arxiv.org/help/api/.",
        "[25] OPENALEX. OpenAlex Developers Documentation[EB/OL]. https://docs.openalex.org/.",
        "[26] SEMANTIC SCHOLAR. Academic Graph API Documentation[EB/OL]. https://api.semanticscholar.org/.",
    ]
    for ref in refs:
        add_reference(doc, ref)

    settings = doc.settings._element
    zoom = settings.find(qn("w:zoom"))
    if zoom is not None and zoom.get(qn("w:percent")) is None:
        zoom.set(qn("w:percent"), "100")

    out_path = OUT / "虫虫特工队_研问智搜_项目文档_初赛提交版V3.3.docx"
    doc.save(out_path)
    return out_path


if __name__ == "__main__":
    print(build())
