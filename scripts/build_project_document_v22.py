from __future__ import annotations

from pathlib import Path
from typing import Iterable

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "artifacts" / "submission_2026"
FIGS = ASSETS / "figures_nature"
SHOTS = ASSETS / "screenshots"
OUT = ASSETS / "draft"

# Typography follows the official Tsinghua University graduate-thesis guide
# (body: Song 12 pt / Times New Roman 12 pt, exact 20 pt leading).
# The competition cover and chapter order remain unchanged.
INK = "000000"
LINE = "000000"
PAPER = "FFFFFF"
BODY_CJK = "宋体"
BODY_LATIN = "Times New Roman"
# Use one modern screen/print typeface for the whole heading run.  This avoids
# the visually abrupt SimHei + Arial mixture seen in short numbered headings.
HEAD_CJK = "Microsoft YaHei"
HEAD_LATIN = "Microsoft YaHei"
COVER_META_CJK = "仿宋"
CONTENT_WIDTH_CM = 15.0


def set_cell_shading(cell, fill: str = PAPER) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=90, bottom=80, end=90) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_cell_width(cell, width_cm: float) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(Cm(width_cm).twips))
    tc_w.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def set_table_three_line_borders(table) -> None:
    """Apply the 1.5 pt / 1 pt / 1.5 pt three-line-table convention."""
    tbl_pr = table._tbl.tblPr
    old = tbl_pr.find(qn("w:tblBorders"))
    if old is not None:
        tbl_pr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for edge, val, size in (
        ("top", "single", "12"),
        ("left", "nil", "0"),
        ("bottom", "single", "12"),
        ("right", "nil", "0"),
        ("insideH", "nil", "0"),
        ("insideV", "nil", "0"),
    ):
        node = OxmlElement(f"w:{edge}")
        node.set(qn("w:val"), val)
        node.set(qn("w:sz"), size)
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), INK)
        borders.append(node)
    tbl_pr.append(borders)


def set_header_rule(cell) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "8")
    bottom.set(qn("w:space"), "0")
    bottom.set(qn("w:color"), INK)
    borders.append(bottom)


def add_field(paragraph, instruction: str, placeholder: str = ""):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = placeholder
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, text, end])
    return run


def set_page_number_format(section, fmt: str, start: int | None = None) -> None:
    sect_pr = section._sectPr
    pg_num = sect_pr.find(qn("w:pgNumType"))
    if pg_num is None:
        pg_num = OxmlElement("w:pgNumType")
        sect_pr.append(pg_num)
    pg_num.set(qn("w:fmt"), fmt)
    if start is not None:
        pg_num.set(qn("w:start"), str(start))
    elif pg_num.get(qn("w:start")) is not None:
        del pg_num.attrib[qn("w:start")]


def set_paragraph_bottom_border(paragraph, color: str = LINE, size: int = 6) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    borders = p_pr.find(qn("w:pBdr"))
    if borders is None:
        borders = OxmlElement("w:pBdr")
        p_pr.insert_element_before(
            borders,
            "w:shd", "w:tabs", "w:suppressAutoHyphens", "w:kinsoku",
            "w:wordWrap", "w:overflowPunct", "w:topLinePunct",
            "w:autoSpaceDE", "w:autoSpaceDN", "w:bidi",
            "w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind",
            "w:contextualSpacing", "w:mirrorIndents", "w:suppressOverlap",
            "w:jc", "w:textDirection", "w:textAlignment",
            "w:textboxTightWrap", "w:outlineLvl", "w:divId", "w:cnfStyle",
            "w:rPr", "w:sectPr", "w:pPrChange",
        )
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(size))
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), color)
    borders.append(bottom)


def configure_section(section) -> None:
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(3.0)
    section.bottom_margin = Cm(3.0)
    section.left_margin = Cm(3.0)
    section.right_margin = Cm(3.0)
    section.header_distance = Cm(2.2)
    section.footer_distance = Cm(2.2)


def add_front_footer(section, roman: bool = True) -> None:
    section.footer.is_linked_to_previous = False
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = add_field(p, "PAGE", "I" if roman else "1")
    set_run_font(run, BODY_CJK, BODY_LATIN, 10.5)


def add_body_header_footer(section) -> None:
    section.header.is_linked_to_previous = False
    section.footer.is_linked_to_previous = False
    hp = section.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    hp.paragraph_format.space_before = Pt(0)
    hp.paragraph_format.space_after = Pt(0)
    hr = hp.add_run("研问智搜 · 项目文档")
    set_run_font(hr, BODY_CJK, BODY_LATIN, 10.5)
    set_paragraph_bottom_border(hp, INK, 6)
    fp = section.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp.paragraph_format.space_before = Pt(0)
    fp.paragraph_format.space_after = Pt(0)
    field = add_field(fp, "PAGE", "1")
    set_run_font(field, BODY_CJK, BODY_LATIN, 10.5)


def set_run_font(run, cjk: str, latin: str, size: float, *, bold=False, color=INK):
    run.bold = bold
    run.font.name = latin
    run._element.get_or_add_rPr()
    run._element.rPr.rFonts.set(qn("w:eastAsia"), cjk)
    run._element.rPr.rFonts.set(qn("w:ascii"), latin)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), latin)
    run._element.rPr.rFonts.set(qn("w:cs"), latin)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    return run


def add_run_text(
    paragraph,
    text: str,
    *,
    bold=False,
    size=12,
    color=INK,
    font=BODY_CJK,
    latin_font: str | None = None,
):
    run = paragraph.add_run(text)
    aliases = {
        "SimSun": (BODY_CJK, BODY_LATIN),
        "宋体": (BODY_CJK, BODY_LATIN),
        "SimHei": (HEAD_CJK, HEAD_LATIN),
        "黑体": (HEAD_CJK, HEAD_LATIN),
        "FangSong": (COVER_META_CJK, BODY_LATIN),
        "仿宋": (COVER_META_CJK, BODY_LATIN),
        "Cambria Math": ("Cambria Math", "Cambria Math"),
    }
    cjk, default_latin = aliases.get(font, (font, BODY_LATIN))
    return set_run_font(run, cjk, latin_font or default_latin, size, bold=bold, color=color)


def body(doc: Document, text: str, *, first=True, bold_prefix: str | None = None):
    p = doc.add_paragraph(style="BodyTextCN")
    if first:
        p.paragraph_format.first_line_indent = Pt(24)
    if bold_prefix and text.startswith(bold_prefix):
        add_run_text(p, bold_prefix, bold=True)
        add_run_text(p, text[len(bold_prefix):])
    else:
        add_run_text(p, text)
    return p


def bullet(doc: Document, text: str):
    p = doc.add_paragraph(style="BulletCN")
    add_run_text(p, text)
    return p


def heading(doc: Document, text: str, level: int):
    p = doc.add_paragraph(text, style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    if level == 1 and any(x.style.name == "Heading 1" for x in doc.paragraphs[:-1]):
        p.paragraph_format.page_break_before = True
    return p


def caption(doc: Document, text: str, *, kind: str = "figure"):
    style = "FigureCaptionCN" if kind == "figure" else "TableCaptionCN"
    p = doc.add_paragraph(style=style)
    add_run_text(p, text, size=11)
    if kind == "table":
        p.paragraph_format.keep_with_next = True
    return p


def figure(doc: Document, path: Path, text: str, width_in: float = 5.75):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width_in))
    caption(doc, text, kind="figure")
    return p


def table(doc: Document, headers: list[str], rows: list[list[str]], widths_cm: list[float], caption_text: str | None = None):
    if caption_text:
        caption(doc, caption_text, kind="table")
    total = sum(widths_cm)
    if total > CONTENT_WIDTH_CM:
        scale = CONTENT_WIDTH_CM / total
        widths_cm = [w * scale for w in widths_cm]
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    set_table_three_line_borders(t)
    hdr = t.rows[0]
    set_repeat_table_header(hdr)
    prevent_row_split(hdr)
    for j, (cell, text, width) in enumerate(zip(hdr.cells, headers, widths_cm)):
        set_cell_width(cell, width)
        set_cell_shading(cell)
        set_cell_margins(cell, top=60, start=70, bottom=60, end=70)
        set_header_rule(cell)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        add_run_text(p, text, bold=True, size=11)
    for i, row in enumerate(rows):
        new_row = t.add_row()
        prevent_row_split(new_row)
        cells = new_row.cells
        for j, (cell, text, width) in enumerate(zip(cells, row, widths_cm)):
            set_cell_width(cell, width)
            set_cell_margins(cell, top=60, start=70, bottom=60, end=70)
            set_cell_shading(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j in (0, len(headers) - 1) else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            add_run_text(p, text, size=11)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.space_after = Pt(0)
    spacer.paragraph_format.line_spacing = Pt(1)
    return t


def add_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = BODY_LATIN
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_CJK)
    normal._element.rPr.rFonts.set(qn("w:ascii"), BODY_LATIN)
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), BODY_LATIN)
    normal.font.size = Pt(12)
    normal.font.color.rgb = RGBColor.from_string(INK)

    styles = doc.styles
    for level, size, before, after in ((1, 16, 24, 18), (2, 14, 24, 6), (3, 13, 12, 6)):
        s = styles[f"Heading {level}"]
        s.font.name = HEAD_LATIN
        s._element.rPr.rFonts.set(qn("w:eastAsia"), HEAD_CJK)
        s._element.rPr.rFonts.set(qn("w:ascii"), HEAD_LATIN)
        s._element.rPr.rFonts.set(qn("w:hAnsi"), HEAD_LATIN)
        s.font.size = Pt(size)
        # Chapter titles retain a strong weight; section titles use the regular
        # face so that 3.1/3.1.1 headings remain crisp rather than over-black.
        s.font.bold = level == 1
        s.font.color.rgb = RGBColor.from_string(INK)
        s.paragraph_format.space_before = Pt(before)
        s.paragraph_format.space_after = Pt(after)
        s.paragraph_format.keep_with_next = True
        s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER if level == 1 else WD_ALIGN_PARAGRAPH.LEFT
        s.paragraph_format.line_spacing = 1.0 if level == 1 else Pt(20)
        s.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE if level == 1 else WD_LINE_SPACING.EXACTLY
        s.paragraph_format.widow_control = True

    body_style = styles.add_style("BodyTextCN", WD_STYLE_TYPE.PARAGRAPH)
    body_style.font.name = BODY_LATIN
    body_style._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_CJK)
    body_style._element.rPr.rFonts.set(qn("w:ascii"), BODY_LATIN)
    body_style._element.rPr.rFonts.set(qn("w:hAnsi"), BODY_LATIN)
    body_style.font.size = Pt(12)
    body_style.font.color.rgb = RGBColor.from_string(INK)
    body_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    body_style.paragraph_format.line_spacing = Pt(20)
    body_style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    body_style.paragraph_format.space_before = Pt(0)
    body_style.paragraph_format.space_after = Pt(0)
    body_style.paragraph_format.widow_control = True

    bullet_style = styles.add_style("BulletCN", WD_STYLE_TYPE.PARAGRAPH)
    bullet_style.font.name = BODY_LATIN
    bullet_style._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_CJK)
    bullet_style._element.rPr.rFonts.set(qn("w:ascii"), BODY_LATIN)
    bullet_style._element.rPr.rFonts.set(qn("w:hAnsi"), BODY_LATIN)
    bullet_style.font.size = Pt(12)
    bullet_style.font.color.rgb = RGBColor.from_string(INK)
    bullet_style.paragraph_format.left_indent = Pt(24)
    bullet_style.paragraph_format.first_line_indent = Pt(-24)
    bullet_style.paragraph_format.line_spacing = Pt(20)
    bullet_style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    bullet_style.paragraph_format.space_before = Pt(0)
    bullet_style.paragraph_format.space_after = Pt(0)

    for style_name, before, after in (("FigureCaptionCN", 6, 12), ("TableCaptionCN", 12, 6)):
        cap = styles.add_style(style_name, WD_STYLE_TYPE.PARAGRAPH)
        cap.font.name = BODY_LATIN
        cap._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_CJK)
        cap._element.rPr.rFonts.set(qn("w:ascii"), BODY_LATIN)
        cap._element.rPr.rFonts.set(qn("w:hAnsi"), BODY_LATIN)
        cap.font.size = Pt(11)
        cap.font.color.rgb = RGBColor.from_string(INK)
        cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.line_spacing = 1.0
        cap.paragraph_format.space_before = Pt(before)
        cap.paragraph_format.space_after = Pt(after)
        cap.paragraph_format.widow_control = True

    for style_name, indent, bold in (("TOC 1", 0.0, True), ("TOC 2", 0.42, False), ("TOC 3", 0.84, False)):
        if style_name in styles:
            s = styles[style_name]
        else:
            s = styles.add_style(style_name, WD_STYLE_TYPE.PARAGRAPH)
        s.font.name = HEAD_LATIN if bold else BODY_LATIN
        s._element.rPr.rFonts.set(qn("w:eastAsia"), HEAD_CJK if bold else BODY_CJK)
        s._element.rPr.rFonts.set(qn("w:ascii"), HEAD_LATIN if bold else BODY_LATIN)
        s._element.rPr.rFonts.set(qn("w:hAnsi"), HEAD_LATIN if bold else BODY_LATIN)
        s.font.size = Pt(12)
        s.font.bold = bold
        s.font.color.rgb = RGBColor.from_string(INK)
        s.paragraph_format.left_indent = Cm(indent)
        s.paragraph_format.space_before = Pt(0)
        s.paragraph_format.space_after = Pt(0)
        s.paragraph_format.line_spacing = Pt(20)
        s.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY

    if "Hyperlink" in styles:
        hyperlink = styles["Hyperlink"]
        hyperlink.font.color.rgb = RGBColor.from_string(INK)
        hyperlink.font.underline = False


def build() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = Document()
    add_styles(doc)
    cover = doc.sections[0]
    configure_section(cover)
    set_page_number_format(cover, "upperRoman", 1)
    cover.header.is_linked_to_previous = False
    cover.header.paragraphs[0].clear()
    add_front_footer(cover)

    # Cover: same hierarchy and spacing as the school-round document, with official competition information.
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Cm(2.1)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_run_text(p, "第八届中国研究生人工智能创新大赛", bold=True, size=18, font=HEAD_CJK)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Cm(2.4)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_run_text(p, "研问智搜", bold=True, size=28, font=HEAD_CJK)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_run_text(p, "科研场景下复杂学术查询的", bold=True, size=15, font=HEAD_CJK)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_run_text(p, "智能论文搜索与推荐系统", bold=True, size=15, font=HEAD_CJK)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Cm(1.5)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_run_text(p, "项目文档", bold=True, size=18, font=HEAD_CJK)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_run_text(p, "版本 V2.2", size=12, font=COVER_META_CJK)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Cm(2.2)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_run_text(p, "日    期：2026.08.31", size=12, font=COVER_META_CJK)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_run_text(p, "团队名称：虫虫特工队", size=12, font=COVER_META_CJK)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_run_text(p, "参赛组别：企业赛题", size=12, font=COVER_META_CJK)

    # TOC page.
    toc_sec = doc.add_section(WD_SECTION.NEW_PAGE)
    configure_section(toc_sec)
    set_page_number_format(toc_sec, "upperRoman", 2)
    add_body_header_footer(toc_sec)
    fp = toc_sec.footer.paragraphs[0]
    fp.clear()
    field = add_field(fp, "PAGE", "II")
    set_run_font(field, BODY_CJK, BODY_LATIN, 10.5)
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(24)
    p.paragraph_format.space_after = Pt(18)
    p.paragraph_format.line_spacing = 1.0
    add_run_text(p, "目录", bold=True, size=16, font=HEAD_CJK)
    toc_p = doc.add_paragraph()
    add_field(toc_p, 'TOC \\o "1-3" \\h \\z \\u', "目录将在打开文档时自动更新")

    # Change history, present in the official template.
    hist_sec = doc.add_section(WD_SECTION.NEW_PAGE)
    configure_section(hist_sec)
    set_page_number_format(hist_sec, "upperRoman")
    add_body_header_footer(hist_sec)
    hist_sec.footer.paragraphs[0].clear()
    field = add_field(hist_sec.footer.paragraphs[0], "PAGE", "III")
    set_run_font(field, BODY_CJK, BODY_LATIN, 10.5)
    hist_sec.footer.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(24)
    p.paragraph_format.space_after = Pt(18)
    p.paragraph_format.line_spacing = 1.0
    add_run_text(p, "记录更改历史", bold=True, size=16, font=HEAD_CJK)
    table(
        doc,
        ["序号", "更改原因", "版本", "作者", "更改日期", "备注"],
        [
            ["1", "校赛版本", "V1.0", "虫虫特工队", "2026.06.26", "完成原型系统与校赛材料"],
            ["2", "初赛重构", "V2.0", "虫虫特工队", "2026.08.31", "更新算法、实验、界面与文档"],
            ["3", "格式统一", "V2.1", "虫虫特工队", "2026.08.31", "统一全文字体、间距、图表与页面格式"],
            ["4", "标题优化", "V2.2", "虫虫特工队", "2026.08.31", "统一标题字形并降低节标题字重"],
        ],
        [1.1, 2.6, 1.4, 2.4, 2.3, 6.0],
    )

    # Main body starts with Arabic page 1.
    main_sec = doc.add_section(WD_SECTION.NEW_PAGE)
    configure_section(main_sec)
    set_page_number_format(main_sec, "decimal", 1)
    add_body_header_footer(main_sec)

    heading(doc, "1 项目概况", 1)
    heading(doc, "1.1 背景和基础", 2)
    body(doc, "文献检索是科研工作流中最基础、也最耗时的环节之一。真实科研问题通常同时包含研究主题、方法、任务、数据、时间和证据关系等多项约束，例如“哪些论文从理论上分析 MAML 在小样本学习中的收敛性”。这类查询要求系统不仅找到语义相近的论文，还要判断论文是否同时满足方法、研究关系与应用场景等必要条件。传统关键词检索依赖用户反复拆词、组合布尔表达式和人工筛选，难以直接完成这一过程。")
    body(doc, "近年来，PaSa、SPAR、Ai2 Paper Finder 与 PaperQA2 等系统将大语言模型、语义检索和引文网络引入学术搜索，形成了“查询理解—自主检索—候选判定—证据归纳”的新范式。PaSa 采用 Crawler 与 Selector 双 Agent 并以强化学习优化搜索策略；SPAR 使用 RefChain 进行查询分解与演化；Ai2 Paper Finder 结合多索引语义检索、查询改写与引文追踪。这些工作表明，复杂学术检索的关键已从单次关键词匹配转向多轮检索规划与证据判断。")
    body(doc, "本赛题要求构建端到端学术论文智能搜索系统，并将 F1、运行效率和结构化回复共同纳入评测。围绕这一目标，研问智搜持续完成了查询理解、多源混合召回、迭代式检索、论文身份统一、综合排序、关系证据救援、结果集合决策与结构化展示等模块，形成可在线运行、可冻结回放、可离线演示的完整系统。")
    body(doc, "当前稳定版本以 TRACE-Rank v4 为基础排序框架，引入 SCGQ v15 与 QCF v11 探索受控引文前沿，并使用关系证据图 REG v18 对候选论文的条件合取、关系谓词、引文方向、种子桥接与图支持进行显式建模。系统的核心目标不是简单扩大候选数量，而是把真正满足复杂查询条件的论文稳定送入结果前列，并给出可复核的推荐依据。")
    figure(doc, SHOTS / "01_home_full.png", "图 1 研问智搜系统首页与检索模式入口")

    heading(doc, "1.2 国内外研究现状", 2)
    body(doc, "围绕复杂学术查询，现有系统大致可分为传统学术搜索、LLM 检索 Agent 和科研证据问答三类。传统系统依靠倒排索引、关键词相关性与引用排序，覆盖面广、响应稳定，但需要用户自行完成查询拆解与结果筛选。LLM 检索 Agent 将查询理解、工具调用和迭代检索组织为自主决策过程，能够处理更自然、更细粒度的问题。科研证据问答则进一步关注全文证据聚合和答案可溯源性。")
    body(doc, "PaSa 的双 Agent 架构强调完整搜索与论文逐篇判定，SPAR 通过查询演化扩大覆盖，Ai2 Paper Finder 强调多索引召回与引文路径，PaperQA2 侧重全文证据与科学问答。现有工作已覆盖检索规划、语义匹配和引文扩展，但对于多项必要条件能否同时成立、候选论文与高置信种子之间属于何种关系，以及新增检索动作是否值得继续等问题，仍需要更细粒度的证据组织。")
    body(doc, "研问智搜在上述范式上形成了面向比赛场景的完整实现：以 Query Contract 表达复杂查询，以多源词项—语义双通道扩大覆盖，以 TRACE-Rank 统一融合排名，再由 REG 对多条件和关系证据进行受控救援。系统同时保留候选池、排序池、最终结果集合和补充候选，使高召回探索与高精度交付可以在同一条流水线中协同。")
    table(
        doc,
        ["系统", "核心方法", "多轮检索", "关系/引文证据", "结果组织"],
        [
            ["Google Scholar", "关键词与引用排序", "用户手工迭代", "引用网络", "论文列表"],
            ["PaSa", "Crawler + Selector + RL", "支持", "引文扩展", "筛选论文集合"],
            ["SPAR", "RefChain 查询分解与演化", "支持", "参考链", "论文集合"],
            ["Ai2 Paper Finder", "多索引语义检索与采样", "支持", "引文追踪", "排序结果"],
            ["PaperQA2", "全文检索与证据聚合", "面向问答", "全文证据", "可溯源答案"],
            ["研问智搜", "Query Contract + TRACE-Rank + REG", "支持", "条件、关系与引文证据", "集合、矩阵、图谱与导出"],
        ],
        [2.7, 4.2, 2.2, 3.4, 4.0],
        "表 1 代表性学术检索系统的方法对比",
    )

    heading(doc, "1.3 场景和价值", 2)
    body(doc, "研问智搜面向开题调研、综述撰写、研究前沿追踪、跨领域选题探索和科研团队知识整理等场景。用户可直接使用自然语言描述研究问题，系统自动识别研究主题、方法约束、任务场景与时间要求，并规划多条互补检索路径，减少人工反复改写检索式的工作量。")
    body(doc, "系统的实际价值体现在三个方面：第一，以多源检索和引文前沿提高复杂查询的覆盖；第二，以条件合取、关系证据和论文级推荐理由提高结果的可解释性；第三，以 F1 优先、均衡集合和探索列表三种结果视图适配快速决策与深度调研。证据矩阵、论文关系图谱、BibTeX/CSV/JSON 导出和打印报告进一步衔接文献筛选、阅读与引用管理。")
    figure(doc, SHOTS / "03_results_overview_viewport.png", "图 2 查询理解、结果集合与运行指标总览")

    heading(doc, "1.4 所需支持", 2)
    body(doc, "系统后端基于 FastAPI，可在普通 Windows 笔记本的 CPU 环境中运行；本地交叉编码器仅在关系证据重排阶段使用，无需模型训练。论文数据来自 OpenAlex、arXiv、Crossref 与 Semantic Scholar 等公开学术服务，大模型用于查询理解、检索规划和结果归纳。系统同时提供响应缓存、冻结回放和离线演示模式，现场网络波动时仍可复现完整界面与检索证据。")

    heading(doc, "2 项目规划", 1)
    heading(doc, "2.1 整体目标", 2)
    body(doc, "项目目标是形成面向复杂学术查询的端到端智能论文搜索与推荐系统，使自然语言问题能够直接转化为可执行检索计划，并完成多源召回、候选融合、关系证据判断、结果集合决策和结构化交付。系统既服务于比赛自动评测，也满足现场演示、论文复核与导出使用。")
    bullet(doc, "（1）查询理解：解析主题、方法、任务、数据、时间和关系约束，构建 Query Contract；")
    bullet(doc, "（2）自主检索：在词项、语义、arXiv、标题探测、引文扩展和第二轮检索之间规划动作；")
    bullet(doc, "（3）综合排序：以 TRACE-Rank 完成多路径融合、约束覆盖、权威性、新近性和多样性排序；")
    bullet(doc, "（4）证据救援：以 REG 对候选的条件合取、关系谓词、引文方向与种子桥接进行受控比较；")
    bullet(doc, "（5）结构化交付：输出论文集合、补充候选、证据矩阵、关系图谱、推荐理由和标准化导出文件。")
    figure(doc, FIGS / "fig7_algorithm_pipeline.png", "图 3 检索—诊断—救援—选择闭环")

    heading(doc, "2.2 技术创新点", 2)
    body(doc, "本项目围绕复杂查询中的多条件约束、候选噪声和关系证据不足三个问题，形成四项相互衔接的设计。")
    body(doc, "（1）Query Contract 驱动的查询规划。系统不把查询压缩为单个关键词串，而是显式提取研究主题、方法、任务、时间与关系要求，并生成多条互补子查询。HELIOS-Lite 根据约束缺口、结果重复率、语义漂移与调用成本决定继续扩展或停止，使检索动作与当前证据状态相对应。", bold_prefix="（1）Query Contract 驱动的查询规划。")
    figure(doc, SHOTS / "04_iterative_process.png", "图 4 大模型查询理解与自主迭代检索流程")
    body(doc, "（2）多源混合召回与受控引文前沿。系统并行组织 OpenAlex 词项检索与语义检索，并结合 arXiv、标题探测、Crossref、Semantic Scholar、第二轮检索和引文扩展。SCGQ 与 QCF 从高置信种子论文出发，沿正向和反向引文关系探索候选，同时保存引文方向与种子桥接证据。", bold_prefix="（2）多源混合召回与受控引文前沿。")
    figure(doc, SHOTS / "06_reg_action.png", "图 5 REG 关系证据检索与受控晋升动作")
    body(doc, "（3）TRACE-Rank 与 REG 的双阶段证据排序。TRACE-Rank 负责候选融合与全局排序，REG 则进一步判断候选是否同时覆盖查询必要条件、是否满足查询中的关系谓词，以及是否得到引文路径和种子论文支持。REG 仅在候选相对 Top20 尾部结果具有明确证据优势时进行替换，从而将关系证据转化为可控的排序动作。", bold_prefix="（3）TRACE-Rank 与 REG 的双阶段证据排序。")
    figure(doc, FIGS / "fig8_reg_mechanism.png", "图 6 REG 查询约束图与关系证据评分机制")
    body(doc, "（4）双层结果集合与可核验交付。系统将高召回候选池、Top20 排序池和最终结果集合分离，分别提供 F1 优先、均衡集合和探索列表。最终结果默认用于 CSV、BibTeX 与比赛载荷，补充候选保留供用户继续探索；每篇论文同时呈现约束覆盖、摘要证据、引文路径、分项得分和推荐理由。", bold_prefix="（4）双层结果集合与可核验交付。")
    figure(doc, SHOTS / "11_paper_cards_view.png", "图 7 最终论文卡片、分项得分与推荐依据")

    heading(doc, "2.3 关键技术指标", 2)
    body(doc, "系统采用冻结查询集记录检索质量、结果集合质量和运行成本。当前指标覆盖校准集、开发集、盲测集与 validation30，并保留 LIVE—FROZEN—REPLAY 回放记录。")
    table(
        doc,
        ["指标", "当前实测", "说明"],
        [
            ["validation30 F1 优先集合", "P=0.2333，R=0.2553，F1=0.2240", "平均返回 2 篇"],
            ["开发集最终集合", "Recall=0.3517，F1=0.2833", "REG 双视图一致性"],
            ["开发集 REG Top20", "Recall=0.5180，F1=0.1069", "25 篇相关论文进入 Top20"],
            ["盲测集 REG Top20", "Recall=0.5008，F1=0.0799", "无原命中损失"],
            ["低成本模式", "F1=0.2157，8.96 API/查询", "平均延迟 31.38 s"],
            ["缓存命中", "约 0.002 s", "0 API、0 新增 Token"],
        ],
        [4.8, 5.2, 6.5],
        "表 2 当前关键技术指标",
    )
    figure(doc, FIGS / "fig5_efficiency_pareto.png", "图 8 质量模式与低成本模式的运行边界")

    heading(doc, "2.4 与赛题参考系统的对比定位", 2)
    body(doc, "赛题参考系统分别强调强化学习搜索、查询演化、多索引语义检索和全文证据问答。研问智搜吸收这些方向的共同经验，并将重点放在多条件关系证据、冻结可复现评测和可直接运行的端到端产品形态。")
    table(
        doc,
        ["维度", "参考系统代表路线", "研问智搜"],
        [
            ["查询理解", "LLM 改写、RefChain、Agent 规划", "Query Contract + 意图路由"],
            ["召回策略", "单源搜索、语义索引、引文追踪", "多源词项—语义双通道 + SCGQ/QCF"],
            ["综合排序", "Selector、语义匹配或采样策略", "TRACE-Rank + REG 关系证据"],
            ["结果组织", "论文集合或证据答案", "三类结果集合 + 矩阵 + 图谱 + 导出"],
            ["运行复现", "依系统而异", "缓存、冻结回放、离线演示与通道审计"],
        ],
        [3.0, 6.2, 7.3],
        "表 3 本项目与赛题参考系统的方法定位",
    )

    heading(doc, "3 实施方案", 1)
    heading(doc, "3.1 总体架构设计", 2)
    body(doc, "系统由查询认知、自主检索、证据决策和结构化交付四层组成。查询认知层负责 Query Contract、意图路由和子查询生成；自主检索层统一调度多源词项检索、语义检索、引文探索与第二轮动作；证据决策层由 TRACE-Rank、SCGQ/QCF、REG 和集合决策模块组成；结构化交付层输出论文集合、证据矩阵、关系图谱、研究全景和导出文件。")
    body(doc, "底层数据来自 OpenAlex、arXiv、Crossref 与 Semantic Scholar，本地运行层提供向量模型、交叉编码器、响应缓存、冻结回放和离线演示。系统在每一步保存论文身份、检索通道、约束覆盖、引文路径与成本字段，保证最终结果能够回溯到具体检索动作和证据来源。")
    figure(doc, FIGS / "fig6_system_architecture.png", "图 9 研问智搜端到端系统架构")

    heading(doc, "3.2 技术可行性分析", 2)
    body(doc, "数据方面，系统不依赖自建封闭论文库，而是通过公开学术 API 获取标题、摘要、作者、年份、引用量、DOI、arXiv ID 与引文关系；统一论文身份优先使用 DOI、arXiv ID、OpenAlex ID 和 Semantic Scholar ID，标准化标题仅作为补充匹配。该机制既支持多源去重，也便于冻结评测和结果回放。")
    body(doc, "算法方面，TRACE-Rank、SCGQ/QCF 与 REG 均已接入生产检索链路。关系证据阶段使用本地 MiniLM 交叉编码与显式证据特征，主要计算在 CPU 上完成，不需要额外训练。运行方面，FastAPI 后端与单文件前端可在 Windows 笔记本一键启动；外部服务不可用时，系统可切换备用数据源、规则解析、缓存或离线冻结响应。")
    body(doc, "工程方面，系统已具备健康检查、独立模式缓存、错误降级、搜索过程审计、BibTeX/CSV/JSON 导出和打印报告。完整测试共 155 项，覆盖查询理解、检索源、身份匹配、评测协议、结果集合、REG 路由和离线演示。")
    figure(doc, SHOTS / "08_evidence_matrix_view.png", "图 10 可核验证据覆盖矩阵")

    heading(doc, "3.3 核心算法与流程", 2)
    body(doc, "给定复杂学术查询 q，系统首先构建查询约束图并规划检索动作，随后形成高召回候选池，经 TRACE-Rank 得到 Top20 排序池；若候选池中存在满足关系证据门控的论文，REG 允许其受控替换 Top20 尾部结果；最后由集合决策模块生成不同规模的最终结果集合。")
    table(
        doc,
        ["步骤", "输入", "主要处理", "输出"],
        [
            ["1", "自然语言查询", "Query Contract 与意图路由", "结构化约束与子查询"],
            ["2", "子查询", "多源词项、语义与 arXiv 检索", "第一阶段候选池"],
            ["3", "候选与证据缺口", "SCGQ/QCF、第二轮动作与引文前沿", "扩展候选池"],
            ["4", "统一候选", "TRACE-Rank 融合与多样性排序", "Top20 排序池"],
            ["5", "Top20 与前沿候选", "REG 关系证据门控", "受控更新的 Top20"],
            ["6", "排序池", "集合决策与结构化归纳", "最终集合、补充候选与可视化"],
        ],
        [1.3, 3.7, 7.0, 4.5],
        "算法 1 研问智搜端到端检索流程",
    )

    heading(doc, "3.3.1 Query Contract 与意图路由", 3)
    body(doc, "系统把查询表示为由主题、方法、任务、数据、时间、发表范围和关系谓词组成的 Query Contract。对于“理论分析”“比较”“改进”“首次提出”等关系要求，系统单独保留关系谓词，而不是将其与主题关键词合并。检索意图路由进一步区分奠基工作、前沿进展、方法比较、应用检索与综述式查询，用于调整检索通道和排序偏好。")

    heading(doc, "3.3.2 多源混合召回", 3)
    body(doc, "第一阶段同时调用 OpenAlex lexical、OpenAlex semantic 与 arXiv 等通道，并保留 Crossref、Semantic Scholar 与标题探测作为补充。每条子查询均记录来源、返回次序和命中身份，随后通过 RRF 等方式融合路径证据。不同数据源返回同一论文时，系统利用稳定标识合并元数据，保留更完整的摘要、年份、引用量与原文链接。")

    heading(doc, "3.3.3 候选融合与统一身份", 3)
    body(doc, "候选融合阶段优先按 arXiv ID、DOI、OpenAlex ID 和 Semantic Scholar ID 建立论文身份；当稳定标识缺失时，才使用去版本号、去标点和规范化空白后的标题作为 fallback。该机制将“真正未召回”和“已经召回但身份未匹配”分开，为通道贡献、失分来源和冻结回放提供统一基准。")

    heading(doc, "3.3.4 TRACE-Rank 综合排序", 3)
    body(doc, "TRACE-Rank 将多路径 RRF、词项相关性、语义相关性、查询约束覆盖、引用影响力、发表新近性和 MMR 多样性组织为综合排序。排序过程同时保存各分项得分，使论文卡片能够解释候选为何进入结果前列。对于含时间限制的查询，系统在排序前执行统一日期截止，避免未来论文进入评测。")

    heading(doc, "3.3.5 SCGQ 与 QCF 引文前沿", 3)
    body(doc, "SCGQ 与 QCF 从 Top20 中的高置信种子论文出发，批量探索参考文献和被引论文，并根据查询条件、种子质量和重复率压缩前沿。与无差别引文扩展不同，该过程保留候选相对种子的引文方向、共同种子数量和桥接强度，为后续 REG 判断候选是否属于目标研究脉络提供结构化证据。")

    heading(doc, "3.3.6 REG 关系证据图", 3)
    body(doc, "REG 不把复杂查询压缩为单一语义相似度，而是分别计算必要条件覆盖 c_j、条件合取 c、关系谓词 r、引文方向 d、种子桥接 b、图支持 g 以及 TRACE 与本地交叉编码信号。条件合取同时考虑平均覆盖和最弱必要条件：")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_run_text(p, "c(p,q) = √( mean_j c_j(p,q) × min_j c_j(p,q) )", bold=True, size=12, font="Cambria Math")
    body(doc, "最终关系证据分数由上述特征按关系族进行冻结组合。系统设置绝对证据下限、相对证据间隔、Top20 头部保护和最大晋升数量；当前每条查询最多允许 1 篇前沿候选替换 Top20 尾部论文，从而控制新增噪声。")
    figure(doc, FIGS / "fig8_reg_mechanism.png", "图 11 REG 条件合取、引文关系与证据间隔门控")

    heading(doc, "3.3.7 查询级风险路由与集合决策", 3)
    body(doc, "排序完成后，系统根据查询级风险生成 F1 优先、均衡集合与探索列表三类输出。F1 优先集合强调有效密度，均衡集合保留 8 篇核心结果，探索列表完整展示 Top20；未进入当前最终集合的论文保留在补充候选区，不影响默认 CSV、BibTeX 与比赛载荷。该设计使集合规模与用户目标保持一致。")

    heading(doc, "3.3.8 缓存、冻结回放与成本控制", 3)
    body(doc, "系统以检索模式、结果集合、检索策略和查询文本共同构成缓存身份。相同配置命中缓存后不再发生外部 API 与大模型调用。评测链路提供 LIVE、FROZEN 和 REPLAY 三种运行方式：LIVE 获取真实外部结果，FROZEN 固定候选与元数据，REPLAY 在相同候选上重放排序与评测，保证版本比较不受网络变化影响。")

    heading(doc, "3.4 系统实现", 2)
    body(doc, "后端以 FastAPI 组织查询理解、学术检索客户端、身份合并、TRACE-Rank、引文前沿、REG、集合决策和响应缓存。前端采用深色科研驾驶舱布局，集中呈现查询理解、迭代检索过程、数据源、运行效率、REG 晋升动作、研究全景、证据矩阵、论文关系图谱、查询—论文关系和论文详情卡片。")
    body(doc, "系统支持 quick、balanced、efficient、helios 和 deep 五种检索深度，并将检索策略与结果集合独立配置。balanced + relational 为默认质量路径，efficient + lean 用于调用预算受限场景。导出功能覆盖检索式复制、BibTeX、CSV、完整 JSON 和打印报告。")
    figure(doc, SHOTS / "09_relation_graph_view.png", "图 12 查询意图与命中论文的关系图谱")

    heading(doc, "3.5 实验与评测", 2)
    body(doc, "实验围绕检索覆盖、Top20 转化、最终结果集合和运行成本展开。所有版本比较均在固定查询、固定 ground truth、统一论文身份和一致日期规则下完成；校准、开发与盲测查询彼此分离，外部检索结果可冻结回放。")

    heading(doc, "3.5.1 实验设置", 3)
    body(doc, "评测使用 AutoScholarQuery 相关查询划分，包括校准40、开发20、盲测20与独立 validation30。Top20 指标用于衡量候选排序质量，最终集合指标用于衡量直接交付结果的 Precision、Recall 与 F1。通道诊断记录候选 Recall、Top20 命中、最终集合命中、API 调用、延迟与缓存状态。")

    heading(doc, "3.5.2 REG 跨数据划分表现", 3)
    body(doc, "REG 在三个独立划分上均提高了 Top20 命中，并保持原有相关论文不丢失。校准集 Top20 TP 由 23 增至 28，开发集由 22 增至 25，盲测集由 17 增至 18；对应 Macro Recall 分别提高至 0.3628、0.5180 和 0.5008。")
    figure(doc, FIGS / "fig1_reg_generalization.png", "图 13 REG 在校准集、开发集与盲测集上的一致增益")
    table(
        doc,
        ["数据划分", "基础排序：TP / Recall / F1", "REG：TP / Recall / F1"],
        [
            ["校准40", "23 / 0.3051 / 0.0511", "28 / 0.3628 / 0.0620"],
            ["开发20", "22 / 0.4346 / 0.0933", "25 / 0.5180 / 0.1069"],
            ["盲测20", "17 / 0.4758 / 0.0754", "18 / 0.5008 / 0.0799"],
        ],
        [3.2, 6.5, 6.8],
        "表 4 REG 跨数据划分结果",
    )

    heading(doc, "3.5.3 最终集合与多通道贡献", 3)
    body(doc, "开发集最终集合从基础精选逐步加入风险救援、覆盖验证和 REG 双视图一致性后，Macro F1 由 0.2292 提升至 0.2833，Macro Recall 由 0.2750 提升至 0.3517，平均返回规模由 2.00 增至 2.20。该结果表明，受控证据救援可以在保持集合紧凑的同时提高有效覆盖。")
    figure(doc, FIGS / "fig2_final_set_trajectory.png", "图 14 开发集最终结果集合的 F1—Recall 演进")
    body(doc, "多通道贡献分析显示，词项检索构成基础覆盖，语义检索、arXiv、标题探测、引文扩展和第二轮检索继续补足不同类型的论文。冻结盲测查询中各通道累计命中 48 篇相关论文；开发集 27 篇候选相关论文中有 25 篇进入 Top20，最终精选 13 篇。")
    figure(doc, FIGS / "fig3_retrieval_evidence.png", "图 15 多通道新增命中与候选—Top20—最终集合证据漏斗")

    heading(doc, "3.5.4 运行效率与成本", 3)
    body(doc, "质量模式在 blind50 上取得 Macro F1 0.2233，平均每条查询 15.40 次 API 调用、延迟 43.32 秒；低成本模式将平均 API 调用降至 8.96 次、延迟降至 31.38 秒，同时保持 Macro F1 0.2157。重复查询命中缓存后响应约 0.002 秒，不产生新增 API 与 Token。")
    table(
        doc,
        ["运行模式", "Macro F1", "API/查询", "平均延迟", "使用场景"],
        [
            ["质量模式", "0.2233", "15.40", "43.32 s", "完整检索与关系证据"],
            ["低成本模式", "0.2157", "8.96", "31.38 s", "调用预算受限"],
            ["缓存复用", "复用原结果", "0", "约 0.002 s", "重复查询与现场演示"],
        ],
        [3.0, 2.5, 2.5, 2.8, 5.7],
        "表 5 运行质量与成本对比",
    )

    heading(doc, "3.5.5 结构化证据消融", 3)
    body(doc, "在相同候选扩展上，通用局部语义重排在开发集得到 20 篇 Top20 命中，REG 得到 25 篇；校准集基础排序、通用图重排、REG 与原子条件路由分别得到 23、25、28 和 24 篇命中。结果支持将条件合取、关系谓词、引文方向与种子桥接作为独立证据，而不是仅使用单一语义相关性。")
    figure(doc, FIGS / "fig4_structured_evidence_ablation.png", "图 16 结构化证据设计的校准集与开发集消融")

    heading(doc, "3.5.6 缓存、回放与现场复现", 3)
    body(doc, "系统对完整响应、外部检索候选和本地重排分数进行分层缓存。冻结评测可在无网络条件下重放候选身份、排序动作和指标计算；离线演示模式使用真实冻结响应，在不访问外部 API 的情况下复现查询理解、REG 动作、证据矩阵、关系图谱和论文卡片，适合现场网络不稳定时使用。")

    heading(doc, "3.5.7 案例分析", 3)
    body(doc, "以查询“Which papers theoretically analyze MAML convergence for few-shot learning?”为例，系统解析出 MAML、理论分析、收敛性和小样本学习等约束，生成多条子查询并获得 191 篇候选、去重后 118 篇。REG 从引文前沿中识别并晋升《Theoretical Convergence of Multi-Step Model-Agnostic Meta-Learning》，最终均衡集合输出 8 篇核心论文，同时保留 12 篇补充候选供进一步探索。")

    heading(doc, "3.5.8 系统功能验证", 3)
    body(doc, "系统已完成在线检索、五种检索深度、三种结果集合、两种检索策略、查询理解、迭代检索、REG、结构化总结、证据矩阵、关系图谱、论文卡片和五类导出功能验证。155 项自动化测试全部通过，浏览器运行日志未发现前端错误；离线冻结模式能够正常返回 8 篇均衡集合和 12 篇补充候选。")

    heading(doc, "3.6 风险与对策", 2)
    table(
        doc,
        ["风险", "已实现对策"],
        [
            ["外部 API 限流或不可用", "多源检索、错误降级、响应缓存与离线冻结演示"],
            ["大模型调用失败", "规则式查询理解、确定性子查询与模板化总结"],
            ["候选扩展引入噪声", "TRACE-Rank、REG 证据门控、头部保护与最大晋升数量"],
            ["论文身份匹配不稳定", "DOI、arXiv ID、OpenAlex ID、S2 ID 与标题 fallback"],
            ["冷查询延迟波动", "低成本模式、异步检索、分层缓存与离线演示"],
            ["现场环境差异", "一键启动脚本、健康检查、冻结回放与完整运行说明"],
        ],
        [5.2, 11.3],
        "表 6 主要风险与应对措施",
    )

    heading(doc, "3.7 系统局限与未来工作", 2)
    body(doc, "后续工作将围绕三方面继续推进：一是扩大冻结评测规模并补充更多关系类型，使指标覆盖更广；二是提高前沿候选的动作精度，进一步将候选池中的有效论文转化到 Top20；三是完善复杂关系解析和全文证据利用，使系统能够处理更细粒度的方法比较、数据约束和研究结论检索。")
    body(doc, "产品层面将继续优化首次检索延迟、结果交互筛选和本地化部署，并扩展文献专题库、团队协作标注和研究脉络追踪能力，使研问智搜从比赛原型逐步发展为可持续使用的科研检索工具。")

    heading(doc, "3.8 计划和分工", 2)
    body(doc, "项目围绕赛题研究、系统原型、评测审计、算法迭代、界面重构和提交材料六条工作线推进。团队成员在总体方案统一的基础上分别承担算法、后端和展示工作，并共同完成系统联调、现场复现与材料审阅。")
    table(
        doc,
        ["成员", "主要职责", "具体工作"],
        [
            ["贾梓旭", "算法设计与项目统筹", "Query Contract、TRACE-Rank、REG 与集合决策；实验方案、指标分析、项目文档与整体统筹"],
            ["贺凯华", "后端系统与数据链路", "FastAPI、多源检索客户端、身份合并、缓存与冻结回放；接口联调和运行环境"],
            ["刘凯", "前端展示与演示交付", "系统界面、证据矩阵、关系图谱、论文卡片与导出；演示视频、答辩素材与现场展示"],
        ],
        [2.4, 3.7, 10.4],
        "表 7 团队成员分工",
    )

    heading(doc, "4 参考资料", 1)
    refs = [
        "[1] He Y, Huang G, Feng P, et al. PaSa: An LLM Agent for Comprehensive Academic Paper Search. ACL 2025. arXiv:2501.10120.",
        "[2] Ajith A, Xia M, Chevalier A, et al. LitSearch: A Retrieval Benchmark for Scientific Literature Search. EMNLP 2024. arXiv:2407.18940.",
        "[3] Shi X, Li Y, Kou Q, et al. SPAR: Scholar Paper Retrieval with LLM-based Agents for Enhanced Academic Search. arXiv:2507.15245, 2025.",
        "[4] Feldman S, et al. AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite. arXiv:2510.21652, 2025.",
        "[5] Skarlinski M, et al. Language Agents for Answering Questions from Scientific Literature. NeurIPS 2024.",
        "[6] Khattab O, Santhanam K, Li X, et al. Demonstrate-Search-Predict: Composing Retrieval and Language Models for Knowledge-Intensive NLP. arXiv:2212.14024, 2022.",
        "[7] Feng P, He Y, Huang G, et al. AGILE: A Novel Framework of LLM Agents. NeurIPS 2024.",
        "[8] Muennighoff N, et al. GritLM: Generative Representational Instruction Tuning. arXiv:2402.09906, 2024.",
        "[9] OpenAlex API Documentation. https://docs.openalex.org/.",
        "[10] arXiv API User's Manual. https://info.arxiv.org/help/api/.",
        "[11] Crossref REST API. https://api.crossref.org/.",
        "[12] Semantic Scholar Academic Graph API. https://api.semanticscholar.org/.",
    ]
    for ref in refs:
        p = doc.add_paragraph(style="BodyTextCN")
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.left_indent = Cm(0)
        p.paragraph_format.line_spacing = Pt(16)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.left_indent = Cm(1.0)
        p.paragraph_format.first_line_indent = Cm(-1.0)
        add_run_text(p, ref, size=10.5)

    # Keep the default settings schema-valid; Word automation refreshes fields
    # before the PDF export step.
    settings = doc.settings._element
    zoom = settings.find(qn("w:zoom"))
    if zoom is not None and zoom.get(qn("w:percent")) is None:
        zoom.set(qn("w:percent"), "100")

    out_path = OUT / "虫虫特工队_研问智搜_项目文档_标题优化版V2.2.docx"
    doc.save(out_path)
    return out_path


if __name__ == "__main__":
    print(build())
