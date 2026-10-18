from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "submission_2026" / "draft"
OUT.mkdir(parents=True, exist_ok=True)

# 延续校赛简介的写法：一项定位、一项问题、一条流程、主要能力、结果形式和总体价值。
INTRO = (
    "“研问智搜”是一款面向科研场景的复杂学术查询理解与智能论文搜索推荐系统。系统针对"
    "文献调研、科研选题和综述准备中“查询复杂、结果分散、筛选成本高”的问题，构建了从"
    "自然语言需求到结构化论文推荐的端到端流程。作品集成查询约束解析、子查询规划、多源"
    "学术检索、引文前沿扩展、证据补全、关系感知排序和核心结果选择等模块，对接OpenAlex、"
    "arXiv、Crossref和Semantic Scholar等学术数据源。系统能够输出分层论文列表、推荐依据、"
    "证据矩阵和关系图谱，并支持文献导出。作品通过查询规划、证据检索与关系排序的协同，"
    "兼顾检索覆盖度、结果相关性、可解释性和使用成本。"
)


def set_run_font(run, east_asia="SimSun", latin="Times New Roman", size=12, bold=False):
    run.font.name = latin
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)


def build():
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(3.0)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.175)
    section.right_margin = Cm(3.175)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(0)
    title.paragraph_format.space_after = Pt(14)
    title.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    set_run_font(title.add_run("参赛作品简介"), east_asia="SimSun", size=22, bold=False)

    body = doc.add_paragraph()
    body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    body.paragraph_format.first_line_indent = Pt(24)
    body.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    body.paragraph_format.line_spacing = Pt(20)
    body.paragraph_format.space_before = Pt(0)
    body.paragraph_format.space_after = Pt(0)
    set_run_font(body.add_run(INTRO), east_asia="SimSun", size=12, bold=False)

    out = OUT / "虫虫特工队_研问智搜_参赛作品简介.docx"
    doc.save(out)
    print(f"intro_chars={len(INTRO)}")
    print(out)


if __name__ == "__main__":
    build()
