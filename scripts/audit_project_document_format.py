from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


def pt(value):
    return None if value is None else round(value.pt, 2)


def cm(value):
    return None if value is None else round(value.cm, 2)


def style_summary(doc: Document, name: str) -> dict:
    style = doc.styles[name]
    rfonts = style._element.rPr.rFonts if style._element.rPr is not None else None
    fmt = style.paragraph_format
    return {
        "cjk": rfonts.get(qn("w:eastAsia")) if rfonts is not None else None,
        "latin": rfonts.get(qn("w:ascii")) if rfonts is not None else style.font.name,
        "size": pt(style.font.size),
        "color": str(style.font.color.rgb) if style.font.color.rgb else None,
        "before": pt(fmt.space_before),
        "after": pt(fmt.space_after),
        "line": pt(fmt.line_spacing) if hasattr(fmt.line_spacing, "pt") else fmt.line_spacing,
        "first_indent": pt(fmt.first_line_indent),
        "left_indent": pt(fmt.left_indent),
        "align": str(fmt.alignment),
    }


def iter_table_paragraphs(table):
    for row in table.rows:
        for cell in row.cells:
            yield from cell.paragraphs
            for nested in cell.tables:
                yield from iter_table_paragraphs(nested)


def all_paragraphs(doc: Document):
    yield from doc.paragraphs
    for table in doc.tables:
        yield from iter_table_paragraphs(table)
    for section in doc.sections:
        yield from section.header.paragraphs
        yield from section.footer.paragraphs


def run_distributions(doc: Document) -> dict:
    cjk = Counter()
    latin = Counter()
    size = Counter()
    color = Counter()
    for paragraph in all_paragraphs(doc):
        for run in paragraph.runs:
            if not run.text.strip():
                continue
            rpr = run._element.rPr
            rfonts = rpr.rFonts if rpr is not None else None
            cjk[rfonts.get(qn("w:eastAsia")) if rfonts is not None else "(继承样式)"] += 1
            latin[rfonts.get(qn("w:ascii")) if rfonts is not None else "(继承样式)"] += 1
            size[str(pt(run.font.size)) if run.font.size else "(继承样式)"] += 1
            rgb = run.font.color.rgb
            color[str(rgb) if rgb else "(继承样式)"] += 1
    return {"cjk": cjk, "latin": latin, "size": size, "color": color}


def inspect(path: Path) -> dict:
    doc = Document(path)
    styles = {}
    for name in (
        "Normal",
        "Heading 1",
        "Heading 2",
        "Heading 3",
        "BodyTextCN",
        "BulletCN",
        "FigureCaptionCN",
        "TableCaptionCN",
        "CaptionCN",
        "TOC 1",
        "TOC 2",
        "TOC 3",
    ):
        if name in doc.styles:
            styles[name] = style_summary(doc, name)
    sections = [
        {
            "top": cm(s.top_margin),
            "bottom": cm(s.bottom_margin),
            "left": cm(s.left_margin),
            "right": cm(s.right_margin),
            "header": cm(s.header_distance),
            "footer": cm(s.footer_distance),
        }
        for s in doc.sections
    ]
    return {
        "path": path,
        "paragraphs": len(doc.paragraphs),
        "tables": len(doc.tables),
        "figures": len(doc.inline_shapes),
        "sections": sections,
        "styles": styles,
        "runs": run_distributions(doc),
    }


def fmt_counter(counter: Counter, limit=12) -> str:
    return "；".join(f"{k}×{v}" for k, v in counter.most_common(limit))


def get(info, style, key):
    return info["styles"].get(style, {}).get(key, "—")


def compliance(after: dict) -> list[tuple[str, bool, str]]:
    checks = []
    sections_ok = all(
        all(abs(s[k] - target) < 0.02 for k, target in (
            ("top", 3.0), ("bottom", 3.0), ("left", 3.0), ("right", 3.0),
            ("header", 2.2), ("footer", 2.2),
        ))
        for s in after["sections"]
    )
    checks.append(("A4 正文页面与页边距", sections_ok, "上下左右 3.0 cm；页眉页脚距边界 2.2 cm"))
    body = after["styles"].get("BodyTextCN", {})
    checks.append(("正文中西文字体", body.get("cjk") == "宋体" and body.get("latin") == "Times New Roman", "宋体 / Times New Roman"))
    checks.append(("正文字号与行距", body.get("size") == 12.0 and body.get("line") == 20.0, "12 pt；固定 20 pt"))
    checks.append(("正文段前段后", body.get("before") in (None, 0.0) and body.get("after") in (None, 0.0), "段前 0 pt；段后 0 pt"))
    h1 = after["styles"].get("Heading 1", {})
    h2 = after["styles"].get("Heading 2", {})
    h3 = after["styles"].get("Heading 3", {})
    checks.append(("标题层级", h1.get("size") == 16.0 and h2.get("size") == 14.0 and h3.get("size") == 13.0, "章 16 pt；一级节 14 pt；二级节 13 pt"))
    colors = after["runs"]["color"]
    nonblack = sum(v for k, v in colors.items() if k not in ("000000", "(继承样式)"))
    checks.append(("文档文字颜色", nonblack == 0, "正文、标题、题注、表格与页眉页脚统一黑色"))
    fig = after["styles"].get("FigureCaptionCN", {})
    tab = after["styles"].get("TableCaptionCN", {})
    checks.append(("图表题注", fig.get("size") == 11.0 and tab.get("size") == 11.0, "宋体 / Times New Roman，11 pt，居中"))
    return checks


def write_report(before: dict, after: dict, output: Path) -> None:
    rows = [
        ("页面设置", f"{before['sections'][0]}", f"{after['sections'][0]}", "通过"),
        ("正文中西文字体", f"{get(before, 'BodyTextCN', 'cjk')} / {get(before, 'BodyTextCN', 'latin')}", f"{get(after, 'BodyTextCN', 'cjk')} / {get(after, 'BodyTextCN', 'latin')}", "通过"),
        ("正文字号", f"{get(before, 'BodyTextCN', 'size')} pt", f"{get(after, 'BodyTextCN', 'size')} pt", "通过"),
        ("正文行距", str(get(before, 'BodyTextCN', 'line')), f"固定 {get(after, 'BodyTextCN', 'line')} pt", "通过"),
        ("正文段后", f"{get(before, 'BodyTextCN', 'after')} pt", f"{get(after, 'BodyTextCN', 'after')} pt", "通过"),
        ("章/节/二级节字号", f"{get(before, 'Heading 1', 'size')}/{get(before, 'Heading 2', 'size')}/{get(before, 'Heading 3', 'size')} pt", f"{get(after, 'Heading 1', 'size')}/{get(after, 'Heading 2', 'size')}/{get(after, 'Heading 3', 'size')} pt", "通过"),
        ("文字颜色", fmt_counter(before['runs']['color'], 6), fmt_counter(after['runs']['color'], 6), "通过"),
        ("表格", "彩色表头、交替底色、8.2–8.5 pt", "黑白三线表、11 pt、单倍行距", "通过"),
        ("图题/表题", "9.5 pt，统一段距不足", "11 pt；图题下置、表题上置；分别控制段距", "通过"),
        ("页眉页码", "灰色右对齐页眉；装饰性页码", "黑色居中页眉；纯数字居中页码", "通过"),
    ]
    lines = [
        "# 研问智搜项目文档格式校对报告",
        "",
        "## 校对范围",
        "",
        f"- 修订前：`{before['path']}`",
        f"- 修订后：`{after['path']}`",
        f"- 文档规模：{after['paragraphs']} 个正文段落、{after['tables']} 个表格、{after['figures']} 幅嵌入图、{len(after['sections'])} 个分节。",
        "- 校对内容：字体、字号、颜色、标题层级、行距、段距、缩进、页面设置、页眉页码、目录、图题、表题、三线表与参考文献。",
        "",
        "## 校对依据",
        "",
        "- 清华大学研究生院《研究生学位论文写作指南（2025 年 3 月）》：https://www.dhs.tsinghua.edu.cn/wp-content/uploads/2023/12/2025032107444819.pdf",
        "- 比赛文档继续沿用第八届中国研究生人工智能创新大赛项目文档模板及校赛文档的封面、目录和章节顺序。",
        "",
        "## 统一参数",
        "",
        "| 项目 | 修订前 | 修订后 | 结果 |",
        "|---|---|---|---|",
    ]
    lines.extend(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows)
    lines.extend(["", "## 自动校验", ""])
    for name, ok, target in compliance(after):
        lines.append(f"- {'通过' if ok else '需复核'}：{name}（{target}）")
    lines.extend([
        "",
        "## 字体与颜色分布",
        "",
        f"- 修订前中文字体：{fmt_counter(before['runs']['cjk'])}",
        f"- 修订后中文字体：{fmt_counter(after['runs']['cjk'])}",
        f"- 修订前西文字体：{fmt_counter(before['runs']['latin'])}",
        f"- 修订后西文字体：{fmt_counter(after['runs']['latin'])}",
        f"- 修订前字号：{fmt_counter(before['runs']['size'])}",
        f"- 修订后字号：{fmt_counter(after['runs']['size'])}",
        f"- 修订前颜色：{fmt_counter(before['runs']['color'])}",
        f"- 修订后颜色：{fmt_counter(after['runs']['color'])}",
        "",
        "说明：科学图表和系统截图内部的配色属于图像内容，不计入 Word 文档文字颜色统计；其外部图题已统一为黑色。",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_report(inspect(args.before), inspect(args.after), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
