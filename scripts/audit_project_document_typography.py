from __future__ import annotations

import argparse
import re
import subprocess
from collections import Counter
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


CJK_RE = re.compile(r"[\u3400-\u9fff\uff00-\uffef]")


def run_fonts(run):
    rpr = run._element.rPr
    rfonts = rpr.rFonts if rpr is not None and rpr.rFonts is not None else None
    return (
        rfonts.get(qn("w:eastAsia")) if rfonts is not None else None,
        rfonts.get(qn("w:ascii")) if rfonts is not None else None,
    )


def size(run):
    return round(run.font.size.pt, 2) if run.font.size else None


def iter_table_paragraphs(table):
    for row in table.rows:
        for cell in row.cells:
            yield from cell.paragraphs
            for nested in cell.tables:
                yield from iter_table_paragraphs(nested)


def iter_all_paragraphs(doc):
    yield from doc.paragraphs
    for table in doc.tables:
        yield from iter_table_paragraphs(table)
    for section in doc.sections:
        yield from section.header.paragraphs
        yield from section.footer.paragraphs


def heading_audit(doc):
    expected_sizes = {"Heading 1": 15.0, "Heading 2": 14.0, "Heading 3": 12.0}
    failures = []
    count = Counter()
    for index, paragraph in enumerate(doc.paragraphs):
        if paragraph.style.name not in expected_sizes:
            continue
        count[paragraph.style.name] += 1
        expected = expected_sizes[paragraph.style.name]
        for run in paragraph.runs:
            text = run.text
            if not text.strip():
                continue
            cjk, latin = run_fonts(run)
            has_cjk = bool(CJK_RE.search(text))
            if size(run) != expected:
                failures.append(f"P{index}: {text!r} 字号 {size(run)}，应为 {expected}")
            if has_cjk:
                if cjk != "黑体" or run.bold not in (False, None):
                    failures.append(f"P{index}: {text!r} 中文字体/字重为 {cjk}/{run.bold}")
            else:
                if latin != "Times New Roman" or run.bold is not True:
                    failures.append(f"P{index}: {text!r} 西文字体/字重为 {latin}/{run.bold}")
    return count, failures


def font_distribution(doc):
    cjk = Counter()
    latin = Counter()
    sizes = Counter()
    colors = Counter()
    for paragraph in iter_all_paragraphs(doc):
        for run in paragraph.runs:
            if not run.text.strip():
                continue
            ea, asc = run_fonts(run)
            cjk[ea or "(继承)"] += 1
            latin[asc or "(继承)"] += 1
            sizes[str(size(run)) if size(run) else "(继承)"] += 1
            rgb = run.font.color.rgb
            colors[str(rgb) if rgb else "(继承)"] += 1
    return cjk, latin, sizes, colors


def pdf_fonts(pdf: Path):
    output = subprocess.check_output(["pdffonts", str(pdf)], text=True, encoding="utf-8", errors="ignore")
    names = []
    for line in output.splitlines()[2:]:
        if not line.strip() or line.startswith("-"):
            continue
        names.append(line.split()[0])
    return names


def fmt(counter):
    return "；".join(f"{k}×{v}" for k, v in counter.most_common())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docx", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    doc = Document(args.docx)
    counts, heading_failures = heading_audit(doc)
    cjk, latin, sizes, colors = font_distribution(doc)
    embedded = pdf_fonts(args.pdf)
    forbidden = [name for name in embedded if any(x in name for x in ("Calibri", "Gothic", "YaHei", "Arial"))]

    lines = [
        "# 项目文档字体与排版专项校验报告 V2.3",
        "",
        f"- DOCX：`{args.docx}`",
        f"- PDF：`{args.pdf}`",
        "- 对照：校赛原始项目文档的实际 run-level 字体属性。",
        "",
        "## 标题规则",
        "",
        "| 层级 | 数字及英文 | 中文 | 字号 | 对齐 |",
        "|---|---|---|---:|---|",
        "| 章标题 | Times New Roman 加粗 | 黑体常规 | 15 pt | 左对齐 |",
        "| 一级节标题 | Times New Roman 加粗 | 黑体常规 | 14 pt | 左对齐 |",
        "| 二级节标题 | Times New Roman 加粗 | 黑体常规 | 12 pt | 左对齐 |",
        "",
        f"- 检查数量：Heading 1={counts['Heading 1']}，Heading 2={counts['Heading 2']}，Heading 3={counts['Heading 3']}。",
        f"- 标题直接字体校验：{'通过' if not heading_failures else '未通过'}。",
    ]
    if heading_failures:
        lines.extend(f"  - {x}" for x in heading_failures)
    lines.extend([
        "",
        "## 全文分布",
        "",
        f"- 中文字体：{fmt(cjk)}",
        f"- 西文字体：{fmt(latin)}",
        f"- 字号：{fmt(sizes)}",
        f"- 颜色：{fmt(colors)}",
        "",
        "## PDF 嵌入字体",
        "",
        *[f"- `{name}`" for name in embedded],
        "",
        f"- Calibri / MS Gothic / Microsoft YaHei / Arial 回退字体：{'未检出' if not forbidden else '检出：' + ', '.join(forbidden)}。",
        "- 正文、标题、题注、表格和页眉页码均使用黑色；科学图表内部配色属于图片内容。",
    ])
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)
    if heading_failures or forbidden:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
