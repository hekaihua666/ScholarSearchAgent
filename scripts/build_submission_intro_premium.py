from __future__ import annotations

import html
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "artifacts" / "submission_2026" / "draft"
DRAFT.mkdir(parents=True, exist_ok=True)

EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

INTRO = (
    "“研问智搜”是一款面向科研场景的复杂学术查询理解与智能论文搜索推荐系统。系统针对"
    "文献调研、科研选题和综述准备中“查询复杂、结果分散、筛选成本高”的问题，将自然语言"
    "需求解析为主题、方法、任务、数据集和时间等检索条件，自动规划子查询并联动多种学术"
    "数据源；随后结合标题、摘要、引文关系和约束满足情况完成候选补全与综合排序。系统以"
    "分层论文列表、推荐依据、证据矩阵和关系图谱呈现结果，支持文献导出，服务于开题调研、"
    "技术路线比较、综述准备与前沿追踪，形成从查询理解、自主检索、综合排序到结构化整理的"
    "完整流程。"
)


def build_html() -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>虫虫特工队_研问智搜_参赛作品简介</title>
<style>
  @page {{ size: A4 portrait; margin: 0; }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; width: 210mm; height: 297mm; }}
  body {{
    background: #ffffff;
    color: #202428;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}
  .page {{
    width: 210mm;
    height: 297mm;
    padding: 32mm 31.75mm 25mm;
    background: #ffffff;
  }}
  h1 {{
    margin: 0;
    text-align: center;
    color: #172d3b;
    font-family: "SimHei", "Microsoft YaHei", sans-serif;
    font-size: 21pt;
    line-height: 1.2;
    font-weight: 400;
    letter-spacing: 1.5pt;
  }}
  .accent {{
    width: 18mm;
    height: .75mm;
    margin: 6mm auto 9mm;
    background: #169b98;
  }}
  .intro {{
    margin: 0;
    color: #24282c;
    font-family: "SimSun", "Songti SC", serif;
    font-size: 12pt;
    line-height: 1.72;
    text-align: justify;
    text-justify: inter-ideograph;
    text-indent: 2em;
  }}
</style>
</head>
<body>
  <main class="page">
    <h1>参赛作品简介</h1>
    <div class="accent"></div>
    <p class="intro">{html.escape(INTRO)}</p>
  </main>
</body>
</html>
"""


def build() -> tuple[Path, Path]:
    if not EDGE.exists():
        raise FileNotFoundError(f"Microsoft Edge not found: {EDGE}")

    html_path = DRAFT / "虫虫特工队_研问智搜_参赛作品简介_新版.html"
    pdf_path = DRAFT / "虫虫特工队_研问智搜_参赛作品简介_新版_clean.pdf"
    html_path.write_text(build_html(), encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="ywzs-edge-") as profile:
        command = [
            str(EDGE),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={pdf_path}",
            "--print-to-pdf-no-header",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=1000",
            html_path.resolve().as_uri(),
        ]
        completed = subprocess.run(command, capture_output=True, timeout=60)
        if completed.returncode != 0 or not pdf_path.exists():
            raise RuntimeError(
                f"Edge PDF export failed ({completed.returncode}):\n"
                f"{completed.stdout!r}\n{completed.stderr!r}"
            )

    print(f"intro_chars={len(INTRO)}")
    print(html_path)
    print(pdf_path)
    return html_path, pdf_path


if __name__ == "__main__":
    build()
