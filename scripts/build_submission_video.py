from __future__ import annotations

import asyncio
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "submission_2026"
SHOTS = ART / "screenshots"
FIGS = ART / "figures_nature"
VIDEO = ART / "video"
FINAL = ART / "final"
VIDEO.mkdir(parents=True, exist_ok=True)
FINAL.mkdir(parents=True, exist_ok=True)

W, H = 1920, 1080
FPS = 30
VOICE = "Microsoft Huihui Desktop"
RATE = 1
AUDIO_EXT = "wav"

FONT_REG = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")

NAVY = "#081321"
NAVY_2 = "#0D1E31"
BLUE = "#27A7FF"
CYAN = "#4FE1C1"
WHITE = "#F7FAFC"
MUTED = "#AFC0D1"
GOLD = "#F2C46D"
ROSE = "#D46A8C"


@dataclass
class Scene:
    key: str
    title: str
    kicker: str
    narration: str
    layout: str
    assets: list[str]
    bullets: list[str]


SCENES = [
    Scene(
        "01_title",
        "研问智搜",
        "复杂学术查询的智能论文搜索与推荐",
        "研问智搜面向科研场景中的复杂学术查询。与只返回关键词匹配结果的传统检索不同，系统需要理解多维约束，主动组织检索路径，并把候选论文整理为可核验、可继续探索的结果。下面展示系统的完整工作流程。",
        "title",
        ["01_home_full.png"],
        ["查询认知", "自主检索", "证据排序", "结构化交付"],
    ),
    Scene(
        "02_query_contract",
        "从自然语言到 Query Contract",
        "查询认知与约束分解",
        "用户输入自然语言问题后，系统首先提取主题、方法、数据、时间、发表范围和关系谓词，并区分必要条件与偏好条件。Query Contract 将这些信息组织为可执行约束，再生成多条互补子查询，为后续检索提供稳定、可解释的搜索计划。",
        "split",
        ["03_results_overview_viewport.png", "04_iterative_process.png"],
        ["多维条件解析", "必要条件与偏好条件", "可执行子查询"],
    ),
    Scene(
        "03_retrieval",
        "多源自主检索与查询演化",
        "并发检索 · 身份统一 · 二轮补缺",
        "在自主检索阶段，系统并发访问 OpenAlex 词法与语义通道、arXiv、Crossref 和 Semantic Scholar。不同来源的 DOI、arXiv 标识、OpenAlex 标识和标题被统一归并。第一轮完成广覆盖召回，系统再依据证据缺口演化查询，并通过引文前沿展开第二轮检索。",
        "split",
        ["04_iterative_process.png", "05_iterative_details.png"],
        ["至少四类学术数据源", "稳定身份去重", "基于缺口的迭代检索"],
    ),
    Scene(
        "04_architecture",
        "TRACE-Rank：多视图候选融合",
        "从候选池到全局排序",
        "候选进入 TRACE-Rank 后，系统综合交叉编码、语义相似度、词法匹配、图关系和多源融合信号。排序不依赖单一高分，而是同时考虑查询约束覆盖、论文质量、时效性与来源互补性，形成可供证据决策层继续处理的全局候选序列。",
        "figure",
        ["report_v33/fig1_system_architecture.png"],
        ["Cross", "Semantic", "Lexical", "Graph", "Fusion"],
    ),
    Scene(
        "05_reg_meq",
        "REG-MEQ：关系与多视图证据",
        "条件合取 · 关系谓词 · 受控晋升",
        "REG 将复杂问题表示为条件合取与关系约束，判断候选是否满足必要条件、关系谓词和引文路径支持。MEQ 把五路分数转换为查询内百分位，并以第三强视图约束单点高分。配合头部保护、证据间隔和最大晋升数量，排序动作保持可解释、可控制。",
        "split",
        ["06_reg_action.png", "../figures_nature/fig8_reg_mechanism.png"],
        ["必要条件门控", "第三强视图一致性", "证据间隔与头部保护"],
    ),
    Scene(
        "06_ebp",
        "EBP：证据加权引文反投影",
        "把已知证据沿引文边回传到缺失候选",
        "EBP 将全部已评分候选视为证据节点，把五路一致性得分沿参考文献边反投影到尚未解析的论文。投票按参考列表长度平方根归一化，多篇高证据候选共同指向的论文获得更高支持度。目标通过一次 OpenAlex 批量请求完成物化，不增加大模型调用和 Token。",
        "figure",
        ["evidence_backprojection_v21/fig3_ebp_architecture.png"],
        ["全候选证据聚合", "归一化引文投票", "+1 批量 API · +0 LLM"],
    ),
    Scene(
        "07_selection",
        "证据校准与三类结果视图",
        "核心结果保持精炼，补充候选保留探索空间",
        "证据校准选择器把 TRACE、局部语义、关系图、约束覆盖、引文来源和排序先验组织为四十维查询内证据向量。核心结果返回一篇高置信论文，均衡集合返回八篇，探索列表保留二十篇与补充候选。三类视图共享同一检索过程，兼顾快速判断和继续探索。",
        "split",
        ["report_v33/fig11_selector_mechanism.png", "report_v33/fig2_result_modes.png"],
        ["核心结果：1 篇", "均衡集合：8 篇", "探索列表：20 篇"],
    ),
    Scene(
        "08_interface",
        "结构化结果与可核验证据",
        "检索过程、证据矩阵、关系图谱与论文卡片",
        "前端在同一流程中展示查询理解、检索动作和来源贡献。证据矩阵呈现每篇论文对主题、方法、数据与关系约束的覆盖，关系图谱连接查询意图和命中论文，论文卡片给出分项得分与推荐依据。结果可导出为 BibTeX、CSV、JSON 和打印报告。",
        "triptych",
        ["08_evidence_matrix_view.png", "09_relation_graph_view.png", "11_paper_cards_view.png"],
        ["证据矩阵", "关系图谱", "多格式导出"],
    ),
    Scene(
        "09_results",
        "实验结果",
        "候选召回、Top20 排序与核心结果 F1",
        "Blind50 包含五十条复杂查询和一百二十三篇目标论文。EBP 将候选 Recall 从零点五三二三提高到零点五六二三；REG-MEQ 将 Top20 Recall 从零点四零七零提高到零点四三七零；在相同的一篇输出规模下，证据校准把核心结果 F1 从零点二三三三提高到零点二五三三。三项提升分别对应候选发现、关系排序和最终选择。",
        "metrics",
        ["report_v33/fig12_final_evidence_chain.png"],
        ["核心 F1：0.2333 → 0.2533", "Top20 Recall：0.4070 → 0.4370", "候选 Recall：0.5323 → 0.5623"],
    ),
    Scene(
        "10_delivery",
        "成本、稳定性与交付",
        "可运行、可追溯、可展示",
        "EBP 在 Blind50 上平均增加二点四三秒检索和零点六一秒本地计算，每条查询只增加一次批量 API；REG-MEQ 与证据校准不增加外部调用。提交版一百六十八项自动化测试全部通过，并完成实时数据源和端到端查询验证。研问智搜由此形成从查询认知到结构化交付的完整闭环。",
        "final",
        ["01_home_full.png", "report_v33/fig1_system_architecture.png"],
        ["168 项测试通过", "+1 API · +0 LLM", "多源降级 · 成本记录"],
    ),
]


def font(size: int, bold: bool = False):
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REG), size=size)


def rounded(draw: ImageDraw.ImageDraw, box, radius=26, fill=NAVY_2, outline="#263B51", width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def contain(im: Image.Image, box: tuple[int, int, int, int], bg="white", radius=20):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    scale = min(bw / im.width, bh / im.height)
    nw, nh = max(1, int(im.width * scale)), max(1, int(im.height * scale))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    card = Image.new("RGB", (bw, bh), bg)
    card.paste(im, ((bw - nw) // 2, (bh - nh) // 2))
    mask = Image.new("L", (bw, bh), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, bw, bh), radius=radius, fill=255)
    return card, mask


def asset_path(name: str) -> Path:
    if name.startswith("../figures_nature/"):
        return FIGS / name.replace("../figures_nature/", "")
    if "/" in name:
        return FIGS / name
    return SHOTS / name


def paste_card(canvas: Image.Image, path: Path, box, bg="white", dim=False):
    im = Image.open(path).convert("RGB")
    if dim:
        im = ImageEnhance.Brightness(im).enhance(0.35).filter(ImageFilter.GaussianBlur(2))
    card, mask = contain(im, box, bg=bg)
    canvas.paste(card, (box[0], box[1]), mask)


def draw_header(draw, idx: int, scene: Scene):
    draw.text((90, 58), f"{idx:02d}  {scene.kicker}", font=font(27, True), fill=CYAN)
    draw.text((90, 112), scene.title, font=font(49, True), fill=WHITE)
    draw.line((90, 190, 1830, 190), fill="#24384B", width=2)
    progress = int(1740 * idx / len(SCENES))
    draw.rounded_rectangle((90, 1010, 1830, 1018), radius=4, fill="#223448")
    draw.rounded_rectangle((90, 1010, 90 + progress, 1018), radius=4, fill=BLUE)
    draw.text((1732, 960), f"{idx}/{len(SCENES)}", font=font(22), fill=MUTED)


def build_scene_image(scene: Scene, idx: int, out: Path):
    canvas = Image.new("RGB", (W, H), NAVY)
    draw = ImageDraw.Draw(canvas)
    if scene.layout == "title":
        im = Image.open(asset_path(scene.assets[0])).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
        im = ImageEnhance.Brightness(im).enhance(0.32).filter(ImageFilter.GaussianBlur(1.2))
        canvas.paste(im)
        overlay = Image.new("RGBA", (W, H), (4, 14, 28, 150))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        draw.text((110, 250), scene.title, font=font(92, True), fill=WHITE)
        draw.text((115, 375), scene.kicker, font=font(38), fill=CYAN)
        draw.rounded_rectangle((115, 465, 430, 475), radius=5, fill=BLUE)
        x = 115
        for b in scene.bullets:
            tw = draw.textbbox((0, 0), b, font=font(25))[2]
            draw.rounded_rectangle((x, 530, x + tw + 44, 582), radius=20, fill="#132A42", outline="#2B5578", width=2)
            draw.text((x + 22, 540), b, font=font(25), fill=WHITE)
            x += tw + 62
        draw.text((115, 820), "虫虫特工队 · 初赛项目视频", font=font(25), fill=MUTED)
        draw.rounded_rectangle((115, 900, 1805, 909), radius=5, fill=BLUE)
        canvas.save(out, quality=95)
        return

    draw_header(draw, idx, scene)

    if scene.layout == "split":
        boxes = [(90, 230, 940, 790), (980, 230, 1830, 790)]
        for p, b in zip(scene.assets, boxes):
            paste_card(canvas, asset_path(p), b, bg="white")
        y = 840
    elif scene.layout == "triptych":
        boxes = [(90, 235, 655, 770), (678, 235, 1243, 770), (1266, 235, 1830, 770)]
        for p, b in zip(scene.assets, boxes):
            paste_card(canvas, asset_path(p), b, bg="white")
        y = 825
    elif scene.layout == "figure":
        paste_card(canvas, asset_path(scene.assets[0]), (205, 225, 1715, 790), bg="white")
        y = 840
    elif scene.layout == "metrics":
        paste_card(canvas, asset_path(scene.assets[0]), (90, 220, 1310, 820), bg="white")
        cards = [
            (1360, 235, 1815, 395, "0.2533", "Blind50 · 核心结果 F1"),
            (1360, 430, 1815, 590, "0.4370", "Blind50 · Top20 Recall"),
            (1360, 625, 1815, 785, "0.5623", "Blind50 · 候选 Recall"),
        ]
        for x0, y0, x1, y1, n, s in cards:
            rounded(draw, (x0, y0, x1, y1), radius=24, fill="#10243A", outline="#31506D", width=2)
            draw.text((x0 + 28, y0 + 22), n, font=font(52, True), fill=CYAN)
            draw.text((x0 + 28, y0 + 100), s, font=font(22), fill=MUTED)
        y = 860
    elif scene.layout == "final":
        paste_card(canvas, asset_path(scene.assets[0]), (90, 235, 930, 770), bg="white")
        paste_card(canvas, asset_path(scene.assets[1]), (970, 235, 1830, 770), bg="white")
        y = 825
    else:
        raise ValueError(scene.layout)

    x = 90
    for i, b in enumerate(scene.bullets):
        tw = draw.textbbox((0, 0), b, font=font(24, True))[2]
        color = [BLUE, CYAN, GOLD, ROSE][i % 4]
        draw.rounded_rectangle((x, y, x + tw + 52, y + 56), radius=18, fill="#12263B", outline=color, width=2)
        draw.ellipse((x + 18, y + 20, x + 30, y + 32), fill=color)
        draw.text((x + 38, y + 12), b, font=font(24, True), fill=WHITE)
        x += tw + 72

    canvas.save(out, quality=95)


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def synthesize_sapi(scene: Scene, path: Path):
    """Generate local Mandarin narration with the installed Windows SAPI voice."""
    text_path = path.with_suffix(".txt")
    text_path.write_text(scene.narration, encoding="utf-8")
    ps = " ".join([
        "Add-Type -AssemblyName System.Speech;",
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;",
        f"$s.SelectVoice({_ps_quote(VOICE)});",
        f"$s.Rate={int(RATE)};",
        f"$s.SetOutputToWaveFile({_ps_quote(str(path))});",
        f"$t=Get-Content -Raw -Encoding UTF8 -LiteralPath {_ps_quote(str(text_path))};",
        "$s.Speak($t);",
        "$s.Dispose();",
    ])
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    finally:
        text_path.unlink(missing_ok=True)


def run(cmd: list[str], cwd: Path | None = None):
    print("RUN", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        text=True,
    )
    return float(out.strip())


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def split_clauses(text: str):
    parts = [x.strip() for x in re.split(r"(?<=[。！？；])", text) if x.strip()]
    result = []
    for p in parts:
        if len(p) <= 34:
            result.append(p)
        else:
            sub = [x.strip() for x in re.split(r"(?<=[，、])", p) if x.strip()]
            buf = ""
            for x in sub:
                if buf and len(buf) + len(x) > 32:
                    result.append(buf)
                    buf = x
                else:
                    buf += x
            if buf:
                result.append(buf)
    return result


def build_global_srt(durations: list[float], out: Path):
    lines = []
    idx = 1
    offset = 0.0
    for scene, dur in zip(SCENES, durations):
        clauses = split_clauses(scene.narration)
        weights = [max(3, len(c)) for c in clauses]
        total = sum(weights)
        cursor = offset + 0.12
        usable = max(0.1, dur - 0.24)
        for c, w in zip(clauses, weights):
            span = usable * w / total
            start, end = cursor, min(offset + dur - 0.06, cursor + span)
            lines.extend([str(idx), f"{srt_time(start)} --> {srt_time(end)}", c, ""])
            idx += 1
            cursor += span
        offset += dur
    out.write_text("\n".join(lines), encoding="utf-8-sig")


def write_narration(durations: list[float]):
    lines = ["# 研问智搜项目视频旁白", "", f"- 配音：{VOICE}", f"- 语速：{RATE}", ""]
    for i, (s, d) in enumerate(zip(SCENES, durations), 1):
        lines.extend([f"## {i:02d}. {s.title}（{d:.1f} 秒）", "", s.narration, ""])
    lines.append(f"总时长：{sum(durations):.1f} 秒")
    (VIDEO / "项目视频旁白.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    scene_dir = VIDEO / "scenes"
    audio_dir = VIDEO / "audio"
    seg_dir = VIDEO / "segments"
    for p in (scene_dir, audio_dir, seg_dir):
        p.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(SCENES, 1):
        build_scene_image(s, i, scene_dir / f"{s.key}.png")

    for s in SCENES:
        synthesize_sapi(s, audio_dir / f"{s.key}.{AUDIO_EXT}")
    durations = [duration(audio_dir / f"{s.key}.{AUDIO_EXT}") for s in SCENES]
    total = sum(durations)
    print("durations", durations, "total", total)
    if total >= 294:
        raise RuntimeError(f"Narration too long: {total:.1f}s")

    for s, dur in zip(SCENES, durations):
        run(
            [
                "ffmpeg", "-y", "-loop", "1", "-i", str(scene_dir / f"{s.key}.png"),
                "-i", str(audio_dir / f"{s.key}.{AUDIO_EXT}"),
                "-vf", "zoompan=z='min(zoom+0.00010,1.018)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1920x1080:fps=30,format=yuv420p",
                "-t", f"{dur:.3f}", "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-shortest", str(seg_dir / f"{s.key}.mp4"),
            ]
        )

    concat_txt = VIDEO / "concat.txt"
    concat_txt.write_text("\n".join(f"file '{(seg_dir / f'{s.key}.mp4').as_posix()}'" for s in SCENES), encoding="utf-8")
    raw = VIDEO / "研问智搜_项目视频_无字幕.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(raw)])

    srt = VIDEO / "研问智搜_项目视频字幕.srt"
    build_global_srt(durations, srt)
    write_narration(durations)

    final_video = FINAL / "虫虫特工队_研问智搜_项目视频.mp4"
    run(
        [
            "ffmpeg", "-y", "-i", str(raw),
            "-vf", "subtitles=研问智搜_项目视频字幕.srt:force_style='FontName=Microsoft YaHei,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H90000000,BackColour=&H60000000,BorderStyle=3,Outline=1,Shadow=0,MarginV=34,Alignment=2'",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-movflags", "+faststart", str(final_video),
        ],
        cwd=VIDEO,
    )

    shutil.copy2(srt, FINAL / "虫虫特工队_研问智搜_项目视频字幕.srt")
    shutil.copy2(VIDEO / "项目视频旁白.md", FINAL / "虫虫特工队_研问智搜_项目视频旁白.md")
    (VIDEO / "scene_manifest.json").write_text(
        json.dumps({"voice": VOICE, "rate": RATE, "durations": durations, "total": total, "scenes": [asdict(s) for s in SCENES]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(final_video)


if __name__ == "__main__":
    main()
