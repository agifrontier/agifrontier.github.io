#!/usr/bin/env python3
"""Backfill deterministic SEO metadata and content fixes for tutorials."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TUTORIALS_DIR = ROOT / "_tutorials"
MANAGED_FIELDS = {"description", "topics", "related_tutorials"}
BROKEN_IMAGE_URLS = {
    "https://s2.loli.net/2024/09/20/ChYxXz3pU9Jqf5T.png",
    "https://arxiv.org/html/2502.02985/x1.png",
    "https://image.uc.cn/s/wemedia/s/upload/2024/762c95e54d3d75865a7f9202534f37fa.png",
    "https://ppt-arena.onrender.com/static/images/teaser.png",
    "https://arxiv.org/html/2511.08128v1/S4.T2",
    "https://arxiv.org/html/2501.00895v1/extracted/6106660/figures/arch_diagram.png",
    "https://raw.githubusercontent.com/wylAImoreira/img-bed/main/202405231718919.png",
}
BROKEN_IMAGE_PATHS = {
    "figures/minigrid_doorkeys_300k.png",
    "figures/minigrid_hard_300k.png",
    "figures/metaworld_results.png",
    "latex/Figures/model_performance_over_rounds.pdf",
    "latex/Figures/cooperation_rates_all_strategies.pdf",
    "latex/Figures/recovery_curves_all_strategies.pdf",
    "latex/Figures/multiple_condition_overlay.pdf",
    "imagese/2512.01335v1/x1.jpg",
    "x1.png",
    "imagese/2511.18870v2/x3.jpg",
    "acl_latex/imgs/framework.png",
    "acl_latex/imgs/query.png",
    "acl_latex/imgs/case_SearchR1.png",
    "acl_latex/imgs/upper_gain_v3.png",
    "acl_latex/imgs/info.png",
    "imagese/2508.19740v1/x2.png",
    "imagese/2508.19740v1/x9.png",
}
GENERIC_HEADINGS = {
    "tl;dr",
    "关键定义",
    "相关工作",
    "本文方法",
    "实验结论",
    "总结",
    "总结与展望",
    "结论",
}
TOPIC_RULES = [
    ("具身智能与机器人", r"\b(robot|robotics|embodied|manipulation|slam|vision.language.action|vla|autonomous.driving)\b|具身|机器人|自动驾驶|运动控制"),
    ("多模态与视觉", r"\b(multimodal|vision|visual|image|video|audio|ocr|vlm|diffusion)\b|多模态|视觉|图像|视频|语音"),
    ("RAG与知识系统", r"\b(rag|retrieval|rerank|embedding|knowledge.graph|knowledge.base|memory)\b|检索|知识图谱|知识库|记忆"),
    ("AI Agent", r"\b(agent|agents|agentic|multi.agent|tool.use|computer.use)\b|智能体|代理"),
    ("推理与强化学习", r"\b(reasoning|reinforcement.learning|rlvr|grpo|dpo|preference|reward.model|planning)\b|推理|强化学习|偏好优化|奖励模型"),
    ("AI安全与评测", r"\b(safety|security|privacy|benchmark|evaluation|hallucination|judge|alignment|jailbreak)\b|安全|隐私|评测|幻觉|对齐"),
    ("模型训练与优化", r"\b(training|fine.tun|post.training|pretrain|optimization|optimizer|lora|quantization|distillation|model.merging|sparse|attention|moe|inference)\b|训练|微调|量化|蒸馏|稀疏|注意力|推理优化"),
    ("数据与AI工程", r"\b(data|dataset|platform|infra|serving|deployment|cuda|gpu|kernel|database|software.engineering)\b|数据|平台|部署|工程化|算子"),
    ("行业应用", r"\b(medical|medicine|health|education|finance|scientific|recommendation|search|advertising)\b|医疗|教育|金融|科研|推荐|广告"),
]
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "based", "by", "for", "from", "in", "is",
    "large", "language", "llm", "llms", "model", "models", "of", "on", "the", "to",
    "toward", "towards", "via", "with",
}
MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
HTML_IMAGE_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
ALT_RE = re.compile(r"\balt\s*=\s*([\"'])(.*?)\1", re.IGNORECASE | re.DOTALL)
TITLE_RE = re.compile(r"^title:\s*(.+?)\s*$")
HEADING_RE = re.compile(r"^#{2,6}\s+(.+?)\s*$")


@dataclass
class Tutorial:
    path: Path
    frontmatter_lines: list[str]
    body: str
    title: str
    slug: str
    description: str = ""
    topics: tuple[str, ...] = ()
    related: tuple[str, ...] = ()


def split_document(text: str) -> tuple[list[str], str]:
    if not text.startswith("---\n"):
        raise ValueError("missing frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("unterminated frontmatter")
    return text[4:end].splitlines(), text[end + 5 :]


def yaml_scalar(value: str) -> str:
    value = value.strip()
    if value.startswith(("\"", "'")) and value.endswith(value[0]):
        try:
            return json.loads(value) if value[0] == '"' else value[1:-1].replace("''", "'")
        except json.JSONDecodeError:
            return value[1:-1]
    return value


def load_tutorial(path: Path) -> Tutorial:
    frontmatter, body = split_document(path.read_text(encoding="utf-8"))
    title = ""
    for line in frontmatter:
        match = TITLE_RE.match(line)
        if match:
            title = yaml_scalar(match.group(1))
            break
    if not title:
        raise ValueError(f"missing title: {path}")
    return Tutorial(path=path, frontmatter_lines=frontmatter, body=body, title=title, slug=path.stem)


def clean_text(value: str) -> str:
    value = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[`*_>#]", "", value)
    value = re.sub(r"\$+[^$]*\$+", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip(" -:：。")


def truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    shortened = value[: limit - 1].rstrip("，,。.;；:： ")
    return shortened + "。"


def extract_description(tutorial: Tutorial) -> str:
    lines = tutorial.body.splitlines()
    heading_candidate = ""
    paragraph_parts: list[str] = []
    collecting = False

    for raw_line in lines:
        line = raw_line.strip()
        heading = HEADING_RE.match(line)
        if heading:
            cleaned_heading = clean_text(heading.group(1))
            if cleaned_heading.casefold() not in GENERIC_HEADINGS and len(cleaned_heading) >= 12:
                heading_candidate = cleaned_heading
                break

        if not line:
            if collecting and paragraph_parts:
                break
            continue
        if (
            "arxiv url" in line.casefold()
            or "http://arxiv.org" in line.casefold()
            or line.startswith(("#", "!", "<img", "<p", "|", "```", "$$"))
            or re.match(r"^[-*+]\s+", line)
            or re.match(r"^\d+[.)]\s+", line)
        ):
            continue
        cleaned = clean_text(line)
        if len(cleaned) < 12:
            continue
        paragraph_parts.append(cleaned)
        collecting = True
        if sum(len(part) for part in paragraph_parts) >= 90:
            break

    candidate = heading_candidate or " ".join(paragraph_parts)
    title = clean_text(tutorial.title)
    if len(candidate) < 50:
        if candidate:
            candidate = f"{candidate}。本文系统梳理其研究背景、核心方法、关键实验结果、现有局限以及后续工程实践启示。"
        else:
            candidate = f"本文解读《{title}》，梳理核心方法、关键实验结果与工程实践启示。"
    elif candidate[-1] not in "。！？.!?":
        candidate += "。"
    return truncate(candidate, 155)


def classify_topics(title: str) -> tuple[str, ...]:
    normalized = title.casefold().replace("-", " ")
    topics = [name for name, pattern in TOPIC_RULES if re.search(pattern, normalized, re.IGNORECASE)]
    if not topics:
        topics = ["基础模型与理论"]
    return tuple(topics[:2])


def title_tokens(title: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", title.casefold())
        if len(token) > 2 and token not in STOPWORDS
    }


def assign_related(tutorials: list[Tutorial]) -> None:
    document_frequency: Counter[str] = Counter()
    tokens_by_slug: dict[str, set[str]] = {}
    for tutorial in tutorials:
        tokens = title_tokens(tutorial.title)
        tokens_by_slug[tutorial.slug] = tokens
        document_frequency.update(tokens)

    total = len(tutorials)
    for tutorial in tutorials:
        source_tokens = tokens_by_slug[tutorial.slug]
        source_topics = set(tutorial.topics)
        candidates: list[tuple[float, str]] = []
        for candidate in tutorials:
            if candidate.slug == tutorial.slug:
                continue
            shared_tokens = source_tokens & tokens_by_slug[candidate.slug]
            shared_topics = source_topics & set(candidate.topics)
            if not shared_tokens and not shared_topics:
                continue
            token_score = sum(math.log((total + 1) / (document_frequency[token] + 1)) + 1 for token in shared_tokens)
            score = token_score + (4.0 * len(shared_topics))
            candidates.append((score, candidate.slug))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        tutorial.related = tuple(slug for _score, slug in candidates[:4])


def contextual_alt(title: str, heading: str) -> str:
    context = clean_text(heading) or clean_text(title)
    return truncate(f"{context} 图示", 80).replace('"', "&quot;")


def fix_body(tutorial: Tutorial) -> tuple[str, dict[str, int]]:
    lines: list[str] = []
    heading = ""
    stats = {"arxiv_https": 0, "empty_alt": 0, "missing_alt": 0, "broken_images": 0}

    for raw_line in tutorial.body.splitlines():
        has_broken_url = any(target in raw_line for target in BROKEN_IMAGE_URLS)
        has_broken_path = any(f"]({target})" in raw_line for target in BROKEN_IMAGE_PATHS)
        if has_broken_url or has_broken_path:
            stats["broken_images"] += 1
            continue

        stats["arxiv_https"] += raw_line.count("http://arxiv.org/")
        line = raw_line.replace("http://arxiv.org/", "https://arxiv.org/")
        heading_match = HEADING_RE.match(line.strip())
        if heading_match:
            heading = heading_match.group(1)
        alt = contextual_alt(tutorial.title, heading)

        def replace_markdown_image(match: re.Match[str]) -> str:
            current_alt, target = match.groups()
            if current_alt.strip():
                return match.group(0)
            stats["empty_alt"] += 1
            return f"![{alt}]({target})"

        line = MARKDOWN_IMAGE_RE.sub(replace_markdown_image, line)

        def replace_html_image(match: re.Match[str]) -> str:
            tag = match.group(0)
            alt_match = ALT_RE.search(tag)
            if alt_match:
                if alt_match.group(2).strip():
                    return tag
                stats["empty_alt"] += 1
                return ALT_RE.sub(f'alt="{alt}"', tag, count=1)
            stats["missing_alt"] += 1
            return tag[:-1].rstrip() + f' alt="{alt}">'

        line = HTML_IMAGE_RE.sub(replace_html_image, line)
        lines.append(line)

    body = "\n".join(lines)
    if tutorial.body.endswith("\n"):
        body += "\n"
    return body, stats


def remove_managed_fields(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    index = 0
    while index < len(lines):
        key = lines[index].split(":", 1)[0].strip() if ":" in lines[index] else ""
        if key not in MANAGED_FIELDS:
            cleaned.append(lines[index])
            index += 1
            continue
        index += 1
        while index < len(lines) and (lines[index].startswith((" ", "\t")) or not lines[index].strip()):
            index += 1
    return cleaned


def render(tutorial: Tutorial, body: str) -> str:
    frontmatter = remove_managed_fields(tutorial.frontmatter_lines)
    title_index = next(index for index, line in enumerate(frontmatter) if TITLE_RE.match(line))
    managed = [
        f"description: {json.dumps(tutorial.description, ensure_ascii=False)}",
        "topics:",
        *[f"  - {json.dumps(topic, ensure_ascii=False)}" for topic in tutorial.topics],
        "related_tutorials:",
        *[f"  - {json.dumps(slug, ensure_ascii=False)}" for slug in tutorial.related],
    ]
    frontmatter[title_index + 1 : title_index + 1] = managed
    return "---\n" + "\n".join(frontmatter) + "\n---\n" + body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Apply changes; default is check-only")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    tutorials = [load_tutorial(path) for path in sorted(TUTORIALS_DIR.glob("*.md"))]
    descriptions: set[str] = set()
    for tutorial in tutorials:
        description = extract_description(tutorial)
        if description in descriptions:
            description = truncate(f"{description.rstrip('。')}：{clean_text(tutorial.title)}。", 155)
        descriptions.add(description)
        tutorial.description = description
        tutorial.topics = classify_topics(tutorial.title)
    assign_related(tutorials)

    totals: Counter[str] = Counter()
    changed_files: list[str] = []
    for tutorial in tutorials:
        body, stats = fix_body(tutorial)
        totals.update(stats)
        expected = render(tutorial, body)
        current = tutorial.path.read_text(encoding="utf-8")
        if current != expected:
            changed_files.append(str(tutorial.path.relative_to(ROOT)))
            if args.write:
                tutorial.path.write_text(expected, encoding="utf-8")

    report = {
        "tutorials": len(tutorials),
        "changed_files": len(changed_files),
        "unique_descriptions": len(descriptions),
        "topic_counts": dict(Counter(topic for item in tutorials for topic in item.topics).most_common()),
        "content_fixes": dict(totals),
        "files": changed_files,
    }
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, ensure_ascii=False, indent=2))
    return 0 if args.write or not changed_files else 1


if __name__ == "__main__":
    raise SystemExit(main())
