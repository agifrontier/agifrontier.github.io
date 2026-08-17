#!/usr/bin/env python3
"""Canonical topic taxonomy and deterministic classification rules."""

from __future__ import annotations

import re
from dataclasses import dataclass


TOPIC_NAMES = (
    "AI Agent",
    "RAG",
    "知识系统",
    "推理",
    "强化学习",
    "模型训练",
    "模型优化",
    "多模态&视觉",
    "具身智能",
    "AI安全",
    "AI评测",
    "数据工程",
    "AI工程",
    "行业应用",
    "基础模型",
    "AI理论",
)


@dataclass(frozen=True)
class TopicRule:
    name: str
    pattern: str


TOPIC_RULES = (
    TopicRule(
        "具身智能",
        r"\b(robot|robotics|embodied|manipulation|slam|vision.language.action|vla|autonomous.driving)\b|具身|机器人|自动驾驶|运动控制",
    ),
    TopicRule(
        "RAG",
        r"\b(rag|retrieval|retriever|rerank|embedding|information.seeking)\b|检索增强|检索器|重排序|信息检索",
    ),
    TopicRule(
        "知识系统",
        r"\b(knowledge.graph|knowledge.base|memory|memorization|long.term.memory)\b|知识图谱|知识库|知识系统|长期记忆|智能体记忆",
    ),
    TopicRule(
        "AI Agent",
        r"\b(agent|agents|agentic|multi.agent|tool.use|computer.use|web.agent|gui.agent)\b|智能体|代理系统|工具调用",
    ),
    TopicRule(
        "强化学习",
        r"\b(reinforcement.learning|rlhf|rlvr|grpo|dpo|ppo|policy.learning|preference.optim|reward.model)\b|强化学习|偏好优化|奖励模型|策略学习",
    ),
    TopicRule(
        "推理",
        r"\b(reasoning|chain.of.thought|cot|planning|search|test.time.compute)\b|推理|思维链|规划|测试时计算",
    ),
    TopicRule(
        "AI安全",
        r"\b(safety|security|privacy|hallucination|alignment|jailbreak|adversarial|trustworthy)\b|安全|隐私|幻觉|对齐|越狱|可信",
    ),
    TopicRule(
        "AI评测",
        r"\b(benchmark|evaluation|evaluate|judge|leaderboard|metric|assessment)\b|基准测试|评测|评价指标|裁判模型",
    ),
    TopicRule(
        "多模态&视觉",
        r"\b(multimodal|multi.modal|cross.modal|vlm|mllm|audio.language|vision.language|computer.vision|vision|visual|image|video|ocr|diffusion|segmentation|detection)\b|多模态|跨模态|视觉语言模型|计算机视觉|视觉|图像|视频|目标检测|分割",
    ),
    TopicRule(
        "模型训练",
        r"\b(train(?:ing)?|fine[ ._-]?tun(?:e|ing)|finetun(?:e|ing)|post[ ._-]?train(?:ing)?|pre[ ._-]?train(?:ing)?|sft|lora|instruction[ ._-]?tun(?:e|ing))\b|训练|微调|预训练|后训练|指令调优",
    ),
    TopicRule(
        "模型优化",
        r"\b(optimization|optimizer|quantization|distillation|pruning|sparse|attention|moe|model.merging|compression|inference.optim|speculative.decod)\b|优化器|量化|蒸馏|剪枝|稀疏|注意力|模型合并|压缩|推理优化|投机解码",
    ),
    TopicRule(
        "数据工程",
        r"\b(data|dataset|database|data.curation|data.synthesis|synthetic.data|annotation)\b|数据集|数据库|数据治理|数据合成|数据标注",
    ),
    TopicRule(
        "AI工程",
        r"\b(platform|infrastructure|infra|serving|deployment|cuda|gpu|kernel|software.engineering|system)\b|平台|基础设施|部署|服务化|算子|软件工程|系统工程",
    ),
    TopicRule(
        "行业应用",
        r"\b(medical|medicine|health|education|finance|scientific|recommendation|advertising|legal|manufacturing)\b|医疗|教育|金融|科研|推荐|广告|法律|制造",
    ),
    TopicRule(
        "AI理论",
        r"\b(theory|theoretical|theorem|bound|convergence|expressivity|complexity|proof)\b|理论|定理|界限|收敛性|表达能力|复杂度|证明",
    ),
)

RULES_BY_NAME = {rule.name: rule for rule in TOPIC_RULES}


LEGACY_TOPIC_OPTIONS = {
    "AI Agent": (("AI Agent", None),),
    "RAG与知识系统": (
        ("RAG", RULES_BY_NAME["RAG"].pattern),
        ("知识系统", RULES_BY_NAME["知识系统"].pattern),
    ),
    "推理与强化学习": (
        ("强化学习", RULES_BY_NAME["强化学习"].pattern),
        ("推理", RULES_BY_NAME["推理"].pattern),
    ),
    "模型训练与优化": (
        ("模型训练", RULES_BY_NAME["模型训练"].pattern),
        ("模型优化", RULES_BY_NAME["模型优化"].pattern),
    ),
    "多模态与视觉": (("多模态&视觉", None),),
    "多模态": (("多模态&视觉", None),),
    "计算机视觉": (("多模态&视觉", None),),
    "具身智能与机器人": (("具身智能", None),),
    "AI安全与评测": (
        ("AI安全", RULES_BY_NAME["AI安全"].pattern),
        ("AI评测", RULES_BY_NAME["AI评测"].pattern),
    ),
    "数据与AI工程": (
        ("数据工程", RULES_BY_NAME["数据工程"].pattern),
        ("AI工程", RULES_BY_NAME["AI工程"].pattern),
    ),
    "行业应用": (("行业应用", None),),
    "基础模型与理论": (
        ("AI理论", RULES_BY_NAME["AI理论"].pattern),
    ),
}

LEGACY_TOPIC_FALLBACKS = {
    "AI Agent": "AI Agent",
    "RAG与知识系统": "RAG",
    "推理与强化学习": "推理",
    "模型训练与优化": "模型训练",
    "多模态与视觉": "多模态&视觉",
    "多模态": "多模态&视觉",
    "计算机视觉": "多模态&视觉",
    "具身智能与机器人": "具身智能",
    "AI安全与评测": "AI安全",
    "数据与AI工程": "AI工程",
    "行业应用": "行业应用",
    "基础模型与理论": "基础模型",
}


def matches(pattern: str | None, evidence: str) -> bool:
    return pattern is not None and re.search(pattern, evidence, re.IGNORECASE) is not None


def classify_topics(title: str) -> tuple[str, ...]:
    """Classify a new tutorial title into at most two independent topics."""
    normalized = title.casefold().replace("-", " ")
    topics = [
        rule.name
        for rule in TOPIC_RULES
        if re.search(rule.pattern, normalized, re.IGNORECASE)
    ]
    return tuple(topics[:2] or ["基础模型"])


def migrate_legacy_topics(existing_topics: tuple[str, ...], evidence: str) -> tuple[str, ...]:
    """Split legacy compound topics while retaining at most two topics per article."""
    primary_topics: list[str] = []
    secondary_topics: list[str] = []

    for legacy_topic in existing_topics:
        if legacy_topic in TOPIC_NAMES:
            options = ((legacy_topic, None),)
            fallback = legacy_topic
        else:
            options = LEGACY_TOPIC_OPTIONS.get(legacy_topic)
            fallback = LEGACY_TOPIC_FALLBACKS.get(legacy_topic)
        if options is None:
            raise ValueError(f"unknown legacy topic: {legacy_topic}")
        if fallback is None:
            raise ValueError(f"missing fallback for legacy topic: {legacy_topic}")

        matched = [name for name, pattern in options if matches(pattern, evidence)]
        primary = matched[0] if matched else fallback
        if primary not in primary_topics:
            primary_topics.append(primary)
        for name in matched[1:]:
            if name not in primary_topics and name not in secondary_topics:
                secondary_topics.append(name)

    selected = primary_topics[:2]
    for topic in secondary_topics:
        if len(selected) >= 2:
            break
        if topic not in selected:
            selected.append(topic)
    return tuple(selected or ["基础模型"])
