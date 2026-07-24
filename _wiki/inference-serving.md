---
layout: "wiki"
title: "Inference Serving"
description: "Inference serving covers latency, throughput, batching, KV cache, routing, quantization, and cost for running models in production。"
wiki_kind: "topic"
materials_count: 16
source_url: "http://agifrontier.duckdns.org:1684/ai-wiki/topic/inference-serving"
generated_at: "2026-07-24T16:30:35.569035+00:00"
published_at: "2026-07-24T16:30:35.569035+00:00"
last_modified_at: "2026-07-24T16:30:35.569035+00:00"
permalink: "/wiki/inference-serving/"
---
**一句话定义：** Inference serving covers latency, throughput, batching, KV cache, routing, quantization, and cost for running models in production.

## 页面状态

- 状态：`source-backed`
- 来源数量：16
- 更新方式：由 source-crawler 资料池生成，人工/LLM 综合写入 `当前综合`。

## 这是什么

Inference Serving 是 AI Wiki 中的一个长期知识节点。它不是一次性新闻，而是持续汇集官方博客、工程实践、newsletter、benchmark 和人物观点的主题页。

## 当前综合

- 这个主题已经有足够材料支撑第一版 wiki 页，适合继续把来源拆成概念、产品、人物和争议子页。
- 当前页面的结论应优先来自一手来源和研究/评测来源；专家观点可用于解释趋势，但不应替代原始事实。

## 为什么值得关注

这个主题同时出现在 16 条资料中，说明它已经跨越单篇文章，成为一个需要持续跟踪的知识簇。关键词包括：inference serving、LLM serving、vLLM、SGLang、KV cache、batching、quantization。

## 近期信号

- **Heterogeneous inference serving across three GPU vendors with llm-d \| l...**：来自 llm-d.ai link target，约 11869 字符。
- **How to easily migrate LLM inference serving from vLLM to Friendli Conta...**：来自 friendli.ai link target，约 5841 字符。
- **Grouped Query Attention (GQA) vs. Multi Head Attention (MHA): LLM Infer...**：来自 friendli.ai link target，约 8539 字符。
- **Iteration Batching (a.k.a. Continuous Batching): Accelerate LLM Inferen...**：来自 friendli.ai link target，约 4515 字符。
- **Why goodput matters more than throughput for LLM serving**：来自 www.cncf.io link target，约 11858 字符。
- **Prefill-Decode Disaggregation for LLM Serving at Scale**：来自 particula.tech link target，约 19249 字符。

## 关键问题

- **What is inference serving?** It is the production system layer that turns a model into an available, fast, measurable service.
- **Why track it in AI Wiki?** Serving determines product cost and responsiveness, and often explains why a better model is not the better product choice.

## 待追踪问题

- 哪些来源是一手事实，哪些只是围绕 inference serving 的二次解读？
- 这个主题的证据是否足以支撑对比页、指南页或 newsletter 选题？

## 来源覆盖

当前页面引用了 16 条资料，主要来自：friendli.ai link target 7 篇、Berkeley Sky ADRS 1 篇、Cohere Blog 1 篇、Together AI Blog 1 篇、aibrix.github.io link target 1 篇、llm-d.ai link target 1 篇、particula.tech link target 1 篇、pytorch.org link target 1 篇。

## 证据类型

| 类型 | 数量 | 阅读建议 |
| --- | ---: | --- |
| 一手来源 | 3 | 官方或研究机构来源，适合支撑模型发布、方法、产品和政策相关事实。 |
| 背景资料 | 13 | 可作为补充上下文，阅读时需要留意发布时间和来源权威性。 |

## 来源列表

### 一手来源

- **[LLM Serving Fairness](https://cohere.com/blog/serving-fairness)** — Cohere Blog，约 10855 字符
- **[Cache-aware prefill–decode disaggregation (CPD) for up to 40% faster long-context LLM serving](https://www.together.ai/blog/cache-aware-disaggregated-inference)** — Together AI Blog，约 14881 字符
- **[Automating Algorithm Discovery: A Case Study in Scheduler Design for Multi-LLM Serving Systems \| ADRS — AI-Driven Research for Systems](https://ucbskyadrs.github.io/blog/scheduler-design-for-multi-llm-serving-systems)** — Berkeley Sky ADRS，约 13869 字符
### 背景资料

- **[Heterogeneous inference serving across three GPU vendors with llm-d \| llm-d](https://llm-d.ai/blog/heterogeneous-inference-3-vendor-sovereign-cluster)** — llm-d.ai link target，约 11869 字符
- **[How to easily migrate LLM inference serving from vLLM to Friendli Container.](https://friendli.ai/blog/migrating-vllm-friendli-container)** — friendli.ai link target，约 5841 字符
- **[Grouped Query Attention (GQA) vs. Multi Head Attention (MHA): LLM Inference Serving Acceleration](https://friendli.ai/blog/gqa-vs-mha)** — friendli.ai link target，约 8539 字符
- **[Iteration Batching (a.k.a. Continuous Batching): Accelerate LLM Inference Serving with Flexible Scheduling](https://friendli.ai/blog/llm-iteration-batching)** — friendli.ai link target，约 4515 字符
- **[Why goodput matters more than throughput for LLM serving](https://www.cncf.io/blog/2026/07/20/why-goodput-matters-more-than-throughput-for-llm-serving/)** — www.cncf.io link target，约 11858 字符
- **[Prefill-Decode Disaggregation for LLM Serving at Scale](https://particula.tech/blog/prefill-decode-disaggregation-llm-serving-scale)** — particula.tech link target，约 19249 字符
- **[SMG: The Case for Disaggregating CPU from GPU in LLM Serving](https://pytorch.org/blog/lightseek-smg/)** — pytorch.org link target，约 17213 字符
- **[PrisKV: A Colocated Tiered KVCache Store for LLM Serving](https://aibrix.github.io/posts/2025-11-26-priskv-intro/)** — aibrix.github.io link target，约 19439 字符
- **[LLM Serving Frameworks](https://www.hyperbolic.ai/blog/llm-serving-frameworks)** — www.hyperbolic.ai link target，约 26558 字符
- **[Friendli TCache: Optimizing LLM Serving by Reusing Computations](https://friendli.ai/blog/friendli-tcache)** — friendli.ai link target，约 5977 字符
- **[LLM Serving Engine Comparative Analysis: Friendli Inference vs. vLLM vs. TensorRT-LLM](https://friendli.ai/blog/friendli-engine-tensorrt-llm-vllm)** — friendli.ai link target，约 7430 字符
- **[Groundbreaking Performance of the Friendli Inference for LLM Serving on an NVIDIA H100 GPU](https://friendli.ai/blog/friendli-engine-llm-serving-nvidia-h100)** — friendli.ai link target，约 7017 字符
- **[Comparing two LLM serving frameworks: Friendli Inference vs. vLLM](https://friendli.ai/blog/comparing-friendli-engine-vllm)** — friendli.ai link target，约 6041 字符
## 相关页面

- **[Inference Serving](http://agifrontier.duckdns.org:1684/ai-wiki/entity/inference-serving)**
- **[vLLM](http://agifrontier.duckdns.org:1684/ai-wiki/entity/vllm)**
- **[LMSYS / SGLang](http://agifrontier.duckdns.org:1684/ai-wiki/entity/sglang)**
- **[AI Infrastructure](/wiki/ai-infra/)**
