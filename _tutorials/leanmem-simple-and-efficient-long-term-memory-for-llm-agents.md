---
layout: default
title: "LeanMem：告别盲目摘要，三级异构记忆让Agent准确率提升15.1点"
description: "针对这种“高开销”与“信息丢失”的死锁，合肥工业大学的研究团队提出了轻量级长期记忆框架 LeanMem。该方案的核心洞察在于： 人类对话产生的信息天然具有异构性，试图用单一、均匀的“摘要-检索”流水线去处理所有对话历史，在底层逻辑上就是行不通的。"
arxiv_id: "2608.03463"
paper_published: "2026-08-04"
published_at: "2026-09-09T13:15:08.606362+08:00"
topics:
  - "知识系统"
  - "AI Agent"
tags:
  - "Dynamic retrieval budgeting"
  - "Event memory"
  - "LLM agents"
  - "LeanMem"
  - "LoCoMo"
  - "LongMemEval-S"
related_tutorials:
  - "lycheememory-v2-efficient-long-term-memory-for-llm-agents-via-semantic-segment-l"
  - "agentic-memory-learning-unified-long-term-and-short-term-memory-management-for-l"
  - "filesystem-based-memory-for-llm-agents-organization-evolution-and-sustainability"
  - "from-passive-retrieval-to-active-memory-navigation-learning-to-use-memory-as-a-structured-action"
---

<p class="paper-original-title" lang="en">LeanMem: Simple and Efficient Long-Term Memory for LLM Agents</p>

<img src="/images/2608.03463v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在构建能够长期陪伴、跨会话执行复杂任务的大语言模型智能体（LLM Agent）时，记忆系统始终是一道难以逾越的工程与理论门槛。随着上下文窗口的物理限制以及模型注意力在超长序列中的衰减，赋予 Agent 持久化外部记忆几乎成了行业共识。然而，当前主流的记忆框架大多陷入了一种两难困境：要么为了保留信息而对每一次交互都调用大模型进行全文重写与全量摘要，导致记忆构建成本居高不下；要么为了控制成本进行激进压缩，导致代码片段、清单细节等细粒度证据永久丢失。

> ArXiv URL：https://arxiv.org/abs/2608.03463v1

针对这种“高开销”与“信息丢失”的死锁，合肥工业大学的研究团队提出了轻量级长期记忆框架 LeanMem。该方案的核心洞察在于：**人类对话产生的信息天然具有异构性，试图用单一、均匀的“摘要-检索”流水线去处理所有对话历史，在底层逻辑上就是行不通的。**

对话中的稳定事实、动态事件与密集细节，在可压缩性、时间动态性和保真度需求上存在巨大差异。LeanMem 通过受控写入、选择性演进以及自适应证据组装三套机制，实现了记忆表示的三级分流。实验表明，在 LoCoMo 与 LongMemEval-S 两个长程多轮评测基准上，LeanMem 面对 GPT-4.1-mini 与开源模型 Qwen3-8B，均以极低的 Token 消耗打破了现有最强记忆基线的记录，准确率最高提升达 15.1 个百分点。

<img src="/images/2608.03463v1/intro1.webp" alt="对话信息在可压缩性与存储需求上的差异" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 为什么均匀摘要会成为长期记忆的死胡同？

现有 Agent 记忆框架通常将对话历史机械化地切分成 Chunk，然后调用大模型对每个切片进行信息抽取、总结并写入向量数据库。这种设计看似规整，但在实际多轮交互中暴露出严重的效率和保真度矛盾。

正如论文起步处指出的那样，对话信息并不是质地均一的面团。第一类信息是关于用户画像与偏好的稳定属性，例如用户的饮食习惯、编程语言偏好或家庭成员构成。这类信息高度可压缩，一旦提取出来几乎长期不变，不需要日后反复修改。第二类信息是具有演进特质的状态事件，例如“正在准备毕业论文”、“开题报告通过”、“下周进行预答辩”。这类事件高度依赖时间锚点，其有效性会随时间推移被更新或覆盖。第三类信息则是密集的技术细节、任务清单或分步指导，例如一段报错日志、旅行路线具体班次。这类内容对保真度要求极高，任何由大模型生成的模糊抽象都会导致关键事实的永久丢失。

如果采用统一的摘要压缩，系统为了防止丢失细节，往往在构建时频繁唤醒大模型反复总结，消耗数以百万计的构建 Token；如果进行强行压缩，面对未来的精确回溯查询时，又只能在推理阶段通过迭代检索、多跳反思等高昂手段试图“猜”回细节。构建侧的过度投入与检索侧的疲软补救，让记忆系统陷入了越优化越臃肿的恶性循环。

### LeanMem 核心机制：异构写入与按需演化

LeanMem 的破局思路非常克制：**只为合适的数据提供合适精度的容器，不该存的丢弃，该压缩的紧凑存储，不能压缩的保留指针。** 整个框架分为受控记忆写入、选择性记忆演进和自适应证据组装三个阶段。

<img src="/images/2608.03463v1/llllll.webp" alt="LeanMem 整体架构" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在记忆构建的第一步，LeanMem 并不急于让大模型阅读整个对话，而是先执行受控记忆写入（Controlled Memory Writing）。系统首先通过轻量规则过滤掉寒暄、礼貌应答等非实质性内容，并将对话按照语义切分为连贯的主题段落（Topic Segmentation）。在人机不对称对话中，用户的输入往往是驱动状态转移和提出需求的核心，因此系统以用户的话语作为切分段落的主要锚点。

紧接着，写入调度器会根据信息特征将段落分流至三种异构存储形态中：

1. **画像记忆（Profile Memory）**：针对稳定的用户偏好与属性，抽取为极简的三元组或键值对，占用极低空间且在后续流程中被视为只读或低频更新项。

2. **事件记忆（Event Memory）**：针对涉及动态进程的信息，形式化抽取为包含事件主题 $e_i$、时间锚点 $t_i$ 以及当前状态描述 $z_i$ 的三元结构 $\langle e_i, t_i, z_i \rangle$。

3. **记录记忆（Record Memory）**：针对包含复杂长文本细节的段落，LeanMem 放弃了大模型的自由总结，转而使用轻量级命名实体识别模型抽取实体关键词 $\kappa_i$，配合一句超简短的主旨 $g_i$，同时最关键的是保存一个直接指向原始对话切片物理位置的源指针 $p_i = I_i$。这样一来，高密度事实无需在记忆库中冗余存储，却保留了随时回溯原文上下文的能力。

在记忆库维护层面，传统方案往往定期执行全局记忆合并（Memory Consolidation），把所有记忆重新喂给大模型重写一遍，带来巨大的幻觉风险和计算浪费。LeanMem 提出了选择性记忆演进（Selective Memory Evolution）。系统明确限定：Profile 记忆是稳定的，无需演进；Record 记忆是指向历史事实的不可变快照，同样禁止演进。整个演进机制完全且唯一作用于 Event 记忆。当新的事件产生时，系统仅在局域范围内检索相同主题的历史事件，按时间戳将其串联成一条逻辑清晰的状态变迁链条，在写入阶段就完成了局部状态更新，从而免除了在推理阶段执行全局多跳整合的计算负担。

最后在推理使用层面，面对用户的提问 $q$，系统并不采用机械化的全局 Top-$k$ 向量检索，而是通过轻量规划器分析该查询的证据需求，形成检索计划 $\pi_q$。如果问题问的是用户基本偏好，则优先调取 Profile 记忆；如果涉及某个事件的最新进展，则定向检索 Event 记忆链条；如果涉及细粒度技术参数，则利用 Record 记忆的关键词索引快速定位并拉取底层原始对话段落。

<img src="/images/2608.03463v1/case123new.webp" alt="三类记忆的写入与使用案例" style="width:min(600px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; display:block;">

### 实验评测：准确率与运行开销的双重跨越

为了验证异构记忆分流的有效性，研究团队在涵盖丰富推理类型的 LoCoMo 数据集以及长达 40 至 50 个 Session、平均文本量达 115K Token 的超长对话基准 LongMemEval-S 上进行了深入评测，底座模型分别选用了闭源的 GPT-4.1-mini 和开源的 Qwen3-8B。

实验数据表明，LeanMem 在所有评测设置中，均取得了优于当前所有显式记忆基线系统的表现。在 LoCoMo 基准上，以 GPT-4.1-mini 为底座，LeanMem 取得了 84.87 的问答准确率与 83.80 的检索召回率，相比表现第二名的强基线系统，准确率提升了 5.54 个百分点；在更具挑战性、更考察超长程状态追踪的 LongMemEval-S 基准上，LeanMem 的准确率更是直接拉开了 15.07 个百分点的巨大差距。即便切换到参数量仅为 8B 的开源模型 Qwen3-8B，这种优势依然稳固，准确率在两个数据集上分别取得了 5.84 和 2.80 个百分点的提升。

更为关键的是效率指标。以往记忆系统提升准确率的代价往往是极其昂贵的上下文消耗。例如代表性的图记忆方案 A-Mem 在 LongMemEval-S 上的记忆构建 Token 超过了 120 万，而 LeanMem 仅花费了 11.7 万 Token，构建开销缩减了 17.24 倍。同时，得益于精准的证据组装与源指针机制，LeanMem 单次查询的推理 Token 消耗降低了 8.41 倍，端到端延迟降至 2.45 秒，展现了在生产级落地场景中的极强可用性。

<img src="/images/2608.03463v1/locomo1.webp" alt="LoCoMo 各问题类型上的准确率表现" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

<img src="/images/2608.03463v1/long1.webp" alt="LongMemEval-S 上的准确率表现" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

细分到具体的问题类别来看，异构存储的优势更加直观。在 LoCoMo 的细分评测中，LeanMem 在单跳事实问题（Single-Hop）上提升了 10.57 个百分点，在多跳关联问题（Multi-Hop）上提升了 12.02 个百分点。单跳问题的飞跃得益于 Profile 和 Record 记忆的高保真存储，避免了模型在被抽象过的模糊摘要中胡乱猜测；而多跳与时间问题（Temporal）的跃升，则直接验证了 Event 记忆带时序演进设计的有效性。

而在 LongMemEval-S 这种跨越数十个会话的超长评测中，针对跨会话事实整合（Multi-Session）、动态知识更新（Knowledge Update）和用户偏好追踪，LeanMem 的准确率均大幅抛离基线。这一结果印证了一个重要推论：**长期记忆系统不需要无死角地记下所有抽象语义，只要把状态演变理顺、把高精细节留好回溯指针，大模型在推理时就能像查阅组织良好的档案库一样迅速得出正确结论。**

### 模块消融与设计反思：为什么“做减法”反而带来提升？

在论文的消融实验部分，研究团队系统性剥离了 LeanMem 的各个组件，揭示了不同设计对整体效果的支撑强度。

当研究人员移除了记忆存储调度器（w/o Memory Storage Scheduling），强行让所有对话内容都退化为单一的传统文本摘要时，系统在两个基准上的准确率出现了最为剧烈的断崖式下跌，同时 Token 消耗显著反弹。这直接证明了“根据信息属性分类定级”是 LeanMem 能够超越同类方案的最核心支柱。

移除事件演进机制（w/o Event Memory Evolution）对超长会话数据集 LongMemEval-S 的打击尤为致命。如果不对具有时间延续性的事件进行局部的状态合并，相关的状态描述就会碎裂散落在不同会话周期的记忆条目中，迫使检索阶段去搜刮海量碎片，不仅推理 Token 剧增，还会因信息冲突引发模型幻觉。

此外，如果去掉提问阶段的自适应证据规划（w/o Retrieval Planning），退化回常规 RAG 常用的固定 Top-$k$ 嵌入检索，准确率同样发生明显缩水。这说明提问本身的意图也是异构的：询问用户习惯时塞入事件链条只会引入噪声，而询问执行步骤时给几个抽象事实同样无法作答，检索必须与证据类型相匹配。

### 对 Agent 记忆架构的工程启示

长期以来，很多针对大模型记忆的研究往往倾向于建立极其宏大复杂的图谱，或者试图让模型不断进行递归总结以达到全知全能的理想状态。然而这类尝试在落地时往往被惊人的 API 账单、难以控制的累积幻觉以及漫长的端到端延迟所拖垮。

LeanMem 的工作给出了一种极具实用主义色彩的解题范式：

- 长期记忆工程的第一准则应当是“抑制不必要的写入与更新”，而不是“全面覆盖”。

- 对话中的绝大多数细节根本不需要也不应该被模型总结，留下结构化关键词和指向原始对话的指针，既保护了原始保真度，又免除了高昂的摘要重构开销。

- 将动态推演收拢在极少数具有状态属性的事件节点上，才能在长期运行中把系统维护复杂度控制在可接受的线性甚至近亚线性级别。

对于正在探索具身智能、企业级个人助手和跨会话工作流 Agent 的开发者而言，LeanMem 证明了轻量化与高保真并不冲突。通过更敏锐地理解信息本身的特征，在架构上做精确的分流与克制的设计，大模型在长程交互中的表现完全可以做到既敏捷又精准。
