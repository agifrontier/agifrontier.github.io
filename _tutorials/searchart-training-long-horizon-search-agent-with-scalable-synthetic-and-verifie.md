---
layout: default
title: "SearchArt：华为提出长程搜索智能体训练框架，27B模型多项基准比肩顶尖闭源Agent"
description: "大语言模型（LLM）驱动的搜索智能体（SearchAgent）正从简单的单轮关键词检索，演进为需要多轮规划、长程探索与交叉验证的深度推理系统。然而，训练一个真正可用的长程搜索智能体始终面临结构性困难：现实中极度缺乏高质量、长周期的搜索与研究任务数据，人工标注完整轨迹成本高昂，且模型在多步工具调用过程中的幻觉与。"
arxiv_id: "2607.24850"
published_at: "2026-09-04T11:26:52.370742+08:00"
topics:
  - "AI Agent"
  - "推理"
tags:
  - "BrowseComp"
  - "Deepresearch-bench"
  - "RL-based policy optimization"
  - "SFT"
  - "SearchArt"
  - "evidence graphs"
related_tutorials:
  - "a-survey-of-reasoning-and-agentic-systems-in-time-series-with-large-language-mod"
  - "harnessx-a-composable-adaptive-and-evolvable-agent-harness-foundry"
  - "skillos-learning-skill-curation-for-self-evolving-agents"
  - "agent0-unleashing-self-evolving-agents-from-zero-data-via-tool-integrated-reason"
---

<p class="paper-original-title" lang="en">SearchArt: Training Long-Horizon Search Agent with Scalable Synthetic and Verified Task</p>

<img src="/images/2607.24850v1/A__title.webp" alt="" style="width:90%; max-width:700px; margin:auto; display:block;">

大语言模型（LLM）驱动的搜索智能体（Search Agent）正从简单的单轮关键词检索，演进为需要多轮规划、长程探索与交叉验证的深度推理系统。然而，训练一个真正可用的长程搜索智能体始终面临结构性困难：现实中极度缺乏高质量、长周期的搜索与研究任务数据，人工标注完整轨迹成本高昂，且模型在多步工具调用过程中的幻觉与错误很难被精确定位与纠正。许多依赖模型参数记忆或浅层检索就能回答的“伪复杂”问题，无法有效迫使智能体学会长程规划与自适应搜索。

> ArXiv URL：https://arxiv.org/abs/2607.24850v1

针对这一瓶颈，华为团队提出了 **SearchArt**，一个面向长程搜索智能体（Long-Horizon Search Agent）的可扩展训练框架。该工作跳出了以往仅依靠终局答案正确性进行弱监督的局限，构建了覆盖三类任务形态的合成与验证流水线，并结合监督微调（SFT）与强化学习（RL）实现多阶段后训练。

<img src="/images/2607.24850v1/browsecomp_leaderboard_overview_v2.webp" alt="不同基准能力综合对比" style="width:85%; max-width:600px; margin:auto; display:block;">

评估结果显示，基于 Qwen3.5-27B 参数底座训练的 SearchArt，在 BrowseComp-ZH 上取得 74.39 分，在 BrowseComp 上取得 70.06 分，在 Deepresearch-bench 上取得 52.55 分，全面对齐甚至超越了当前主流的前沿闭源商业智能体。这项工作的核心突破在于：**长程搜索能力的本质不是让模型在更长的时间里盲目试错，而是通过抗捷径、强分散的结构化合成数据，引导智能体建立起高质量的中间推理行为与动态探索策略。**

### 突破浅层检索：三轨并行的长程任务合成体系

训练长程搜索智能体面临的首要难题是合成任务往往“徒有其表”。如果一个多跳问题的大部分证据都可以从单一网页或摘要卡片中打包获取，或者模型仅凭参数内部知识即可猜出答案，智能体就会迅速退化为依赖表面线索的捷径求解器。为此，SearchArt 设计了涵盖三类典型场景的数据合成框架。

#### 1. 抗捷径与证据分散的 DeepSearch QA

DeepSearch 流水线旨在生成需要高强度、多路径检索的客观事实型任务。其核心目标是切断参数记忆与单点检索的捷径，迫使模型在复杂网络中收集并整合离散证据。

<img src="/images/2607.24850v1/Web_Enhanced_Complex_QA_Pipeline_Visual_Clarity_Optimized_v2.webp" alt="DeepSearch QA 合成流水线总览" style="width:85%; max-width:450px; margin:auto; display:block;">

该流程首先进行**种子实体初始化**。研发团队并没有在知识图谱中随机选点，而是从维基百科等大规模多语言图谱中精选低入度、低出度的长尾实体（Long-tail Entities）。这类实体在模型预训练语料中的曝光度极低，从源头上遏制了模型脱离检索直接作答的可能。同时，算法利用多轮随机游走评估候选实体局部的拓扑丰富度，优先保留那些能延伸出多样关系类型与多分支证据路径的种子，并配合分层领域分类法，动态平衡不同学科领域的覆盖配额。

在获得种子后，系统采用**网页增强的智能体知识扩展（Web-enhanced Agentic Knowledge Expansion）**机制。不同于以往局限于静态图谱遍历的方案，SearchArt 部署了一个自主探索智能体，在开放网络中循环执行“查询生成—网页检索—知识抽取—图谱构建—知识空白发现”。

<img src="/images/2607.24850v1/deepresearch_figure_v2.webp" alt="网页增强知识扩展执行流程" style="width:90%; max-width:700px; margin:auto; display:block;">

该智能体主动探寻种子实体周边未被静态图谱收录的深层关联，如历史事件、组织变迁、跨域脉络等。针对网络抓取中常见的单向关系或事件断裂，算法设立了**知识空白发现机制（Knowledge-gap Discovery）**，识别出稀疏节点与未决引用，触发针对性的追问检索。多轮探索完成后，再经由实体对齐、消歧与规范化，汇聚为拓扑复杂的全局知识图谱。

为了从海量图谱中筛选出真正需要长程搜索的子图，SearchArt 引入了量化的**难度排序标准**，结合结构复杂度与证据来源分散度两个维度进行评分。其中度数复杂度定义为：




{% raw %}$$C_{\text{degree}}(G)=\frac{1}{|V|}\sum_{v\in V}\deg(v)$${% endraw %}



系统在此基础上兼顾图的深度与宽度。更关键的是，研究团队提出了**证据源分散度（Evidence Source Dispersion, ESD）**指标：




{% raw %}$$\text{ESD}(G)=\frac{\sigma(n)}{\mu(n)}$${% endraw %}



该指标度量了支撑事实在不同独立信息源之间的分布方差。即使一个子图逻辑推导很长，如果证据集中在少数网页，其难度评级依然会被压低。只有拓扑复杂度高且 ESD 得分高的子图才会被送入下游生成。

最后，在 QA 生成环节，流水线通过实体别名掩蔽、属性模糊化等手段抹去表面常数，构造反向或溯因推理任务。再通过**硬轨迹条件化生成（Hard-trajectory-conditioned Generation）**，收集模型在历史高难度样本上的求解轨迹（基于 Pass@$K$ 筛选），将其作为 In-context 示例输入给生成模型，指导其模拟多轮线索解构与频繁查询重写的真实困难模式。

#### 2. 覆盖长文理解与报告撰写的 DeepResearch QA

对于科研论证、产业竞品分析这类深度研究场景，任务目标通常不是寻找单一实体，而是撰写包含多方论证的长篇分析报告。

<img src="/images/2607.24850v1/DeepResearch_QA_Synthesis_Model_v1.webp" alt="DeepResearch QA 合成模型架构" style="width:85%; max-width:600px; margin:auto; display:block;">

SearchArt 将高质量的学术综述、行业研报、深度调查文章视作现实世界中自然存在的“标准参考答案”。流水线通过建立时间截断边界，利用大模型逆向推导其对应的初始研究课题。在此基础上，系统进一步构建证据追溯链条，确保整个长篇综述中的关键结论都有明确的原始信息来源可供追踪，以此训练智能体在大规模长文档阅读中的全局规划与信息整合能力。

#### 3. 贴合复杂意图的现实用户体验导向 QA

标准基准测试中精心设计的谜题式提问往往脱离普通用户的表达习惯。真实场景下的查询往往充满模糊性、隐含约束或决策权衡。

<img src="/images/2607.24850v1/Real-world_User_Experience_QA_Model_v1.webp" alt="现实用户体验导向 QA 合成架构" style="width:85%; max-width:600px; margin:auto; display:block;">

SearchArt 补充构建了面向现实决策场景的数据流，模拟真实用户在生活建议、跨领域比选、策略规划等方面的交互。这类任务的特点是条件约束分散在长提示词中，智能体不仅要具备深度搜索能力，还必须具备对不确定性需求的澄清能力、意图对齐能力以及面向开放性结论的综合权衡能力。

### 联合校验流水线与多阶段后训练

单纯扩大合成数据规模并不能消除合成数据中的逻辑断层。如果数据中包含错误的推理轨迹或失真证据，监督微调只会放大模型的幻觉倾向。SearchArt 建立了一套严苛的联合验证流水线，从三个层面执行过滤：

1. **问答一致性校验**：评估逆向推导出的问题是否具备唯一且合乎逻辑的答案空间，剔除存在歧义或前提假设不成立的伪问题。

2. **检索证据相关性判定**：逐条审查从开放网页中捕获的支撑材料，确认其是否在时间窗口、事实表述上严格支撑对应结论，过滤含有噪声的死链与陈旧信息。

3. **推导轨迹质量评估**：追踪推理链中的每一步动作，验证工具调用（Tool-use）与检索返回结果之间是否存在严密的因果逻辑，剔除包含无效循环或跳跃性假设的解答过程。

通过这套机制筛选出的高质量轨迹，随后被注入到两阶段后训练流程中：

首先是**结构化监督微调（SFT）**阶段，模型学习如何规范地拆解子目标、生成高效的检索 Query、在工具调用间传递结构化状态变量，并掌握针对长篇材料的摘要和聚合模式。

随后进入**强化学习策略优化（RL-based Policy Optimization）**阶段。区别于仅依赖最终答案对错的稀疏奖励函数，SearchArt 将中间探索质量、证据召回完整性与检索开销惩罚融入优化目标。这使得智能体能够学会依据任务的实际难易度自适应分配搜索预算（Search Budget）：遇到浅层事实快速终结，面对高分散的复杂多跳任务时，则能主动拓展证据分支、自我纠偏并展开多步验证。

### 实验结论：27B 开源架构打破闭源壁垒

在包含 BrowseComp-ZH、BrowseComp、BrowseComp-Plus、WideSearch 以及 DeepResearch-Bench 等代表性中英文评测基准上，基于 Qwen3.5-27B 训练的 SearchArt 展现出显著的性能跃迁。

从基准表现来看，SearchArt-27B 在事实检索基准 BrowseComp-ZH 达到 74.39 分，在英文 BrowseComp 达到 70.06 分。在更强调长程研究、结构化输出与大范围证据归纳的 Deepresearch-bench 上，取得了 52.55 分的成绩。这一结果表明，一个参数量仅为 27B 的中等规模开源模型，在经过结构化合成任务与过程验证训练后，其在复杂搜索规划和长程推理任务上的综合表现，不仅大幅拉开了与同尺寸基线模型的差距，还成功对齐甚至部分超越了参数规模大得多的顶尖商业闭源 Agent 接口。

消融分析与轨迹观察进一步证实了数据生成策略的关键作用：

- **截断捷径的必要性**：依托长尾实体与证据源分散度（ESD）指标合成的任务，有效阻止了模型在 SFT 阶段退化为依赖内生参数记忆猜答案；

- **自适应搜索预算机制的形成**：强化学习阶段将过程轨迹质量纳入反馈后，模型在长程任务中的冗余查询量明显下降，能够更平稳地在未见网页和长文本中进行线索追踪，有效抑制了“检索未充分即草率下结论”或“在无关页面无限死循环”的传统顽疾。

### 启示与未来展望

SearchArt 的核心价值不仅在于刷新了开源搜索智能体的基准得分，更在于提供了一套可工业化复制的、基于过程可验证性的数据演进范式。它证明了对于复杂的长程决策与检索系统而言，**训练数据的“搜索拓扑深度”和“证据离散度”比单纯的数据吞吐量更加重要**。

这一技术路径清晰地指明，未来高性能智能体的构建重点正在从“扩大模型底座参数”向“精细化合成与高阶环境交互后训练”倾斜。利用结构化知识图谱配合网络自适应扩展来生产高质量监督信号，再借助验证流水线反哺强化学习，将成为破局长程复杂任务训练数据匮乏的关键通道。
