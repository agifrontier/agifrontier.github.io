---
layout: default
title: "不是历史累赘而是新文档太吵？港中文等提出CRRL破解搜索Agent干扰"
description: "针对这一瓶颈，研究团队提出了一种轻量级的上下文提纯器（Context Refiner），并通过强化学习框架 CRRL 将动态提问提纯机制无缝融入 Agent 训练全流程，实现了在显著降低检索频次与上下文长度的同时，大幅拉升问答准确率。"
arxiv_id: "2608.10743"
paper_published: "2026-08-11"
published_at: "2026-09-23T13:15:08.398930+08:00"
topics:
  - "AI Agent"
  - "推理"
tags:
  - "Context interference"
  - "Context refinement"
  - "Distill-based context refiner"
  - "LLMs"
  - "Multi-turn search agents"
  - "RL training pipelines"
related_tutorials:
  - "skyrl-agent-efficient-rl-training-for-multi-turn-llm-agent"
  - "beyond-turn-limits-training-deep-search-agents-with-dynamic-context-window"
  - "thinker-training-llms-in-hierarchical-thinking-for-deep-search-via-multi-turn-in"
  - "deepdive-advancing-deep-search-agents-with-knowledge-graphs-and-multi-turn-rl"
---

<p class="paper-original-title" lang="en">Mitigating Context Interference for Reliable and Efficient Search Agents</p>

让大语言模型（LLM）化身为能够自主规划、多轮调用搜索引擎并持续推理的“搜索智能体”（Search Agent），已经成为解决复杂问答与开放域知识推理的主流范式。然而，在实际运行中，许多看似能力强大的搜索 Agent 常常出现令人费解的失误：明明检索引擎已经返回了包含正确答案的维基百科段落，Agent 却在接下来的推理中视而不见，甚至被检索内容中的无关噪音带偏，最终给出一个南辕北辙的答案。

> ArXiv URL：https://arxiv.org/abs/2608.10743v1

香港中文大学、伦敦大学学院（UCL）、浙江大学、香港大学与爱丁堡大学等机构的研究团队，针对这一现象展开了系统性研究，首次深入解剖了多轮搜索智能体中的“上下文干扰”（Context Interference）问题。研究发现，导致 Agent 决策漂移和推理失败的主要元凶，并不是过往轮次累积的漫长思考链或历史检索词，而是**当轮最新检索出来的大段文档**。针对这一瓶颈，研究团队提出了一种轻量级的上下文提纯器（Context Refiner），并通过强化学习框架 CRRL 将动态提问提纯机制无缝融入 Agent 训练全流程，实现了在显著降低检索频次与上下文长度的同时，大幅拉升问答准确率。

<img src="/images/2608.10743v1/x1.webp" alt="上下文干扰对搜索智能体表现的影响" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 搜索 Agent 为什么总是“看到了却答不对”？

在传统的检索增强生成（RAG）系统中，模型通常只进行单次检索与单次回答，上下文结构相对扁平。但自主搜索 Agent 面临的环境要复杂得多。为了完成一个复杂的跨实体、多跳推理任务（例如 HotpotQA、2WikiMultiHopQA 等），Agent 需要在多个轮次之间不断交替执行“思考（Thinking）- 生成检索词（Query）- 获取外部文档（Documents）- 整合更新状态”的循环。

这一交互过程本质上是一个马尔可夫决策过程（MDP）。Agent 的最终输出由模型的参数化内部知识与外部检索引入的知识共同决定。为了保证召回率，搜索引擎每轮通常会返回 Top-3 或更多的长文本段落。这些外部文档不可避免地夹杂着海量不相关实体、背景噪音甚至是误导性表述。

研究团队通过实验定量揭示了一个尴尬的现实（如上图所示）：在各类问答基准中，“召回率”（Recall Rate，即检索文档包含正确答案的比例）与“召回准确率”（Recall Accuracy，即检索命中且最终答对的比例）之间存在巨大鸿沟。这意味着，大量失败案例并不是因为检索系统能力不足，而是因为模型在被塞入冗长的新文档后，受到了严重的上下文干扰，丧失了准确定位关键信息并激活自身内部知识的能力。此外，被噪音误导的 Agent 往往会误以为当前线索不足，进而触发额外的不必要检索，造成时间和 Token 计算资源的严重浪费。

<img src="/images/2608.10743v1/x2.webp" alt="上下文干扰与提纯前后对比示例" style="width:min(600px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; display:block;">

### 抽丝剥茧：干扰究竟藏在上下文的哪一部分？

为了对症下药，首先需要明确：在漫长的多轮对话历史中，究竟是哪个组件向模型注入了最大毒素？研究团队将 Agent 的上下文历史细分为四个部分进行消融掩码实验分析：当前任务指令、过往轮次的思考步骤、过往生成的检索词，以及过往和当前检索到的文档。

通过在 IRCoT（交织检索与思维链）框架下对 Qwen-2.5-7B/3B-Instruct 进行控制变量评测，团队对比了不同历史剪裁策略的表现：

第一种策略是只保留当轮最新检索到的文档，将历史所有轮次的旧文档彻底剔除（IRCoT-$o$）。实验结果显示，这一简单操作不仅没有破坏长程记忆，反而在绝大多数数据集上直接带来了准确率（EM）的提升，同时显著减少了平均检索次数（ART）。这说明历史累积的旧文档中存在大量过期和无关干扰，持续堆叠只会让模型迷失方向。

第二种策略是在剔除历史文档的基础上，进一步剔除历史轮次生成过的检索词（IRCoT-$oq$）。结果表明模型的性能仍然保持稳健甚至略有改善，证明过往检索词本身携带的信息量极低，同样构成轻微干扰。

然而，当实验进一步剔除过往轮次的中间思考步骤（IRCoT-$oqp$）时，模型性能发生了显著滑坡，平均检索次数急剧飙升。这表明，中间推理步骤构成了多轮任务的“认知骨架”，负责记录子目标完成进度；一旦缺失思考步骤，Agent 就会陷入重复探索和盲目检索的死循环。

更为关键的洞察在于：即使完全清理了历史检索文档与查询词，模型与理论最优性能之间依然存在明显差距。通过深入追踪模型在每一轮的注意力分配与决策走向，研究团队确认，多轮搜索 Agent 面临的最致命干扰，正是**当前轮次刚刚检索返回的未经提炼的原始长文档**。搜索引擎返回的内容覆盖面越广，局部段落中夹带的实体冲突与无关细节就越多，直接诱导下一步推理走入死胡同。

### 从全量注入到精准蒸馏：构建 Context Refiner

明确了干扰的核心来源，解决方案的雏形也随之浮现：在每次检索得到原始文档后，不能直接将其全量拼接至上下文，而必须先插入一个轻量化的过滤中间件，完成“先提纯、再生成”（Refine Context and then Generate）的过程。

<img src="/images/2608.10743v1/x3.webp" alt="上下文提纯器的蒸馏训练流程" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

理想的提纯器需要达成严苛的目标：它必须依据当轮特定的搜索词，把长篇段落中真正相关的关键陈述榨取出来，同时严格压缩篇幅，且绝对不能凭空捏造事实或引入外部幻觉。直接采用通用的大语言模型通过 Prompt 压缩长文（如 GPT-Compress 或模型自我提炼 Self-Refine）往往差强人意——较小参数量的基座模型缺乏在复杂噪音中精准抽取事实的能力，而通用的文本摘要又容易抹去具体的细粒度线索。

为此，研究团队设计了一套基于教师模型的高质量数据蒸馏流水线，如上图所示：

1. **轨迹采样与关键抽取**：利用高性能教师模型（如 GPT-4）在多轮问答轨迹中运行，指示其在拿到检索文档集合与当轮检索词后，专门提取与该检索词直接相关的关键命题。

2. **蕴含验证与幻觉过滤**：仅保留最终答对题目的成功轨迹，并引入严密的文本蕴含模型（Entailment Model）进行自动双向核验，严格确保提取出的高纯度文本完全被原始文档蕴含，彻底剔除模型利用自身隐式记忆补充常识所造成的幻觉。

3. **监督微调注入能力**：将清洗后的三元组数据构建为提纯微调集，对开源基座模型（如 Qwen2.5-7B/3B）实施有监督微调（SFT），训练出专门用于搜索场景的轻量级上下文提纯器 $\mathcal{F}$。

经过训练的上下文提纯器展现出极具竞争力的表现。实验数据显示，在多跳复杂问答测试集上，经过提纯器处理后的上下文，其平均长度相比传统的 IRCoT 基线出现了断崖式下降（如下图所示）。

<img src="/images/2608.10743v1/context_len.webp" alt="各方法在 QA 测试集上的平均上下文长度对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

大幅瘦身的上下文并没有损害信息完整性，反而使小模型展现出逼近甚至追平大参数商业模型（Prompt-driven GPT-Refine）的抗干扰推理表现。模型不再需要在一长串与当前子问题无关的新闻背景或人物生平中费力寻找线索，直接依据高密度的提纯事实即可顺畅推进下一步思考。

### 走向端到端优化：强化学习框架 CRRL

仅仅在推理阶段以外挂插件的形式调用提纯器，虽然能够缓解即时干扰，但并未触及 Agent 策略网络本身的参数演进。如果 Agent 在训练阶段接触到的全都是充满冗余噪音的上下文，其学到的策略分布就会天然带有对噪音的妥协与迟钝。

为了彻底释放该架构的潜力，研究团队进一步将上下文动态提纯机制内嵌到了 Agent 的强化学习（RL）训练管线中，提出了 CRRL（Context Refinement in Reinforcement Learning）训练方法。

<img src="/images/2608.10743v1/x4.webp" alt="CRRL 强化学习训练框架总览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在 CRRL 框架中，整个交互环境与奖励设计围绕着“可靠性与效率并重”展开：

在 Agent 与检索环境交互的每一步状态转移中，检索器返回的原始文档都会先实时流经 Context Refiner 进行清洗，被提纯的高纯度观察结果再作为全新状态反馈给 Policy Agent。这一设计直接重构了 Agent 面临的状态空间马尔可夫链，消除了状态表征中的高频噪声，使得价值函数与策略梯度的估计更加稳定。

在策略更新层面，CRRL 不仅考察最终回答的准确匹配度（Exact Match），还将检索轮次消耗与 Token 开销隐式或显式地纳入长期回报评估。在纯净的上下文支持下，Policy Agent 能够极快地感知到每一次检索动作带来的真实信息增益，从而学会在获得足够依据时果断终止检索、直接输出最终结论，避免了以往 Agent 常见的“死循环检索”和“无效试探”。

全面的对比实验证明，相比于标准的直接强化学习微调方案以及单纯依靠数据增强的基线模型，经过 CRRL 训练的搜索 Agent 展现出了全局性的飞跃：不仅在 Natural Questions、HotpotQA、TriviaQA 等多跳及单跳复杂基准上的问答准确度大幅领先，更实现了平均检索交互轮次与推理时延的双重压缩。

### 总结与未来启示

这项工作为当下火热的 AI Agent 研发提供了一个至关重要的底层认知：**在智能体系统中，更长的上下文并不等同于更强的感知能力；相反，低质且未经提纯的信息正在成为吞噬 Agent 推理能力的“隐形杀手”。**

传统的 RAG 和 Agent 开发往往热衷于追求更大的检索召回窗口（Top-K）、更长的上下文承载能力，但这篇论文用翔实的定量证据指明：导致 Agent 性能崩塌的核心阻碍恰恰就来自于当轮检索返回的文本堆叠。通过解耦“原始文档检索”与“高密信息摄入”，建立“先提纯再生成”的双阶段范式，不仅可以打破小参数开源模型处理复杂搜索任务的能力天花板，更为构建超低时延、高确定性的工业级 Agent 架构开辟了一条兼顾可靠性与能效比的实用路线。
