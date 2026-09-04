---
layout: default
title: "智源研究院AREX：基于发现-验证不对称性，让大模型递归自我改进"
description: "让大语言模型进行“深度研究”（DeepResearch）是当下智能体（Agent）演进的核心方向。但当前的许多研究系统陷入了一个误区：以为把上下文窗口拉长、放开检索步数限制，智能体就能自然找到复杂的最终答案。"
arxiv_id: "2607.21461"
published_at: "2026-09-04T13:15:08.122237+08:00"
topics:
  - "AI Agent"
tags:
  - "AI Agent"
  - "AI论文解读"
related_tutorials:
  - "alita-g-self-evolving-generative-agent-for-agent-generation"
  - "learning-on-the-job-an-experience-driven-self-evolving-agent-for-long-horizon-ta"
  - "mars-optimizing-dual-system-deep-research-via-multi-agent-reinforcement-learning"
  - "a-multi-agent-framework-for-stateful-inference-time-search"
---

<p class="paper-original-title" lang="en">AREX: Towards a Recursively Self-Improving Agent for Deep Research</p>

让大语言模型进行“深度研究”（Deep Research）是当下智能体（Agent）演进的核心方向。但当前的许多研究系统陷入了一个误区：以为把上下文窗口拉长、放开检索步数限制，智能体就能自然找到复杂的最终答案。事实恰恰相反，无节制的长程搜索往往会导致早期的检索噪音被放大、翻盘证据被遗忘、上下文充满失效尝试，最终模型在海量信息中迷失方向。

> ArXiv URL：https://arxiv.org/abs/2607.21461

面对多重约束交织的现实研究任务，北京智源人工智能研究院（BAAI）提出了一个根本性的洞见：**深度研究任务存在显著的“发现-验证不对称性”（Discovery–Verification Asymmetry）**。寻找一个同时满足多重约束的最终答案极其昂贵且搜索空间巨大；但如果反过来，对一个已经成型的候选解按约束逐项进行拆解核验，难度却低得多。

<img src="/images/2607.21461/arex_benchmark_results_paper.webp" alt="AREX 性能总览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

基于这一认知，智源团队推出了递归自我改进深度研究智能体——**AREX**。AREX 不把“验证”当成搜索完成后的终审打分器，而是将其作为驱动研究多轮迭代的核心状态转移机制。配合模型自主触发的上下文压缩工具，仅有 10B 激活参数的 AREX-Base（基于 122B MoE 架构）不仅击败了 397B 的稠密基座，还在 WideSearch、DeepSearchQA 等高难度基准上比肩甚至超越了业内前沿的大规模商用模型。

### 为什么盲目“搜得更久”无法带来更好的研究？

在复杂的深度研究场景中，用户的问题通常暗含多个强耦合的限制条件。例如，要求找出同时满足特定年代、学术成就、社会经历和地理限制的人物或事件。在这种任务中，单向线性推进的搜索智能体往往暴露出两个致命缺陷：

其一是**搜索空间的不可控扩散**。传统 Agent 面对此类任务，通常采用单一轨迹的端到端展开（One-shot Trajectory），一边搜索一边推理。一旦初始阶段采纳了一个看似合理但实际上违背了某项子约束的假设，模型就会沿着错误方向疯狂深入。当它终于意识到线索对不上时，整个上下文已经被杂乱的搜索结果填满，回溯成本高昂。

其二是**无脑截断或全量保留的历史诅咒**。长时间与外部环境交互，智能体的历史上下文会飞速膨胀。保留全部原始网页和交互过程会极大稀释模型的注意力和推理能力；而简单的滑动窗口或外部模型粗暴摘要，又极易把前期辛苦挖掘出的关键事实、引用凭据随手丢弃。

AREX 给出的解法打破了这种“单向搜索”思路。它意识到，一个中间答案即使不完全正确，也绝非毫无价值。通过约束维度的细粒度审计，智能体可以清楚地知道：哪些条件已经被坐实，哪些证据存在冲突，哪些陈述仍然缺乏支撑。将这些部分验证的成果固化下来，把剩下的未决条件转化成下一轮更聚焦、更精准的“定向子任务”，才是破局的关键。

<img src="/images/2607.21461/method_recursive.webp" alt="AREX 递归双循环机制" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 双循环架构：把验证转化为行动的指南针

为了让“发现-验证不对称性”真正落地为可执行的工程算法，AREX 设计了内外嵌套的递归双循环结构（Bi-level Recursive Self-Improvement Loop）。

#### 1. 内部研究循环（Inner Research Loop）

内部循环负责针对具体的“研究目标”调动工具展开作业。在第一轮中，这个目标来自用户的原始提问 $x$；而在后续轮次中，目标则是由外层循环分解出的精准补盲任务 $q^{(k)}$（例如“核实候选人 A 是否在 1998 年之前获得过该奖项”）。

在这一过程中，AREX 会反复执行如下状态推进：




{% raw %}$$ h_{t}^{(k)}=\left[\left(m_{i}^{(k)},a_{i}^{(k)},o_{i}^{(k)}\right)\right]_{i=1}^{t} $${% endraw %}






{% raw %}$$ \left(m_{t+1}^{(k)},a_{t+1}^{(k)}\right)=\pi_{\theta}\left(x,q^{(k)},h_{t}^{(k)}\right),\qquad o_{t+1}^{(k)}=\mathcal{T}\left(a_{t+1}^{(k)}\right) $${% endraw %}



当目标被充分挖掘，或者模型判断继续搜索收益递减时，它不会直接吐出最终答案，而是通过专门的 `finish` 接口导出结构化产物：




{% raw %}$$ r^{(k)}=F_{\theta}\left(\bar{h}_{T_{k}}^{(k)}\right)=\left(y^{(k)},\mathcal{E}^{(k)},s^{(k)}\right) $${% endraw %}



这里包含了临时答案 $y^{(k)}$、支撑证据集 $\mathcal{E}^{(k)}$ 以及该轮的置信度评分 $s^{(k)}$。

#### 2. 自主上下文更新（ACU）

长程研究对上下文管理提出了严苛要求。AREX 没有使用外部模型来充当“摘要器”，而是将上下文维护内化为自身的原生工具调用能力——自主上下文更新（Autonomous Context Updating, ACU）。

模型在研究过程中一旦感知到上下文冗杂，或达到 128K Token 的活跃上下文阈值，就会主动调用 `update_context`。它并不是在机械概括，而是站在当前研究目标的视角，把探索轨迹压缩为紧凑的“改进状态”（Improvement State）：




{% raw %}$$ z_{t}^{(k)}=f_{\theta}\left(h_{t}^{(k)}\right) $${% endraw %}



该状态保留已经坐实的证据凭证、记录各项约束的满足进度、醒目标注尚未填补的信息缺口，并制定接下来的检索计划。那些被证伪的死胡同和重复的页面噪音则被剔除，为后续深度推理腾出清晰的工作记忆。

#### 3. 外部自我改进循环（Outer Self-Improvement Loop）

外层循环接收到内层产出的 $r^{(k)}$ 和终态上下文后，行使裁决权：




{% raw %}$$ d^{(k)}=\begin{cases}\textsc{Accept},&s^{(k)}\geq\tau\\[3.0pt] \textsc{Refine},&s^{(k)}<\tau\ \land\ v^{(k)}=1\\[3.0pt] \textsc{Restart},&s^{(k)}<\tau\ \land\ v^{(k)}=0\end{cases} $${% endraw %}



当置信度达标时，任务完成并输出；若置信度不足，外层循环会评估当前轨迹是否具有挽救价值：若主体方向正确，则保留有效发现，生成下一轮定向研究目标 $q^{(k+1)}$ 并触发精炼（Refine）；若判断轨迹已经被严重误导、充满噪音，则果断重置状态（Restart），重新立项探索。通过将多轮递归的上限约束在预设范围内，系统在计算预算与回答质量之间取得了严格的平衡。

### 训练突破：攻克长程轨迹中的“关键步归因”

智能体在长达几十甚至上百步的搜索链路中，常常面临严重的奖励稀疏难题：最终答案是错的，并不代表中间每一步搜索都一无是处；最终答案是对的，也绝不意味着路径上的每次点击都合理。

智源团队采用了“渐进式多阶段中训（Mid-training）+ 关键步强化学习”的方案，系统化培养模型的科研素养。

在数据合成阶段，研究人员通过多源证据交织、中间假设构造等手段，生成了海量可验证的合成研究任务，并使用强教师模型在真实环境中探索出高质量样本。

更关键的革新在于训练目标的设计。团队发现，长轨迹中绝大多数动作只是常规的状态转移，真正决定成败的往往是少数几个“关键决策点”（Key Steps）——例如在海量链接中锁定核心证据的瞬间、断然放弃错误方向掉头重来的时刻、或者将散落事实穿针引线的转折点。

在有监督合并阶段，研究人员针对关键步骤施加重点损失约束：




{% raw %}$$ \mathcal{L}_{\mathrm{key}}=-\mathbb{E}_{s_{j}\sim\mathcal{K}}\left[\frac{1}{|s_{j}|}\sum_{k=1}^{|s_{j}|}\log\pi_{\theta}(a_{j,k}\mid c_{j,k})\right] $${% endraw %}



在强化学习阶段，团队设计了面向步骤感知的群组策略优化（Step-aware Group Policy Optimization），并引入轻量级的步级奖励塑形（Step Reward Shaping）：




{% raw %}$$ A_{i,j}=A^{\mathrm{out}}_{i}+\lambda_{\mathrm{key}}\widetilde{B}_{i,j} $${% endraw %}



其中 $\widetilde{B}_{i,j}$ 仅在轨迹最终结果为正时激活，向关键决策步骤注入辅助奖励。这种设计既保留了终局正确性作为主要的优化指南针，又为长程探索提供了高精度的局部航标，让模型在反复试错中精准掌握“何时该压缩、何时该转向、何时该质疑”。

### 实验评测：以 10B 激活参数跨级对抗庞然大物

AREX 家族推出了两个主力版本：基于 Qwen3.5-4B 打造的致密模型 AREX-Turbo，以及基于 Qwen3.5-122B-A10B（总参数 122B，每次推理仅激活 10B 参数）打造的 AREX-Base。

评测涵盖了 BrowseComp、WideSearch、DeepSearchQA、GAIA、xbench-2510 以及极高难度的带工具版 Humanity’s Last Exam (HLE) 等多个权威基准。

在以长程深度检索著称的 BrowseComp 和 WideSearch 上，AREX-Base 展现出极具统治力的能力。在 WideSearch-en 上，AREX-Base 取得了优于主流闭源商业系统和开源竞品的顶尖表现。即便面对总参数量几乎是其四倍的 Qwen3.5-397B，仅激活 10B 参数的 AREX-Base 依然在综合表现上实现全面反超。同时，在 DeepSearchQA 和文本类 HLE 评测中，AREX-Base 也超越了 MiroThinker-H1 与 DeepSeek-V4-Pro 等强劲对手。

即使是仅有 4B 参数的微型选手 AREX-Turbo，其表现同样令人瞩目：在六大基准中的五个项目上，AREX-Turbo 均直接跨级战胜了参数量近乎九倍于己的 Qwen3.5-35B。这充分印证了该框架的高效性——合理的认知架构和自我纠错回路，能够大幅弥补纯粹参数规模的不足。

消融实验进一步揭示了各个模块的核心价值。在针对内部循环上下文工具的对比测试中，如果剥离 ACU 机制，仅让模型在未经处理的冗长历史中死磕，BrowseComp 的准确率会从 71.4% 骤降至 59.6%，跌落达 11.8 个百分点。这一显著落差清晰地表明：**在长程推理中，智能体的核心竞争力往往不在于它的瞬时记忆有多庞大，而在于它是否具备自主组织工作记忆、提炼有效认知的高阶心智。**

### 深度研究走向“工程化批判”

从技术演进脉络来看，AREX 标志着大模型智能体从“被动的工具调用者”向“具备批判性思维的探索者”跃迁。过去我们在构建 Agent 时，往往把力气花在如何编写更复杂的 Prompt、如何接入更多搜索引擎、或者如何堆砌外挂的反思模块上；而 AREX 将反思、审计、任务二次分解直接变成了模型内生的一等公民能力。

这项研究留下的最重要启示在于：探索真实世界的复杂知识没有捷径可走，单次推理命中终极答案更像是一种概率侥幸。与其指望模型在茫茫大海中一次蒙对，不如赋予它一套可靠的脚手架——把大问题拆碎、把确定性的证据存盘、把不确定的疑点留作下一阶段的靶标。这种沿着“部分验证到精准发起”循环往复的机制，正是自主智能系统迈向严谨严肃科研场景的坚实一步。
