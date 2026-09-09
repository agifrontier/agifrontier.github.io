---
layout: default
title: "ABSeeker：从答案倒推线索，4B小模型长程搜索能力比肩30B"
description: "来自上海交通大学的研究团队提出了名为 ABC（Answer-Backtracked Credit Assignment，答案倒推信用分配） 的全新框架，并基于 Qwen3.5-4B 训练出了长程搜索智能体 ABSeeker 。"
arxiv_id: "2608.05102"
paper_published: "2026-08-05"
published_at: "2026-09-09T13:15:08.606362+08:00"
topics:
  - "AI Agent"
  - "推理"
tags:
  - "ABC"
  - "ABC-GRPO"
  - "ABC-SFT"
  - "ABSeeker"
  - "Answer-Backtracked Clue Recovery"
  - "BrowseComp"
related_tutorials:
  - "the-horizon-gap-planning-memory-execution-training-and-evaluation-for-long-horiz"
  - "searchart-training-long-horizon-search-agent-with-scalable-synthetic-and-verifie"
  - "preventing-error-propagation-in-multi-agent-ai-through-runtime-monitoring"
  - "agentgym-rl-training-llm-agents-for-long-horizon-decision-making-through-multi-t"
---

<p class="paper-original-title" lang="en">ABSeeker: Training Long-Horizon Search Agents via Answer-Backtracked Credit Assignment</p>

<img src="/images/2608.05102v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在复杂信息检索和深度研究（Deep Research）场景中，大语言模型已经不再满足于单轮的“检索-生成”模式，而是演进为能够自主执行数十次甚至上百次交互的长程搜索智能体（Long-Horizon Search Agents）。这类智能体在网络环境中反复执行查询构建、页面浏览、证据交叉验证以及假设修正等复杂操作。

> ArXiv URL：https://arxiv.org/abs/2608.05102v1

然而，训练这种多步交互智能体面临着一个经典的强化学习痛点：**信用分配（Credit Assignment）难题**。

长期以来，业界主流的监督微调（SFT）和强化学习（RL，如 GRPO）训练范式，普遍采用轨迹级别的二元奖励机制。只要智能体最终找对了答案，整个轨迹中的所有步骤都会被一视同仁地赋予正向奖励；而一旦最终答案错误，轨迹中所有步骤都会被打上负向标签。

这种粗粒度的奖励机制在长程搜索中存在致命缺陷。一条最终成功的轨迹中，完全可能混杂着大量的无效检索、死循环和错误推理；相反，一条最终遗憾失败的轨迹，其前几十步往往已经精准命中了极其关键的中间实体与核心证据。如果把整条轨迹简单地视为全对或全错，模型不仅会在 SFT 中学到冗余和误导性行为，在 RL 阶段也会面临极度稀疏的奖励信号与高方差梯度。

来自上海交通大学的研究团队提出了名为 **ABC（Answer-Backtracked Credit Assignment，答案倒推信用分配）** 的全新框架，并基于 Qwen3.5-4B 训练出了长程搜索智能体 **ABSeeker**。该方法巧妙利用了搜索任务“从前向后找极难，但从答案反向倒推极清晰”的内在特性，将稀疏的轨迹级最终结果分解为密集的步级细粒度监督信号。在仅使用 8.5k 训练样本的情况下，4B 尺寸的 ABSeeker 在 BrowseComp 上取得了 55.3% 的成绩，不仅大幅超越同体量的 4B 智能体，甚至打平并超越了参数量接近 30B 的主流搜索模型。

<img src="/images/2608.05102v1/teasor.webp" alt="ABSeeker 与其他大中型搜索模型的表现对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 为什么搜索任务需要“从答案倒推”？

在常规的数学或代码推理任务中，中间步骤的验证往往依赖单元测试或严格的形式化逻辑规则。但在开放域网络搜索中，环境极其嘈杂，信息极其零散，缺乏通用的形式化验证器。

传统方法往往直接依赖一个强大的判别模型，在智能体每走一步时实时打分。然而在没有先验全貌的情况下，判别模型很难界定智能体发起的某次特定搜索到底是在漫无目的地游荡，还是在为深层次的线索做前置铺垫。

上海交大团队抓住了一个关键本质：**对于训练数据而言，标准答案（Ground-Truth Answer）是已知的**。一旦终点固定，整个信息迷宫就具备了逆向追踪的可行性。从最终正确答案出发，我们可以反向推导出解开该谜题所必经的核心实体、关联事实、逻辑链条与约束条件。

例如，面对一个包含多重复杂限制的问题：“某护肤品牌的核心专利成分是什么？该品牌的母公司创始人毕业于哪所学校？”，从问题正向搜索需要经历漫长的分支排查；但如果已知答案指向“CeraVe”以及欧莱雅创始人的母校，系统就可以反向锚定 6 个核心线索：从神经酰胺（Ceramides）的关联，到欧莱雅的收购事件，再到欧仁·舒莱尔（Eugène Schueller）的毕业院校。

这些逆向抽取的线索链条，构成了稳定且不可动摇的“客观锚点”。无论模型在推理时走出多么曲折奇特的路径，系统都可以拿这些中间线索作为尺子，对智能体的每一个动作进行公正且细致的步级打分。

### ABC 框架的双阶段设计：从线索恢复到密集步级奖励

ABC 框架的整体运行流程可以拆解为两个阶段：首先是离线的线索恢复，其次是轨迹滚动展开后的细粒度步级评分。

<img src="/images/2608.05102v1/fig1.webp" alt="ABC 框架整体架构示意图" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

#### 第一阶段：答案倒推线索恢复（Answer-Backtracked Clue Recovery）

针对每个训练样本，输入包含原始查询 $q$ 和已验证的确定答案 $a^*$。线索恢复模块的任务，是构建一个连接问题与答案的有向证据集合：




{% raw %}$$ \mathcal{C}=\{c_1, c_2, \ldots, c_K\} $${% endraw %}



这些线索不是空泛的搜索建议，而是具体的事实三元组、实体名称以及多跳推理中的连接节点。在具体实现中，研究者采用强大的后台模型（如 DeepSeek-V4-Flash），在已知问题和标准答案的前提下，进行逆向因果链分析，提炼出解答该问题不可或缺的最小充分证据集。这套线索在后续的训练循环中保持固定，避免了评分标准随策略迭代而产生漂移。

#### 第二阶段：线索锚定步级评分（Clue-Anchored Step Scoring）

在智能体与网页环境进行多步交互生成搜索轨迹 $\tau=(s_1, s_2, \ldots, s_T, a)$ 时，评分器会对每一个交互步骤 $s_t$（包含当前步的模型思维链思考、调用的搜索工具、工具返回的网页结果）进行独立评估。

传统的二元轨迹反馈只有在最后输出 $a$ 时给出 $r_{\text{ans}}(\tau) \in \{0, 1\}$，而 ABC 则为每个时间步赋予密集的标量分数 $r_t$。每一个步骤都默认拥有 $1.0$ 的基准分数，以确保合理的探索性动作不会无辜受罚。评分器对照前述线索库 $\mathcal{C}$，执行预定义的增量评价逻辑：

- **发现新线索**：智能体首次检索并明确识别出尚未被确认的核心线索，给予大幅正向奖励（$\Delta = +0.8$）；

- **验证与精炼**：对已发现的线索进行交叉确认或进一步细化，给予中等奖励（$\Delta = +0.4$）；

- **偏离与推倒重来**：在已经掌握充分线索的情况下无故放弃正确方向，退回到错误候选，给予惩罚（$\Delta = -0.8$）；

- **直接命中或错误作答**：在最后动作中正确推导最终答案给予强化奖励（$\Delta = +1.0$），反之若给出错误结论则予以严厉扣分（$\Delta = -1.0$）。

最终每步的奖励被截断在区间内：




{% raw %}$$ r_t = \operatorname{clip}\left(1.0 + \sum_{j \in \mathcal{A}_t} \Delta_j, \, 0, \, 2.0\right) $${% endraw %}



<img src="/images/2608.05102v1/fig2.webp" alt="线索恢复与步级评分具体案例解析" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

上图清晰地展示了这种步级信贷分配的优势：即使整条轨迹在第 64 步因为最终推导失误导致答题失败，但在第 22 步首次命中神经酰胺并关联至目标品牌时，该步依然斩获了 $1.8$ 的高分；在第 35 步同时验证了母公司收购事实与创始人学历时，步骤奖励直接拉满到上限 $2.0$。这种“论功行赏、失误定责”的机制，彻底盘活了原本被当作垃圾丢弃的失败交互样本。

### 如何将步级奖励注入训练：ABC-SFT 与 ABC-GRPO

拥有了高质量的密集步级奖励 $r_t$ 之后，接下来的关键是如何将它高效注入到模型的参数更新中。研究团队在监督微调和强化学习两个阶段分别对目标函数进行了重塑。

在 SFT 阶段，传统做法是将所有收集到的成功轨迹放在一起计算交叉熵损失，甚至粗暴丢弃所有失败轨迹。ABC-SFT 则将失败轨迹同样纳入训练集，通过重加权机制调整每个交互轮次的损失贡献：




{% raw %}$$ \mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{T} w(r_t) \sum_{j} \log p_\theta(x_{t,j} \mid x_{t,<j}) $${% endraw %}



其中权重系数采用 Sigmoid 映射 $w(r_t) = \sigma(\alpha \cdot (r_t - \beta))$。通过控制锐度 $\alpha$ 和基准阈值 $\beta$，得分极高的优质检索动作在反向传播中提供强劲梯度，而那些冗余、低分甚至错误的步骤，其权重被严重压缩，对参数更新的影响微乎其微。这使得模型在模仿学习阶段就能剔除“杂质动作”，学会高效率的信息收集模式。

在强化学习阶段，团队基于流行的群体相对策略优化（GRPO）构建了 ABC-GRPO。不同于标准 GRPO 将整条轨迹的单一正负分数在群体内归一化，ABC-GRPO 将步级奖励 $r_{i,t}$ 视为即时回馈，并通过时间折扣因子 $\gamma$ 向前传递未来步骤的影响：




{% raw %}$$ A_{i,t} = \sum_{k=t}^{T_i} \gamma^{k-t} \widehat{R}_{i,k} $${% endraw %}



这种优势函数 $A_{i,t}$ 使得强化学习能够针对某一个具体工具调用或推理抉择提供针对性的梯度推动，从根本上缓解了长程交互中“由于第 100 步的随机失误抹杀第 5 步精妙操作”的信贷模糊问题。

### 实验结果：4B 模型的越级表现与深度分析

研究团队选用开源轻量模型 Qwen3.5-4B 作为基础底座，采用 OpenSeeker 的开源数据进行长程交互轨迹采样，最长允许智能体与环境交互 200 个回合。最终用于 SFT 的有效轨迹仅有 8.5k 条，RL 阶段仅对 1000 个问题进行在线采样扩展。

评估基准覆盖了多个高难度开放域搜索评测集，包括由人类专家精心构造的复杂长链检索基准 BrowseComp、中文长程基准 BrowseComp-ZH、综合多任务 Agent 基准 GAIA-text，以及聚焦极难信息聚合的 xbench-2505 和 xbench-2510。

从实验评测来看，ABSeeker 展现出了极其反常识的越级战斗力：

在不开启上下文管理的原始长程测试下，ABSeeker-4B 在 BrowseComp 上跑出了 37.3%，在 BrowseComp-ZH 上取得了 39.1% 的准确率。这一成绩直接拉开了与同级别 4B 智能体（如 QUEST-4B、DR-Venus）的差距。

而当配合长文本场景常用的上下文管理（Context Management）机制以避免冗余网页挤占有效注意空间后，ABSeeker 的成绩进一步提升至 BrowseComp **55.3%** 和 BrowseComp-ZH **52.9%**。这一表现不仅全面压制了包括 Tongyi-DeepResearch、OpenSeeker、MiroThinker-1.7-mini 在内的多款 30B 级别大模型，甚至在部分综合检索集上逼近了百亿乃至千亿级前沿闭源模型的表现。

<img src="/images/2608.05102v1/rl_step_vs_bc_acc.webp" alt="不同训练轮次与推理动态的对比曲线" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

从强化学习的训练动态曲线可以观察到，采用传统轨迹级二元反馈的基线模型，其探索策略极易陷入震荡，准确率在经过有限步数提升后迅速遭遇平台期；而搭载 ABC-GRPO 的智能体，其在 BrowseComp 上的准确率曲线展现出了稳健且持续向上的攀升趋势。密集且精准的步级奖励，使得策略优化能够稳定感知到每一次检索方向的边际改善。

<img src="/images/2608.05102v1/context.webp" alt="不同上下文预算下的性能对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

消融实验进一步验证了双阶段步级信贷分配的各自价值。

单独引入 ABC-SFT 时，模型在 BrowseComp 上就比常规 SFT 取得了显著的性能增益，这证实了“在监督数据中惩罚冗余操作、保留失败轨迹中的闪光点”对小模型学习复杂探索路径具有重要价值。而在强化学习阶段，ABC-GRPO 带来的提升跨越了所有测试基准。相比于简单将轨迹结果粗暴分配给每一步的机制，细粒度的线索锚定奖励使得模型能够在长达几十步的探索迷雾中始终对准关键证据链。

### 总结与未来启示

长程搜索智能体的竞争正在从“调用搜索工具的简单工程封装”演变为“复杂推理决策的底层信贷优化”。ABSeeker 带来的核心启示在于：**在复杂的多步决策任务中，奖励稀疏性不仅是样本效率的敌人，更是模型产生幻觉与冗余动作的温床。**

通过利用离线训练中“答案已知”的先验，将正向极难的无监督推理转化为逆向清晰的有向线索网，ABC 框架为长程智能体的对齐提供了一种高度可落地的工程范式。它打破了“长程智能体必须依赖超大规模基座参数才能稳住推理链条”的刻板印象，证明了 4B 级别的轻量模型只要在每一步的选择上获得清晰的信用归因，同样能够展现出极具深度的深层探索与证据整合能力。

未来，这种“答案倒推中间子目标或关键证据”的思想，很可能会进一步向网络安全渗透测试、跨软件复杂工作流自动化以及大规模代码库漏洞排查等领域扩散。只要任务的终局结果具备因果回溯的可解释性，细粒度的过程监督就足以彻底改变复杂智能体的训练范式。
