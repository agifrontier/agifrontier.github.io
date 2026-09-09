---
layout: default
title: "MemPrism：不是记忆没找对，而是格式给错了！长程Agent成功率提升7.4分"
description: "由北京邮电大学、复旦大学、南洋理工大学、北京大学、上海交通大学等多所高校联合提出的最新研究 MemPrism ，指出了一个长期被忽视的底层瓶颈： 检索后表征失配（Post-retrieval Representation Mismatch） 。"
arxiv_id: "2608.06745"
paper_published: "2026-08-07"
published_at: "2026-09-09T13:15:08.606362+08:00"
topics:
  - "知识系统"
  - "AI Agent"
tags:
  - "Deterministic composer"
  - "Event stream"
  - "Long-horizon agents"
  - "MemPrism"
  - "Render transform"
  - "Task-conditioned relational views"
related_tutorials:
  - "stream-scaling-up-mechanistic-interpretability-to-long-context-in-llms-via-spars"
  - "context-as-an-environment-programmatic-context-management-for-long-horizon-agent"
  - "a-subgoal-driven-framework-for-improving-long-horizon-llm-agents"
  - "abot-world-0-infinite-interactive-world-rollout-on-a-single-desktop-gpu"
---

<p class="paper-original-title" lang="en">MemPrism: Task-Conditioned Relational Memory Views for Long-Horizon Agents</p>

<img src="/images/2608.06745v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在长时间跨度的具身导航、网页操作或复杂工具调用任务中，大模型智能体（Agent）经常需要跨越数十步甚至数百步的交互链条。为了防止上下文爆炸或灾难性遗忘，业界开发了形形色色的外部记忆库、层次化摘要和知识图谱。然而，当工程师把所有自认为关键的上下文检索出来拼入 Prompt 后，模型往往依然在关键决策节点上“鬼打墙”或彻底迷失。

> ArXiv URL：https://arxiv.org/abs/2608.06745v1

这种现象过去常被归咎于检索精度不足或模型本身的推理缺陷。由北京邮电大学、复旦大学、南洋理工大学、北京大学、上海交通大学等多所高校联合提出的最新研究 **MemPrism**，指出了一个长期被忽视的底层瓶颈：**检索后表征失配（Post-retrieval Representation Mismatch）**。简单来说，记忆系统即便把 100% 正确的历史事实找了出来，但如果以一种固定、僵化的线性格式丢给决策模型，模型依然无法直接看清当前动作所需的逻辑依赖。

<img src="/images/2608.06745v1/Difference.jpg" alt="长程智能体不同记忆范式的概念对比" style="width:min(600px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; display:block;">

为了解决这一问题，研究团队打破了“记忆如何存储就如何呈现”的固有模式，将智能体的持久化经验存储与决策时的临时工作记忆彻底解耦。MemPrism 把所有真实交互记录为客观事件流，在每一次做决策时，由轻量级的视图策略网络根据当前状态动态合成特定关系的工作记忆视图，再以二维光学图像的形式喂给冻结的任务大模型。实验表明，MemPrism 在视觉具身基准 EB-ALFRED 上将任务成功率提升了 7.4 个百分点，在 ALFWorld 上降低了 33.6% 的 Token 消耗，且学到的视图选择策略无需微调即可直接迁移至完全不同的多模态大模型上。

### 为什么拿到正确事实依然会做错决策？

长期以来，针对长程智能体记忆机制的探索主要沿着三条线索展开：第一条线索聚焦于持久化结构构建，例如利用分层摘要、知识图谱（AriGraph）或原子笔记动态链接（A-MEM）将冗长的轨迹沉淀为结构化知识；第二条线索探索可学习的记忆控制，让模型自主学习何时写入、更新或遗忘（如 Memory-R1、AgeMem）；第三条线索则改变载体媒介，利用 AgentOCR 等技术将大量文本历史渲染成压缩图像，以视觉 Token 换取更长的上下文容量。

这些方案虽然在“存什么、怎么存、怎么省 Token”上取得了长足进展，但在读出端大多沿用着一个隐性假设：只要正确的信息被保存并检索出来，下游的策略模型就有能力直接消费。

现实中的决策依赖却呈现出完全不同的复杂形态：

当智能体需要排查动作死循环时，它需要的是将重复操作与其环境反馈紧密对齐的“动作-结果视图”；当它需要跟踪某个关键物体是否已被放入容器时，它需要的是按实体维度折叠的“状态变化表”；当它需要回溯某个前置条件为何未满足时，它需要的是沿着因果或时间跨步串联的“局部依赖链”；而当它仅仅按部就班推进流程时，传统的按时间正序排列的“时间轨迹”才最有效。

如果用单一固定的文本或图结构去硬套所有决策节点，模型在推理时就必须在庞大的上下文中自己做注意力重排与关系推理。一旦上下文变长，注意力分散和长文本干扰就会迅速引发失误。这正是表征失配的根源：**信息在场，但关系隐蔽**。

### MemPrism 的核心架构：解耦存储与呈现

面对这一痛点，MemPrism 确立了一项基本原则：持久化记忆负责记录绝对保真、可复用的历史事实，而工作记忆必须针对当前这一步的决策按需临时构建。

<img src="/images/2608.06745v1/Memprism.jpg" alt="MemPrism 框架全景架构" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

整个 MemPrism 体系包含五个紧密配合的模块，其中前四个模块构成在线推理闭环，最后一个模块负责离线与在线策略优化：

1. **记录器（Recorder）**：维护一条按时间追加的统一事件流 $\mathcal{E}_{<t} = (\tilde{e}_1, \dots, \tilde{e}_{t-1})$。系统在每一步只记录真实执行的操作、环境观测、环境反馈和实体状态变化量 $\Delta s_t$。任何由模型中间生成的推理、摘要或渲染图绝不写回事件流，防止历史记录被不可逆的偏差或幻觉污染。

2. **路由器（Router）**：扮演工作记忆调度大脑的角色。它根据当前观察 $o_t$、总任务目标 $g$ 以及历史简要统计，实时预测一个四维视图动作 $a_t^v = (\tau_t, w_t, c_t, \gamma_t)$。这四个维度分别决定了关系结构类型、回溯的时间窗口、结果过滤条件以及信息粒度。

3. **合成器（Composer）**：依据 Router 给出的离散配置，从统一事件流中切片并提取相关事实，按所选的关系拓扑重新组织为特定的数据结构 $S_t$。

4. **渲染器（Render）**：将合成器输出的关系结构转化为一张二维“光学工作记忆视图” $V_t$。这一临时视图作为当前步的视觉 Prompt 输入给冻结的通用任务模型，在当前决策动作生成后即被销毁。

5. **适配器（Adapter）**：专门用于训练 Router 的策略学习机制。

在整个执行过程中，核心的任务策略大模型（如 Qwen2.5-VL-7B-Instruct）从头到尾处于冻结状态。MemPrism 不去改变任务模型的参数，而是改变呈现在它眼前的“认知界面”。

### 四种关系视图与二维光学空间的设计精髓

MemPrism 预定义了四种经典的关系视图类型，覆盖了长程任务中绝大多数决策依赖场景：

- **时间轨迹（TemporalTrace）**：严格按照时间戳正序排列历史事件，重点展示任务进展与宏观执行节奏。

- **动作效果（ActionEffect）**：按动作类型和目标实体进行双重聚合，让相同的尝试动作及其对应的环境反馈强制横向对齐。如果智能体连续三次推门都返回“门已锁”，这种对齐能以最直观的视觉冲突打破模型的盲目重试。

- **实体状态（EntityState）**：以场景中的物理实体为主键，汇总其属性随时间发生的所有跃变，极大降低了长期状态跟踪的记忆负荷。

- **依赖追踪（DependencyChain）**：以最近一步事件为锚点，跨时间步倒查共享相同实体、状态或紧密因果关联的前序事件，将散落在轨迹各处的因果链条压缩呈现在一起。

值得关注的是，MemPrism 之所以选用光学图像（Optical View）作为工作记忆的载体，并不仅仅是为了借用视觉压缩比来省 Token，而是看中了二维视觉排版的**关系表现力**。

在纯文本流中，Token 之间的关系完全依赖单向的相对位置编码。而在二维渲染图像中：

- 物理空间的临近性（Spatial Proximity）可以直接代表动作语义的关联度；

- 行列对齐（Alignment）让状态对比和循环检测一目了然；

- 箭头符号（Arrows）直观表达因果推进与前后依赖；

- 框选高亮（Highlights）则能将任务模型的视觉注意力无损引导至关键报错信息上。

通过控制信息粒度 $\gamma_t$，Render 模块可以在极高密度的紧凑表格与包含完整文本细节的展开框之间切换，在空间预算与信息无损之间达成平衡。

### 两阶段策略学习：从软蒸馏到分组 GRPO

Router 的离散动作空间大小为 $\lvert \mathcal{A}^v \rvert = \lvert \mathcal{T} \rvert \times \lvert \mathcal{W} \rvert \times \lvert \mathcal{O} \rvert \times \lvert \mathcal{G} \rvert = 144$。要在多达 144 种视图像素排列中为每一步精准匹配最优配置，绝非靠人工规则所能胜任。研究团队为此设计了一套轻量级但针对性极强的两阶段优化方案。

第一阶段是**动作条件下的软监督初始化**。研究团队构建了一个离线评估教师模型，在给定真实交互历史、当前状态和参考动作 $u_t^{\text{ref}}$ 的前提下，反向推导不同视图动作对推导正确行为的贡献度，生成软概率分布 $q_T(a \mid t)$。Router 通过最小化 KL 散度进行监督微调（SFT）：




{% raw %}$$\mathcal{L}_{\mathrm{SFT}} = \operatorname{KL}\bigl(q_T(\cdot \mid t) \,\|\, \pi_\theta^v(\cdot \mid z_t)\bigr) + \lambda_C \sum_{a \in \mathcal{A}^v} \pi_\theta^v(a \mid z_t) \widehat{C}_t(a)$${% endraw %}



公式中引入的成本项 $\widehat{C}_t(a)$ 至关重要，它对调取过大时间跨度或过细粒度的高成本操作施加惩罚，迫使模型学会“非必要不展开”。

第二阶段则是**基于结果驱动的在线分组强化学习（Grouped GRPO）**。离线轨迹无法预料智能体在实际部署中遇到的偏航状态，研究团队在环境交互中利用分组相对策略优化（Grouped Relative Policy Optimization）对 Router 进行强化：




{% raw %}$$\mathcal{L}_{\mathrm{GRPO}} = -\mathbb{E}_{i,t} \left[ \min\left(\rho_{i,t} A_i, \, \operatorname{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) A_i\right) \right] + \beta_{\mathrm{KL}} D_{\mathrm{KL}}(\pi_{\mathrm{ref}}^v \,\|\, \pi_\theta^v) - \beta_H \mathcal{H}(\pi_\theta^v)$${% endraw %}



在每一次迭代中，同一个任务实例会并行采样 16 条轨迹，以组内奖励均值和方差计算优势值 $A_i$。在整个强化学习训练期间，**任务模型本身始终冻结**。这意味着环境的奖励信号完全用来指导 Router“如何重组记忆”，把性能收益精准归因于记忆交互界面的改良，消除了任务模型自身参数变化带来的混淆变量。

### 实验验证：长程优势显著，跨模型直接泛化

为了检验 MemPrism 的实际效能，论文在三大代表性基准上展开了严格测试：包括文本具身环境 ALFWorld、复杂的视觉具身交互基准 EB-ALFRED（基于 EmbodiedBench 评测封装），以及包含真实网页操作的离线基准 Mind2Web。

在视觉具身基准 EB-ALFRED 上，任务极具挑战性。若完全不给历史记忆（No Memory），任务模型的平均成功率仅有 4.7%；即便是把完整的纯文本历史全量塞入 Prompt（Full History），成功率也仅艰难爬升至 10.3%。在外部记忆基准中，基于 Neo4j 图数据库的 Mem0g 成功率为 11.0%，而经过监督初始化的 MemPrism-SFT 直接打破天花板达到 15.7%。在经过 GRPO 在线对齐后，**MemPrism-SFT+GRPO 更是达到了 17.7% 的成功率**，相比 Full History 实现了 7.4 个百分点的绝对提升。

在 ALFWorld 上，研究团队进一步分析了轨迹变长对系统的影响。随着环境步数从 5 步递增到 50 步，Full History 的 Prompt 消耗呈现出近乎线性的暴涨，从 574 个 Token 激增到 1201 个 Token（增加 2.1 倍）。而 MemPrism 依靠光学生成和动态裁切，其 Token 占用在整个长程推演中牢牢压制在 642 至 852 之间，并在 15 步之后几乎不再上涨。在大于 50 步的超长任务场景下，MemPrism 不仅减少了 33.6% 的 Token 消耗，成功率还反超 Full History 达 9.3 个百分点。

更具实用价值的发现来自**跨模型零样本迁移实验**。研究团队将基于 Qwen2.5-VL-7B 训练出的 Router 策略网络直接挂接到未参与训练的 Qwen3-VL-4B、Qwen2.5-VL-3B 以及 MiniCPM-V-2.6 上，下游模型表现出了惊人的一致性提升：

- 在所有模型上，MemPrism 均带来了 30.64% 至 48.15% 的显著 Prompt 压缩。

- 在 MiniCPM-V-2.6 上，由于该模型具有极高的视觉 Token 密度（180 万像素仅映射为 640 个视觉 Token）和极强的 OCR 能力，其长文本冗余问题最为严重。MemPrism 在该模型上激发出最大的效能红利——不仅 Token 消耗锐减 48.15%，任务成功率还实现了 9.40 个百分点的暴涨。

### 架构演进的关键启示

MemPrism 的机制验证表明，长期困扰大模型长程推理的瓶颈，很多时候不是“记不住”，也不是“算力不够”，而是人机交互界面与智能体内部认知界面的错位。

过去人们习惯将大模型当作全知全能的黑盒，试图让它在海量混杂的文本序列中自行抽丝剥茧。但认知科学早已证实，人类工作记忆之所以高效，正是因为大脑能够根据当前面临的具体动作意图，对感知经验进行极度特异化的瞬时重构。

MemPrism 通过“只记录客观事件”保障了底层事实的绝对保真，通过“离散视图路由”建立了认知层面的任务特异化，又通过“二维光学渲染”借用了视觉模型天然的几何对齐优势。这项工作清晰地表明：构建高可靠性的自主 Agent，未来的重心或许应当从无限制地扩展上下文长度，转向如何在正确的时间、以正确的结构把关键信息呈现给模型。随着多模态大模型视觉理解能力的持续进化，这种利用动态光学界面重塑工作记忆的范式，极有可能成为长程智能体系统的标准基础设施。
