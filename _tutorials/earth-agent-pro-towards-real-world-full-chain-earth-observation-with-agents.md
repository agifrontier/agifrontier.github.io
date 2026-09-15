---
layout: default
title: "Earth-Agent-Pro：全链条遥感Agent，解耦规划与执行提升20.95分"
description: "为了打破这种“纸上谈兵”的困境，来自哈尔滨工业大学、上海人工智能实验室、上海交通大学、中山大学、天津大学与清华大学的研究团队联合推出了针对全链条对地观测的完整解决方案——不仅构建了首个覆盖全流程的评测基准 Earth-Bench-Pro ，还提出了执行自适应的智能体框架 Earth-Agent-Pro 。"
arxiv_id: "2609.12533"
paper_published: "2026-09-11"
published_at: "2026-09-15T13:15:08.126145+08:00"
topics:
  - "AI Agent"
  - "推理"
tags:
  - "Earth-Agent-Pro"
  - "Earth-Bench-Pro"
  - "LLM-as-Judge accuracy"
  - "Plan-and-Execute framework"
  - "large language model adapters"
  - "open-world Earth observation"
related_tutorials:
  - "architecting-resilient-llm-agents-a-guide-to-secure-plan-then-execute-implementa"
  - "judging-llm-as-a-judge-with-mt-bench-and-chatbot-arena"
  - "llm-as-a-judge-toward-world-models-for-slate-recommendation-systems"
  - "swe-bench-can-language-models-resolve-real-world-github-issues"
---

<p class="paper-original-title" lang="en">Earth-Agent-Pro: Towards Real-World Full-Chain Earth Observation with Agents</p>

<img src="/images/2609.12533/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在计算机视觉与多模态大模型的研究中，对地观测（Earth Observation, EO）一直被视为极具专业门槛的硬核领域。然而，审视当下各类遥感智能体（EO Agent）的研究，会发现一个普遍存在的“温室现象”：绝大多数基准测试和智能体系统，默认都会将预先裁剪好的卫星影像切片、对齐的数据集甚至备选项直接“喂”到模型嘴边。模型所需要做的，往往只是做做视觉问答（VQA）或者单步的目标检测。

> ArXiv URL：https://arxiv.org/abs/2609.12533

但真实的遥感科学研究根本不是这样运作的。面对一个诸如“评估某流域过去五年植被水分胁迫动态”的宏观科研需求，真实的遥感专家必须经历一套极为漫长且严密的链路：从根据时空坐标检索卫星数据源、下载多光谱或合成孔径雷达影像、进行辐射定标与波段重组，到计算各类地学指数、执行统计推断，最终基于运行时产生的一系列数字证据给出可追溯的结论。当现有的 Agent 被直接扔进这种真实开放环境时，原本亮眼的准确率往往迅速雪崩。

<img src="/images/2609.12533/fig1.webp" alt="研究概览：从传统静态评测走向全链条真实执行" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了打破这种“纸上谈兵”的困境，来自哈尔滨工业大学、上海人工智能实验室、上海交通大学、中山大学、天津大学与清华大学的研究团队联合推出了针对全链条对地观测的完整解决方案——不仅构建了首个覆盖全流程的评测基准 **Earth-Bench-Pro**，还提出了执行自适应的智能体框架 **Earth-Agent-Pro**。该框架将宏观科学工作流规划与微观参数具象化彻底解耦，引入专家技能约束与结构化记忆，并结合角色特化训练。实验结果显示，在 GPT-5 基座下，Earth-Agent-Pro 的准确率相比主流的 ReAct 架构大幅提升了 20.95 个百分点；在开源 9B 模型上，仅通过双适配器微调便实现了 11.69 个百分点的跃升。这项工作标志着遥感领域的研究正真正从“静态读图”走向“全自动科学调查”。

### 告别“喂饭式”评测：Earth-Bench-Pro 设立的真实考场

要让智能体学会真正的对地观测，首先需要一个不掺水分的实战考场。过去诸如 RSVQA、GeoBench-VLM 等基准，关注的都是单一感知输入下的静态答案，缺乏对工具调用轨迹与中间产物的检验；而近期的工具调用评测，又大多割裂了数据检索、预处理与最终的自然语言推断。

研究团队构建的 Earth-Bench-Pro 改变了这一游戏规则。该基准以“完整可执行的证据链”作为核心标注单元，由计算机、遥感与地球科学三个领域的 8 位专家联合把关，从 248 个专家级任务内核出发，衍生出了 744 道配对评估任务，涵盖了高分辨率光学（RGB）、多光谱/高光谱观测以及高阶遥感专题产品。

<img src="/images/2609.12533/fig5.webp" alt="Earth-Bench-Pro 任务构成与全球时空分布" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了诊断 Agent 到底在哪个环节掉队，Earth-Bench-Pro 巧妙地设计了三层进阶模式：

1. **指令遵循模式（Earth-Bench-IF）**：为模型提供现成数据以及明确排好序的标准工具调用序列，纯粹测试模型的单步参数填充与执行能力；

2. **自主规划模式（Earth-Bench-AP）**：依然提供预先准备好的数据，但隐藏了参考工作流，强迫 Agent 自行组合工具完成科学探索；

3. **开放世界执行模式（Earth-Bench-OW）**：完全模拟真实科研场景，只给出一个宏观的科学提问与数据需求描述。Agent 必须自主完成数据源检索与下载、多步骤管线预处理、地学运算，并最终给出一份以运行时证据为支撑的开放式科学报告。

在 Earth-Bench-OW 模式下，任务平均包含 7.1 次工具调用，涉及多达 84 种跨领域的专业工具。评测不仅核对最终答案的数值精度与物理单位，还要严密回溯中间栅格文件、特征图谱与统计指标的数据血统（Provenance）。这种评测设计彻底堵死了大模型靠幻觉蒙混过关的可能性。

### 为什么 ReAct 在长程遥感中会溃败？

在以往的通用智能体研究中，ReAct（Reasoning + Acting）是一种近乎标准配置的范式：思考一步，执行一步，根据环境报错再思考下一步。然而，在面对数十步工具调用、环境极度异构的遥感任务时，ReAct 几乎必然陷入瘫痪。

首要问题在于**规划空间的组合爆炸与失控**。遥感工具库涵盖空间几何计算、传感器波段运算、大气校正等，工具总数动辄数十上百。如果让大模型在每一步都从全部工具列表中挑选，它极易陷入“盲人摸象”的状态，调用毫无关联的算子。

更致命的是**上下文污染与级联崩溃**。在执行复杂的影像变换时，工具经常因为参数维度、投影坐标系不匹配而抛出冗长的异常报错。在传统的单流交互记忆中，这些海量的错误调用与堆栈信息会迅速塞满上下文窗口。不仅会导致模型注意力涣散，还会让模型在后续推理中反复“重蹈覆辙”，甚至把上一轮试错产生的无效临时文件作为后续计算的输入。一旦中间某个节点彻底失败，ReAct 往往只能从头再来，白白丢弃前面耗时数十分钟下载和校准的高价值数据。

### Earth-Agent-Pro：技能约束与结构化记忆的双重保险

针对上述结构性缺陷，Earth-Agent-Pro 确立了一套“规划与执行解耦、执行驱动自适应修复”的全新架构。

<img src="/images/2609.12533/fig2-new.webp" alt="Earth-Agent-Pro 框架机制设计" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

整个框架运转的第一道防线是**专家技能引导（Skill-Guided Workflow Planning）**。研究团队将地学专家的程序性先验提炼为 6 类高阶技能模板，分别对应光谱指数分析、遥感产品反演、目标感知检测等核心任务族。每个技能明确界定了当前任务允许调用的工具子集、先后执行依赖、参数约束以及迭代规程。当接收到科学问题后，Planner 首先进行技能检索与绑定，在生成工作流骨架前直接剪除无关工具的干扰，使规划阶段的搜索空间急剧收敛。

<img src="/images/2609.12533/fig4-new.webp" alt="专家技能引导工作流规划与参数约束" style="width:min(600px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; display:block;">

第二道核心防线是**以工作流为中心的结构化记忆（Workflow-Centered Structured Memory）**。系统将运行时状态精确表达为二元组 $\mathcal{S}_k = (W_k, M_k)$：

- $W_k$ 代表动态工作流序列，包含当前活跃的节点管线以及历史修订归档；

- $M_k$ 代表已被验证合法的执行证据库，按时间顺序仅记录工具类型、经过验证的参数、工具产生的观测结果以及完整的数据溯源信息。

这种设计的精妙之处在于**隔离失败污染**。Executor 在执行某个特定节点时，如果尝试失败，该尝试产生的报错信息仅用于当前节点的局部重试（In-node Retry），绝不会被写入全局证据库 $M_k$。后续节点只能看到“干净且经过校验”的上游证据，从根本上切断了错误在长程上下文中的累积路径。

第三道防线则是**保全前缀的局部后缀修复（Suffix Repair）**。当某个节点经过多次重试依然不可挽回地失败，或者在最终质检阶段发现某些波段的证据链不完整时，系统不会触发代价高昂的全局重置。相反，控制模块会锁定已经成功的历史节点序列，保留这部分计算进度与中间产物作为前缀，仅将出错节点及其后续拓扑结构提取出来交给 Planner。Planner 会根据失败诊断结果，动态合成一段替代性的“局部后缀”（Suffix），并严丝合缝地拼接回主干工作流中。这种机制大幅降低了真实环境下与遥感计算环境交互的时间损耗与令牌成本。

### 角色特化训练：把规划与具象化彻底分开

有了精密的运行架构，底层的语言模型又该如何适配？过去很多端到端微调方法试图用一个统一的轨迹损失函数来训练模型，但这实际上混淆了两种本质不同的认知行为：**全局的工作流拓扑推演**与**局部的工具参数精细落地**。把这两者混在一起，只会导致长程信用分配（Credit Assignment）极度模糊。

Earth-Agent-Pro 采取了“各司其职、分道扬镳”的训练策略，分别训练两个专职 Adapter，并将底座模型完全冻结。

<img src="/images/2609.12533/fig3.webp" alt="依赖保持的合成训练数据生成管线" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了支撑这一训练体系，研究人员首先基于 248 个专家工作流种子，通过大模型编写代码提取出不可变骨架（保留工具依赖与算子拓扑）与可变实例状态（替换经纬度范围、时间跨度、产品源等），自动化合成了 2,480 条严格保持拓扑依赖的高质量轨迹。

对于负责宏观编排的 **Planner**，采用**序列级监督微调（Sequence-level SFT）**。Planner 的上下文仅接收自然语言问题、当前激活的技能规范以及候选工具 Schema，目标是自回归生成标准的工作流节点序列。在这个阶段，完全屏蔽任何运行时的文件路径与具体参数，使 Planner 专心学习遥感任务的步骤分解与算子协同规律：




{% raw %}$$\mathcal{L}_{\mathrm{plan}}(\psi_{\mathrm{P}})=-\mathbb{E}_{\mathcal{D}_{\mathrm{P}}}\sum_{\ell}\log p_{\psi_{\mathrm{P}}}(\omega_{\ell}^{\mathrm{P}}\mid x^{\mathrm{P}},\boldsymbol{\omega}_{<\ell}^{\mathrm{P}})$${% endraw %}



对于负责微观落地的 **Executor**，则采用了基于**组相对策略优化（GRPO）的强化学习训练**。Executor 面对的是一个确定的单步节点，它的职责是在给定当前上下文、上游产物引用和工具 Schema 的前提下，输出绝对合规且精确的参数字典。由于工具本身是固定的，评测可以通过纯规则对生成的参数进行本地确定性验证。研究团队设计了分层奖励函数：格式崩溃（如 JSON 损坏）给予最重惩罚 $-2$；未能满足参数 Schema 给予 $-1$；参数完全合规且键值命中参考状态则根据吻合程度给予正向奖励。通过在同节点下采样多组候选参数计算相对优势，Executor 迅速学会了如何在遥感领域复杂的地理坐标系、传感器通道代号与文件路径之间实现精准接地（Grounding）。

### 实测检验：全链条落地的真实表现

在 Earth-Bench-Pro 的全套测试中，Earth-Agent-Pro 展现出了显著超越传统智能体框架的鲁棒性。

在使用 GPT-5 作为底座模型的严苛评测中，面对真实开放的 Earth-Bench-OW 任务，传统的 ReAct 范式由于工具调用顺序混乱和长程状态丢失，不仅在执行正确性上频频受挫，在工具调用时序一致性（Tools-In-Order）指标上也表现不佳。而 Earth-Agent-Pro 凭借专家技能的先验锚定与结构化记忆的稳健驱动，在最终的 LLM-as-Judge 科学结论准确率上达到了 66.13%，足足超出 ReAct 达 **20.95 个百分点**；在工具序列合规度上更是建立了 **24.44 个百分点**的巨大优势。

更具实用价值的发现体现在开源小参数模型上。对于未经过特化训练的原生 Qwen3.5-9B，在面对复杂的遥感科学分析时往往不知所措，在开放世界评测中的准确率仅有 38.31%。而在接入了分别经过 SFT 和 GRPO 强化的双 Adapter 之后，该模型的准确率直接攀升到了 50.00%，净胜 **11.69 个百分点**。

后续的消融实验进一步印证了这种角色解耦训练的必要性。当研究团队将经过微调的 Planner 置于“纯规划”场景下测试时，其序列合成质量相较基线显著提高；而在固定标准工作流、纯粹测试参数具象化能力时，经过 GRPO 训练的 Executor 展现出极高的一致性，有效抑制了模型在经纬度范围拼写、波段名称对应等细节上的幻觉。两者相辅相成，构成了长程科学任务中不可或缺的推拉力量。

### 从静态感知到自动科研的范式演进

回顾整个技术方案，Earth-Agent-Pro 的核心价值不仅在于几项评测指标的刷新，更在于它为垂直领域的专业智能体落地提供了一套具有普遍参考意义的方法论。

传统的 Agent 开发者往往对 LLM 的泛化推理寄予过高期望，试图通过“大模型 + 完整 Prompt + 自由交互循环”来包打天下。然而真实世界的科学与工业工程，其内在逻辑是高度严谨且充满专业约束的。Earth-Agent-Pro 证明了：将**领域专业规范以 Skill 的形式显式下沉**、将**执行证据与失败日志以结构化形式强行解耦**，并针对**宏观规划与微观接地分配不同的训练范式**，才是突破长程复杂任务“瓶颈”的真正钥匙。

当遥感智能体不再局限于识别一张给定图片里有几架飞机，而是能够自主从卫星数据归档中心调取数年尺度的立体观测数据，在云端拉起一套多级计算管线并产出严谨的科学报告时，人工智能赋能对地观测才真正迈出了走向自主科学发现的关键一步。
