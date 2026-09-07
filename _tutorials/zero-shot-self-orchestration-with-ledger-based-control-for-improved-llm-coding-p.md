---
layout: default
title: "不是模型越大越好：Zero-Shot自编排以1/5成本逼平顶级模型"
description: "本文提出的架构则走出了一条轻量级的动态编排路线。整个系统由 Manager 角色主导，但 Manager 不负责硬编码的流水线，而是维护一个共享文件空间作为工作记忆。针对输入的代码难题，Manager 首先调用生成式提示词生成初步方案与头脑风暴，建立待办任务清单（Task Ledger）。"
arxiv_id: "2608.26480"
paper_published: "2026-08-27"
published_at: "2026-09-07T13:15:08.344824+08:00"
topics:
  - "基础模型"
tags:
  - "LLM"
  - "Ledger-Based Control"
  - "LiveCodeBench"
  - "Zero-Shot Self-Orchestration"
  - "context management"
  - "manager-worker scaffold"
related_tutorials:
  - "zero-shot-performance-prediction-for-probabilistic-scaling-laws"
  - "livecodebench-holistic-and-contamination-free-evaluation-of-large-language-model"
  - "molecular-representations-for-large-language-models"
  - "predicting-task-performance-with-context-aware-scaling-laws"
---

<p class="paper-original-title" lang="en">Zero-Shot Self-Orchestration with Ledger-Based Control for Improved LLM Coding Performance</p>

<img src="/images/2608.26480v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在多智能体（Multi-Agent）系统席卷大模型应用的今天，几乎所有宣称“超越单模型”的方案都笼罩在一种模糊的技术迷雾中。业界普遍的直觉是，由多个角色组成的团队在分工协作、互相校验和汇总成果时，必然能超越单次推理的基准模型。然而，大量既有研究的对比实验往往夹杂着严重的混淆变量：研究者在引入多智能体框架的同时，悄悄改变了 Token 消耗预算、提示词工程策略、外部工具调用甚至检索增强方式。最终，整体性能提升究竟来自智能体协作本身，还是仅仅因为多花了数倍的计算量与更充沛的提示词信息，往往难以厘清。

> ArXiv URL：https://arxiv.org/abs/2608.26480v1

来自 Persis Capital 的最新研究《Zero-Shot Self-Orchestration with Ledger-Based Control for Improved LLM Coding Performance》针对这一问题展开了严格的控制变量评估。研究团队构建了一个基于共享文件系统与任务账本的 Manager-Worker（管理者-执行者）控制框架。整个架构既不需要对协调策略进行任何训练，也不针对特定基准进行微调，所有角色完全由同一个基座模型在各自独立的上下文中零样本（Zero-Shot）运行。在最新的 100 道 LiveCodeBench 困难级别编程题目上，该研究跨越 9 款模型进行了严格对照评测。实验证明：多智能体编排带来的性能收益是真实的，但具有高度的条件性；它大致让 Token 账单翻了三倍，却以极具性价比的方式打破了模型能力的成本天花板——在搭载该框架后，中等规格的 GPT-5.6-Terra 仅花费约五分之一的成本，就逼平了顶尖闭源模型 Claude Fable 5 的单次调用准确率。

### 抛弃死板流水线：基于共享账本的零样本自编排

以往的多智能体框架大致可以归结为两类极端方案：一类如 Sakana AI 的 Conductor 等依赖强化学习专门训练调度器，训练成本和通用性门槛极高；另一类如 MetaGPT、ChatDev 等预设固定的 SOP（标准作业程序），通过产品经理、架构师、程序员等刚性流程串联，无法根据题目的实际进展动态调优。

<img src="/images/2608.26480v1/how_the_manager_arm_answers_one_question_multiagent_py_current_tree_light.webp" alt="Manager-Worker 架构执行流程图" style="width:min(600px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; display:block;">

本文提出的架构则走出了一条轻量级的动态编排路线。整个系统由 Manager 角色主导，但 Manager 不负责硬编码的流水线，而是维护一个共享文件空间作为工作记忆。针对输入的代码难题，Manager 首先调用生成式提示词生成初步方案与头脑风暴，建立待办任务清单（Task Ledger）。在后续的每一轮迭代中，Manager 动态审视当前产物，调整任务列表，向 Worker 发起具体的子任务委托。每一个 Worker 都是该基座模型的一次全新调用，拥有干净独立的上下文，只接收与当前子任务相关的规范协议、当前代码产物与累积笔记。Worker 执行完成后将产物写入共享区，Manager 随后检查结果并决定下一步行动，直至验证通过或触发终止。

这种设计的巧妙之处在于，它完全依赖通用大模型的零样本规划能力进行自组织调度，没有任何经过微调的路由器或固定的多轮对话历史包袱，从而将变量完全限制在“框架组织方式”与“推理期计算量投入”本身。

### 真实提升有多大？前沿模型硬核实测

为了彻底消除接口路由波动造成的测试噪声，研究团队将最具代表性的四款模型固定在官方端点与本地受控实例上，在统一的 128k 最大输出 Token 限制且开启思考推理（Reasoning On）的条件下，对每道题目独立跑满 5 次配对测试。

<img src="/images/2608.26480v1/manager_vs_single_call_four_models_lcb-100_5_passes_128k_max_tokens_reasoning_on_bars_light.webp" alt="四款模型在 128k 思考开启下的五轮测试表现" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

测试结果展现出非常强烈的统计显著性，但各模型受益程度呈现出显著的分层。开源阵营中的 Qwen3.8-27B 表现最为惊人：在单次调用基线中，其 pass@1 准确率仅为 $63.0\%$，但在引入 Manager 编排框架后直接拉升到 $86.4\%$，净增 $23.4$ 个百分点。在逐题胜负配对统计中，Manager 架构在 125 次测试中战胜了单次调用，而仅有 8 次落败，胜负比接近十六比一。

闭源商业模型同样从中获得了确凿的提升。GPT-5.6-Luna 从 $67.2\%$ 提升至 $77.8\%$（净增 $10.6$ 个百分点），GPT-5.6-Terra 则从 $77.0\%$ 稳步提升至 $85.0\%$（净增 $8.0$ 个百分点）。对比作为单次调用顶尖天花板的 Claude Fable 5（单次调用得分为 $87.4\%$），GPT-5.6-Terra 借助该框架将差距缩小到统计学上无法区分的范畴（$p = 0.59$）。在早期单轮全尺寸对比中，顶级的 Opus-5 在 Manager 框架协助下拿下了全场最高的 $91\%$ 准确率。

### 成本帕累托前沿：花 3 倍 Token 换取 5 倍性价比

引入多智能体框架必然伴随计算资源的额外开销。数据显示，Manager 带来的规划、头脑风暴、验证和多次 Worker 迭代，导致 Qwen3.8-27B 的 Token 消耗增加了 $153\%$（每轮从 20.44 美元折算成本上升至 51.75 美元），GPT-5.6-Luna 增加了 $266\%$，GPT-5.6-Terra 增加了 $244\%$。粗略来看，自编排框架让整体 Token 账单涨到了单次调用的三倍左右。

然而，如果从准确率与总花费构成的帕累托前沿（Pareto Frontier）来看，这种额外消耗展现出极具说服力的经济效益。

<img src="/images/2608.26480v1/what_accuracy_costs_lcb-100_5_passes_128k_max_tokens_reasoning_on_all_seven_pinned-backend_arms_light.webp" alt="推理成本与代码准确率的帕累托前沿对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

实验揭示了一个大模型应用落地中非常关键的经济学法则：直接购买更大参数量的昂贵模型，其边际准确率成本极为陡峭；而将预算花在中等规格模型的推理期编排上，性价比反而显著占优。以 GPT-5.6-Terra 加上 Manager 框架为例，其在 100 题基准上的实测开销为 11.71 美元，准确率达到 $85.0\%$；而直接单次调用 Claude Fable 5 的花费高达 61.11 美元，准确率仅为 $87.4\%$。两者准确率没有统计学差异，但前者的运行成本仅有后者的五分之一不到。

更具实用价值的是，开源模型 Qwen3.8-27B 依靠自编排框架拿下了 $86.4\%$ 的高分，单轮 100 题测试成本约为 51.75 美元。这意味着任何拥有自建算力基础设施的机构，完全不需要依赖昂贵且不可控的商业闭源 API，仅凭可自托管的开源权重加上一套零样本编排逻辑，就能在顶尖困难代码竞赛题上逼近前沿模型单次调用的第一梯队水平。

### 收益从何而来？截断救援与认知解耦的真实机理

多智能体究竟为什么能在不更新模型权重的情况下击败单次推理？研究团队对模型输出轨迹的深入分析排除了许多玄学假设，锁定了两个核心物理机制：

首先是**上下文管理与截断救援（Truncation Rescue）**。在极其复杂的竞赛级代码生成中，单次调用的模型往往需要在长程思维链中展开巨量推演，极易撞上推理提供商设置的单次调用 Token 上限。一旦超出上限，模型会被强制截断（`finish_reason=length`），直接导致未能输出可执行代码而被判 0 分。在单次调用的 Qwen3.8-27B 中，500 次题目测试里有 150 次触发了截断，最终导致 35 道题完全交白卷。然而在 Manager-Worker 体系下，Worker 每次只处理被高度精简的子任务，单次调用长度大幅压缩。即便某个 Worker 意外被截断，损失的也只是一次中间任务迭代，Manager 能够捕获异常并在下一轮重新指派，从而实现对“长尾失败”的系统级吸收。数据显示，单次调用因超长而弃考的 35 道题中，Manager 成功挽救并答对了 25 道题，仅此一项就为 Qwen3.8-27B 贡献了 5.0 个百分点的净收益（占其总提升的约五分之一）。

<img src="/images/2608.26480v1/manager_vs_single_call_across_model_scale_lcb-100_5_passes_16k_max_tokens_reasoning_off_bars_light.webp" alt="不同模型规模在无推理模式下的 5 轮表现" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

其次是**问题分解（Problem Decomposition）**。当截断因素被排除后，剩余的性能增量完全来自于分步解构的收益。在关闭显式思维链（Reasoning Off）的实验组中，模型的单步逻辑推理能力受到限制，此时 Manager 的解题调度展现出更加夸张的放大效应。在 16k 较短输出限制下，无长推理机制的 Kimi-K3 凭借 Manager 协助，准确率暴增了 $30.4$ 个百分点（$p < 2\times 10^{-5}$），Minimax-M3 提升了 $11.0$ 个百分点。

但值得警惕的是，框架的增益并非普适法则。实验发现，对于某些基础遵循能力稍逊的模型（如 Qwen3.6-35B），引入 Manager 甚至导致性能下滑（下降 1 到 9 个百分点）。这是因为如果模型自身无法严格理解并执行分工合同，Manager 分配任务时的指令偏差和 Worker 的偏题输出会在多轮迭代中互相放大，产生严重的协调噪声（Coordination Tax）。

这篇研究所呈现的完整证据链表明，多智能体系统既非包治百病的银弹，也不是纯粹的数字游戏。其本质是一种通过在测试期增加计算量，以结构化上下文管理换取解空间稳定性的工程机制。当模型本身具备足够的指令遵从度、但受制于单次输出长度和上下文退化时，使用轻量级的零样本自编排框架，往往能以远低于升级更大模型维度的经济成本，实现跨层级的能力跃迁。
