---
layout: default
title: "告别黑盒！GaP多智能体生成计算图，机器实操成功率飙升至99%"
description: "当前，端到端的视觉-语言-动作大模型在通用机器人领域风头正盛。然而，当这些模型真正走入商业和工业流水线时，往往会遭遇“水土不服”。在真实的工厂或商用厨房中，机器人需要日复一日地执行极高可靠性的任务。传统的“黑盒”大模型缺乏可解释性，一旦目标物体的位置或姿态发生微小偏移，成功率便会断崖式下跌。"
arxiv_id: "2607.05369"
topics:
  - "AI Agent"
tags:
  - "Directed computation graphs"
  - "GaP"
  - "Internal simulation rehearsal"
  - "MORSL"
  - "Multi-agent self-learning harness"
  - "ROS"
related_tutorials:
  - "ai-agent-systems-architectures-applications-and-evaluation"
  - "llmtimesmapreduce-v3-enabling-interactive-in-depth-survey-generation-through-a-m"
  - "code-as-agent-harness"
  - "measuring-harness-induced-belief-divergence-in-multi-step-llm-agents"
---

<p class="paper-original-title" lang="en" style="font-size:1rem; line-height:1.5; color:var(--global-text-color-light, #6c757d); margin:-0.5rem 0 1.5rem;">GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks</p>

当前，端到端的视觉-语言-动作大模型在通用机器人领域风头正盛。然而，当这些模型真正走入商业和工业流水线时，往往会遭遇“水土不服”。

> **ArXiv URL**：http://arxiv.org/abs/2607.05369v1

在真实的工厂或商用厨房中，机器人需要日复一日地执行极高可靠性的任务。传统的“黑盒”大模型缺乏可解释性，一旦目标物体的位置或姿态发生微小偏移，成功率便会断崖式下跌。

为了跨越这道可靠性鸿沟，Bosch、卡内基梅隆大学、NVIDIA与加州大学伯克利分校的研究团队提出了一种全新范式。

该研究摒弃了纯粹的端到端黑盒，引入了名为 **GaP**（**Graph-as-Policy**） 的多智能体自学习框架。它巧妙融合了传统机器人工程的可解释性与大模型的开放世界适应能力。

### 变分自动化任务

在探讨 GaP 的核心机制前，我们需要先理解该研究界定的一个关键场景：**变分自动化**（**Variational Automation, VA**）。

传统的固定自动化（如汽车点焊）盲目重复相同的动作，环境极度单一。而变分自动化则不同，它要求机器人在已知的工作空间中，持续处理几何形状和姿态存在非平凡变化的对象。

例如在商业厨房制作三明治，或在物流中心分拣包裹。这类任务既不能完全依赖死板的预设程序，也无法承受纯数据驱动模型带来的试错成本与幻觉风险。

在 VA 任务中，工作台、机器人和传感器是固定的。物体的种类和初始姿态范围是已知的。这为引入模块化技能和物理仿真提供了绝佳的先决条件。

### Graph-as-Policy：策略即计算图

面对复杂的 VA 任务，如果仅依赖单个大语言模型直接生成 Python 控制代码，随着上下文窗口的膨胀，模型极易出现约束违规或“幻觉”。

为了解决这一难题，GaP 引入了有向计算图 $\mathcal{G}=(V,E)$ 的概念。

我们可以将这个生成过程比作建立一条“模块化加工流水线”。

在这个流水线中，节点（$V$）相当于各个标准的“加工站”。它们是从 **模块化开放机器人技能库**（**Modular Open Robot Skill Library, MORSL**） 中提取的原子功能单元。

这些技能包含了感知模型（如 SAM2、Grounding DINO）、抓取规划（GraspGen）、运动规划（cuRobo）等 51 种经过验证的工具。

边（$E$）则相当于“传送带”与“控制阀”。数据边负责将上一个加工站的输出传递给下一个节点，控制边则根据条件判断执行走向。

<img src="/images/2607.05369v1/x1.webp" alt="[Uncaptioned image]" style="width:85%; max-width:600px; margin:auto; display:block;">

通过这种图结构，GaP 极大地限制了单个智能体的上下文负担，并杜绝了模型虚构不存在技能的“作弊”行为。

### 多智能体协作与内部仿真自学习

那么，这条“流水线”是如何被设计和优化的呢？GaP 采用了一个层级化的多智能体调度系统。

首先，**编排智能体**（**Orchestration Agent**） 作为总工程师，接收自然语言任务指令。它会将宏观任务拆解为语义片段（如“打开旋钮”、“拿起爆米花锅”）。

随后，**技能智能体**（**Skill Agents**） 作为各部门负责人，针对分配到的片段，从 MORSL 库中挑选合适的节点，合成局部的功能子图。总工程师再将这些子图拼接成完整的初始执行图 $\mathcal{G}_0$。

这仅仅是开始。GaP 最核心的亮点在于其 **自学习**（**Self-Learning**） 机制。

框架会在内部基于 NVIDIA Isaac 物理模拟器生成仿真环境。针对目标任务，系统会并行采样 $N$ 个具有不同物体初始位姿的任务实例 $\tau_i$。

GaP 会在仿真中反复“彩排”这些实例。如果某次执行失败，系统会记录物理执行数据，定位几何层面的根本原因。

随后，智能体们会触发图更新机制。它们会迭代地修改计算图架构，例如替换功能相似的节点、调整连接边，或是修改代码参数。这一过程会持续进行，直到任务的成功率与吞吐量达到平台期。

<img src="/images/2607.05369v1/rehearsal-new.webp" alt="Refer to caption" style="width:90%; max-width:700px; margin:auto; display:block;">

### 核心实验与惊人表现

为了评估 GaP，研究团队设计了 8 个全新的变分自动化基准测试（4 个在仿真中，4 个在现实世界中）。

在处理杂货订单与打包杂货的任务中，当目标物体的位置发生变化时，现有的视觉-语言-动作模型如 $\pi_{0.5}$ 和 MolmoAct2 的成功率暴跌至 0.20。

相比之下，GaP 展现出了惊人的鲁棒性，在各类位姿变化下依然保持着 0.93 至 0.99 的超高成功率。

更有趣的是，GaP 还可以作为现有端到端模型的“辅助轮”。当结合 GaP 优先执行摄像机居中移动后，$\pi_{0.5}$ 重新回到了其熟悉的分布范围内，成功率实现了翻倍增长。

在真实的物理实验中，GaP 的表现同样优异。

在真实世界完成杂货订单任务时，传统的任务与运动规划系统 TipTop 仅有 8/25 的成功率。因为它无法为复杂形状的物体和高篮子找到可行的运动学解。而 GaP 达成了 25/25（100%）的完美成功率。

在工业气息更浓的 **插入USB-C线缆**（**Insert Cables**） 真实任务中，由于引入了 ROS 节点，GaP 在 130 次试验中取得了 0.93 的高成功率。

<img src="/images/2607.05369v1/usb-insertion_setup.webp" alt="Refer to caption" style="width:85%; max-width:450px; margin:auto; display:block;">

而在仿真的 **清洗周转箱**（**Wash Crates**） 双臂协作任务中，GaP 自主生成的策略图达到了 0.953 的成功率。这几乎媲美了人类专家手工精心调试的策略性能（0.987）。

### 局限与工程启示

GaP 的成功表明，在工业应用中，大模型不应被视为直接输出机器指令的“黑盒控制器”。

相反，大语言模型更适合作为“高级架构师”。它们通过编写和优化结构化的计算图，来调用经过严密数学与物理验证的底层算法库。

不过，该研究也坦言了目前的局限性。

首先是执行效率。尽管成功率极高，但 GaP 目前的执行周期仍远低于工业标准界定的高吞吐量（例如每件 7 秒以内）。VLM 推理请求和逆运动学规划的高昂时间成本仍需进一步压缩。

其次，当前的 8 个基准测试主要集中在准静态的抓取与放置任务上。除了线缆插入涉及力感知外，未来还需要探索 GaP 在处理柔性物体、动态受力以及移动目标等更复杂场景下的表现。

总之，GaP 框架为弥合经典工程（GOFE）与前沿大模型之间的鸿沟提供了一条极具潜力的可行路径。随着技能库的持续丰富，这种策略即图的范式必将在未来的机器人自动化浪潮中占据重要的一席之地。
