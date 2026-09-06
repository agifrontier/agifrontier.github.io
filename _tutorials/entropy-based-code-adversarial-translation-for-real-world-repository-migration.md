---
layout: default
title: "ECAT：解耦生成与评估，华为等将安卓转鸿蒙迁移质量提升至74.7%"
description: "为了攻克工程级长程代码迁移的难题，华为技术有限公司与香港中文大学的研究团队提出了一种基于代码对抗熵最小化的多智能体协同框架—— ECAT （Entropy-based Code Adversarial Translation）。"
arxiv_id: "2608.09273"
paper_published: "2026-08-10"
published_at: "2026-09-06T13:15:07.886648+08:00"
topics:
  - "AI安全"
tags:
  - "A2H-RepoBench"
  - "Android-to-HarmonyOS"
  - "Code Entropy"
  - "ECAT"
  - "adversarial entropy minimization"
  - "generator-discriminator"
related_tutorials:
  - "statistical-reinforcement-learning-in-the-real-world-a-survey-of-challenges-and-"
  - "toolllm-facilitating-large-language-models-to-master-16000-real-world-apis"
  - "empowering-real-world-a-survey-on-the-technology-practice-and-evaluation-of-llm-"
  - "mai-ui-technical-report-real-world-centric-foundation-gui-agents"
---

<p class="paper-original-title" lang="en">Entropy-based Code Adversarial Translation for Real-world Repository Migration</p>

<img src="/images/2608.09273v2/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在大模型驱动软件工程的浪潮中，单文件函数生成和针对小型代码仓库的缺陷修复已经取得了令人瞩目的进展。然而，当工程师试图让大模型接管整个工程代码库的跨平台重构时，现有的多智能体系统往往难以交付一个真正可运行的软件。尤其是在“安卓（Android）迁移至鸿蒙（HarmonyOS）”这类跨平台、跨框架的系统级迁移场景中，工程代码量动辄从数万行飙升至数十万行。此时，不仅编程语言从 Java/Kotlin 变为 ArkTS，UI 范式、底层生命周期、硬件通信 API 以及模块依赖也发生了颠覆性的变化。错误在成百上千个跨文件调用间层层传导，极易让现有的代码智能体迷失在长程推理的泥潭中。

> ArXiv URL：https://arxiv.org/abs/2608.09273v2

为了攻克工程级长程代码迁移的难题，华为技术有限公司与香港中文大学的研究团队提出了一种基于代码对抗熵最小化的多智能体协同框架——**ECAT**（Entropy-based Code Adversarial Translation）。该方法跳出了传统代码生成系统“由同一个智能体边写边自我检查”的常见范式，借鉴生成对抗网络的思路，将代码仓迁移抽象为一种对抗性的信息熵缩减过程。与此同时，研究团队开源了首个面向真实工程级跨平台迁移的基准测试集 **A2H-RepoBench**，覆盖从 5 万行到 30 万行代码规模的工业级安卓项目。在严苛的功能评估体系下，ECAT 将综合迁移质量从现有最强基准的 46.4% 大幅提升至 74.7%，并成功验证了跨项目经验的自主演化与知识迁移。

<img src="/images/2608.09273v2/x1.webp" alt="ECAT 架构总览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 为什么现有智能体做不好整仓跨平台迁移？

要理解 ECAT 的突破，首先需要剖析大模型智能体在长周期跨平台重构时屡屡失利的本质机理。

此前主流的代码智能体多依赖于迭代式自我反思（Self-refinement）机制。在面对一个模块时，智能体一边撰写目标平台的代码，一边通过编译器报错或内置提示进行反思并打补丁。在规模较小的任务中，这种自问自答的方式行之有效；但在十几万行代码的真实项目中，这种机制暴露出致命缺陷。

由于大模型普遍存在自我确认偏差（Self-confirmation Bias），让生成代码的同一个智能体去自我审查，极易高估未完成代码的完整度。许多智能体为了平息编译报错，倾向于在跨平台接口处写出大量的空占位符（Stubs）或浅层逻辑。表面上看，项目没有语法报错甚至通过了基础结构解析，但一旦部署运行，内部的数据流与业务逻辑完全断裂。不仅如此，现有的自动化迁移流水线缺乏有效的跨项目记忆积累，每一次面对新代码仓库，智能体都在从零探索相同的适配陷阱，算力消耗巨大且收敛极慢。

### 对抗熵最小化：把代码仓的混乱程度显式量化

ECAT 没有把代码迁移视作单纯的语言翻译任务，而是将其重新定义为一个对抗熵最小化问题：




{% raw %}$$ \pi^*=\arg\min_{\pi}\mathbb{E}\!\left[H(\mathcal{R}^{T})\right] $${% endraw %}



在大型工程由安卓向鸿蒙转换的过程中，任何未解决的跨文件依赖、未适配的资源文件以及隐藏的接口悬空，都会在宏观上体现为目标代码仓 $\mathcal{R}^{T}$ 的“混乱度”。作者团队借用信息熵的比喻，提出了用于度量代码混乱状态的综合指标——**Code Entropy（代码熵）**。代码熵 $H(\mathcal{R}_t)$ 并不是一个抽象的单一分数，而是静态维度与动态维度的规范化加权聚合：




{% raw %}$$ H(\mathcal{R}_{t})=\frac{\mathbf{W}^{\top}\mathbf{E}}{\mathbf{1}^{\top}\mathbf{W}} $${% endraw %}



其中，静态熵负责拦截代码编译错误、项目目录结构错位以及跨平台 AST 迁移失真；而动态熵则更为苛刻，它依赖于真实鸿蒙仿真器中的自动化部署结果，抓取运行时的崩溃堆栈、空白渲染以及 UI 状态异常。只有在静态与动态两个维度均被完全熨平的前提下，代码熵才会趋向于零，这意味着目标仓库不仅结构完整，而且在行为上真实可运行。

有了清晰的量化标尺，ECAT 进一步搭建了由生成器智能体 $G$ 与判别器智能体 $D$ 构成的对抗闭环。在每个迭代轮次中，判别器充当极其苛刻的代码审查官，它不仅负责计算当前代码熵，更会生成一种被称为**文本梯度**（Text Gradients）的高阶反馈信号 $\mathbf{g}_t$：




{% raw %}$$ (H_t, \mathbf{g}_t) = D(\mathcal{M}_t; \mathcal{R}^S, \mathcal{R}_t) $${% endraw %}



这种文本梯度不仅精确指出代码仓库中存在逻辑缺陷的文件位置与根因，还会明确指出修复该缺陷所必需的技能与范式。生成器根据这一指令更新代码仓库；如果更新后的代码熵未能有效降低，新版本将被无情拒绝并回滚：




{% raw %}$$ \mathcal{R}_{t+1}=\begin{cases}\widehat{\mathcal{R}}_{t+1},&H(\widehat{\mathcal{R}}_{t+1})<H(\mathcal{R}_{t}),\\ \mathcal{R}_{t},&\text{otherwise}\end{cases} $${% endraw %}



值得一提的是，在每一次迭代中，生成器和判别器都会在隔离的上下文中重新实例化，两者无法互相读取中间思维链（Reasoning Traces）。这种物理级别的上下文隔离，彻底切断了自我确认偏差的形成路径，倒逼生成器必须拿出经得起多维度检验的修复方案。

### 告别扁平存储：自演化记忆树让踩坑经验可迁移

在大模型工程应用中，长期记忆通常采用扁平的向量检索或文本拼接存储。但随着迁移迭代次数的增加，这种粗放式的记忆管理会迅速撑爆模型的上下文窗口，引入大量无关噪声。

<img src="/images/2608.09273v2/x2.webp" alt="自演化记忆树架构" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

ECAT 提出了一种**自演化记忆树**（Self-evolving Memory Tree）结构。该记忆树由根节点、中间分类节点和叶节点组成：

1. **根节点**存放全局跨平台迁移规范与宏观策略；

2. **中间节点**按照粗粒度的缺陷类型进行聚类，例如页面路由导航适配、跨平台存储抽象层重写、空接口检测等；

3. **叶节点**则收录经过成功低熵轨迹提炼出的具体修复模式，包含代表性缺陷、根本原因分析以及完备的验证标准。

关键在于，叶节点在沉淀经验时会主动剔除与特定业务强绑定的代码细节，只保留具备泛化价值的通用迁移模式。在生成器和判别器工作时，智能体沿着树形索引逐级检索，只加载当前缺陷相关的上下文分支，既保证了召回的精准度，又将上下文负载降到了最低。更重要的是，在一套开源项目上积累的记忆树，可以直接挂载到全新的目标代码库中，使智能体拥有了“在实战中学习并自我进化”的工程经验沉淀能力。

### A2H-RepoBench：真实复杂项目下的硬核检验

为了验证整仓级跨平台迁移的真实水平，研究团队构建了目前业内首个长程安卓转鸿蒙真实基准 **A2H-RepoBench**。

基准挑选了三个代表性的开源安卓项目，覆盖了不同的代码规模与技术栈深度：

- **Gallery（5 万行代码）**：多媒体相册应用，重点考查复杂 UI 交互范式转换与本地媒体资源管理；

- **AntennaPod（12 万行代码）**：主流播客客户端，涉及大量后台音频流播放、复杂的异步任务调度以及持久化数据库同步；

- **Meshtastic（30 万行代码）**：离线网状通信系统，高度依赖底层蓝牙硬件交互、底层协议解析与繁复的长跨度文件引用。

为了摆脱“能否编译成功”这种低维度评价指标，A2H-RepoBench 设立了双重严苛指标：一方面是通过抽象语法树计算跨平台代码结构的**节点对齐度**（Node Alignment）；另一方面则是采用基于真实用例清单的 **Agent-as-Judge**，在鸿蒙仿真器中逐项验证实际功能的运行情况。值得注意的是，负责评判的测试智能体拥有完整的独立功能检查表，而这一检查表在迁移过程中对 ECAT 的判别器是严格不可见的，杜绝了数据泄露。

底座模型采用 DeepSeek-V4-Pro 配合多模态模型 Qwen3.7-Plus（负责真机仿真器截图的动态熵分析）。在三个规模梯度的严苛对抗中，ECAT 展现出了压倒性的优势。

在 AntennaPod（12 万行代码）的主界面迁移对比中，传统的基线方案要么遭遇布局塌陷，要么遗漏了底部的导航栏与交互微件。而 ECAT 在完整的对抗修复后，生成的鸿蒙 ArkTS 界面在视觉布局、数据列表渲染以及交互控件的保真度上，几乎完美复刻了原版安卓应用的体验。

<img src="/images/2608.09273v2/x3.webp" alt="AntennaPod 迁移 UI 质感对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

即使在高达 30 万行代码的 Meshtastic 上，面对极为庞杂的底层驱动调用，ECAT 依旧保持了极高的结构对齐度与功能可运行性。对比实验进一步表明，如果剥离掉基于真机画面的“动态熵”反馈，模型在 AntennaPod 这类重度依赖运行时异步调度的项目上评分明显下滑；静态分析可以保证语法无误，但只有动静结合的熵减闭环，才能杜绝空白屏幕和运行期静默崩溃。

### 经验迁移的价值：收敛步数与推理开销近乎减半

记忆树的跨项目迁移能力在消融实验中得到了更为直观的数据印证。在对 Gallery 仓库的迁移实验中，研究团队将另外两个项目（AntennaPod 与 Meshtastic）优化成功后沉淀出的自演化记忆树作为先验输入，观察系统的优化动力学变化。

<img src="/images/2608.09273v2/x4.webp" alt="优化动力学与 Token 消耗对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

<img src="/images/2608.09273v2/x5.webp" alt="优化动力学与 Token 消耗对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

从上方的代码熵收敛曲线（a）可以清晰看到，在初期阶段，各类跨平台的系统级宏观缺陷被快速修补，代码熵呈现出陡峭的断崖式下降；随着迭代深入，剩余的细微边界错误被逐步清除，熵值趋于平缓。在未加载记忆树（w/o MT）的情况下，系统需要进行 66 轮对抗博弈才能使代码熵达标收敛；而借助迁移而来的结构化记忆，ECAT 在第 38 轮就成功锁定了低熵状态。

这种加速直接反映在推理成本上。如图（b）所示，引入记忆树后，达成同等迁移质量所需的累积 Token 消耗量从大约 125M 骤降至约 65M，开销近乎腰斩。这表明，树状结构沉淀下来的跨平台接口映射与缺陷处置方案，有效避免了生成器在未知暗礁中盲目试错，使大模型智能体真正具备了工业级项目间的“经验复用”属性。

### 从单纯代码生成到工程系统治理

跨平台代码库迁移，本质上是大模型走向全自动软件工程（Agentic Software Engineering）必须跨越的一座高山。它不再是一道 LeetCode 式的单题算法求解，而是在巨大的依赖图谱、异步状态机和异构运行环境中维持高度秩序的复杂系统工程。

ECAT 带来的核心启示在于：**面对极度庞杂的长程任务，试图依靠单一智能体的自我完善，终究会受困于上下文噪声与自我评判的盲目性。** 将“生成代码”与“找茬诊断”在架构层面上坚决切断，引入可显式优化的度量标尺（代码熵），并通过结构化记忆将单次成功经验转化为可迁移的资产，是让大模型稳定驾驭几十万行代码级任务的一条有效路径。随着 HarmonyOS 原生应用生态的快速扩张，这种全自动、可闭环、具备经验自演化能力的迁移范式，展现出了极具想象力的工业落地前景。
