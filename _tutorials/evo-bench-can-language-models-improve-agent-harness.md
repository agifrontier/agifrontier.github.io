---
layout: default
title: "Evo-Bench：让大模型自主重构支架，最高提升16.6分逼近人类专家"
description: "由中国人民大学 AI Box 团队与 BOSS 直聘等机构联合提出的研究给出了坚实的实证回应。他们推出了首个专门用于评测大模型自主迭代 Agent 支架能力的基准测试 Evo-Bench 。"
arxiv_id: "2608.09096"
paper_published: "2026-08-10"
published_at: "2026-09-08T13:15:08.162312+08:00"
topics:
  - "AI Agent"
tags:
  - "Evo-Bench"
  - "LLMs"
  - "autonomous evolution"
  - "auxiliary-task evolution"
  - "harness evolution"
  - "harness transferability"
related_tutorials:
  - "a-survey-on-large-language-model-based-autonomous-agents"
  - "evo-harness-context-to-harness-skill-compilation-for-self-evolving-agents"
  - "multi-agent-evolve-llm-self-improve-through-co-evolution"
  - "swe-bench-can-language-models-resolve-real-world-github-issues"
---

<p class="paper-original-title" lang="en">Evo-Bench: Can Language Models Improve Agent Harness?</p>

<img src="/images/2608.09096v2/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

过去两年里，AI Agent 领域最被广泛接受的共识之一，是“决定 Agent 上限的往往不是基础模型本身，而是包裹在模型外面的工程支架”。无论是最早让模型学会写代码调工具的 CodeAct，还是近期风靡开发者社区的 Claude Code，其核心突破都在于一套极其严密的支撑架构（Agent Harness）——它规定了模型何时调用终端、如何清理网页噪音、以何种格式读写工作区文件，以及在遇到报错时怎样回溯重试。

> ArXiv URL：https://arxiv.org/abs/2608.09096v2

为了让 Agent 变得更聪明，人类软件工程师花费了海量时间手写复杂的控制流、状态机与防御性补丁。于是一个自然演进的技术问题浮出水面：如果大语言模型本身已经具备长程规划与写代码的能力，**它能否不再只作为被调用的“工人”，而是化身“系统架构师”，自主重构、调试并迭代自己所依赖的运行支架？**

由中国人民大学 AI Box 团队与 BOSS 直聘等机构联合提出的研究给出了坚实的实证回应。他们推出了首个专门用于评测大模型自主迭代 Agent 支架能力的基准测试 **Evo-Bench**。实验表明，顶尖前沿模型在完全自主的代码重构闭环中，能带来最高 16.6 分的绝对性能跃升，并在通用任务上全面超越人类专家耗时数月打磨的手工支架；但同时，实验也暴露出了令人深思的技术瓶颈——大模型极易在演化初期迅速达到能力上限，并在后续轮次中因为“过度工程化”而改崩代码。

<img src="/images/2608.09096v2/x1.webp" alt="Evo-Bench 的核心定位与任务环境示意" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 为什么评估“支架进化”极其困难？

让大模型自我优化并不新鲜，但此前大部分工作都停留在“Prompt 搜索”或针对单一垂直任务的微调上。真正意义上的“支架进化（Harness Evolution）”，要求模型面对的是一个包含数千行代码的真实 Python 代码库。在这个代码库中，模型需要读懂调度循环、工具定义、上下文截断与异常处理机制，然后通过运行验证集、分析报错日志、提出改进假设并提交代码补丁，实现长程迭代。

要为这种“元级（Meta-level）”的自主研发能力建立衡量基准，面临着三个极度严苛的工程挑战：

其一是**支架敏感度难题**。如果一个测试任务本身的成败完全取决于底层大模型的先验知识储备，那么无论外部支架写得多精妙，得分都不会有明显波动。这种任务对于测试“支架改进能力”来说就是纯粹的噪音。基准任务必须对代码架构的优劣保持高度敏感。

其二是**过拟合与分布漂移风险**。在长程进化中，负责演化的模型很容易为了刷高验证集分数，硬编码针对某些特定样本的特异性规则。如果验证集（Validation）与测试集（Evaluation）的设计缺乏统计学上严格的敏感度对齐，最终选拔出的所谓“优秀架构”就只是在死记硬背。

其三是**可控的长程归因**。在传统的 Agent 评测中，基座模型能力的波动往往会与运行框架的改进混在一起。为了精确剥离出“纯粹的系统架构改进贡献”，必须在评测协议中严格锚定底层的执行模型，让进化模型作为独立的“外部研究员”进行受控优化。

<img src="/images/2608.09096v2/x2.webp" alt="Evo-Bench 交互界面与演化流程概览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 两阶段支架引导设计：如何淘出真正敏感的测试题？

为了解决上述问题，研究团队放弃了传统基准直接随机切分数据集的粗放做法，提出了一套“两阶段支架引导构建框架（Two-Stage Harness-Guided Benchmark Construction Framework）”。

<img src="/images/2608.09096v2/x3.webp" alt="Evo-Bench 两阶段运行支架引导的基准构建框架" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在第一阶段，研究者首先构建了一个包含 320 个任务的辅助任务集（涵盖搜索、办公、通用三大场景），并让 GPT-5.6 Sol、Claude Opus 4.8、Claude Sonnet 5 和 GLM-5.2 四个前沿模型在上面独立运行完整的进化实验。在收集到的 73 个可运行支架变体中，通过多样性感知算法筛选出 12 个结构迥异的代表性支架 $\mathcal{H}_{\mathrm{aux}}$。这些支架在工具调度、程序控制流、记忆缓存机制上各有千秋，构成了刻画任务敏感度的“标尺”。

在第二阶段，研究团队使用这 12 个结构各异的支架，对来自 APEX-Agents、BrowseComp、Claw-Eval、GDPval 和 HLE 五大权威基准的 2,329 个候选任务进行了穷举式交叉评测。团队为每一个任务 $x$ 计算了两个核心统计量：平均表现 $\mathrm{Perf}(x)$，以及任务得分与支架全局性能之间的皮尔逊相关系数，即支架敏感度 $\mathrm{Sens}(x)$：




{% raw %}$$ \mathrm{Sens}(x) =\operatorname{corr}\left(\{m_{h}(x)\}_{h\in\mathcal{H}_{\mathrm{aux}}},\{Q_{h}^{(-x)}\}_{h\in\mathcal{H}_{\mathrm{aux}}}\right) $${% endraw %}



如果某个任务在不同架构下的得分完全随机，或者无论架构怎么优化都纹丝不动，其 $\mathrm{Sens}(x) \le 0$。这类任务在第一轮筛选中被直接剔除。剩余的高响应度任务则按照难度和敏感度进行分层抽样，最终被严格划分为互不相交的验证集与评测集。这种机制确保了模型在验证集上摸索出的系统优化策略，能够极其稳定地迁移到评测集上，从根本上杜绝了特异性投机。

最终落地的 Evo-Bench 锁定了三大核心领域：涵盖复杂多跳网络检索与信息综合的**搜索领域（Search）**；涉及大规模表格计算、PDF 数据对齐与审计报表生成的**办公领域（Office）**；以及考察复杂工作流恢复、跨工具状态维持的**通用智能体领域（General）**。

### 9 大模型同台竞技：代码进化能力的真实梯队

在统一的基准实验中，所有进化模型（Evolver）均从最基础的单一循环交互框架（CodeAct 种子支架）出发。为了严格隔离变量，底层的执行策略模型（Policy Model）被统一固定为 DeepSeek-V4-Flash，每次进化被赋予 20 轮迭代、1,000 个交互步数以及 48 小时的计算时间上限。

评测阵容覆盖了 7 款顶尖前沿商业模型与 2 款开源主力模型。实验结果展现出极具冲击力的分化格局：

<img src="/images/2608.09096v2/x4.webp" alt="模型在不同领域上的雷达图与表现对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在总分榜单上，GPT-5.6 Sol 与 Claude Opus 4.8 展现出压倒性的工程重构水准，分别在基准上斩获 46.3 分与 45.8 分，相比初始 CodeAct 种子框架的 29.7 分，取得了 16.6 与 16.1 分的绝对增幅。这一表现已经极其逼近人类专家数月来手工堆砌出的多领域支架组合（Artificial Baseline，47.5 分）。这明确证实了，前沿大语言模型已经真正具备以系统工程师的视角、自主修改并大幅提升复杂软件系统性能的本能。

然而，不同场景下的表现分化揭示了模型进化的深层机制：

在**通用任务（General）**中，自主进化的支架展现出了对人类手工设计的全面碾压。顶尖模型演化出的支架得分高达 47.7，显著击败了人类工程基线的 44.4 分。原因在于人类工程师在编写通用 Agent 流程时，往往基于确定性预设；而大模型在经历多轮试错后，自主加入了一套极其细致的动态容错逻辑——例如智能截断超长空回复、自动抹除终端环境变量中的敏感凭证、增加针对邮件草稿箱的误发拦截器等，这类防御性工程设计使得系统面对长尾异常时极其鲁棒。

在**搜索任务（Search）**中，模型同样表现优异，通过自主编写 Python 网页爬取、HTML 噪音清洗（智能去除 CSS 与无用脚本但保留超链接结构）以及多轮次检索聚合模块，将基础框架的得分几乎翻倍。

但在**办公任务（Office）**中，所有进化模型均遭遇了明显的滑铁卢，最高得分（41.6 分）仍无法逾越人类手工设计的 43.9 分。深入分析发现，处理真实的商业审计与跨表对齐（如 GDPval 和 APEX）需要高度特定、甚至有些“繁琐晦涩”的工作流规范——比如跨 Sheet 单元格级精度的缓存回填、公式重新求值校验、以及特定二进制报表渲染检查机制。现阶段的模型进化倾向于寻找通用的抽象调度模式，很难在没有任何外部指引的情况下，凭空“发明”出契合高度专业化工业软件规范的处理套路。

### 早期饱和与过拟合困境：模型为何容易在后期把代码写崩？

在跟踪模型进化的时间轨迹时，研究团队捕捉到了一个极为反直觉的现象：**进化的早期饱和（Early Saturation）**。

从直觉上看，随着迭代轮次（Iterations）的增加，大模型应该不断吸收测试失败的经验，系统性能应该呈现单调上升的平滑曲线。但实际情况截然相反：

<img src="/images/2608.09096v2/x5.webp" alt="GPT-5.6 Sol 与 Opus-4.8 在演化过程中的轨迹动态与分值跃迁" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

无论是 Claude Opus 4.8 还是国产旗舰 GLM-5.2，其“任意时刻历史最高验证集分数（Anytime Validation Score）”都在前 5 到 8 轮迭代中迅速冲顶。在极短的几轮尝试中，模型就能敏锐地定位出 CodeAct 原生框架的致命缺陷，并迅速补齐关键工具、引入分流路由。

但在随后的第 10 到 20 轮中，进化过程往往陷入停滞甚至倒退。当通用模块修补完毕后，模型开始面对极其棘手的长尾边缘 Corner Cases。此时模型展现出了某种“过度工程化”的焦虑：它开始频繁重写核心调度模块、生造层次复杂的子类抽象、或者针对验证集里的某几个报错添加生硬的条件分支。这些改动不仅没能解决深层逻辑矛盾，反而破坏了系统早先建立的稳定性，甚至引入语法死锁和超时异常。

这一发现给业界敲响了警钟：**在当前的自主 Agent 研发管线中，盲目拉长自迭代的轮次反而具有破坏性。** 如何让模型具备“代码回滚（Reversion）”的自省机制，以及如何设计能够惩罚代码膨胀的收敛准则，是通向更高级自我进化的必经之路。

### 架构师的解题思路：GPT-5.6 Sol 是如何重构系统的？

为了具象化大模型的“架构设计模式”，研究对表现最好的 GPT-5.6 Sol 的最终代码进行了系统级逆向剖析：

<img src="/images/2608.09096v2/x6.webp" alt="GPT-5.6 Sol 自主演化出的系统架构全貌与成本效益分析" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

可以看到，GPT-5.6 Sol 并没有在原来的平面循环代码里打补丁，而是近乎重构出了一个优雅的**分层领域路由器（Hierarchical Domain Router）**：

在入口处，它增加了一个意图鉴别层，能够自主识别当前任务是偏向网络检索、报表审计还是跨应用系统操作；针对搜索，它下挂了带管道清洗的数据抓取工具；针对办公场景，它构建了跨文件跨页签的证据跟踪器（Evidence Tracker）；而在底层的执行循环中，它引入了状态自检与证据反思逻辑。当执行策略出现重复动作或死循环倾向时，运行支架会在代码层强行切断操作流，并向策略注入明确的诊断提示。

在成本效率维度，研究给出了极具实用价值的帕累托前沿图：

- **顶级性能区间**：以 GPT-5.6 Sol 为代表，单次 48 小时演化探索的 API 消耗超过 500 美元。它虽然拿下了全场最高分，但成本极其高昂。

- **效率最佳平衡点（Knee of the Curve）**：国内的 GLM-5.2 与 Qwen3.7-Max 展现了极强的性价比，单次演化成本控制在 30 到 40 美元以内，却换来了仅次于顶尖梯队的架构进化收益，性价比表现惊人。

- **极致轻量区间**：DeepSeek-V4-Pro 以不到 1 美元的推理成本，成功交付了一个具备完备容错与网页检索能力的改版支架，为小算力预算下的自我改进提供了现实样本。

### 进化的支架是“独立推理结构”：不可思议的跨模型迁移性

这项研究中最具理论价值的发现，是**自主进化生成的支架具备高度的跨模型可迁移性**。

在主实验中，支架是在 DeepSeek-V4-Flash 的环境下迭代出来的。那么，这套支架会不会只是对 DeepSeek 某些特定输出习惯的“过拟合补丁”？如果换一个完全不同的模型来当底层执行者，系统还会生效吗？

研究团队进行了一组跨架构消融验证：将优化完毕的支架剥离出来，分别塞入 Qwen3.6-35B-A3B 以及 GLM-5.2 作为执行策略。

<img src="/images/2608.09096v2/x7.webp" alt="进化预算与策略模型对性能表现的缩放曲线" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

实验结果显示：无论底层的 Policy Model 切换成参数量较小的 Qwen 开源模型，还是结构不同的 GLM，由先进 Evolver 进化出的支架均能带来极其稳健的跨模型涨分。即使底座换掉，由 GPT 或 Claude 搭建的高质量调度流、清洗工具与容错断言，依然能稳定地为新的策略模型兜底赋能。

这意味着，大模型在代码进化过程中提炼出的，不是表层的提示词魔法，而是具有普适价值的**外部推理与执行结构（Transferable Reasoning Structures）**。优秀的软件工程原则，在硅基智能体之间呈现出高度的通用性。

### 走向真正的自我进化：从写支架到改权重

回顾整个 Evo-Bench 的评测结果，我们看到了一条清晰的智能体演进路径：

在大语言模型被赋予代码执行权限之后，它不仅能充当终端应用，更展示出了作为一名“AI 研发工程师”的雏形。从最初简单的 CodeAct 循环，到自主构建包含路由、容错、清洗与校验的复合系统，前沿模型用 16.6 分的跨越证明了自主工程迭代的现实可行性。

但研究指出的天花板同样不容忽视：在面对高度专业化、涉及复杂外部系统规约的办公任务时，模型的闭门造车依然难敌人类专家的领域工程经验；在经历多轮演进后，代码库的混乱度增加与长尾回退问题，也亟待更先进的演化算法去约束。

无论如何，Evo-Bench 的出现为 Agent 社区提供了一个极为稀缺的“标尺”。它让大家清晰地意识到：评估大模型的终局，或许不再是看它能做对多少道固定的问答题，而是看它在给定的时间和计算预算内，能在多大程度上重塑自己生存和运行的系统框架。
