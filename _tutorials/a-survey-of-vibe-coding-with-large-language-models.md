---
layout: default
title: "A Survey of Vibe Coding with Large Language Models"
---

# A Survey of Vibe Coding with Large Language Models

- **ArXiv URL**: http://arxiv.org/abs/2510.12399v1

- **作者**: Lingrui Mei; Xueqi Cheng; Yuyao Ge; Yujun Cai; Shenghua Liu; Tianyu Liu; Jiafeng Guo; Jiayu Yao; Yujia Zheng; Tianhao Li; 等14人

- **发布机构**: Chinese Academy of Sciences; Duke University; Peking University; State Key Laboratory of AI Safety; University of California; University of Chinese Academy of Sciences; University of Queensland

---

# 关于使用大型语言模型进行 Vibe Coding 的综述

## 引言

大型语言模型 (Large Language Models, LLMs) 通过能够流畅理解和生成自然语言的对话系统，显著推动了人工智能的发展。在软件开发领域的早期应用中，LLMs 主要作为辅助工具，开发者通过自然语言提示生成代码片段，但由于准确性有限，仍需大量手动审查和迭代调试。

随着GPT-4和Claude Sonnet 4等先进架构的出现，LLM的能力实现了质的飞跃，能够通过与环境的动态交互（如执行shell命令、文件操作和测试）自主完成编程任务。这些智能体在真实世界的编程任务上取得了快速进展。例如，在SWE-bench基准测试上，SWE-agent达到了12.5%的解决率，而OpenHands在SWE-bench Verified上达到了53%。

更强大的LLM（如GPT-5 Pro）的进步催生了一种新的开发范式——**“Vibe Coding”**。在这种模式下，开发者不再逐行审查代码，而是通过观察AI生成代码的执行结果来验证其正确性，并进行迭代式的自然语言需求沟通和反馈。编码智能体 (Coding Agents) 不仅生成代码，还能自主配置环境、执行程序、自我诊断错误并更新实现，这标志着人对AI信任度的显著提升，从传统的代码理解转向了结果导向的验证。

然而，仅有强大的智能体是不够的。任务的复杂性暴露了非结构化自然语言指令的局限性，它难以传达细致的需求和架构约束。实证研究甚至发现，经验丰富的开发者在使用AI工具时，任务完成时间反而增加了19%。有效的人-AI协作需要系统性的提示工程和上下文工程 (context engineering)、结构化的指令以及在不同交互类型中平衡人与智能体的主导权。

为填补这一关键空白，本文首次对基于大型语言模型的Vibe Coding进行了全面而系统的综述。本文旨在：(1) 为理解软件开发中的人-智能体协作建立严谨的理论基础；(2) 为开发者选择和实施合适的开发策略提供可行的指导；(3) 识别涵盖技术基础设施、安全机制和人因工程的关键挑战与未来方向。这项工作为新兴的AI增强软件工程领域奠定了概念基础，并为研究人员和实践者提供了技术路线图。

<img src="/images/2510.12399v1/x1.jpg" alt="图1：Vibe Coding生态系统概述，包含理论基础、开发模型、基础设施和反馈机制。" style="width:85%; max-width:600px; margin:auto; display:block;">

## 相关工作

### 相关综述

#### 基础LLM
多篇综述记录了大型语言模型的发展，涵盖了LLM架构、训练范式和能力。这些工作研究了Transformer的变体，重点关注高效架构和长上下文能力。从BERT到ChatGPT的发展轨迹追溯了基础模型的演进，并探讨了其机遇与风险。还有专门的综述讨论了评估方法、效率（从模型中心和数据中心的角度）以及文本生成和知识增强型预训练语言模型等特定能力。

#### 上下文学习 (In-Context Learning)
在架构基础之上，研究转向了如何在不进行额外训练的情况下利用预训练模型。提示工程 (Prompt engineering) 和上下文学习已成为基础技术，其分类体系涵盖了广泛的提示方法、应用、安全考量以及在自然语言处理 (NLP) 任务中的性能。随着上下文工程成为一门正式的学科，上下文学习机制得到了深入探索。思维链 (Chain-of-Thought, CoT) 推理被证明尤为有效，相关的分类学研究了“思维链-X” (Chain-of-X) 范式，并探索了长思维链和多模态思维链推理。多模态大型语言模型是一个快速发展的领域，相关综述研究了其架构、训练方法以及跨多种数据模态的视觉-语言集成。

#### 后训练 (Post-Training)
当上下文学习不足时，后训练方法为模型对齐特定需求和增强推理能力提供了途径。强化学习方法，如近端策略优化 (PPO)、Q学习和演员-评论家方法等得到了综述，特别关注了基于人类反馈的强化学习 (RLHF)、基于AI反馈的强化学习 (RLAIF) 和直接偏好优化 (DPO)。指令微调 (Instruction tuning) 和监督微调方法的研究涵盖了数据集构建和训练策略，并探讨了用于增强指令遵循能力的数据选择方法。对齐研究将方法分为外部对齐和内部对齐，并考虑了对抗性因素，同时探索了免训练对齐和个性化对齐技术。DPO作为一种无需强化学习的RLHF替代方案，其分类体系涵盖了数据策略、学习框架和约束机制。后训练范式涵盖了微调、对齐、推理、效率和领域适应，其中参数高效的方法如低秩适应 (LoRA) 和适配器 (adapters) 提供了计算开销的实验比较。

#### 智能体系统 (Agent Systems)
工具使用和规划能力的整合将LLM从被动的模型转变为主动的智能体。基础性综述建立了涵盖智能体构建、大脑-感知-行动架构和自主决策能力的框架，并提供了跨越推理和代码生成等基准的统一分类体系。多智能体系统 (Multi-agent systems) 的研究涵盖了智能体配置、通信协议和跨复杂任务解决场景的协作工作流。智能体能力的专门综述包括：工具使用与检索增强生成 (RAG) 和反馈学习的结合；规划机制（包括任务分解和记忆）；用于推理和工具执行的单智能体和多智能体架构；以及短期和长期记忆机制。评估方法涵盖了规划、工具使用、自我反思和特定应用的基准。领域特定应用包括网页自动化、科学发现（跨生命科学和材料科学）、操作系统智能体与GUI交互，以及具备反馈循环和终身学习能力的自进化智能体。与本综述尤其相关的是，近期综述研究了编码智能体，涵盖了软件开发生命周期中的单智能体和多智能体架构、规划、上下文管理和工具集成，并带有基准测试框架。

### 基础技术

#### 用于代码生成的强化学习
将强化学习应用于代码生成需要可执行的反馈信号。早期方法将预训练语言模型与深度强化学习结合，利用单元测试反馈和关键采样，在竞争性基准上取得了良好表现。基于执行的方法利用PPO和编译器反馈进行实时优化。更先进的RL框架采用多粒度单元测试反馈、将生成任务分解为课程子任务进行细粒度优化、使用基于排序的对齐机制，以及利用组相对策略优化与编译器反馈来获得有竞争力的性能。

#### 自主编码智能体系统
除了监督生成，自主智能体通过专门的架构和多智能体协作来解决完整的软件工程任务。单智能体系统引入了定制的智能体-计算机接口，在基准测试上表现出色；或结合结构感知的代码搜索和基于频谱的故障定位，以实现低成本的问题解决。多智能体框架在开发过程中采用专业角色分工，例如，分配不同的程序员、测试设计员和测试执行员智能体，以较低的Token消耗实现高通过率。

#### 函数调用 (Function Calling)
有效的智能体系统需要与外部系统和API交互的机制。函数调用框架通过简单的API和少量示例，教会语言模型自监督地使用工具。执行和交互环境为多种编程语言提供了安全的执行环境，并支持轻量级的RL框架。优化和部署方面的进展包括自动识别可并行的函数调用以减少延迟和成本，以及协调专门的智能体进行工具选择、执行和校准以提高成功率。

#### 监督微调 (Supervised Fine-Tuning)
监督微调和指令微调是代码模型训练的基础。指令进化方法通过迭代地进化代码指令来提升性能。自指令方法可以自举模型的指令遵循能力。专门的微调解决了以安全为中心的生成、优化代码生成、代码编辑、调试和改进等问题。用于代码的基础模型采用了仓库级的预训练、扩展的上下文窗口，并在多样化的语言上进行训练。

<img src="/images/2510.12399v1/x2.jpg" alt="图2：Vibe Coding 中的三方关系与智能体上下文构建示意图。" style="width:85%; max-width:600px; margin:auto; display:block;">

## Vibe Coding：管理编码智能体的工程学

### Vibe Coding 的定义

本文将 Vibe Coding 定义为一种基于大型语言模型的软件开发工程方法论。其核心是一种**人、项目和编码智能体之间的动态、迭代和协同进化关系**。在这种范式中，人类从直接的代码编写者转变为**意图阐述者、上下文管理者和质量仲裁者**。项目也从静态的代码仓库扩展为包含代码库、数据库和领域知识的多方面信息空间。编码智能体作为智能执行者，在人类意图和项目约束的双重指导下，执行代码生成、修改和调试。

#### 三方关系的公式化
本文将 Vibe Coding 建模为一个由三元组 $\mathcal{V}=\langle\mathcal{H},\mathcal{P},\mathcal{A}\_{\theta}\rangle$ 定义的动态交互系统，其中：
- $\mathcal{H}$：人类开发者，具备需求认知能力 $\mathcal{H}\_{\text{req}}:\mathcal{D}\rightarrow\mathcal{I}$（将领域需求 $\mathcal{D}$ 转化为指令 $\mathcal{I}$）和质量判别能力 $\mathcal{H}\_{\text{eval}}:\mathcal{O}\rightarrow\{0,1\}\times\mathcal{F}$（对产出 $\mathcal{O}$ 做出接受/拒绝的判断并提供反馈 $\mathcal{F}$）。
- $\mathcal{P}$：软件项目，表示为一个项目上下文空间 $\mathcal{P}=\langle\mathcal{C}\_{\text{code}},\mathcal{C}\_{\text{data}},\mathcal{C}\_{\text{know}}\rangle$，分别对应代码库、数据库和领域知识。
- $\mathcal{A}\_{\theta}$：编码智能体，一个由参数 $\theta$ 化的大型语言模型，执行条件生成函数 $\mathcal{A}\_{\theta}:\mathcal{I}\times\mathcal{P}\times\mathcal{E}\rightarrow\mathcal{O}$。

三方协作可被建模为一个**约束马尔可夫决策过程 (Constrained Markov Decision Process, Constrained MDP)**，其中人类定义目标空间和约束边界，项目提供状态空间和转移约束，智能体执行策略和状态转移：




{% raw %}$$
\mathcal{V}_{\text{MDP}}=\langle\mathcal{S}_{\mathcal{P}},\mathcal{A}_{\mathcal{H}\rightarrow\mathcal{A}_{\theta}},\mathcal{T}_{\mathcal{A}_{\theta} \mid \mathcal{P}},\mathcal{R}_{\mathcal{H}},\gamma\rangle
$${% endraw %}




| 组件 | 描述 |
| --- | --- |
| $\mathcal{S}\_{\mathcal{P}}$ | 状态空间，由项目的当前状态定义。 |
| $\mathcal{A}\_{\mathcal{H}\rightarrow\mathcal{A}\_{\theta}}$ | 动作空间，由人类给智能体的指令触发。 |
| $\mathcal{T}\_{\mathcal{A}\_{\theta} \mid \mathcal{P}}$ | 转移函数，受项目规范约束。 |
| $\mathcal{R}\_{\mathcal{H}}$ | 奖励函数，由人类评估决定。 |
| $\gamma$ | 折扣因子。 |

#### 智能体的条件生成过程
给定人类意图 $\mathcal{I}$、项目上下文 $\mathcal{K}\subseteq\mathcal{P}$（从项目信息空间中检索到的相关子集）和执行环境 $\mathcal{E}$，智能体以自回归方式生成代码序列 $Y=(y\_{1},\ldots,y\_{T})$，其联合概率分解为：




{% raw %}$$
P_{\theta}(Y \mid \mathcal{I},\mathcal{K},\mathcal{E})=\prod_{t=1}^{T}P_{\theta}(y_{t} \mid y_{<t},\mathcal{C}_{t})
$${% endraw %}



其中 $\mathcal{C}\_{t}=\mathcal{A}(\mathcal{I},\mathcal{K},\mathcal{E},y\_{<t})$ 表示在步骤 $t$ 的动态上下文。上下文的组件 $c\_i$ 对应于三方关系中的不同信息源，包括：
- $c\_{\text{instr}}$: 系统指令和任务需求。
- $c\_{\text{code}}$, $c\_{\text{data}}$, $c\_{\text{know}}$: 分别是代码库、数据库和领域知识。
- $c\_{\text{tool}}$, $c\_{\text{mem}}$, $c\_{\text{tasks}}$: 分别是可调用工具的定义、历史交互记忆和当前任务状态。

#### Vibe Coding 的优化目标
从三方视角看，Vibe Coding 的核心挑战是在有限的上下文窗口 $L\_{\max}$ 内，找到最优的上下文编排策略 $\mathcal{F}^{\*}=\{\mathcal{A},\text{Retrieve},\text{Filter},\text{Rank}\}$，以最大化生成质量。其优化目标是：




{% raw %}$$
\mathcal{F}^{\*}=\arg\max_{\mathcal{F}}\mathbb{E}_{\tau\sim\mathcal{T}}[R(P_{\theta}(Y \mid \mathcal{C}_{\mathcal{F}}(\tau)),Y_{\tau}^{\*})]\quad\text{s.t.}\quad \mid \mathcal{C}_{\mathcal{F}}(\tau) \mid \leq L_{\max}
$${% endraw %}



其中，$\mathcal{C}\_{\mathcal{F}}(\tau)$ 是策略 $\mathcal{F}$ 为任务 $\tau$ 从项目 $\mathcal{P}$ 中检索和过滤的上下文，$Y\_{\tau}^{\*}$ 是人类心智模型中的理想输出。

#### 人-智能体协同循环与任务演进
Vibe Coding 的核心机制是**通过持续反馈进行的人类指导，以引导智能体实现项目目标**，并在此过程中**动态扩展需求空间**。其迭代演化过程可以表示为：




{% raw %}$$
(o_{k+1},\mathcal{I}_{k+1})=\begin{cases}(o_{k},\mathcal{I}_{k})&\text{若 }\mathcal{A}_{k}=o_{k}\text{ (完全接受, 终止)}\\ (\mathcal{A}_{\theta}(o_{k}\setminus\mathcal{A}_{k};\delta_{k},\mathcal{I}_{k},\mathcal{K}),\mathcal{I}_{k})&\text{若 }\delta_{k}\in\mathcal{F}\text{ (局部修正)}\\ (\mathcal{A}_{\theta}(\mathcal{I}_{k}\cup\{\delta_{k}\},\mathcal{K}),\mathcal{I}_{k}\cup\{\delta_{k}\})&\text{若 }\delta_{k}\in\mathcal{I}_{\text{new}}\text{ (需求扩展)}\end{cases}
$${% endraw %}



这里，$(\mathcal{A}\_{k},\delta\_{k})$ 是人类在观察了执行结果 $\mathcal{R}\_{k}$ 后给出的反馈。

#### 迭代式任务扩展的公式化
Vibe Coding 支持**需求的动态演进**。任务演进轨迹被定义为一个指令集序列 $\{\mathcal{I}\_{0},\mathcal{I}\_{1},\ldots,\mathcal{I}\_{K}\}$，其中第 $k$ 次扩展被公式化为：




{% raw %}$$
\mathcal{I}_{k+1}=\mathcal{I}_{k}\oplus\Delta\mathcal{I}_{k}=\mathcal{I}_{k}\cup\{\delta_{k}^{(1)},\delta_{k}^{(2)},\ldots,\delta_{k}^{(m_{k})}\}
$${% endraw %}



这种机制体现了两个关键特性：(1) **认知需求延迟满足**：人类无需在初期详尽规划所有细节，而可以在观察智能体输出后逐步完善约束。(2) **机会主义需求发现**：当智能体输出暴露了隐性需求或边界情况时，人类可以立即补充约束。

整个开发周期被建模为一个多阶段优化问题，在每个阶段 $k$ 对应一个任务空间 $\mathcal{I}\_{k}$：




{% raw %}$$
\max_{\{o_{k}\}_{k=0}^{K}}\sum_{k=0}^{K}\omega_{k}\cdot R(o_{k},Y_{\mathcal{I}_{k}}^{*})\quad\text{s.t.}\quad o_{k}=\mathcal{A}_{\theta}(\mathcal{I}_{k},\mathcal{K},\mathcal{E}),\quad\mathcal{I}_{k}\subseteq\mathcal{I}_{k+1}
$${% endraw %}



这个公式捕捉了 Vibe Coding 的精髓：通过持续的人类干预和任务空间的动态扩展，系统逐步收敛到最终的软件目标。这种人、智能体和项目三者的协同构成了一个**自适应、需求可演进的闭环软件开发系统**。

<img src="/images/2510.12399v1/x3.jpg" alt="图3：Vibe Coding 的优势，将团队级别的能力赋予个人，实现持续开发和质量收敛，并拓宽软件创造者生态系统。" style="width:90%; max-width:700px; margin:auto; display:block;">

### 为何需要 Vibe Coding

Vibe Coding 将软件开发从被动辅助转变为协作伙伴关系，解决了在普及化、工作流效率和生态系统扩展方面的挑战。

##### 个人开发者的团队级能力
Vibe Coding 使个人开发者能够交付团队规模的功能。传统上，生产级应用需要协调前端、后端、数据库、安全、DevOps 和 QA 等多个专家。而编码智能体可以提供跨领域的专业知识。开发者专注于需求，而智能体则负责跨技术栈的实现。这使得资源有限的实体能将原型开发时间从几周压缩到几天。

##### 持续开发与质量收敛
Vibe Coding 旨在平衡开发速度和代码质量。传统工作流常常需要在交付速度和测试严谨性之间做出权衡。Vibe Coding 通过与人类约束解耦的自主迭代来同时提升两者。智能体可以进行全天候的自动化测试、重构和性能分析，从而将人类的认知资源解放出来，用于更高层次的设计和优化。

##### 拓宽软件创造者生态
Vibe Coding 通过降低技术门槛来普及开发。传统开发要求在实现想法前具备广泛的编程知识。在 Vibe Coding 中，自然语言成为主要的创造界面。领域专家（如医生、教育家、设计师）可以直接表达他们的需求，而无需计算机科学教育。这使得创新来源多样化，并可能通过“创作者经济”的扩展产生经济影响，标志着软件素养从专业技能向普适能力的演进。

## 用于编码的大型语言模型

### 代码LLM的数据基础

#### 预训练代码语料库
代码LLM需要来自多样化来源的大规模训练数据。这些模型主要依赖于从GitHub和Stack Overflow等开放平台获取的大规模代码语料库，并根据仓库星标数、文档完整性和社区参与度等指标进行质量过滤。

训练数据集在构成和策划策略上差异显著，主要有两种方法：**深度优先策略**，侧重于流行语言以保证质量；**广度优先策略**，涵盖多种语言以确保覆盖面。