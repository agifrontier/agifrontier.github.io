---
layout: default
title: "SigLeak：无需恶意提示词，仅凭良性交互即可逆向私有Agent技能"
description: "为此，研究团队设计了黑盒推断框架 SigLeak 。实验表明，该方法在五个任务场景、三个模型家族与三个 Agent 框架下，均能高保真地逆向出私有技能，并使下游任务执行成功率相比未挂载技能的基准平均提升 6.88 个百分点。这一发现从根本上动摇了依靠“黑盒托管”即可保护 Agent 核心资产的传统安全假设。"
arxiv_id: "2607.25560"
paper_published: "2026-07-28"
published_at: "2026-09-22T13:15:08.426544+08:00"
topics:
  - "AI Agent"
tags:
  - "SigLeak"
  - "Skill Leakage"
  - "SkillSim"
  - "behavioral side channel"
  - "black-box inference"
  - "diagnostic tasks"
related_tutorials:
  - "daydreaming-stealing-hidden-agent-skills-through-black-box-task-interaction"
  - "beyond-the-black-box-theory-and-mechanism-of-large-language-models"
  - "jailbreaking-black-box-large-language-models-in-twenty-queries"
  - "when-agents-learn-to-be-you-benchmarking-privacy-leakage-impersonation-risk-and-"
---

<p class="paper-original-title" lang="en">Agent Skills Matter: Inferring Proprietary Skills from Execution Trajectories</p>

<img src="/images/2607.25560v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在智能体（Agent）生态正迅速迈向商业化的今天，模块化“技能”（Skills）被视为最具商业价值的软件资产之一。无论是如 Claw Mart 这类允许开发者靠售卖技能获利的应用市场，还是云端托管的 Agent-as-a-Service（AaaS）平台，服务商普遍遵循一种保护商业机密与知识产权的标准范式：把包含复杂工作流、专家经验和少样本提示词的技能指令深度隐藏在受控后端，前端仅向用户暴露标准的问题交互接口以及必要的执行过程监控。

> ArXiv URL：https://arxiv.org/abs/2607.25560v1

然而，服务商在界面中实时展示的工具调用参数、终端命令、中间解析与执行状态，原本是为了方便人类监督、纠错和随时打断，却无形中打开了一扇隐蔽的侧信道（Side Channel）。

<img src="/images/2607.25560v1/Fig1-new.webp" alt="技能泄露（Skill Leakage）威胁总览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

来自阿里巴巴、北京航空航天大学、华东理工大学、南开大学以及上海人工智能实验室的研究团队，在一项最新研究中首次系统性指出了这一新型安全威胁——**技能泄露（Skill Leakage）**。不同于传统通过越狱提示词（Jailbreak Prompts）强行诱导大模型吐出系统提示词的对抗性攻击，这种威胁不需要任何恶意或越狱指令，也不依赖任何参考答案或成功标签，仅仅通过发送普通的良性任务探针，攻击者便能从模型在外部展现出的行为轨迹中，完整逆向出被隐藏的私有程序性知识。

为此，研究团队设计了黑盒推断框架 **SigLeak**。实验表明，该方法在五个任务场景、三个模型家族与三个 Agent 框架下，均能高保真地逆向出私有技能，并使下游任务执行成功率相比未挂载技能的基准平均提升 6.88 个百分点。这一发现从根本上动摇了依靠“黑盒托管”即可保护 Agent 核心资产的传统安全假设。

### 执行轨迹中的行为指纹：技能为何无法彻底隐藏？

要理解技能泄露为什么可能发生，首先需要看清当前大语言模型 Agent 挂载技能的底层机制。所谓的 Agent 技能，本质上是打包好的程序性知识（Procedural Knowledge）。它通常包含结构化的系统指令、领域规则、工具使用的最佳实践以及错误恢复逻辑。

当 Agent 在云端接收到用户请求时，为了节省上下文窗口，通常采用渐进式披露（Progressive Disclosure）机制：模型首先读取技能的元数据（名称与描述），一旦判定当前任务需要该技能，再动态加载完整的技能内容。这一机制使得模型在完成任务时的外部行为不可避免地被打上特定印记。

研究团队在电子表格处理基准 SpreadsheetBench 上进行了一组先验实验，直观展示了不同技能在执行轨迹上留下的显著行为指纹。

<img src="/images/2607.25560v1/Graph1.webp" alt="SpreadsheetBench 上不同技能配置与模型架构下的成功率对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在面对“统计特定表格元素并将结果写回工作簿”的相同任务时，未搭载技能的基础 Agent 往往直接尝试估算数值并写入错误结果。而挂载了 Anthropic 官方技能的 Agent，则展现出清晰的“公式优先”行为模式——优先生成 Excel 公式、尝试触发工作表重算，并在缓存失效时主动修复。相比之下，通过大规模轨迹蒸馏生成的 Trace2Skill 展现出带有条件回退与单元格最终校验的严密指令模式，而文本空间迭代优化的 SkillOpt 则偏向从工作簿中读取具体数据计算后直接输出字面值。

这些在工具选择、资源检索、错误补救与结果构建中反复出现的规律性行为模式，被研究团队定义为**技能签名（Skill Signatures）**。只要 Agent 的交互界面对外展示每一步的操作动作与环境反馈，这些技能签名就会如实记录在可观测的执行轨迹 $\tau = [(a_1, o_1), (a_2, o_2), \dots]$ 中。这就构成了一条天然的行为侧信道：虽然服务商从未直接暴露技能文档本身，但技能引导 Agent 做出的决策动作，已经把其核心业务逻辑“全盘招供”。

### 溯因难题与 SigLeak 的双引擎设计

然而，从几条执行轨迹中推断出通用的技能描述绝非易事。攻击者面临的最核心阻碍是**行为归因难题（Attribution Problem）**：在一段观察到的执行序列中，某一项具体的工具调用到底是因为用户任务本身的显式约束、基础大语言模型自带的偏好风格，还是因为私有技能的指导？此外，模型的随机性还会带来大量偶发性噪声。

为了剥离无关变量并精准定位私有技能的代码与流程规范，SigLeak 确立了两大核心支柱：**诊断性任务构建（Diagnostic Task Construction）**与**对比轨迹比对（Contrastive Trajectory Comparison）**。

<img src="/images/2607.25560v1/Fig2-new.webp" alt="SigLeak 双阶段逆向推断框架总览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在整体系统架构上，SigLeak 并不依赖针对大模型的欺骗性越狱语句，而是将逆向过程拆解为生成器（Generator）与技能合成器（Skill Synthesizer）协同推进的双阶段流程。

整个流程仅需输入一段对目标任务背景的公开发布描述 $d$。在第一阶段，生成器仅依靠 $d$ 构建一组覆盖面广、决策点密集的初始诊断探针。技能合成器在成对的运行轨迹中捕捉初始签名，拼装成初步技能草案 $\hat{s}_0$。进入第二阶段后，生成器根据当前技能草案的未竟之处以及历史探针摘要，发起有针对性的补充测试探针，技能合成器则据此执行差异引导的精炼更新，直到新生成的轨迹无法再提取出有效修改，或探针调用预算耗尽。

在具体运作机制上，框架围绕以下两项核心设计展开：

1. **高决策密度与多样性探针生成**：生成器在构造输入任务 $q$ 时，严格遵循多样性与决策密度原则。多样性要求探针跨越不同的任务家族、输入格式与约束边界，避免逆向出的技能过拟合于某一种单一操作模式；决策密度则要求探针故意包含复杂的长链条操作，强制 Agent 在需求解释、资源预检、工具调用组合、异常恢复和输出格式化等关键决策节点上做出抉择，促使目标技能全面触发其隐藏逻辑。

2. **配对执行协议与对比签名差分**：为了消除基础模型固有偏好与任务硬性要求的影响，技能合成器采用了成对执行协议。由于现代 Agent 平台的技能调用依赖模型的上下文决策，SigLeak 在向 Agent 提交良性查询时，分别测试开启技能与显式指令关闭技能（例如添加中立的“请仅使用基础工具完成此任务，勿加载额外特定流程指南”等无害引导约束）两种情境：




{% raw %}$$\tilde{q}^{\mathrm{on}} = q, \qquad \tilde{q}^{\mathrm{off}} = c^{\mathrm{off}} \,\|\, q$${% endraw %}




{% raw %}$$\tau^{\mathrm{on}}(q) = \mathcal{A}_{s^*}\left(\tilde{q}^{\mathrm{on}}\right), \qquad \tau^{\mathrm{off}}(q) = \mathcal{A}_{s^*}\left(\tilde{q}^{\mathrm{off}}\right)$${% endraw %}



在保持输入探针和底层模型完全相同的前提下，对比算子 $\operatorname{Delta}(q; \tau^{\mathrm{on}}, \tau^{\mathrm{off}})$ 能够直接过滤掉两端共有的任务刚性操作和模型底层惯性，从差异部分提取出真正归属于目标私有技能 $s^*$ 的候选特征集合 $\mathcal{S}$。

### 递进式精炼：从粗粒度草案到指令补完

在第一阶段的初始构建中，生成器构建包含 6 个互补维度的初始探针集，分别针对复杂推理、工具调度策略、中间状态校验机制、错误自愈路径、终态输出规范化以及任务描述泛化进行全面扫描。技能合成器将提取到的高频行为模式整合为初始重构技能 $\hat{s}_0$。

但仅凭初步测试生成的技能往往存在细节缺失、条件模糊甚至过度泛化的问题。在第二阶段的迭代精炼中，上下文信息被扩充为：




{% raw %}$$\mathcal{C}^{(r)} = \left\{ d, \, \hat{s}_{r-1}, \, \operatorname{Summ}(\mathcal{Q}_{<r}), \, \mathcal{S}_{<r} \right\}$${% endraw %}



生成器根据历史探针摘要与已有技能的表述缺陷，针对性地设计更具边缘检验性的新探针。在更新函数 $\operatorname{Update}(\hat{s}_{r-1}, \mathcal{S}^{(r)})$ 的作用下，算法能够精准执行三类动作：为缺失的行为环节补写规则、为过于简略的操作步骤补充上下文边界条件、以及修正被最新对比轨迹所推翻的错误猜测。这种渐进式推断既保证了技能逻辑的完备性，又极大减少了盲目探测对 API 调用预算的消耗。

### 跨多模型与框架的实验验证

为了全面验证 SigLeak 对私有技能的逆向还原能力，研究团队构建了涵盖 5 个异构任务场景的基准测试，包括专注于数据分析与电子表格推理的 SpreadsheetBench、面向文档级问答的 OfficeQA、在噪音干扰与冲突网页环境下进行事实检索的 SealQA、基于前沿 arXiv 论文的定理级严谨数学推理 LiveMathematicianBench，以及家庭多步具身决策环境 ALFWorld。

实验评估了包括 GPT-5.4-mini、GPT-5.5、MiniMax-M3 和 GLM-5.2 在内的三大家族模型，并结合了 Codex、OpenCode 与 Hermes 等三种主流 Agent 运行时框架。

评估从两个关键维度展开：第一是**功能实用性（Functional Utility）**，即把逆向出的技能 $\hat{s}$ 重新挂载到没有任何先验的基础 Agent 上，检验其在真实下游任务中的独立执行成功率（SR）；第二是**语义保真度（Semantic Fidelity）**，利用细粒度评测指标 SkillSim 计算逆向技能与原始私有技能在语义意图、操作指示上的重合度（Precision、Recall 与 $F_1$ 值）。

在下游任务执行的实用性上，实验将 SigLeak 与基于纯任务描述直接生成（Direct Generation）、朴素轨迹摘要（Naive Trace Summarization）以及黑盒提示词窃取基线（BBS）进行了全面横向对比。

数据表明，在几乎所有模型与环境配置下，SigLeak 逆向出的技能均取得了最高或并列最高的下游成功率。与完全禁用技能的基础 Agent 相比，搭载 SigLeak 逆向技能的 Agent 在下游任务上的平均成功率提升了 6.88 个百分点。在 GPT-5.4-mini 和 GPT-5.5 驱动的 Codex 框架下，平均提升幅度均超过 5 个百分点；而在搭载 MiniMax-M3 模型的 OpenCode 与 Hermes 框架下，逆向技能带来的成功率平均涨幅更是分别达到了 11.11 与 8.19 个百分点。

值得注意的是，ALFWorld 是唯一的特例。在该场景中，朴素轨迹摘要方法（直接将带技能的成功执行轨迹扔给大模型做文本总结）取得了与 SigLeak 相当甚至微幅领先的性能。研究人员分析发现，ALFWorld 环境下的技能签名高度外显为具体的状态跟踪、搜索计数和动作排序，这类序贯控制逻辑在单个执行日志中表现得极为显眼，使得直接摘要就能捕获大半有效信息。然而，在其余需要复杂决策判断和策略性工具调用的高阶场景中，基础模型自身的干扰与任务噪声显著增加，SigLeak 独特的差分对比过滤机制便展现出了不可替代的优势。

### 能否精准分辨“同一领域的相近技能”？

逆向跨领域的宏观流程固然有效，但攻击者更渴望知道的是：如果面对多个同处于一个细分垂直领域、使用完全相同的基础工具集，但执行策略和业务偏好截然不同的私有技能，这种仅凭轨迹的侧信道推断是否会混淆？

为了测试这一极限边界，研究团队设计了一项更严苛的对照实验：在相同的 arXiv 文献检索背景下，选定三个分别偏重深度定理验证、跨论文关联对比和综述式快速汇总的不同技能。向 SigLeak 输入完全相同的公共任务描述背景，仅允许其通过各自不同的黑盒 Agent 实例观察成对轨迹。

语义保真度评测结果显示，SigLeak 在 5 个通用场景中取得了高达 22.8% 的平均细粒度 $F_1$ 值与 20.1% 的召回率（显著高于直接生成的 18.3% 与 13.5%）。而在三款同领域 arXiv 技能的交叉相似度矩阵比对中，每个被逆向出的技能与自己对应的真实目标技能之间的软召回率、精确率和 $F_1$ 分数均呈现出鲜明的对角线峰值。这意味着，SigLeak 提取到的并不是针对该领域的泛化提示词模板，而是极其精准地锚定了不同开发者编码在执行逻辑中的个性化程序细节。

消融实验进一步揭示了第二阶段差异引导精炼的效率特征。消融曲线表明，在经历第 0 轮的初始构建后，下游成功率和细粒度 $F_1$ 提升最快、斜率最大的阶段均集中在前两轮精炼中。这说明即便在极其有限的交互预算限制下，攻击者只需发起少数几次针对性的探针追问，就能逼近私有技能的核心逻辑。

### 黑盒安全假设的失效与防御反思

这项研究给大模型智能体安全领域带来了一个明确且具有警示意义的结论：**在 Agent 商业化落地中，将核心能力以“黑盒服务”封装并不能形成可靠的技术护城河。**

长期以来，业界的安全防线普遍针对的是直接的越狱注入（Prompt Injection）和系统提示词反编译。服务商普遍认为，只要拦截了含有恶意指令的敏感请求，并拒绝在界面输出技能原文，后端的专有工作流就是绝对安全的。然而，SigLeak 的成功证明，只要 Agent 为了保证可信度、可纠错性而在前端实时展现完整的工具调用链和执行环境观察，这一连串由技能驱动的连续决策动作就已经构成了事实上的信息泄露通道。

针对这一新型行为侧信道，未来的 Agent 架构设计势必需要在系统透明性与知识产权保护之间做出更为艰难的权衡。单纯的文本内容过滤已不再奏效，如何扰乱工具调用的特征分布、对非关键路径引入差分隐私式的决策噪声，或者对高度可预测的工作流进行动态混淆，将成为下一阶段 Agent 安全防护必须攻克的新课题。
