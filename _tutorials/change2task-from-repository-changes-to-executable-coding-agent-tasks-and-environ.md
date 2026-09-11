---
layout: default
title: "Change2Task：把历史PR迁移到现代代码分支，存储成本降低71.2%"
description: "为了破解“真实性”与“现代可执行性”之间的两难困境，研究者提出了名为 Change2Task 的系统性框架。该系统的核心逻辑在于： 不再费尽周折地去维护老旧历史快照，而是将代码库历史中已合并的真实 Pull Request（PR）所承载的开发者意图，精准“重构”到该项目当前健康、可运行的现代分支上 。"
arxiv_id: "2607.28591"
paper_published: "2026-07-30"
published_at: "2026-09-11T13:15:08.164389+08:00"
topics:
  - "AI Agent"
tags:
  - "AI Agent"
  - "AI论文解读"
related_tutorials:
  - "issuetrojanbench-benchmarking-ai-coding-agents-against-malicious-issue-requests"
  - "mechgeo-autoformalizing-and-proving-euclidean-geometry-in-lean-4"
  - "deep-agentic-search-for-repository-level-code-question-answering-an-empirical-st"
  - "dba-bench-a-production-fidelity-benchmark-for-llm-based-database-operations-agen"
---

<p class="paper-original-title" lang="en">Change2Task: From Repository Changes to Executable Coding Agent Tasks and Environments</p>

在大模型驱动软件工程（Coding Agent）的演进过程中，评估与训练的瓶颈正在发生根本性转移。过去，大模型刷题依赖于静态文本补全；而当下的 SWE-agent、OpenHands 等智能体系统，则需要在真实代码库中自主搜索文件、执行编译、调用测试并依据终端反馈动态修正。这种范式转变意味着，每一个有效的数据样本不再是一段简短的 Prompt，而必须是一个完整挂载了依赖项、运行环境、验证脚本与确定性测试套件的“可执行环境”。

> ArXiv URL：https://arxiv.org/abs/2607.28591

搭建这类环境极其昂贵。以近期学界的基准为例，daVinci-Env 维护了超过 1.28 万个代码库的 4.5 万个 Docker 镜像，构建成本接近 90 万美元；SWE-Universe 为了获得可验证环境，甚至构建了专门的分布式基础设施。更为严峻的是，像 SWE-bench 这样的传统基准，直接将任务锚定在几年前提交发生时的历史快照上。随着第三方源失效、编译工具链断代、Python 运行时废弃，这些历史镜像的冷启动失败率和存储开销居高不下。而另一条路线——纯粹通过模型合成变异的 Synthetic Task，虽然易于批量生成，却往往脱离真实软件工程的维护意图。

<img src="/images/2607.28591/change2task_concrete_instance_aaai.webp" alt="Change2Task具体任务案例" style="width:min(600px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; display:block;">

为了破解“真实性”与“现代可执行性”之间的两难困境，研究者提出了名为 Change2Task 的系统性框架。该系统的核心逻辑在于：**不再费尽周折地去维护老旧历史快照，而是将代码库历史中已合并的真实 Pull Request（PR）所承载的开发者意图，精准“重构”到该项目当前健康、可运行的现代分支上**。通过这种方式，团队仅需维护少量高质量的现代基础镜像，就能在单个镜像上动态挂载源自不同历史时期的多个工程任务。实验显示，Change2Task 在 1,130 个候选变更中实现了 79.6% 的验证成功率，比传统 PR 镜像基线多挽救了 29.2% 的任务，并将环境存储开销大幅削减了 71.2%。

### 核心机制：将历史变更投射到演化后的现代基座

软件代码库随着时间推移会不断重构、重命名甚至更改依赖接口，直接将几年前的历史补丁应用到现代代码上几乎必然引发冲突。Change2Task 将任务构建形式化为两组代码状态的演化与还原关系：

历史空间中，PR 前的旧状态记为 $V_{\mathrm{pre}}$，合入开发者补丁 $P_s$ 后变为已修复状态 $V_{\mathrm{post}}$，即：




{% raw %}$$V_{\mathrm{pre}}\xrightarrow{P_s}V_{\mathrm{post}}$${% endraw %}



而在现代空间中，系统首先选定一个已知健康、依赖完备且测试通过的后代分支修订版本 $H$。Change2Task 的目标是构造一个“任务劣化补丁” $D$，将现代健康分支退行到包含目标缺陷或特性的任务状态 $C$；同时必须确保存在一个对应的“恢复补丁” $G$，能够将代码重新拉回功能正常的健康状态 $H'$：




{% raw %}$$H\xrightarrow{D}C\xrightarrow{G}H'$${% endraw %}



这里的核心约束是：$H'$ 并不要求在文本字符上与 $H$ 完全逐字一致，但必须在行为层面上完整恢复 $H$ 所具备的健康特性。

<img src="/images/2607.28591/change2task_workflow_aaai.webp" alt="Change2Task工作流概览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

整个系统由四个级联阶段构成。首先是**任务证据提取（Deriving Task Evidence）**，系统从历史合并 PR 中抽取付款意图、实现补丁以及伴随的测试代码，将其划分为两组检查：目标检查（Target Checks）专门用来观测待解决的任务条件，而回归检查（Regression Checks）则用来守卫周围既有逻辑不受干扰。同时，系统生成一份涵盖修改文件、AST 符号及代码块范围的“源变更画像”（Source Change Profile）。如果一个历史 PR 的修改意图含混不清，或者目标测试根本无法独立解耦，该样本就会在入口阶段被直接剔除。

第二阶段是**现代基座确立（Establishing the Modern Task Base）**。Change2Task 会锁定代码库中可稳定编译运行的现代后代版本，冻结其 Commit Hash，并结合具体任务类型载入对应的适配器，界定允许 Agent 修改的代码范围。

### 阶梯式状态重构：从补丁逆转到智能体级代码迁移

当现代代码基座确定后，最艰巨的挑战在于如何生成任务补丁 $D$。历史代码与现代代码之间可能隔着数千次提交，代码行号漂移、函数重命名或文件拆分是常态。为了在自动化生成与计算开销之间取得平衡，Change2Task 设计了三级逐步升级的重构策略。

<img src="/images/2607.28591/change2task_task_construction_aaai.webp" alt="任务状态重构的三级机制" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

第一级是**补丁逆转（Level 1: Patch Reversal）**。在代码演化较为平缓的模块中，开发者的历史补丁往往可以直接逆向应用。如果能直接将历史修改干净地从现代分支剥离，且不破坏周围语法结构，系统就直接完成构建。这一级几乎不消耗额外的大模型推理 Token，执行速度最快。

如果直接逆转失败（例如出现 Git 冲突或上下文失配），流程升级至第二级**代码映射（Level 2: Code Mapping）**。系统借助抽象语法树（AST）分析与符号追踪技术，定位历史变更涉及的核心类、函数和调用点在现代代码中的落脚点。通过语法感知的对齐算法，将历史增删逻辑迁移到重构后的代码块中。

当代码结构发生剧烈变化、出现大规模接口替换或逻辑搬迁时，前两级规则均告失效，流程进入第三级**智能体级重构（Level 3: Agent Reconstruction）**。此时，系统调用具备工具调用能力的底层编码大模型（论文中采用了集成 Claude Code 的 Opus 4.8），为智能体提供历史 PR 的描述、原补丁意图以及现代代码结构视图。智能体在受控的循环反馈回路中生成候选补丁，直接在隔离环境中试运行编译与测试，并依据运行时报错进行多轮自适应修复，直至生成合法的任务状态。

每一级输出的候选补丁，都必须经历严苛的**三态生命周期验证（Lifecycle and Scope Validation）**：

- 在初始现代基座 $H$ 上：目标检查通过，回归检查通过（基座本身必须是健康的）。

- 施加任务补丁 $D$ 进入状态 $C$：目标检查必须明确失败（暴露出待修复的 Bug 或缺失的特性），但回归检查必须保持全部通过（不能引入非预期的次生故障）。

- 施加恢复补丁 $G$ 变为 $H'$：目标检查与回归检查必须再度全部通过。

只有在多次重复运行中完全满足“通-断-通”确定性状态转换，且改动范围完全落在限定 Scope 内的候选用例，才会被最终收录。这种严谨的闭环设计，彻底杜绝了依靠断言作弊或引入脆弱不稳定测试的脏数据。

### 五大核心任务全覆盖：适配器架构的实战检验

为了验证 Change2Task 是否能够适应不同的研发场景，研究团队没有局限在单一的 Bug 修复上，而是构建了涵盖五种主流代码智能体工作负载的评测体系：

1. **缺陷修复（Bug Fix）**：智能体面对未修复的代码状态，要求在不触碰回归测试的前提下修复问题。

2. **特性新增（Feature Addition）**：给定需求描述，智能体需在保留既有系统稳定性的前提下，实现一套全新的接口功能并使新增特性测试通过。

3. **测试生成（Test Generation）**：智能体需要编写出能够精准捕获特定逻辑分支的高质量测试用例。

4. **API 迁移（API Migration）**：模拟上游依赖升级场景，智能体需要将旧版 API 调用平滑替换为新版协议规范。

5. **安全修复（Security Repair）**：要求在沙箱隔离环境中，修补已知安全漏洞同时避免引入功能退化。

这五类任务形态差异巨大，测试断言与目标产物各不相同。Change2Task 通过通用的适配器层将不同的输入输出契约接入底层统一的重构与验证核心，充分展现了框架的高扩展性。

<img src="/images/2607.28591/rq1_construction_results_aaai.webp" alt="RQ1任务恢复与构建路径统计" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在针对 12 个公开开源基准集合筛选出的 1,130 个可用变更样本中，Change2Task 最终成功输出了 900 个高质量的成对评估任务（包括 500 个 Bug Fix 任务以及其余四类各 100 个任务）。整体构建成功率达到了 79.6%。

深入分析构建路由可以发现，三级梯度设计起到了关键支撑作用。在最基础的 Level 1 补丁逆转中，仅有部分代码演化极少的用例能够通过；Level 2 代码映射将通过率进一步向上推升；而当遇到涉及较长演化周期的复杂任务时，Level 3 智能体级重构力挽狂澜，贡献了近半数的有效恢复。在与 SWE-smith PR Mirror 的横向对比中，针对相同的 621 个 Bug Fix 候选样本，传统基线仅能复原 387 个任务，而 Change2Task 成功重构了 500 个，任务产出数量直接提升了 29.2%。这表明，面对真实开源项目频繁重构的现实，缺乏语义重构能力的传统方案会丢弃大量宝贵的历史数据，而智能体自适应介入大幅拓宽了数据生产漏斗。

### 任务保真度与评估一致性：现代任务还“真”吗？

将历史补丁迁移到现代代码后，一个必须严肃回答的问题是：**重构后的任务，是否还保留了原历史 PR 的工程复杂度和考核价值？**如果重构过程把一个原本需要跨文件修改的复杂 Bug 退化成了一个单行修复，那么这种扩充数据在训练和评测中就会失真。

研究团队从六个维度量化了现代恢复补丁与历史原始补丁之间的“源变更画像保真度”（Source Change Profile Fidelity）：涉及修改的文件数、代码块数（Hunks）、修改行数、修改的 AST 符号数量，以及两类测试断言表面。在权重分配中，代码实现足迹占据 76%，测试验证表面占据 24%。

<img src="/images/2607.28591/rq2_fidelity_results_aaai.webp" alt="历史到现代变更画像保真度" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

统计结果显示，整个生成语料库的加权保真度高达 0.894。无论是修改规模还是测试逻辑复杂度，重构后的现代版本都与真实开发者的原始工作高度贴合。即使是在经历过深层重构、需要进入 Level 3 智能体重构的代码库中，保真度指标也依然保持在健康区间，没有出现明显的任务空心化或退化现象。

更为关键的验证来自智能体在两个分支上的实测表现。研究人员构建了严格的成对对比实验：Original Branch 代表运行在历史原始快照上的任务，Change2Task Branch 代表在现代健康分支上重构的同一任务。评测选用了四个不同技术路线的头部代码智能体：Codex CLI (GPT-5.5)、Claude Code (Sonnet 5)、Gemini CLI (Gemini 3.1 Pro) 以及 GitHub Copilot (GPT-5.6 Terra)。所有实验均严格锁定提示词、工具权限、Token 上限和环境预算。

实验表明，智能体在两套分支上的解决结果表现出极高的吻合度，原始一致性最高达到了 98.0%（Cohen's $\kappa$ 系数高度一致）。无论是在历史分支还是现代分支上，智能体之间的相对排名完全保持一致，未发生任何倒挂。这充分证明：**Change2Task 在现代分支上重构出来的任务，完整保留了历史用例所具备的判别能力与难度梯度**，完全可以无缝替代陈旧且难以维护的历史环境。

### 系统工程红利：环境复用与吞吐量跃升

从系统工程和计算基础设施的角度来看，Change2Task 带来的收益极为可观。

在传统的评测与训练流程中，每个任务几乎都需要绑定专属的基础镜像或独立的虚拟机快照。评测 900 个任务通常意味着要冷启动、构建并拉取 900 套独立的执行环境。而在 Change2Task 的体系中，这 900 个跨越不同历史年份的真实任务，被聚合收敛到了仅 388 个现代健康基座上。

<img src="/images/2607.28591/rq4_amortization_results_aaai.webp" alt="RQ4环境摊销与开销对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

基座的复用直接摊薄了基础设施的固定成本。数据显示，在相同硬件规格和镜像保留策略下：

- **环境构建耗时缩减了 58.4%**：现代基座编译一次后即可多次用于挂载不同任务，避免了重复解决老旧依赖破损的问题。

- **存储与镜像注册表负载暴跌 71.2%**：由于不需要为每个任务单独打包巨型 Docker 层，任务仅以轻量补丁和验证脚本的形式按需动态注入，磁盘和网络流量大幅缓解。

- **全流程端到端成本降低了 10.8%**：尽管在 Level 3 阶段调用了前沿模型辅助重构，但该计算开销在环境构建与持久存储节省的机房资源面前被迅速摊薄，整体系统经济性展现出明显优势。

这种“一次重构环境、多次动态挂载任务”的思路，改变了 Coding Agent 数据基建的供需模式。长期以来，团队往往把大量工程时间耗费在解决某一个旧项目特定 GCC 版本不兼容、某些已经下架的 PyPI 包无法下载等纯消耗性琐事上。Change2Task 表明，把精力集中在把现代分支维护成少数几个“坚固的执行基座”，然后利用历史 PR 沉淀的海量人类智慧反向投影，能够获得更高质量、更具可持续性的合成数据流。

### 展望与演进方向

Change2Task 的价值不仅在于提供了一个自动化测试构建工具，更在于指明了软件工程智能体数据供给的新范式。过往的研究要么止步于数量有限的历史静态 Benchmark，面临严重的数据污染与环境腐烂风险；要么走向完全由 LLM 捏造的伪任务，陷入分布偏移的怪圈。Change2Task 证明了：人类开源社区数十年累积的 Pull Request 是一座极其富饶的矿藏，即便其物理代码已经演化，其背后的维护逻辑、测试攻防意图与系统演进模式依然可以通过现代技术无缝“转世”。

当然，这项技术也有其边界。当代码库发生跨主版本的大规模重构、依赖生态彻底更迭时，部分极具价值的历史变更可能完全无法在现代接口中找到对应语义，这正是目前三态验证中被 Reject 的主要原因。随着未来代码智能体语义对齐能力的进一步增强，以及跨项目、跨架构接口映射技术的发展，这种从历史动态投射高保真训练数据的能力，将为大模型在复杂软件系统中的长周期持续进化提供源源不断的动力。
