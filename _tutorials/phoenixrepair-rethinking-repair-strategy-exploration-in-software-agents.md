---
layout: default
title: "PhoenixRepair：多位置采样与反思迭代，SWE-bench解决率达76.0%"
description: "针对这一瓶颈，来自华为、华中科技大学、中山大学与浙江大学的研究团队提出了多智能体修复框架 PhoenixRepair 。该框架重新审视了软件智能体的策略探索机制，通过“多位置采样”与“迭代反思与提炼”两大核心设计，打破了单点定位与盲目修补的局限。"
arxiv_id: "2607.18859"
paper_published: "2026-07-21"
published_at: "2026-10-03T13:15:07.852414+08:00"
topics:
  - "AI Agent"
tags:
  - "Fault Localization"
  - "Graph-Based Localization"
  - "Iterative Reflection and Refinement"
  - "Multi-Agent"
  - "Multi-Location Sampling"
  - "Pass@1"
related_tutorials:
  - "from-experience-to-strategy-empowering-llm-agents-with-trainable-graph-memory"
  - "the-two-stage-decision-sampling-hypothesis-understanding-the-emergence-of-self-r"
  - "sampling-and-loss-weights-in-multi-domain-training"
  - "exploration-vs-exploitation-rethinking-rlvr-through-clipping-entropy-and-spuriou"
seo_title: "PhoenixRepair：多位置采样与反思迭代，SWE-bench解决率达76.0%"
---

<p class="paper-original-title" lang="en">PhoenixRepair: Rethinking Repair Strategy Exploration in Software Agents</p>

<img src="/images/2607.18859v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在软件工程自动化领域，基于大语言模型的智能体（Software Agents）解决 GitHub 真实 Issue 的能力被视为衡量 AI 编程水平的关键试金石。然而，当业界将大量算力押注在更长的上下文窗口、更复杂的工具调用链路或单纯增加采样次数时，许多智能体在实际修复代码缺陷（Issue Resolution）时依然频频受挫。根本原因在于现有智能体的修复策略探索空间极其狭窄：它们往往过早锁定在某一个可能错误的修改位置，并在该位置上进行缺乏有效反馈的单向尝试。

> ArXiv URL：https://arxiv.org/abs/2607.18859v1

针对这一瓶颈，来自华为、华中科技大学、中山大学与浙江大学的研究团队提出了多智能体修复框架 **PhoenixRepair**。该框架重新审视了软件智能体的策略探索机制，通过“多位置采样”与“迭代反思与提炼”两大核心设计，打破了单点定位与盲目修补的局限。实验结果显示，在权威基准 **SWE-bench-Verified** 上，PhoenixRepair 搭配 MiniMax-M2.5 取得了 **76.0%** 的 Pass@1 解决率；在使用 DeepSeek-V3.1 时，相比强基线 SWE-agent 实现了 **7.8%** 的相对提升。

<img src="/images/2607.18859v1/a-1.webp" alt="智能体在软件缺陷修复中的探索空间局限" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 为什么现有代码修复智能体总是“一条路走到黑”？

传统的自主代码修复智能体（如 SWE-agent、OpenHands 等）通常依赖智能体与计算机交互环境（ACI）逐步排查问题、定位故障点并提交 Patch。这类方法在处理逻辑直白的中小型缺陷时表现尚可，但在真实工业级代码仓库中，往往暴露出策略探索严重不足的缺陷。

这种探索不足主要体现在两个维度。首先是**候选修改位置的探索极度匮乏**。概率语言模型存在向高置信度模式坍缩的天然倾向。一旦智能体在第一轮推理中认定了某处文件或函数存在问题，后续的所有操作都会围绕这个局部展开。即使定位本身出现偏差，智能体也极少主动跳出当前视野重新审视其他可能引发连锁反应的代码位置。如果在根源上找错了位置，后续再精细的修改动作也只是徒劳。

<img src="/images/2607.18859v1/motivation.webp" alt="智能体修复探索不足的动机示例" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

其次是**在单个修改位置上的修补尝试缺乏系统性迭代**。即便幸运地锁定了正确的修改区域，生成一次性完美通过所有测试的补丁仍然概率极低。现有方案要么依赖朴素的重试采样，要么将多次失败的尝试日志无节制地塞入上下文中，导致上下文窗口迅速膨胀、关键信号被海量报错噪声淹没。智能体无法从过往的失败轨迹中提炼出结构化的反思与改进指引，导致后续生成往往陷入反复犯同样低级错误的死循环。

### PhoenixRepair 框架：系统化展开策略搜索空间

为了打破这种“单点固着”与“盲目试错”的双重束缚，PhoenixRepair 构建了一个由多个专门化智能体协作的三阶段流水线。整个框架不再试图一次性赌对正确方案，而是主动拓宽候选解的搜索树，并在推进过程中逐步剪枝与蒸馏经验。

<img src="/images/2607.18859v1/overview.webp" alt="PhoenixRepair 框架整体架构" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

#### 第一阶段：分级多位置采样与图增强定位

在多位置采样（Multi-location Sampling）阶段，定位智能体并不只输出一个绝对自信的修改点，而是通过多次连续交互获取 $N$ 个候选位置。为了避免生成重复的冗余猜测，在执行第 $i$ 次定位采样时，系统会将此前采样的候选位置集合 $\{\ell_1, \ell_2, \dots, \ell_{i-1}\}$ 作为显式约束注入提示词，促使智能体主动探索仓库中的替代代码区域。

每个采样得到的修改位置均精确定位至文件路径、起止行号以及待修改代码片段。在完成多次采样后，系统执行去重操作：如果两个候选位置位于同一文件、对应相同的程序实体（例如类中的同一个方法）且行号存在交叠，则被判定为重复项合并，得到精简候选集 $\mathcal{L}'$。

PhoenixRepair 的巧妙之处在于根据候选集规模动态评估任务难度，并施加差异化策略：

1. **简单与中等任务**：当去重后的候选位置数量不多时（$\lvert \mathcal{L}' \rvert \le k$），说明故障定位的模糊度较低，系统直接将 $\mathcal{L}'$ 作为后续补丁生成的基准集。
2. **高难度复杂任务**：当去重后的候选位置数量超过阈值（$\lvert \mathcal{L}' \rvert > k$），意味着代码逻辑高度耦合，传统文本级智能体已无法清晰界定责任边界。此时，框架引入基于代码依赖图的外部定位组件（如 LocAgent），抽取实体调用链与依赖子图 $\mathcal{L}_{\text{graph}}$ 注入候选池，合并去重后形成最终的候选修改位置集合 $\mathcal{L}_{\text{final}}$。

这种分层设计使得计算开销不会无脑浪费在简单任务上，同时保证了极端复杂场景下的候选覆盖率。

#### 第二阶段：双指标驱动的迭代反思与剪枝

进入第二阶段后，系统针对入选的各个代码位置并行或轮番推进修复。代码生成智能体（Coder Agent）在系统提示词的严格约束下，为每一个候选位置分别生成专属修复补丁，并执行问题复现与本地测试验证，产生各自对应的完整执行轨迹。

面对多个位置生成的候选补丁，如何选出最有前途的分支？PhoenixRepair 引入了选择智能体（Selector Agent），基于两项客观与主观相结合的多维质量指标打分，并遵循“二分淘汰”策略：每轮评估后仅保留排名前 $\lceil w/2 \rceil$ 的候选补丁及对应位置，持续迭代收敛，直至最终只保留一个最优位置。这两项关键指标分别为：

*   **复现与边界测试质量（$Q_{\text{test}}$）**：修改补丁不能仅仅满足于当前报错消失，还必须验证问题复现测试与边界测试的完备性。由于原生执行轨迹中充斥着重复命令与冗长报错，分析智能体（Analysis Agent）首先对轨迹进行结构化压缩，剔除重复报错和无状态改变的试探步骤，提炼出核心测试构造流程，进而对测试用例的有效性进行高质量评分。

*   **回归测试通过率（$R_{\text{pass}}$）**：防止“修好一个 Bug，引入十个新 Bug”。系统从原工程中提取全量原有测试，并由专职测试智能体进行语义过滤，剔除那些本身就因预期修改而应当失效的测试，提纯出严格代表既有功能的回归测试集 $\mathcal{T}_{\text{final}}$。补丁必须在该测试集上取得高通过率，确保没有引入附带破坏。

#### 第三阶段：经验提炼与最终轮生成

当候选位置经过多轮迭代与剪枝收敛至唯一的最终位置 $\ell_{\text{final}}$ 后，PhoenixRepair 并不直接将上一轮的中间代码当作最终交付成果。在最终生成阶段，分析智能体全面复盘该位置在历史所有尝试中的执行轨迹、测试失败日志以及边缘案例。

这些散落的历史试错过程被蒸馏为高密度的先验指引（Guidance Information），包含避坑指南、必须注意的隐式约定以及关键逻辑补丁形态。这份指引被注入最终轮 Coder Agent 的系统提示词中，使其在拥有“全景记忆”的前提下，打磨出兼具鲁棒性与规范性的最终 Patch。

### 实验评测：在真实复杂缺陷上的全方位突破

为了全面检验 PhoenixRepair 的实际战斗力，研究团队在基准测试集 **SWE-bench-Verified** 的 500 个真实人工校验任务上进行了系统评测，并覆盖了包括 DeepSeek-V3.1、DeepSeek-V3.2、Qwen-Coder-Plus、GLM-4.7 以及 MiniMax-M2.5 在内的多款前沿大模型底座。

#### 整体修复率（Pass@1）持续领跑

实验数据显示，不论接入哪种底层模型，PhoenixRepair 相比业界公认的高性能基线 SWE-agent 均取得了稳定且显著的性能超越。

在 DeepSeek-V3.1 下，PhoenixRepair 取得了 **7.8%** 的相对提升，这是所有模型配置中相对增幅最大的一组。这一现象反映出：当底层模型的单步推理与直接命中率处于中等水平时，PhoenixRepair 的多路径探索与自纠错机制能够发挥最大的补位效应，系统性扩大有效解的搜索空间。

随着模型基座能力提升，该机制的收益依旧稳健。在 Qwen-Coder-Plus 与 GLM-4.7 上，PhoenixRepair 均取得了 6.4% 的相对提升；而在 DeepSeek-V3.2 上，Pass@1 达到了 74.4%（相对提升 7.2%）。即使在最强底座 MiniMax-M2.5 下，原本基线已处于高位的背景下，PhoenixRepair 依然将解决率进一步推高至 **76.0%**（相对提升 5.3%）。这证明了更优质的搜索策略并不会被强模型内化吞噬，反而能与更高级的模型推理能力形成协同增益。

#### 缺陷定位精度显著细化

补丁修复率的提升，首要归功于定位精度的改善。研究团队在文件级、模块级与函数级三个不同粒度上统计了 Acc@1 定位准确率。

结果表明，PhoenixRepair 在各个层级均明显优于基线方案，其中**函数级精度**的提升尤为突出。在大型代码仓库中，找到被报错波及的文件并不算太难，但准确命中必须动手术的具体方法或函数却极度依赖全局依赖理解。PhoenixRepair 借助图增强定位与多样本横向比对，成功将搜索范围精准收敛至具体的函数片段，为后续 Coder Agent 的精准修改扫清了最大的前置障碍。

#### 跨框架通用性与模块消融

为了证明该机制不是某种特定智能体骨架下的“特异功能”，团队将多位置采样和迭代反思模块无缝挂载至轻量级框架 Mini-SWE-agent 与支持运行时自演进的 Live-SWE-agent。实验结果表明，在挂载新机制后，两种变体在 DeepSeek-V3.2 上的 Pass@1 和定位精度均获得了显著同向增长，体现出极强的即插即用泛化性。

<img src="/images/2607.18859v1/ablation_chart-2.webp" alt="消融实验各项指标变化趋势" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

随后的消融实验进一步厘清了各模块的贡献度：

*   **移除额外图结构定位信息**：在 Qwen3-Coder-Plus 下导致 Pass@1 相对下降 0.3%，说明图信息主要在少数高度复杂的隐蔽任务中起决定性兜底作用；

*   **移除多位置采样（仅保留单点迭代）**：导致性能大幅下降 2.9%；

*   **移除迭代反思与提炼机制（仅保留多位置暴力生成）**：导致性能下降 3.4%。

消融数据清晰印证了一个判断：单靠广撒网（仅多位置）或单靠钻牛角尖（仅单点反思）都无法达到最优，只有将“广度维度的候选拓宽”与“深度维度的经验提炼”结合起来，才能真正释放策略探索的威力。

### 真实案例剖析：攻克 Pylint 复杂路径过滤缺陷

真实案例最能直观体现 PhoenixRepair 的策略差异。在针对 Python 代码分析工具的著名缺陷 `pylint-dev#6528` 的修复中，问题核心在于 Pylint 在某些参数配置下无法正确忽略指定目录，反而穿透进入被忽略的文件夹执行语法检查。

<img src="/images/2607.18859v1/casestudy2.webp" alt="Pylint issue 6528 案例对比与修复轨迹" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

基线智能体在分析报错后，注意力迅速被目录遍历的主循环所吸引，并在遍历逻辑中反复打补丁。然而，这种单点修改只治标不治本，不仅破坏了现有的子目录发现测试，还导致了多处非预期文件的回归测试报错。

PhoenixRepair 在第一阶段的多位置采样中，除了常规的遍历模块外，还挖掘到了负责文件匹配规则的预过滤层。在后续的二分筛选中，测试代理敏锐地发现修改预过滤层能保持 100% 的回归测试通过率。在最终轮中，Analysis Agent 总结了前面几轮补丁在边缘通配符匹配上的缺陷，指导 Coder Agent 生成了既阻断忽略目录遍历、又避免其下文件被隐式解析的完备 Patch，干净利落地一次性通过了 SWE-bench 的全套严苛验证。

### 成本考量：探索开销是否可控？

多位置采样与多轮反思必然带来推理 Token 的增加。在成本分析中，PhoenixRepair 单任务在 DeepSeek-V3.2 下的原始平均 API 成本为 0.154 美元，而 SWE-agent 为 0.083 美元，算力开销约为基线的 1.85 倍。

然而，在实际工程部署中，这一开销能够被现代推理架构大幅对冲。当前主流大模型服务商（包括 DeepSeek 与 Anthropic）均全面普及了**前缀缓存（Prefix Caching）**机制。在 PhoenixRepair 的执行流程中，针对同一任务的多位置采样和同一位置的多轮修正，其仓库上下文提示词前缀具有极高的重合度。由于命中缓存的输入 Token 成本通常可降低 80% 至 90%，PhoenixRepair 引入的实际有效经济成本远低于单纯按调用次数估算的理论值。用不到两倍的合理计算代价换取 76.0% 的顶级修复成功率，在绝大多数工业化自动化研发场景中都是极具性价比的权衡。

### 总结与展望

PhoenixRepair 的核心贡献并不在于堆砌更长的主动执行步数，而在于**重塑了软件智能体探索策略的结构**。它指出了当前代码 Agent 研发中的盲区：大模型天生的概率倾向会导致其在错误的位置上越陷越深，唯有在架构层面强行引入多候选空间采样、引入客观回归反馈驱动的二分剪枝、以及利用历史轨迹提炼指引，才能真正构建具备工业级可靠性的软件修复系统。随着推理端扩展（Inference-time Scaling）成为大模型演进的主流方向，PhoenixRepair 提供了一条行之有效的系统化搜索与自校正路线。
