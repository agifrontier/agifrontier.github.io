---
layout: default
title: "SWE-Pruner Pro：模型内生剪枝信号，Token省39%解决率反增3.8%"
description: "来自抖音集团（Douyin Group）与上海交通大学的研究团队提出了 SWE-Pruner Pro，推翻了这种“依赖外部判别器”的惯性路径。他们发现了一个极具启发性的现象： 当代码大模型在阅读环境返回的工具输出时。"
arxiv_id: "2607.18213"
paper_published: "2026-07-20"
published_at: "2026-09-12T13:15:08.779692+08:00"
topics:
  - "模型优化"
tags:
  - "Coder LLM"
  - "MiMo-V2-Flash"
  - "SWE-Bench"
  - "SWE-Pruner Pro"
  - "agent-internal pruning"
  - "keep-or-prune head"
related_tutorials:
  - "mimo-v2-flash-technical-report"
  - "online-monitoring-and-corrective-steering-of-programming-agents"
  - "swe-bench-can-language-models-resolve-real-world-github-issues"
  - "towards-flash-thinking-via-decoupled-advantage-policy-optimization"
---

<p class="paper-original-title" lang="en">SWE-Pruner Pro: The Coder LLM Already Knows What to Prune</p>

<img src="/images/2607.18213v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在软件工程智能体（Coding Agent）的实际运行中，上下文膨胀几乎是一个无法回避的工程瓶颈。当模型在代码仓库中执行 `cat`、`grep`、`find` 或运行测试脚本时，终端往往会倾泻出成百上千行的环境输出。为了保持多轮推理的连贯性，这些日志、堆栈跟踪和源文件片段会被原封不动地推入上下文窗口，不仅导致推理费用成倍上涨，还会引发大模型在长序列下注意力涣散的退化现象。

> ArXiv URL：https://arxiv.org/abs/2607.18213v1

传统应对方案通常走向两个极端：要么采用通用的提示词压缩技术（如基于困惑度的 LLMLingua），但这种方式完全忽略了代码语法的完整性和 Agent 当前的任务意图；要么像先前的 SWE-Pruner 那样，引入一个外置的分类小模型，并强迫 Agent 在每轮交互前先写下一段“目标提示查询”（Goal-Hint Query），由外置模型来判断哪些输出该留、哪些该删。这种外部打补丁的思路不仅增加了调用延迟，其自身生成查询和额外推理的开销，甚至经常反噬压缩所节省下来的 Token。

来自抖音集团（Douyin Group）与上海交通大学的研究团队提出了 SWE-Pruner Pro，推翻了这种“依赖外部判别器”的惯性路径。他们发现了一个极具启发性的现象：**当代码大模型在阅读环境返回的工具输出时，其自身的深层隐藏状态（Hidden States）中已经高度编码了哪些代码行重要、哪些无关的判别信号。** 换言之，大模型在 Prefill 阶段就已经“心知肚明”，根本不需要额外的判别模型或意图生成步骤。基于这一洞察构建的轻量内生剪枝方案，不仅在多个基准测试中砍掉了最高 39% 的 Token 消耗，更在 MiMo-V2-Flash 模型上将 SWE-Bench Verified 的任务解决率逆势提升了 3.8 个百分点。

<img src="/images/2607.18213v1/comparison.webp" alt="SWE-Pruner 与 SWE-Pruner Pro 的架构对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 线性探测：隐藏状态里早已写好的答案

为什么过去的研究总倾向于在外部分析代码的重要性？直觉上，判断某段日志是否对解决 Bug 有用，似乎需要结合任务目标进行专门的语义检索与相关性重排。然而研究团队通过探测实验（Linear Probe）揭示了 Transformer 骨干网络内部鲜为人知的信息组织方式。

在 Agent 的多轮循环中，历史交互上下文与新返回的工具输出 $r_t$ 会共同参与前向传播。研究人员提取了模型最后一层在工具输出位置所产生的隐藏状态序列 $\{h_i\}$，并在一个保留轨迹验证集上训练了一个极简的线性探测器（即通过 LDA 线性判别分析与逻辑回归），来测试这组表征能否区分人工标注的“保留行”与“丢弃行”。

结果令人意外：在正例样本率（即需要保留的代码行）仅约 30% 的极度失衡分布下，仅凭单一线性探测器，模型表征就跑出了 0.83 的 AUC 和 0.63 的最佳 $F_1$ 分数，远超多数类基线的上限（0.46）。这表明，大语言模型在顺应自注意力机制摄入上下文信息时，其自身的深层表征已经完成了对上下文行级相关性的初步解耦与表征赋权。既然模型自身已经完成了最繁重的语义理解工作，那么重新挂载外部判别器不仅是算力浪费，更是一种信息冗余。

当然，线性探测在得分分布的中间地带仍存在一定的重叠区间，且它无法直接感知工具输出的宏观尺度，这就需要一个极轻量但非线性的输出头，把骨干网络早已形成的潜意识“翻译”成明确的剪枝指令。

### 零额外前向计算的内生剪枝架构

SWE-Pruner Pro 的核心工程哲学是完全复用推理引擎原生的计算流，将剪枝开销压制在理论下限。

<img src="/images/2607.18213v1/overview.webp" alt="SWE-Pruner Pro 的全局流水线与端到端工作流" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在常规交互中，Agent 发出调用命令 $c_t$ 后，环境返回原始响应 $r_t$。在生成下一步行动前，推理引擎无论如何都必须对新输入的 $r_t$ 执行一次 Prefill 计算以构建 KV 缓存。SWE-Pruner Pro 并不打断这个过程，它直接挂接在引擎内部，在这次必须执行的 Prefill 中直接截取 $r_t$ 对应的最后一层隐藏状态 $\{h_1, \dots, h_L\}$。随后，剪枝头在毫秒级时间内对每个 Token 进行二分类预测，并聚合得到行级丢弃决策。

这个剪枝行为发生在“轮次之间”：在第 $t$ 轮，Agent 自身的生成仍可借助刚刚计算好的完整 KV 缓存保证全局感知不受损；但当进入第 $t+1$ 轮时，输入历史中冗长的原始响应 $r_t$ 会被剔除无用行后的剪枝版本 $\tilde{r}_t$ 所替代。由于剪枝后的响应长度平均仅有原始长度的 30% 左右，后续所有轮次所累积的注意力计算量和 KV 缓存显存占用都得到了显著释放。

<img src="/images/2607.18213v1/arch.webp" alt="SWE-Pruner Pro 的轻量剪枝头内部结构" style="width:min(600px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; display:block;">

为了让剪枝头既轻量又鲁棒，架构中引入了两个针对代码交互场景量身定制的关键设计：

#### 1. 长度感知偏置嵌入（Length-Aware Embedding）

在代码排障过程中，剪枝失误的代价对于不同长度的工具输出是高度不对称的。如果一条工具输出总共只有 5 行（例如关键变量的 `print` 结果），哪怕错误地删掉 1 行都可能直接破坏关键线索；但如果一条输出足足有 300 行（例如冗长的构建日志或完整函数实现），误删掉一两行边缘代码的影响几乎可以忽略不计。

为了将这种先验融入模型，SWE-Pruner Pro 根据原始工具输出的总行数 $N$，查询一个可学习的长度嵌入向量 $\mathbf{e}(N) \in \mathbb{R}^d$，并将其直接以广播相加的形式叠加到骨干网络的隐藏状态上：




{% raw %}$$ \tilde{h}_i = h_i + \mathbf{e}(N) $${% endraw %}



该嵌入在训练初始阶段被全零初始化，以保证不破坏预训练表征的基线分布。这种机制使得后续仅由两层 GELU 激活的全连接层构成的极简分类网络 $f_\theta$，能够动态自适应不同规模上下文的裁剪激进程度。

#### 2. 样本内平衡焦点损失（Per-Sample Balanced Focal Loss）

在训练该分类头时，研究团队遭遇了显著的数据分布偏移。真实任务轨迹中保留行的比例天然处于少数，且不同工具输出的保留比例剧烈波动。常规的全局交叉熵或全局加权往往会导致分类头要么过度保守不敢动剪刀，要么过度激进破坏代码骨架。

SWE-Pruner Pro 采用了一种基于单个样本内部重平衡的 Focal Loss。首先计算每个 Token 的基础焦点损失：




{% raw %}$$ \mathcal{L}^{\mathrm{tok}}_i = (1 - p_{t,i})^\gamma \cdot \mathrm{BCE}(p_i, y_i), \quad \gamma=2 $${% endraw %}



随后，在每个响应样本 $s$ 内部，分别对正类（Keep）和负类（Prune）的损失求均值，并强制两者的贡献权重各占一半：




{% raw %}$$ \mathcal{L}^{\mathrm{keep}}_s = \frac{\sum_i y_i \mathcal{L}^{\mathrm{tok}}_i}{\sum_i y_i}, \quad \mathcal{L}^{\mathrm{prune}}_s = \frac{\sum_i (1 - y_i) \mathcal{L}^{\mathrm{tok}}_i}{\sum_i (1 - y_i)} $${% endraw %}






{% raw %}$$ \mathcal{L}_s = \frac{1}{2} \mathcal{L}^{\mathrm{keep}}_s + \frac{1}{2} \mathcal{L}^{\mathrm{prune}}_s $${% endraw %}



这一设计确保了即使在极度冗长、有效代码仅占 5% 的极端样例中，模型对少数保留行的判别能力也能获得充足的梯度驱动，避免了由于正样本稀释导致的骨架丢失。在推理时，模型通过对行内所有 Token 的概率以 0.5 为阈值进行投票，从而稳健地输出整行的保留与删除标签。

### 实验全景：唯一实现全程负 Token 膨胀的剪枝器

为了验证这一机制的普适性，研究人员在当前开源社区顶级的长程代码 MoE 模型上展开了评测：参数量高达 309B（激活 15B）、具备 256K 超长窗口的 MiMo-V2-Flash，以及专精于代码智能体的 Qwen3-Coder-Next（80B 总参数，激活 3B）。评测基准覆盖了代码问答与真实执行环境下的 SWE-QA、SWE-QA-Pro、超长上下文聚合任务 Oolong，以及业界黄金标准 SWE-Bench Verified。

对比基准包括了六种主流策略：LLMLingua2、Selective Context、基于语义重排的 RAG、让主模型自我提示剪枝的 Self-Prune、基于代码困惑度压缩的 LongCodeZip，以及前代外置独立打分模型的 SWE-Pruner。

在纯阅读类的多轮任务中，评测揭示了一个残酷的现状：**现存的大多数剪枝器在端到端统计中根本不能省 Token。** 许多压缩方法为了剪枝而引入的辅助提示词、检索描述或重排开销，抵消了被裁减掉的内容。例如 LLMLingua2 在长上下文 Oolong 基准上，端到端 Token 消耗反而恶化了惊人的 $+190\%$；即使是引入额外模型的 SWE-Pruner，在 SWE-QA 上也带来了正向的 Token 膨胀。

唯独 SWE-Pruner Pro 在全部四项基准、两种骨干模型下，毫无例外地实现了总 Token 消耗的全面下降。在 SWE-QA-Pro 上，Qwen3-Coder-Next 的端到端 Token 使用量大幅锐减了 39%；在 Oolong 任务上，MiMo-V2-Flash 的 Token 消耗同样缩减了 30%。

更为关键的是下游任务的表现。在传统的长上下文评测中，压缩往往以精度塌陷为代价。但在 Oolong 基准上，SWE-Pruner Pro 在缩减 30% Token 的同时，将 MiMo-V2-Flash 的 Exact Match 准确率拉升了 2.2 个百分点；在极具挑战性的软件工程实战基准 SWE-Bench Verified（500 个复杂 Issue 修复任务）中，SWE-Pruner Pro 更是跑出了 $+3.8\%$ 的解决率提升。

对于这一反常的“省了计算却变强了”的现象，代码智能体的决策特性给出了合理解释：模型面对海量冗余日志时，有限的有效注意力往往会被散落在各处的报错栈和打印信息稀释。SWE-Pruner Pro 精准清洗掉了干扰视线的冗余输出，相当于被动为 Agent 构建了一个信息密度更高的推理上下文，使得长程规划时的注意力集中度大幅改善。

而在更小巧的 Qwen3-Coder-Next 上，受限于较小的参数容量，所有剪枝方法在 SWE-Bench Verified 上都略微出现了解法回退。但 SWE-Pruner Pro 展现出了极强的韧性：仅微跌 1.2 个百分点（仅多失误 6 个 Issue），却换取了高达 13.5% 的输入 Token 净节省，在准确率保留与资源节约之间交出了远优于竞品的帕累托前沿曲线。

### 消融实验：为什么两个细节缺一不可？

为了拆解各个组件的真实价值，研究人员在由 GPT-5.4-mini 充当评委的保留验证集上进行了一系列严密的消融对比：

首先是损失函数的选取。如果将样本内平衡 Focal Loss 替换为常规的二元交叉熵（BCE），评委评分瞬间从 7.08 暴跌至 5.95，行级 $F_1$ 也下滑了 0.16；如果改用语义分割中常见的 Dice 损失或 Tversky 损失，虽然单纯看行级 $F_1$ 能够维持在 0.61 附近，但 LLM 评委给出的可读性评分却呈现雪崩式下跌（骤降至 4.88 和 5.23）。这是因为代码天然具备语法连贯性，简单的行级重合度指标无法区分“保留了可用代码骨架”与“虽然精度匹配但语法七零八落”的本质区别。只有样本内平衡的 Focal 机制，能在惩罚误删与保障保留行语义完整性之间取得平衡。

其次是长度感知嵌入。当剥离掉长度嵌入 $\mathbf{e}(N)$、退化为纯粹的长度无关分类器时，模型在行级 $F_1$ 上虽然数字没有明显变化，但综合评分直接从 7.08 掉到了 6.86。消融细节显示，这一机制精准地促使模型在短文本上倾向于保守保留、在几百行的超长输出上果断动刀，有效避免了关键短信息被误伤的毁灭性错误。

在工程落地上，研究团队将该轻量头与知名推理引擎 SGLang 进行了深层融合。在针对真实交互的 16 条重放轨迹分析中，剪枝头在服务端内部直接消费缓存隐藏状态，没有任何跨进程的张量搬运损耗。

在耗时剖析中，这种原生挂接方式仅为整个交互生命周期增加了 15.0% 的微小时间开销（中位数比率仅为 14.7%）。这种代价在每一轮交互中只需支付一次，而换回来的上下文大幅缩减，却能够使之后每一个单步的自回归生成速度全面加快。长程任务执行得越久，整体 Wall-Clock 时间的综合收益就越明显。

### 从“外部监管”重回“内生认知”

SWE-Pruner Pro 的意义不仅在于提供了一个高效的代码剪枝工具，更在于它向 Agent 系统架构的演进模式发出了重要的反思信号。

在过去很长一段时间里，面对大模型长程规划中的上下文遗忘、幻觉与冗余问题，学界的流行做法是不断在外围搭设架构脚手架：设计更复杂的重排器、增加反思步骤、挂接外部分类判别器，甚至设计专门的辅助 Agent 来监视主 Agent。这种思路虽然直观，却人为割裂了模型自身的认知连贯性，并引入了大量的附加开销。

SWE-Pruner Pro 证明了另一种可能性的存在：大模型本身远比外部附加的模型更理解自己此刻的注意力焦点与信息诉求。那些被外置模块费尽心机重新推导的“相关性”指标，早就在模型自身的深层表征中被计算、沉淀下来了。

未来的智能体系统优化，或许不必执着于在外围建造越发臃肿的辅助系统，而是应当学会直接从大模型内部读取并解码那些已然存在的潜意识信号。将这种内生认知与轻量化的执行逻辑结合，才是让智能体在大规模工程落地中兼具轻盈与敏锐的最优解法。
