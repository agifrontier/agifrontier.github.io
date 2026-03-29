---
layout: default
title: "Learning to Discover at Test Time"
---

## TTT-Discover：开源模型+测试时训练，仅需数百美元刷新多领域SOTA

<img src="/images/2601.16175v1/A__title.jpg" alt="" style="width:90%; max-width:700px; margin:auto; display:block;">

当面对一个从未见过的科学难题时，人类专家通常不会只依赖“既有知识”去猜测答案，而是会通过不断的尝试、失败、总结经验，在解决问题的过程中“现场学习”。

> ArXiv URL：http://arxiv.org/abs/2601.16175v1

然而，目前的AI范式——即便是最强的推理模型（如o1系列）——在测试时（Test Time）通常也是“大脑冻结”的。它们可以通过思维链（CoT）或搜索策略（如Best-of-N）来探索解空间，但模型本身的权重是固定的。这就好比一个学生在考试时只能靠回忆，而不能从刚才做错的草稿中通过学习变得更聪明。

近日，来自Astera Institute、NVIDIA、Stanford等机构的研究团队提出了一种全新的范式：**Test-Time Training to Discover (TTT-Discover)**。

该研究打破了“测试时模型冻结”的铁律，允许大模型在解决特定问题时，利用强化学习（RL）实时更新自身权重。令人震惊的是，该方法仅使用开源模型（gpt-oss-120b），在每个问题上花费仅数百美元，就在数学、GPU内核优化、算法竞赛和生物学等多个领域刷新了SOTA（State of the Art），甚至超越了人类专家和闭源前沿模型。

### 从“搜索”到“发现”：为什么我们需要测试时训练？

在解决科学发现类问题（Discovery Problem）时，AI面临的核心挑战是**分布外泛化**（Out-of-Distribution Generalization）。真正的发现，往往位于模型训练数据的边界之外。

此前的主流方法是**测试时计算扩展**（Test-time Scaling），例如AlphaEvolve。这类方法通过提示（Prompting）一个冻结的LLM进行搜索。虽然可以通过进化算法优化提示词，但LLM本身并没有“进步”。

TTT-Discover则更进一步：**它在测试时直接对LLM进行强化学习训练。**

这种“持续学习”的形式非常特殊，因为它与传统的强化学习目标截然不同：

1.  **目标是极值，而非均值**：传统RL试图最大化所有尝试的平均奖励；而科学发现只需要**一个**最好的解决方案。

2.  **专注于特例，而非泛化**：模型不需要学会解决所有问题，只需要解决**当前这一个**特定的难题。

<img src="/images/2601.16175v1/x1.jpg" alt="Refer to caption" style="width:90%; max-width:700px; margin:auto; display:block;">

*图1：TTT-Discover在测试时针对单个问题持续训练LLM。随着训练步数（Step 0到49）的增加，奖励分布显著向高分移动，最终超越了人类最佳水平（Prior Art）。*

### 技术核心：为“发现”而生的强化学习

该研究将每个科学问题定义为一个独立的马尔可夫决策过程（MDP）。为了适应上述特殊目标，TTT-Discover在算法设计上做出了关键调整。

#### 1. 独特的训练目标

传统的RL算法可能会让策略坍缩到“安全但平庸”的高分区域。为了鼓励发现，TTT-Discover设计了一个加权的训练目标 $J\_{\beta}(\theta)$：




{% raw %}$$ J_{\beta}(\theta)=\mathbb{E}\_{s\sim\texttt{reuse}(\mathcal{H})}\left[\log\mathbb{E}\_{a\sim\pi\_{\theta}(\cdot\mid s)}\left[e^{\beta(s)R(s,a)}\right]\right] $${% endraw %}



这个目标函数通过指数加权，极大地偏向于那些**最有希望的解决方案**。简单来说，模型不仅是从错误中学习，更是疯狂地从那些“灵光一现”的高分尝试中汲取养分，迅速调整权重以生成更多类似的解。

#### 2. 搜索与复用

在探索策略上，TTT-Discover结合了**PUCT**（Predictor + Upper Confidence Bound applied to Trees）算法来管理复用缓冲区（Reuse Buffer）。这确保了模型既能利用已知的高分路径，又能保持一定的探索多样性，避免过早陷入局部最优。

### 实战战绩：全面刷新SOTA

研究团队在四个截然不同的领域进行了测试，结果令人印象深刻。值得注意的是，所有结果都是基于开源模型 **gpt-oss-120b** 取得的，且代码已开源。

#### 1. 数学：Erdős 最小重叠问题

这是一个经典的组合数论问题。自1955年提出以来，人类数学家一直在寻找上下界的突破。

TTT-Discover 发现了一种非对称的构造方法，成功将该问题的上界从之前的 $0.380924$（由AlphaEvolve保持）进一步降低。这不仅是数值上的微小提升，更是数学结构上的新发现。此外，在自相关不等式（Autocorrelation Inequalities）问题上，它也构建出了优于现有最佳结果的阶跃函数。

<img src="/images/2601.16175v1/x2.jpg" alt="Refer to caption" style="width:90%; max-width:700px; margin:auto; display:block;">

*图2：TTT-Discover发现的算法通过FFT加速梯度下降，找到了最小化相关性边界的新解。*

#### 2. 工程：GPU 内核优化 (2倍速提升)

在GPU编程领域，每一微秒的优化都价值连城。研究团队在GPUMode竞赛的任务中测试了TTT-Discover。

结果显示，TTT-Discover编写的GPU内核（TriMul competition）比现有最佳人类解决方案快了近 **2倍**。

*   **人类专家评价**：GPUMode组织者指出，AI生成的方案极其激进地进行了算子融合（Operator Fusion），减少了内存带宽压力，这是大多数人类选手未能做到的。

#### 3. 算法设计：AtCoder 竞赛

在AtCoder的启发式算法竞赛（Heuristic Competitions）中，TTT-Discover 在两个历史比赛（ahc039 和 ahc058）中均取得了超越已知最佳AI结果的成绩，其生成的代码能够处理极其复杂的调度和规划问题。

#### 4. 生物学：单细胞分析去噪

在单细胞RNA测序数据分析中，去噪是一个关键步骤。TTT-Discover 发现的去噪算法在均方误差（MSE）和泊松指标上均优于目前的SOTA方法（如MAGIC和ALRA），得到了MIT生物学教授的高度评价。

### 总结与启示

TTT-Discover 的成功向我们展示了一个反直觉的事实：**解决最难的科学问题，可能并不需要更强的预训练模型，而是需要更强的“临场学习”能力。**

该研究证明，通过在测试时进行针对性的强化学习，即使是参数量较小的开源模型，也能在特定领域超越闭源的顶级模型。更重要的是，这种方法的成本极低——在Tinker平台上，解决一个问题的成本仅需数百美元。

这或许预示着AI科学发现的新范式：未来的AI科学家，不再是带着装满知识的“死脑筋”进考场，而是带着一本空白的草稿纸，在考场上通过不断的自我进化，推导出人类未知的真理。