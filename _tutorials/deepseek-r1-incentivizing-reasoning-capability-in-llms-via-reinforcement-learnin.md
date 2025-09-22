---
layout: default
title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
---

# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

- **ArXiv URL**: http://arxiv.org/abs/2501.12948v1

- **作者**: K. Yu; Bei Feng; Yuting Yan; Yanping Huang; Shiyu Wang; Jingchang Chen; Xiaodong Liu; Yu-Wei Luo; Jingyang Yuan; Zhean Xu; 等196人

- **发布机构**: DeepSeek

---

# TL;DR
本文通过大规模强化学习（无论是纯粹应用于基础模型还是结合少量冷启动数据），成功地激发并显著增强了大型语言模型的推理能力，推出了DeepSeek-R1系列模型，并验证了可以将这种高级推理能力通过蒸馏有效地迁移到更小的模型中。

# 关键定义
*   **DeepSeek-R1-Zero**: 本文提出的第一代推理模型之一。它的核心特征是直接在基础模型上进行大规模强化学习（Reinforcement Learning, RL），而**不经过**监督微调（Supervised Fine-Tuning, SFT）作为预备步骤。该模型验证了仅通过RL即可自发涌现出强大的推理能力。
*   **DeepSeek-R1**: 在DeepSeek-R1-Zero基础上改进的增强版模型。它采用了一个多阶段训练流程，包括使用少量高质量的“冷启动数据”进行初始SFT，然后进行多轮RL和SFT的迭代，旨在解决R1-Zero的可读性问题并进一步提升性能和通用能力。
*   **GRPO (Group Relative Policy Optimization)**: 本文采用的核心强化学习算法。它是一种无评论家（critic-free）的RL方法，通过在同一提示下采样一组输出，并根据这组输出的奖励（rewards）来估计优势（advantage），从而更新策略模型。这种方法相比传统方法节省了训练成本。
*   **冷启动数据 (Cold-start data)**: 指用于DeepSeek-R1初始微调阶段的一小部分（数千条）高质量、长思维链（Chain-of-Thought, CoT）的监督数据。这些数据为模型提供了推理模式的“种子”，有助于加速RL收敛并确保生成内容具有良好的可读性。

# 相关工作
当前，训练后（post-training）阶段已成为提升大型语言模型（LLM）能力的关键环节，尤其是在推理方面。领域内的前沿（SOTA）工作，如OpenAI的o1系列模型，通过在推理时增加思维链的长度，在数学、编码和科学推理等任务上取得了显著进展。然而，如何有效地实现测试时计算扩展（test-time scaling）对整个研究界来说仍是一个开放性问题。

现有的探索方向包括基于过程的奖励模型（process-based reward models）、强化学习以及蒙特卡洛树搜索（MCTS）等搜索算法。尽管这些方法取得了一定的成果，但尚未有任何一种方法能在通用推理性能上达到与OpenAI o1系列模型相媲美的水平。

本文旨在解决的核心问题是：能否**仅通过纯粹的强化学习**，而不依赖任何监督数据，来激发LLM的推理潜力，并使其达到或超越当前最先进的水平。同时，本文也探索如何通过一个更完善的流程来解决纯RL方法可能带来的可读性差、语言混杂等问题。

# 本文方法

本文的核心方法是利用大规模强化学习来提升LLM的推理能力。作者提出了两种具体的实现路径：DeepSeek-R1-Zero，一种纯粹的RL探索；以及DeepSeek-R1，一个更成熟和用户友好的多阶段训练流程。

### DeepSeek-R1-Zero：在基础模型上进行强化学习

DeepSeek-R1-Zero旨在探索在没有任何监督数据的情况下，LLM通过纯粹的RL过程自我演化出推理能力的潜力。

#### 创新点

该方法的核心创新在于**直接将RL应用于基础模型**（DeepSeek-V3-Base），绕过了传统的SFT预备步骤。它证明了推理能力可以作为一种“涌现”行为，通过奖励信号被激励出来，而非必须通过模仿人类标注的CoT数据来学习。

#### 算法与奖励
*   **RL算法**: 采用了**GRPO (Group Relative Policy Optimization)** 算法。该算法无需训练一个与策略模型同样大小的评论家模型，而是从一组样本的得分中估计基线，从而节省训练成本。其优化目标如下：


{% raw %}$$
\mathcal{J}_{GRPO}(\theta)=\mathbb{E}_{[q\sim P(Q),\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta_{old}}(O \mid q)]} \\ \frac{1}{G}\sum_{i=1}^{G}\left(\min\left(\frac{\pi_{\theta}(o_{i} \mid q)}{\pi_{\theta_{old}}(o_{i} \mid q)}A_{i},\text{clip}\left(\frac{\pi_{\theta}(o_{i} \mid q)}{\pi_{\theta_{old}}(o_{i} \mid q)},1-\varepsilon,1+\varepsilon\right)A_{i}\right)-\beta\mathbb{D}_{KL}\left(\pi_{\theta} \mid  \mid \pi_{ref}\right)\right),
$${% endraw %}


其中优势$A\_i$由组内奖励计算得出：


{% raw %}$$
A_{i}=\frac{r_{i}-{\mathrm{mean}(\{r_{1},r_{2},\cdots,r_{G}\})}}{{\mathrm{std}(\{r_{1},r_{2},\cdots,r_{G}\})}}.
$${% endraw %}



*   **奖励模型**: 采用了基于规则的奖励系统，主要包含两部分：
    1.  **准确性奖励**：通过检查最终答案是否正确来提供奖励。例如，对于数学题，验证答案是否与标准答案匹配；对于编程题，通过编译器运行测试用例。
    2.  **格式奖励**：强制模型将思考过程包裹在 $$<think>$$ 和 $$</think>$$ 标签内。
    本文明确指出**未使用神经奖励模型**，以避免在大规模RL过程中出现奖励 hacking（reward hacking）问题，并简化训练流程。

#### 涌现与不足
训练过程中，DeepSeek-R1-Zero自发地学会了通过增加思考时间（生成更长的CoT）来解决复杂问题，并涌现出如自我反思、探索不同解题路径等高级行为，甚至出现了类似“啊哈时刻（aha moment）”的自我修正。

<img src="/images/2501.12948v1/plot_length.jpg" alt="DeepSeek-R1-Zero在训练过程中平均响应长度的变化" style="width:85%; max-width:600px; margin:auto; display:block;">

然而，这种纯粹RL训练出的模型存在可读性差、中英文混杂等问题，不适合直接面向用户。


| DeepSeek-R1-Zero中间版本的“啊哈时刻”示例 |
| :--- |
| **问题**: 如果 $a>1$，那么 $\sqrt{a-\sqrt{a+x}}=x$ 的实数解之和等于 |
| **响应**: <think> |
| 要解方程 $\sqrt{a-\sqrt{a+x}}=x$，我们先两边平方…… |
| $\left(\sqrt{a-\sqrt{a+x}}\right)^{2}=x^{2}\implies a-\sqrt{a+x}=x^{2}.$ |
| 重新整理以分离内部的平方根项： |
| $(a-x^{2})^{2}=a+x\implies a^{2}-2ax^{2}+(x^{2})^{2}=a+x\implies x^{4}-2ax^{2}-x+(a^{2}-a)=0$ |
| …… |
| 等等，等等。等等。这里有个我可以标记的啊哈时刻。 |
| 让我们逐步重新评估，以确定正确的和是否可以为…… |
| 我们从这个方程开始： |
| $\sqrt{a-\sqrt{a+x}}=x$ |
| 首先，让我们两边平方： |
| $a-\sqrt{a+x}=x^{2}\implies\sqrt{a+x}=a-x^{2}$ |
| 接下来，我可以再次对两边进行平方，处理这个方程：…… |
| …… |

### DeepSeek-R1：带冷启动的强化学习

为了解决DeepSeek-R1-Zero的不足，并进一步提升性能，本文设计了一个包含四个阶段的精细化训练流程来构建DeepSeek-R1。

#### 流程与创新
1.  **阶段一：冷启动 (Cold Start)**
    在RL开始前，先用数千条高质量、人类友好的长CoT数据对基础模型进行SFT。这些数据通过精心设计，确保了输出格式的**可读性**，并为模型后续的RL探索提供了有利的起点。

2.  **阶段二：面向推理的强化学习 (Reasoning-oriented RL)**
    在冷启动模型的基础上，应用与R1-Zero相同的GRPO算法进行RL训练。此阶段专注于提升编码、数学、科学等推理能力。为解决语言混杂问题，额外引入了一个**语言一致性奖励**，鼓励模型在CoT中使用目标语言。

3.  **阶段三：拒绝采样与监督微调 (Rejection Sampling and SFT)**
    当RL收敛后，使用该阶段的模型通过**拒绝采样**（只保留正确答案的生成轨迹）来收集约60万条高质量的推理数据。同时，结合了约20万条来自DeepSeek-V3的非推理数据（如写作、事实问答等）来增强模型的通用能力。最后，使用这约80万条的混合数据对**原始的基础模型**（DeepSeek-V3-Base）进行新一轮的SFT。

4.  **阶段四：全场景强化学习 (RL for all Scenarios)**
    为了进一步对齐人类偏好，对上一阶段微调后的模型进行第二轮RL。此阶段结合了**规则奖励**（用于推理任务）和**神经奖励模型**（用于评估通用任务的有用性和无害性），旨在同时优化模型的推理、有用性和安全性。

### 蒸馏：赋予小模型推理能力

为了让更高效的小型模型也能具备强大的推理能力，本文采用了一种直接的蒸馏方法。
*   **方法**: 使用在DeepSeek-R1训练流程中（阶段三）收集的80万个高质量SFT样本，对Qwen和Llama系列的多个开源模型进行微调。
*   **目的**: 验证大模型通过复杂流程学到的推理模式，可以被有效地“教会”给小模型，这是一种高效且经济的能力迁移方式。

# 实验结论

实验结果有力地证实了本文方法的有效性。

<img src="/images/2501.12948v1/x1.jpg" alt="DeepSeek-R1的基准测试性能" style="width:85%; max-width:600px; margin:auto; display:block;">

#### 关键结果
*   **DeepSeek-R1 vs. 顶级模型**:
    *   在**推理任务**上，DeepSeek-R1表现出色，在AIME 2024 (pass@1 79.8%) 和MATH-500 (pass@1 97.3%) 等数学基准上达到了与OpenAI-o1-1217相媲美甚至略微超越的水平。在编程竞赛任务Codeforces上，其Elo评分达到2029，超过了96.3%的人类参赛者。
    *   在**知识和通用能力**上，DeepSeek-R1在MMLU、GPQA Diamond等知识密集型基准上显著优于其前身DeepSeek-V3。在AlpacaEval 2.0和ArenaHard等评估开放式生成能力的基准上也取得了极高的胜率，证明了RL不仅增强了推理，也泛化到了其他能力。
*   **DeepSeek-R1-Zero的性能**:
    *   仅通过纯RL训练的DeepSeek-R1-Zero在AIME 2024上pass@1达到71.0%，多数投票后达到86.7%，性能与OpenAI-o1-0912相当，验证了纯RL方法的巨大潜力。


| 模型 | AIME 2024 | | MATH-500 | GPQA | LiveCode | CodeForces |
| --- | --- | --- | --- | --- | --- | --- |
| | pass@1 | cons@64 | pass@1 | Diamond pass@1 | Bench pass@1 | rating |
| OpenAI-o1-mini | 63.6 | 80.0 | 90.0 | 60.0 | 53.8 | 1820 |
| OpenAI-o1-0912 | 74.4 | 83.3 | 94.8 | 77.3 | 63.4 | 1843 |
| DeepSeek-R1-Zero | 71.0 | 86.7 | 95.9 | 73.3 | 50.0 | 1444 |

*   **蒸馏模型的性能**:
    *   蒸馏方法极为成功。例如，$$DeepSeek-R1-Distill-Qwen-7B$$在多个推理基准上全面超越了像GPT-4o这样的大型通用模型。$$DeepSeek-R1-Distill-Qwen-32B$$和$$70B$$模型在大多数基准上显著超过了o1-mini，为开源社区树立了新的密集模型（dense model）性能标杆。
*   **蒸馏 vs. 直接RL**: 对比实验表明，从强大的DeepSeek-R1**蒸馏**到32B模型，其性能**远超**直接在32B基础模型上进行大规模RL训练所能达到的水平。这说明，由更强大基础模型发现的“智能模式”对于提升小模型能力至关重要。


| 模型 | AIME 2024 | | MATH-500 | GPQA Diamond | LiveCodeBench |
| --- | --- | --- | --- | --- | --- |
| | pass@1 | cons@64 | pass@1 | pass@1 | pass@1 |
| QwQ-32B-Preview | 50.0 | 60.0 | 90.6 | 54.5 | 41.9 |
| DeepSeek-R1-Zero-Qwen-32B | 47.0 | 60.0 | 91.6 | 55.0 | 40.2 |
| DeepSeek-R1-Distill-Qwen-32B | 72.6 | 83.3 | 94.3 | 62.1 | 57.2 |

#### 存在不足
*   DeepSeek-R1在某些任务上仍有不足。例如，由于安全对齐，它在中文事实问答（C-SimpleQA）上表现不如DeepSeek-V3。在软件工程任务（如Aider）上的提升也相对有限。
*   该模型对提示词（prompt）较为敏感，使用零样本（zero-shot）提示效果最好，而少样本（few-shot）提示反而可能降低其性能。

#### 最终结论
本文成功证明了通过大规模强化学习可以有效激发和提升LLM的推理能力。DeepSeek-R1的多阶段管线不仅实现了与业界顶尖模型相媲美的推理性能，而且兼顾了输出的可读性和通用性。更重要的是，研究发现通过蒸馏，可以将这种来之不易的推理能力高效地赋予各种规模的开源模型，为整个社区的发展提供了宝贵的资源和途径。