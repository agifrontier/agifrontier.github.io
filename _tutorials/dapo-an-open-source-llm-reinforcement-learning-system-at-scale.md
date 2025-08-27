---
layout: default
title: "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"
---

# TL;DR
本文提出了一种名为 DAPO 的大规模强化学习算法和系统，通过解耦裁剪与动态采样等四项关键技术，有效解决了长思路链（long-CoT）场景下的熵坍塌、训练不稳定等问题，并在Qwen2.5-32B模型上实现了AIME 2024数学竞赛SOTA性能。

# 关键定义
本文提出或沿用了以下对理解本文至关重要的核心概念：

1.  **DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization)**: 本文提出的核心强化学习算法。它基于 GRPO，但引入了四项关键改进：解耦的裁剪范围（Clip-Higher）、动态采样、Token级策略梯度损失和超长回答奖励塑造，旨在解决大规模LLM在长思路链推理任务中训练不稳定的问题。
2.  **Clip-Higher**: 一种改进的PPO裁剪策略。它将传统的对称裁剪范围 `ε` 分解为独立的下限 `ε_low` 和上限 `ε_high`。通过设置一个比 `ε_low` 更大的 `ε_high`，该策略放宽了对低概率“探索性”Token概率提升的限制，从而缓解熵坍塌，促进模型探索。
3.  **动态采样 (Dynamic Sampling)**: 一种数据处理策略，用于解决训练过程中梯度信号减弱的问题。当一个批次中的所有样本都得到相同奖励（全对或全错）时，优势（advantage）为零，导致无效梯度。该策略通过持续采样并过滤掉这些梯度为零的样本组，直到批次被有效样本填满，从而保证训练效率和稳定性。
4.  **Token级策略梯度损失 (Token-Level Policy Gradient Loss)**: 一种损失计算方式。与GRPO中按样本（sample-level）平均损失不同，该方法将批次中所有Token的损失直接相加再平均。这确保了每个Token对梯度的贡献是均等的，避免了长序列中的Token权重被稀释，同时能更有效地惩罚长回答中的无用模式。

# 相关工作
*   **研究现状**: 当前，通过大规模强化学习（Reinforcement Learning, RL）来激发大型语言模型（LLM）进行复杂推理（如长思路链 CoT）已成为前沿方向。以OpenAI的o1和DeepSeek的R1为代表的模型在此类任务上表现卓越。然而，其核心的RL算法和训练细节并未公开，导致社区难以复现其结果。
*   **现有瓶颈**: 直接应用像PPO或GRPO这样的标准RL算法进行大规模训练时，会遇到熵坍塌（模型输出趋同，失去探索能力）、奖励噪声和训练不稳定等严重问题。研究者普遍发现，在缺少关键技术细节的情况下，复现SOTA模型的性能极为困难。
*   **本文目标**: 本文旨在揭示并解决大规模LLM强化学习中的关键挑战，通过提出DAPO算法及配套技术，并开源包括算法、代码和数据在内的完整系统，提供一个可复现、达到业界顶尖水平的解决方案，从而推动该领域的进一步发展。

# 本文方法
本文提出了**DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization)** 算法，其核心目标函数如下：


$$
\mathcal{J}_{\text{DAPO}}(\theta)=\mathbb{E}_{(q,a)\sim\mathcal{D},\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta_{\text{old}}}(\cdot\mid q)}\Bigg{[}\frac{1}{\sum_{i=1}^{G}|o_{i}|}\sum_{i=1}^{G}\sum_{t=1}^{|o_{i}|}\min\Big{(}r_{i,t}(\theta)\hat{A}_{i,t},\ \text{clip}\Big{(}r_{i,t}(\theta),1-{\varepsilon_{\text{low}}},1+{\varepsilon_{\text{high}}}\Big{)}\hat{A}_{i,t}\Big{)}\Bigg{]}
$$


约束条件为，对于每个问题`q`，其采样的一组回答`{o_i}`中，必须既有正确答案也有错误答案，即:


$$
0<\Big{|}\{o_{i}\mid\texttt{is\_equivalent}(a,o_{i})\}\Big{|}<G
$$



该算法在GRPO的基础上引入了四项关键创新技术，以实现稳定高效的大规模长思路链强化学习。

### 创新点1：Clip-Higher
在标准的PPO或GRPO中，策略更新的裁剪范围 `ε` 是对称的，这会限制低概率“探索性”Token的概率提升，导致策略过早收敛和“熵坍塌”现象。
<img src="/images/2503.14476v2/x2.jpg" alt="acc and entropy" style="width:85%; max-width:600px; margin:auto; display:block;">
*(a) AIME准确率*
<img src="/images/2503.14476v2/x3.jpg" alt="acc and entropy" style="width:85%; max-width:600px; margin:auto; display:block;">
*(b) Actor模型熵*
**图2**: 应用Clip-Higher策略前后，RL训练过程中AIME测试集准确率和Actor模型生成概率的熵变化。

本文提出的**Clip-Higher**策略将裁剪范围解耦为下限 `ε_low` 和上限 `ε_high`。通过设置一个较大的 `ε_high` (例如0.28) 并保持较小的 `ε_low` (例如0.2)，算法允许对有益的、但初始概率较低的Token进行更大幅度的概率提升，从而鼓励探索，维持了更高的策略熵，避免了多样性的丧失。

### 创新点2：动态采样
在RL训练中，当一个问题采样的所有回答都正确（或都错误）时，它们的奖励相同，导致计算出的优势(`advantage`)为零，这些样本对梯度更新没有贡献。随着模型能力增强，全对样本的比例会不断增加，导致有效训练数据减少，梯度方差增大，训练效率下降。

<img src="/images/2503.14476v2/x5.jpg" alt="dynamic sampling" style="width:85%; max-width:600px; margin:auto; display:block;">
**图3 (b)**: 训练过程中，所有采样回答都正确的样本比例不断增加。

为了解决这个问题，本文提出了**动态采样**。在每个训练批次，系统会持续采样，并动态过滤掉那些所有回答奖励都相同的样本组，直到收集到足够数量的、包含不同奖励（即有对有错）的“有效”样本组。这确保了每个批次的梯度都是有效的，显著提升了训练的稳定性和样本效率。

### 创新点3：Token级策略梯度损失
原始GRPO采用样本级（sample-level）损失计算，即先计算每个样本内所有Token的平均损失，再对不同样本的损失求平均。这种方式下，长序列中的每个Token对总损失的贡献会被稀释。这不仅使得模型难以学习长推理路径中的关键模式，也无法有效惩罚长回答中出现的无意义重复或“乱码”（gibberish）等低质量内容。

<img src="/images/2503.14476v2/x6.jpg" alt="entropy and length" style="width:85%; max-width:600px; margin:auto; display:block;">
*(a) Actor模型生成概率的熵*
<img src="/images/2503.14476v2/x7.jpg" alt="entropy and length" style="width:85%; max-width:600px; margin:auto; display:block;">
*(b) Actor模型生成的回答平均长度*
**图4**: 应用Token级损失（右侧曲线）可以控制熵和回答长度的健康增长，而样本级损失（左侧曲线）则导致两者不受控制地增加。

本文转而采用**Token级策略梯度损失**，将批次中所有Token的损失直接求和再平均。这保证了每个Token（无论其所在序列长短）对梯度更新的贡献是平等的，从而更有效地学习和惩罚各种生成模式，使得回答长度和熵的变化更加健康。

### 创新点4：超长奖励塑造
在生成长思路链时，模型回答的长度可能超过预设的最大值而被截断。如果简单地给这些被截断的回答一个负奖励，可能会惩罚一个本是正确但过长的推理过程，引入奖励噪声，干扰训练。

<img src="/images/2503.14476v2/x8.jpg" alt="overlong reward shaping" style="width:85%; max-width:600px; margin:auto; display:block;">
*(a) AIME性能*
<img src="/images/2503.14476v2/x9.jpg" alt="overlong reward shaping" style="width:85%; max-width:600px; margin:auto; display:block;">
*(b) Actor模型熵*
**图5**: 应用超长奖励塑造策略后，训练更稳定，性能和熵表现更优。

本文提出了**超长奖励塑造 (Overlong Reward Shaping)** 策略。首先，通过过滤（masking）掉被截断样本的损失来稳定训练。在此基础上，进一步设计了一种**软性超长惩罚 (Soft Overlong Punishment)** 机制。当回答长度超过某个阈值但在最大长度限制内时，会根据其超出的长度施加一个线性的负奖励，引导模型生成更简洁的回答，而不是因过长而被粗暴惩罚。


$$
R_{\text{length}}(y)=\begin{cases}0,&|y|\leq L_{\text{max}}-L_{\text{cache}}\\ \frac{(L_{\text{max}}-L_{\text{cache}})-|y|}{L_{\text{cache}}},&L_{\text{max}}-L_{\text{cache}}<|y|\leq L_{\text{max}}\\ -1,&L_{\text{max}}<|y|\end{cases}
$$



# 实验结论
*   **主要成果**: 基于Qwen2.5-32B基础模型，DAPO算法在AIME 2024测试集上取得了50分的成绩，超过了之前使用相同基础模型的DeepSeek-R1-Zero（47分），并且训练步数仅为其50%。
<img src="/images/2503.14476v2/x1.jpg" alt="main result" style="width:90%; max-width:800px; margin:auto; display:block;">

*   **消融实验**: 实验证明了各项技术改进的有效性。从一个仅能达到30分的朴素GRPO基线开始，逐步加入超长过滤、Clip-Higher、软性惩罚、Token级损失和动态采样等技术后，模型性能稳步提升至50分。

| 模型 | $\textbf{AIME24}\_{\text{avg@32}}$ |
| --- | --- |
| DeepSeek-R1-Zero-Qwen-32B | 47 |
| 朴素GRPO | 30 |
| + 超长过滤 | 36 |
| + Clip-Higher | 38 |
| + 软性超长惩罚 | 41 |
| + Token级损失 | 42 |
| + 动态采样 (DAPO) | 50 |

*   **训练动态分析**: 本文强调监控训练过程中的关键指标至关重要。回答长度、奖励分数和生成熵等指标的动态变化曲线（如下图）揭示了RL训练的复杂性，并可作为诊断问题的依据。例如，通过Clip-Higher策略成功解决了熵坍塌问题，并发现维持熵的缓慢上升有利于模型性能提升。
<img src="/images/2503.14476v2/x11.jpg" alt="training dynamics" style="width:85%; max-width:600px; margin:auto; display:block;">
(a) 平均回答长度
<img src="/images/2503.14476v2/x12.jpg" alt="training dynamics" style="width:85%; max-width:600px; margin:auto; display:block;">
(b) 奖励分数
<img src="/images/2503.14476v2/x13.jpg" alt="training dynamics" style="width:85%; max-width:600px; margin:auto; display:block;">
(c) 生成熵
<img src="/images/2503.14476v2/x14.jpg" alt="training dynamics" style="width:85%; max-width:600px; margin:auto; display:block;">
(d) 平均概率
**图7**: DAPO训练过程中的关键指标动态曲线。

*   **涌现能力**: 实验观察到，随着RL训练的进行，模型不仅会强化已有的正确推理模式，还会自发产生新的高级推理行为，例如对自己的推理过程进行反思和回溯修正。
*   **最终结论**: 本文提出的DAPO算法及其包含的四项关键技术，成功解决了大规模LLM在长思路链推理任务中遇到的核心挑战，提供了一个高效、可复现的开源RL系统，其性能达到了SOTA水平。