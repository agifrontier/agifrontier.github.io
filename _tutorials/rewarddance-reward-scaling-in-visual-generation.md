---
layout: default
title: "RewardDance: Reward Scaling in Visual Generation"
---

# RewardDance: Reward Scaling in Visual Generation

- **ArXiv URL**: http://arxiv.org/abs/2509.08826v1

- **作者**: Jie Wu; Yu Gao; Zeyue Xue; Wei Liu; Hanzhong Guo; Ming Li; Xiaoxia Hou; Jie Liu; Yan Zeng; Zilyu Ye; 等12人

- **发布机构**: ByteDance

---

# TL;DR
本文提出了RewardDance，一个基于生成式范式的可扩展视觉奖励模型框架，它通过将奖励预测重构为“是/否”Token预测任务，实现了对模型尺寸和上下文信息的有效扩展，从而显著提升了视觉生成模型的质量和对齐效果。

# 关键定义
*   **生成式奖励范式 (Generative Reward Paradigm)**：一种新颖的奖励建模方法，它将奖励评分任务从传统的回归预测（输出一个分数）转变为一个生成任务。在该范式下，视觉语言模型 (Vision-Language Model, VLM) 被训练来预测在比较两个样本时，输出“是”（表示第一个更优）或“否”的Token，奖励值即为“是”Token的概率。
*   **上下文扩展 (Context Scaling)**：一种增强奖励模型输入信息丰富度的方法。它不仅使用传统的图像-文本对，还额外融入了任务感知指令 (task-aware instructions)、参考样本 (reference examples) 以及链式思维 (Chain-of-Thought, CoT) 推理过程，使奖励模型的判断更加准确和鲁棒。
*   **模型扩展 (Model Scaling)**：一种通过系统性地增加奖励模型参数量来提升其性能的策略。本文中，作者将奖励模型的规模从10亿 (1B) 参数扩展到260亿 (26B) 参数，并验证了更大的模型能带来更优的生成效果。

# 相关工作
当前的视觉生成领域由扩散模型主导，其性能通过奖励模型 (Reward Model, RM) 和强化学习等范式得到增强。然而，现有奖励模型的设计存在明显瓶瓶颈。

早期的奖励模型基于CLIP架构，其可扩展性差且泛化能力有限。后续基于VLM的模型虽然探索了新范式，但进展分散：一些工作实现了大规模模型，却局限于易受“奖励黑客”（Reward Hacking）攻击的回归式范式；另一些工作采用了更强大的生成式范式，但未能有效进行模型扩展。这种回归式范式与VLM的自回归、下一Token预测的内在机制存在根本性的“范式不匹配” (paradigm mismatch)，限制了VLM预训练知识的充分利用。

本文旨在解决上述问题，即现有视觉奖励模型缺乏统一且可扩展的设计原则。作者的目标是创建一个能够统一模型规模和上下文丰富度的框架，以充分释放VLM在高级视觉奖励建模中的潜力。

<img src="/images/2509.08826v1/x2.jpg" alt="插图" style="width:90%; max-width:700px; margin:auto; display:block;">

# 本文方法
本文提出了RewardDance，一个专为可扩展视觉奖励建模设计的框架。它通过引入新颖的生成式奖励范式，解决了传统回归式范式的局限性，并系统地从上下文和模型尺寸两个维度进行扩展。

<img src="/images/2509.08826v1/x3.jpg" alt="插图" style="width:85%; max-width:600px; margin:auto; display:block;">

### 方法背景：传统回归式范式的局限
传统的奖励模型通常采用“逐点回归”范式。模型独立评估单个图像，并通过一个回归头输出一个标量奖励值。这类模型通常使用布拉德利-特里 (Bradley-Terry, BT) 损失函数进行优化：


{% raw %}$$
\mathcal{L}_{BT}=-\mathbb{E}_{(y,x^{w},x^{l})\sim\mathcal{D}}\left[\log\left(\sigma\left(r(x^{w},y)-r(x^{l},y)\right)\right)\right],
$${% endraw %}


其中 $y$ 是提示词，$x^w$ 和 $x^l$ 分别是偏好和非偏好的图像，$r(\cdot,\cdot)$ 是奖励模型。这种方法存在“范式不匹配”问题，即回归任务与VLM原生的下一Token预测能力不符，从而限制了模型扩展的潜力。

### RewardDance：生成式奖励模型
为解决上述局限，RewardDance将奖励建模重新定义为一个生成任务。具体而言，它将奖励评分问题构建为一个比较判断任务，让模型预测“是/否”Token。奖励分数就是模型预测“是”Token的概率：


{% raw %}$$
r_{\theta}(x_{1},x_{2},y,i)=P_{\theta}(\text{"yes"}\mid x_{1},x_{2},y,i),
$${% endraw %}


其中 $x\_1, x\_2$ 是待比较的图像，$y$ 是提示词，$i$ 是任务指令。这种方法天然地与VLM的自回归机制对齐，为有效的扩展铺平了道路。

### 创新点1：上下文扩展 (Context Scaling)
RewardDance通过融入三种关键信息来扩展奖励模型的上下文：
1.  **任务感知指令**：在输入中加入明确的指令，指导模型根据特定标准进行评估。
2.  **参考图像**：采用成对比较（pairwise comparison）的方式，让模型判断$$image1$$是否优于$$image2$$。
3.  **链式思维 (CoT) 推理**：训练模型在给出“是/否”判断的同时，生成详细的推理过程。这些CoT数据由强大的教师模型蒸馏而来，不仅提升了模型的性能，也使其决策过程更具可解释性。

<img src="/images/2509.08826v1/cot_example.jpg" alt="插图" style="width:90%; max-width:700px; margin:auto; display:block;">

### 创新点2：模型扩展 (Model Scaling)
为了验证模型规模的影响，本文系统地使用了从10亿到260亿参数不等的InternVL系列模型作为奖励模型的骨干网络。实验证明，奖励模型的参数量与最终生成结果的质量存在强烈的正相关关系。

### 训练与对齐
学习到的RewardDance模型被用于指导扩散模型的优化，主要通过以下两种方式：
1.  **奖励微调 (Reward Finetuning)**：采用ReFL算法进行强化学习微调。由于RewardDance是比较模型，它首先通过Best-of-N (BoN) 采样策略从N个候选中选出最优质的图像作为后续微调的参考基准。
2.  **推理时扩展 (Inference-Time Scaling)**：采用“路径搜索” (Search over Paths) 策略在推理时优化生成。该策略在生成过程中动态剪枝，选择最有前景的生成轨迹。为了效率，此阶段使用了一个轻量级的、无需参考图像的“逐点” (pointwise) 生成式奖励模型变体作为验证器。

# 实验结论
本文通过在文生图、文生视频和图生视频等多个任务上的大量实验，验证了RewardDance框架的有效性。

### 奖励模型扩展的效果
*   **性能与模型规模正相关**：实验表明，无论是在RL微调还是推理时扩展的范式下，将奖励模型 (RM) 的规模从10亿参数扩展到260亿参数，都能为FLUX.1-dev和Seedream-3.0等基础模型带来持续且显著的性能提升。
*   **泛化能力是关键指标**：对奖励模型准确率的分析发现，模型在域内 (In-Domain, ID) 数据集上的准确率与其参数规模没有严格正相关。然而，在域外 (Out-Of-Domain, OOD) 数据集上的准确率（代表泛化能力）与最终的RL性能提升更为一致。这表明，评估RM时，**泛化能力比域内准确率更重要**。
*   **视频任务同样有效**：在文生视频 (T2V) 和图生视频 (I2V) 任务中，奖励模型的扩展同样带来了显著的性能增益。使用26B的RM相比于SFT基线模型，T2V任务性能提升了49%，I2V任务提升了47%。

### 与SOTA模型的对比
通过RewardDance优化的模型在多个公开基准测试上均达到了业界顶尖水平。
*   在Bench-240文生图基准上，Seedream-3.0 w RewardDance取得了0.848的最高分，超越了Imagen 3、Luma和Midjourney V6.1等强大的闭源商业模型。


<center>表1：在Bench-240基准上与SOTA模型的比较</center>


| 模型 | 整体得分 | 动作 | 属性 | 概念 | 物体 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Midjourney V6.1 | 0.63 | 0.62 | 0.51 | 0.52 | 0.68 |
| DALLE-3 | 0.67 | 0.70 | 0.57 | 0.53 | 0.71 |
| Luma | 0.77 | 0.71 | 0.68 | N/A | 0.82 |
| Imagen 3 | 0.79 | 0.77 | 0.66 | 0.73 | N/A |
| **Seedream-3.0 w RewardDance (本文)** | **0.848** | **0.87** | **0.89** | **0.78** | **0.89** |


*   在视频生成任务上，使用视频-文本对齐分数进行评估，Seedance 1.0 w RewardDance同样表现出色，在文生视频任务中得分最高 (1.66)，在图生视频任务中与最强模型持平 (1.65)。


<center>表2：在视频生成任务上与SOTA模型的比较（视频-文本对齐分数）</center>


| 模型 | 文生视频 (T2V) | 图生视频 (I2V) |
| :--- | :---: | :---: |
| Sora | 1.37 | – |
| Wan 2.1 | 1.49 | 1.36 |
| Kling 2.1 Master | 1.57 | 1.65 |
| Veo-3.0 | 1.63 | 1.59 |
| **Seedance 1.0 w RewardDance (本文)** | **1.66** | **1.65** |


### 消融研究
*   **抵抗奖励黑客**：更大规模的奖励模型（如26B）在训练后期仍能保持较高的奖励值方差，说明它们维持了更强的探索能力，对奖励黑客攻击的抵抗力更强。相比之下，小模型（1B/2B）很快收敛，停止了有效探索。
*   **范式与上下文的重要性**：实验证实，从回归式范式转为生成式范式能带来性能提升。在此基础上，加入参考图像进行比较评估能进一步提升效果。此外，使用高质量的参考样本 (Best-of-N) 和链式思维 (CoT) 数据进行训练，都能显著提高最终的生成质量。

<img src="/images/2509.08826v1/x4.jpg" alt="插图" style="width:85%; max-width:600px; margin:auto; display:block;">
<img src="/images/2509.08826v1/x5.jpg" alt="插图" style="width:85%; max-width:600px; margin:auto; display:block;">

### 可视化结果
可视化结果直观地展示了奖励模型扩展带来的好处。随着RM模型规模的增大，生成模型在处理包含复杂数量关系和多实例描述的提示词时，表现得越来越准确。

<img src="/images/2509.08826v1/x7.jpg" alt="插图" style="width:85%; max-width:450px; margin:auto; display:block;">
<img src="/images/2509.08826v1/x8.jpg" alt="插图" style="width:80%; max-width:300px; margin:auto; display:block;">

### 总结
本文明确了“可扩展性”应作为视觉奖励模型设计的核心原则。通过提出RewardDance框架及其创新的生成式范式，本文系统地论证了通过扩展奖励模型的尺寸和上下文丰富度，可以稳定、持续地提升视觉生成模型的质量。这一发现为未来更强大、更鲁棒的奖励模型研究开辟了新的道路。