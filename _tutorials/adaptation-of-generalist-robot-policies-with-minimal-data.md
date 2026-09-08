---
layout: default
title: "CMU提出MiDAS：仅靠1次示范+在线残差RL，打通具身大模型自主微调"
description: "针对这一瓶颈，来自卡内基梅隆大学（Carnegie Mellon University, CMU）的研究团队提出了一种更加逼近全自主进化的新范式—— 极简数据自适应（Minimal-Data Adaptation, MDA） 。"
arxiv_id: "2608.11363"
paper_published: "2026-08-11"
published_at: "2026-09-08T13:15:08.162312+08:00"
topics:
  - "具身智能"
  - "模型训练"
tags:
  - "LIBERO"
  - "MiDAS"
  - "RoboCasa"
  - "behavior cloning"
  - "minimal-data adaptation"
  - "offline-to-online RL"
related_tutorials:
  - "octo-an-open-source-generalist-robot-policy"
  - "turbovla-real-time-vision-language-action-model-at-32-hz-on-an-rtx-4090-with-1-g"
  - "behavior-cloning-is-not-all-you-need-the-optimality-of-on-policy-distillation-fo"
  - "bridgedata-v2-a-dataset-for-robot-learning-at-scale"
---

<p class="paper-original-title" lang="en">Adaptation of Generalist Robot Policies with Minimal Data</p>

让通用机器人像人类一样，通过在真实世界中“自主摸索、试错修正”来习得新技能，一直是机器人学习领域的核心追求。然而在现实中，直接让现有的视觉-语言-动作（Vision-Language-Action, VLA）模型进入全新场景从零探索，几乎注定会陷入死局：稀疏奖励环境下，模型的零样本探索极其脆弱，往往连第一次任务成功都碰不到，更谈不上自主迭代。为了跨越这一鸿沟，学术界此前通常依赖几十甚至上百条人类演示数据做离线微调，或者预设初始策略已经具备极高的任务成功率。

> ArXiv URL：https://arxiv.org/abs/2608.11363v1

针对这一瓶颈，来自卡内基梅隆大学（Carnegie Mellon University, CMU）的研究团队提出了一种更加逼近全自主进化的新范式——**极简数据自适应（Minimal-Data Adaptation, MDA）**，并构建了名为 **MiDAS** 的离线到在线（Offline-to-Online）强化学习框架。MiDAS 证明：**即使只有 1 次人类成功示范，配合恰当设计的在线交互，就足以撬动通用机器人策略的高效进化。**

在双臂物理机器人平台 YAM 上，MiDAS 从一个仅有 1 次示范、初始执行极其脆弱的策略出发，经过约 6 小时的自主在线交互，成功补全了末端控制的精度短板并掌握了新的动作模式。这项工作为大模型时代的机器人在线自适应指明了轻量、可落地的算法路径。

<img src="/images/2608.11363v1/main_fig.jpg" alt="MDA核心思想概览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 核心矛盾：大模型有通识，但缺少“临门一脚”的闭环精度

当前以 $\pi_{0.5}$、Octo 为代表的预训练 VLA 模型，通过海量多模态数据与多样化操作数据预训练，已经掌握了丰富的世界通识、语言理解能力以及泛化的视觉表征。然而，当把它们放到全新的物理场景中执行长程复杂任务时，这些泛化先验并不能直接转化为可靠的控制精度。

面对新任务，零样本部署往往完全脱靶；而如果仅依靠极少量示范（例如 $K=1$）进行行为克隆（Behavior Cloning, BC），策略通常只能“照猫画虎”。如下图所示，在 RoboCasa 环境的“双摩卡壶放上灶台”任务中，仅靠 1 次示范微调出的策略，在宏观层面上完全理解任务意图——机械臂能够准确走向目标物体、完成对位，但在真实环境的位姿扰动下，却会在抓取第二个壶把手时屡屡落空。

<img src="/images/2608.11363v1/both_mokapots_filmstrip.webp" alt="单次示范微调的局限性" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

这就揭示了一个关键机制事实：**极少数示范的核心价值不在于提供全状态分布的覆盖，而在于将漫无边际的探索空间收敛到一个任务强相关的狭窄流形上（Task Anchor）。** 示范解决了“往哪探索”的问题，但后续的微调与抗扰动鲁棒性，必须依靠机器人与环境的真实交互（Online Interaction）来自主填补。

### MiDAS 的两阶段配方：基座锚定与在线残差修正

为了在极度缺乏示范（$K=1$）且面临稀疏奖励的极端工况下稳定学习，CMU 团队放弃了直接端到端更新庞大 VLA 的常规思路，而是将自适应过程解耦为两个阶段：

1. **第一阶段：轻量 LoRA 行为克隆锚定任务（Stage I: Anchor）**

   利用这 1 次或极少数示范数据，通过 LoRA 对预训练 VLA 策略 $\pi_{\mathrm{base}}$ 进行参数高效微调，得到粗糙但具备任务指向性的基座策略 $\pi_{\mathrm{base}}^K$。这一步的唯一目的不是追求完美成功率，而是确保机器人在后续探索中有概率触碰到奖励信号，避免完全盲盒探索。

2. **第二阶段：冻结基座，在线强化学习训练残差策略（Stage II: Residual RL）**

   在进入环境交互后，直接对大模型进行高频在线强化学习更新会导致灾难性遗忘与严重的显存瓶颈。MiDAS 彻底冻结 $\pi_{\mathrm{base}}^K$ 的 VLM 主干和流匹配头，在其输出之上外挂一个轻量级的残差动作网络：

   


   {% raw %}$$ \pi_{\theta}^{\mathrm{res}}(\cdot\mid\mathbf{s}_{t},\mathbf{a}^{\mathrm{base}})=\tanh\!\left(\mathcal{N}\!\left(\mu_{\theta}(\mathbf{s}_{t},\mathbf{a}^{\mathrm{base}}),\;\sigma_{\theta}^{2}(\mathbf{s}_{t},\mathbf{a}^{\mathrm{base}})\right)\right) $${% endraw %}



   残差网络以 VLM 提取的高维表征和基座输出的动作块（Action Chunk）为输入，仅学习对基准动作的修正量。

在 Stage II 中，如果直接运行传统的在线强化学习，由于策略初期收集的绝大多数轨迹都是失败样本，Critic 极易崩溃。MiDAS 引入了三项关键技术抉择以稳定学习过程：

- **离线预热（Offline Warmup）**：在机器人正式自主交互前，利用预热损失 $\mathcal{L}_{\mathrm{warm}}(\theta)$ 引导残差 Actor 初始化在恒等映射附近，即 $\tanh(\mu_{\theta}(\mathbf{s},\mathbf{a}^{\mathrm{base}})) \approx \mathbf{a}^{\mathrm{base}}$，防止未训练的残差网络破坏第一阶段好不容易建立的粗糙行为先验。

- **样本均衡回放（Success-Balancing Replay）**：在稀疏奖励下，将成功轨迹以更高比例抽样送入批评网络训练批次，确保 Critic 能在海量失败中敏锐捕获成功的价值梯度。

- **基于策略无关 RL 的残差引导更新（PA-RL Updates）**：利用学到的动作价值函数 $Q_\psi(\mathbf{s}, \mathbf{a})$ 对动作实施一阶梯度上升：

  


  {% raw %}$$ \mathbf{a}_{t}^{\star} \leftarrow\mathbf{a}_{t}+\eta\,\nabla_{\mathbf{a}}Q_{\psi}(\mathbf{s}_{t},\mathbf{a}_{t}) $${% endraw %}



  这一机制赋予了机器人突破基座策略动作支持区（Action Support）的能力，使其能够根据即时价值反馈，做出超越人类那 1 次示范局限的纠错动作。

### 为什么有效？拆解极简示范自适应的三大支柱

MiDAS 之所以能够从单次示范中迅速泛化，并非依赖单一模块的魔法，而是预训练表征、示范引导和在线价值学习三者分工协作的结果。作者在 LIBERO 基准测试中进行了深入的表征与机制分析。

<img src="/images/2608.11363v1/libero_reps.webp" alt="表征与状态空间分析" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

首先，**预训练 VLA 提供了绝佳的状态抽象**。机器人若直接从像素端从零学习 Critic，需要数十万步样本；而直接复用预训练冻结的 VLM 表征，将高维视觉与任务上下文无缝压缩，极大地降低了 Critic 拟合复杂长程价值函数的样本复杂度。

其次，**在线价值学习打破了行为克隆的上限**。在复杂刀具拾取等任务中，纯 BC 策略往往在接近把手时的微小位移处停滞或滑脱。论文中的降维可视化与动作轨迹对比表明，PA-RL 残差更新并非在原地进行高斯抖动，而是明确沿着 Critic 的上升方向，对基座动作进行了关键帧上的位移补偿，精准纠正了末端姿态。

<img src="/images/2608.11363v1/combined_perturbation_comparison.webp" alt="鲁棒性与扰动对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 泛化边界：能扛住何种变化？

在自适应完成后，策略到底学到了什么级别的鲁棒性？研究团队通过对物体的初始位置、朝向以及背景视觉进行系统性扰动评估，得出了冷静且清晰的结论：

- **感知与小范围状态扰动完全可覆盖**：得益于预训练 VLM 的视觉不变性以及在线残差网络学到的局部闭环反馈，策略在面对光照变化、相机轻微视角偏移以及数厘米级的位置漂移时，展现出了极强的适应能力，显著超越了纯 BC 策略。

- **跨越抓取模态的拓扑跃迁仍受限**：如果物体朝向被旋转到极端角度，导致机器人必须采取完全不同的抓取拓扑形态（例如从“正握”变为“侧掏”），单靠当前狭窄重置分布下学出的残差策略仍会失效。

- **重置课程学习（Curriculum）是有效解法**：当研究人员在在线训练中逐步拓宽初始状态的重置分布（Reset Distribution Curriculum），残差 RL 便能进一步扩宽动作搜索空间，将高成功率外推到更大的位置与旋转范围。

### 真实物理部署与启示

在仿真验证之外，CMU 团队将 MiDAS 部署到了实际的双臂 YAM 机器人平台上。仅用人工手把手教学采集了 1 次长程示范，初始微调策略在真实机械公差和物体摆放轻微不一致的情况下频频抓空。随后，系统进入自主在线交互阶段。依靠自主重置与稀疏奖励机制，机器人在大约 6 小时的纯在线交互内完成了策略进化，不仅修复了抓取不稳的致命伤，还在真实环境中稳定重现了双臂协作任务。

这篇论文最重要的价值，在于指出了具身智能落地过程中一种极具性价比的工程路径：**我们不再需要追求一个在任何未见场景下都能“零样本通关”的万能策略，因为这在物理交互的多样性面前极度困难；也不需要为每个新工位采集数以千计的示教数据。**

只要预训练大模型具备足够的通识表征，人类给它一次“打样”，剩下的微调全部交给带有价值引导的轻量残差网络去自主摸索。从“1 次示范”到“稳定胜任”，MiDAS 为大模型时代的具身智能自主进化补上了至关重要的一块拼图。
