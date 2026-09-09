---
layout: default
title: "MobileWAM：前瞻思维链解锁移动操作，推理加速最高达8.7倍"
description: "为此，清华大学智能产业研究院（AIR）、上海交通大学、香港科技大学等多家机构联合提出了 MobileWAM ，首次将视频生成世界动作模型系统性适配到全身移动操作中。"
arxiv_id: "2608.04657"
paper_published: "2026-08-05"
published_at: "2026-09-09T13:15:08.606362+08:00"
topics:
  - "具身智能"
  - "推理"
tags:
  - "CoF"
  - "MobileWAM"
  - "WAMs"
  - "decoupled video-action denoising"
  - "layerwise joint attention"
  - "mixture-of-transformers"
related_tutorials:
  - "mixture-of-depths-attention"
  - "world-tokens-enhancing-embodied-policies-with-training-time-world-modeling"
  - "mixture-of-contexts-for-long-video-generation"
  - "gamewam-a-world-action-model-for-video-games"
---

<p class="paper-original-title" lang="en">MobileWAM: Bridging World Action Models to Mobile Manipulation with Chain-of-Foresight</p>

<img src="/images/2608.04657v2/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在机器人具身智能领域，基于视频生成大模型的“世界动作模型”（World Action Models, WAMs）正逐渐成为一股主流范式。这类模型通过在大规模互联网视频上预训练扩散模型（Diffusion Transformers），使系统天然内嵌了对物理世界运动规律的先验理解，从而在桌面机械臂抓取任务上取得了亮眼成绩。然而，一旦把固定底座的机械臂挪到可移动的底盘上，现有的具身模型便频现破绽。

> ArXiv URL：https://arxiv.org/abs/2608.04657v2

移动操作（Mobile Manipulation）绝非简单的“底盘导航”拼凑“桌面抓取”。固定相机下，背景静止、视角恒定；而在移动底盘上，视角随着底盘移动剧烈晃动，动作与未来画面呈现出高度的自运动相关性。底盘与机械臂构成了庞大的冗余自由度空间，早期底盘哪怕发生微小的航向偏移，都会在数十步之后导致机械臂完全脱离抓取工作空间。更致命的是，底盘移动与机械臂操作属于异质运动动力学（Heterogeneous Dynamics）：前者在开阔空间大范围运动，后者在局部空间追求毫米级的接触与避障，若用单一网络混杂学习，往往相互干扰。为此，清华大学智能产业研究院（AIR）、上海交通大学、香港科技大学等多家机构联合提出了 **MobileWAM**，首次将视频生成世界动作模型系统性适配到全身移动操作中。

<img src="/images/2608.04657v2/figure1.webp" alt="MobileWAM 概览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 从割裂走向统一：不对称联合注意力与移动专家混合

传统移动操作方案常年依赖两种妥协做法：要么采用模块化设计，用独立的导航算法带着底盘到位，再唤醒机械臂执行抓取，但接口处的累积误差无法挽回；要么依赖复杂的感知支路，强行引入3D点云或需要特权环境真值分割的多阶段训练。MobileWAM 选择了一条纯 RGB 输入的端到端路径，通过视频扩散 Transformer（World Expert）作为骨干，与一个轻量级的动作 Transformer（Action Expert）以逐层联合注意力（Joint Attention）的方式紧密咬合。

底干网络采用了预训练的图文到视频扩散模型（30层Block，隐藏层维度3072），通过 3D VAE 在时间和空间维度压缩视觉信号，使一个潜在时间步精准对齐一段时长的动作块（Action Chunk, $H=4$）。机器人的头部视角与手腕视角拼接输入，建立起跨视角的几何一致性。

真正的架构玄机藏在注意力掩码（Attention Mask）与前馈层设计中。为了让底盘运动与手臂抓取各司其职，动作专家的每个前馈网络（FFN）被重构为三专家混合结构（Mobile MoE）：包含一个共享专家（Shared Expert）、一个底盘运动专家（Locomotion Expert）以及一个抓取操作专家（Manipulation Expert）。该模块不采用生硬的硬编码分流，而是通过动作 Token 的运动意图向量驱动轻量软路由（Soft Router）。底盘大范围机动时激活移动专精，接近物体对准时转为抓取专精，而共享专家始终负责底层的本体感觉与空间运动共性表征，巧妙化解了异质动力学的参数更新冲突。

<img src="/images/2608.04657v2/figure2.webp" alt="MobileWAM 架构细节" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 前瞻思维链：在隐空间串联起因果未来

面对长程任务中的因果级联问题，简单的动作模仿容易迷失方向，而视频生成模型在训练时对未来的预测能力成了绝佳的辅助监督源。如果单纯让模型在训练时并行预测未来多步画面，目标之间缺乏时序上的条件约束，模型极易投机取巧，学不到动作的因果传递。MobileWAM 由此提出了 **Chain-of-Foresight (CoF)**，即隐空间的前瞻思维链。

CoF 从骨干网络中均匀抽取的 4 个深浅层中提炼初始信念状态 $h_0$：




{% raw %}$$ \mathbf{h}_{0}=g\!\left(\left[\mathbf{H}^{(4)};\,\mathbf{H}^{(12)};\,\mathbf{H}^{(20)};\,\mathbf{H}^{(30)}\right]\right) $${% endraw %}



随后，模型以类似循环神经网络（RNN）的自回归形式，逐步递推预测未来 $K$ 步的隐空间块：




{% raw %}$$ \left(\hat{\mathbf{v}}_{k},\;\mathbf{h}_{k}\right)=F_{k}\!\left(\mathbf{h}_{k-1},\,\mathbf{z}_{k}^{\tau_{k}};\;\ell,\mathbf{s}_{t},o_{t}\right),\quad k=1,\dots,K $${% endraw %}



每一步前瞻不仅预测当期的未来潜在视频速度场 $\hat{\mathbf{v}}_k$，还将浓缩了环境物理演变规律的信念状态 $\mathbf{h}_k$ 传递给下一个时间步。越远的未来施加越小的折扣损失权重（$w_1 > w_2 > w_3$）。

这项设计最精妙的取舍在于其与部署效率的解耦。在注意力掩码机制中，动作 Token 仅单向读取当前帧的视觉表征，视频生成的噪声 Token 与 CoF 模块绝不反向干扰当前帧的编码。这意味着，前瞻思维链在训练时通过反向传播的梯度，将“预判未来”的物理直觉深度刻入骨干网络的前向表征中；而在实机部署推断时，整个 CoF 预测链条连同未来视频生成分支被直接裁掉。

部署状态下的 MobileWAM 纯粹作为一个高效的“当前帧视觉编码器”，只需对骨干网络执行单次前向缓存 KV 值，动作专家在此基础上进行极少步数的流匹配（Flow Matching）去噪即可生成全身控制指令。在 NVIDIA A800 上的实测推断显示，MobileWAM 单次预测循环仅耗时 938 毫秒，相较于“先生成未来视频、再根据视频规划动作”的 Motus 和 LingBot-VA 等经典生成式 WAM，取得了 5.3 倍至 8.7 倍的显著推断加速，使端到端世界动作模型真正在具身移动底盘上具备了实时控制可行性。

### 模拟与真机验证：全方位超越强基线

在包含复杂厨房重排场景的标准移动操作基准 **ManiSkill-HAB**（SetTable 评测套件）上，MobileWAM 展现出了惊人的稳定性。评测囊括了开冰箱、取苹果、置物、拉抽屉等 7 项极具挑战的子任务序列。

对比当前主流的具身基准模型（包括引入 3D 点云的 DP3、AC-DiT 以及 AnchorVLA 等），MobileWAM 以纯 RGB 双视角输入和单阶段训练策略，取得了 **73.0%** 的平均任务成功率，高居所有参评方法榜首，在 7 个子任务中的 5 项刷新了最佳表现，且没有任何一项任务发生灾难性失败（许多基线在开冰箱或精准放置时成功率跌入个位数）。相较于专为移动操作设计的 AC-DiT，MobileWAM 的平均成功率高出 17.4 个百分点。

<img src="/images/2608.04657v2/figure3.webp" alt="泛化性与更多表现" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在消融实验中，作者层层剖析了各项设计的必要性：

1. **模块贡献拆解**：基础 WAM 的基线成功率为 65.4%，引入前瞻思维链 CoF 后跃升至 68.9%（纯靠训练期梯度塑造，零推断开销）；再叠加 Mobile MoE 路由后，成功率进一步攀升至 73.0%，尤其在需要基座重定位与毫米级末端释放并行的“Place Apple”任务上，成功率单项暴涨 11.4 个百分点。

2. **串行链式与并行监督**：消融表明，无因果递推的并行多步预测对表征提升极其有限（52.3%），而采用表达能力不足的简易 MLP 进行循环串联甚至会污染骨干特征（跌至 46.3%）；只有基于 Transformer 结构的深层信念传递（58.2%）才能将物理演变约束转化为有效的控制先验。

3. **软 MoE 对决硬拆分**：将动作专家直接切分为两个独立的底盘、手臂物理子网，会严重割裂全身协调性并砍半有效训练数据，性能出现显著滑坡；实验证明，在统一特征空间下通过软路由实现任务分流，是协调全身冗余自由度最具泛化性的路径。

为了验证模拟环境到真实物理世界的迁移能力，研究团队在配备升降机构与单臂的 **ARX Lift2** 移动操作机器人上进行了真机微调实验。测试设置了 5 个时间跨度与因果链条依次递增的家庭服务任务（从单步拉抽屉 $T_a$，跨越至长周期的“开抽屉-取物-转运-关抽屉”复合任务 $T_e$）。

在与强劲的具身基线 $\pi_{0.5}$ 正面对决中，MobileWAM 在所有 5 项真机任务上均取得了更高的成功率。更为关键的趋势在于：任务的因果深度越深、视点变化与时间跨度越长，MobileWAM 的领先优势越为悬殊。在极度考验长程规划因果性的复合任务 $T_e$ 中，$\pi_{0.5}$ 在多阶段转移中全军覆没（成功率为 0%），而 MobileWAM 依然保持了 15% 的真实端到端达成率。

### 具身世界模型落地的极简启示

MobileWAM 给机器人学习社区带来了一个至关重要的启示：移动机器人需要的时空因果一致性，并不一定非要依靠昂贵的多模态传感器堆叠（如密集 3D 点云）或多阶段分治算法去生硬构建。互联网海量视频预训练出的时空生成先验，本身就是一座未被充分挖掘的物理金矿。

通过构建类似前瞻思维链（Chain-of-Foresight）的因果辅助机制，在训练阶段倒逼潜在表征理解未来的物理流向，同时借助移动 MoE 在单一网络内优雅解构底盘与手臂的动力学冲突，模型便能在纯 RGB 的极简输入下掌握高鲁棒性的全身移动协同能力。更重要的是，“训练期脑补未来、部署期纯编码动作”的解耦架构，彻底摆脱了生成式世界模型推断拖沓的顽疾，为将更大规模的高清视频基础模型平滑搬上真实移动机器人底盘，指明了一条兼顾物理常识与执行效率的切实路线。
