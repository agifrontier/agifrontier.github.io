---
layout: default
title: "GaussMemory：统一3D高斯读写，长程机械臂操作超越π0-FAST达6%"
description: "东京理科大学团队提出的 GaussMemory 打破了这一僵局。该方案放弃了被动存储的死板逻辑，转向由任务驱动的“主动记忆”范式。论文的核心主张是：机器人不应该只是记录它看到了什么，而应该端到端地学会“如何去记”——自主发现哪些物体需要高精度追踪、以多大的激进度进行位置修正，以及何时丢弃无用信息。"
arxiv_id: "2608.14986"
paper_published: "2026-08-15"
published_at: "2026-09-23T13:15:08.398930+08:00"
topics:
  - "具身智能"
  - "知识系统"
tags:
  - "3D Gaussian Splatting"
  - "GaussMemory"
  - "LIBERO"
  - "MemoryVLA"
  - "VLABench"
  - "end-to-end learned memory updates"
related_tutorials:
  - "a-subgoal-driven-framework-for-improving-long-horizon-llm-agents"
  - "learning-on-the-job-an-experience-driven-self-evolving-agent-for-long-horizon-ta"
  - "end-to-end-test-time-training-for-long-context"
  - "memprism-task-conditioned-relational-memory-views-for-long-horizon-agents"
---

<p class="paper-original-title" lang="en">GaussMemory: Task-Driven 3D Gaussian Scene Memory for Long-Horizon Robotic Manipulation</p>

<img src="/images/2608.14986v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在具身智能领域，基于视觉-语言-动作（Vision-Language-Action, VLA）的大模型正在迅速改变机器人控制范式。无论是开源的 OpenVLA 还是具备强大动作生成能力的 $\pi_0$，都在短程桌面操作中展现出了惊人的泛化能力。然而，一旦将任务视界拉长，这些系统就会暴露出致命的缺陷——它们本质上缺乏持久的空间记忆。

> ArXiv URL：https://arxiv.org/abs/2608.14986v1

想象一个经典的长程交互场景：机器人需要把物品放入两个外观完全相同的抽屉之一，随后抽屉关闭，物体被遮挡；在执行了一系列无关的中间子任务后，机器人需要重新取出该物品。此时，机器人眼前的 RGB 观测已经无法提供任何关于“东西到底在哪个抽屉里”的线索。无记忆的 VLA 策略在这一步只能随机乱撞。

<img src="/images/2608.14986v1/concept1.webp" alt="概念示意与问题场景" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

现有的改进方案通常有两种思路：一种是在提示词或上下文窗口中塞入历史图像序列或文本日志，但这既丢失了精确的 3D 度量，又会迅速撑爆显存；另一种是构建 3D 体素或点云地图，然而现有的 3D 记忆机制大多退化为“被动的记录员”——它们依赖固定的手工规则，将视野内的一切像素机械地投射到空间中，无论是一个关乎抓取成败的核心滑块，还是背景中一块纹理斑驳的静止墙面，在存储和更新时都被赋予了完全相同的权重。

东京理科大学团队提出的 **GaussMemory** 打破了这一僵局。该方案放弃了被动存储的死板逻辑，转向由任务驱动的“主动记忆”范式。论文的核心主张是：机器人不应该只是记录它看到了什么，而应该端到端地学会“如何去记”——自主发现哪些物体需要高精度追踪、以多大的激进度进行位置修正，以及何时丢弃无用信息。凭借以 3D 高斯泼溅（3D Gaussian Splatting, 3DGS）为底座的动态几何基质与一体化注意力读写机制，GaussMemory 在长程任务基准 LIBERO Long-10 上斩获了 10 项任务中的 9 项第一，并在高难度复合基准 VLABench 上显著超越了 $\pi_0\text{-FAST}$。

<img src="/images/2608.14986v1/duibi.webp" alt="GaussMemory与现有记忆方案的范式对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 从被动刻录到动态表征：带时序元数据的 3D 高斯基质

要想在机器人操作中维持可靠的物理世界模型，底层的数据表征必须兼顾几何精确度与计算效率。显式点云缺乏表面连续性，而隐式神经辐射场（NeRF）的渲染与更新又过于沉重。GaussMemory 选择将 3D Gaussian Splatting 作为持久场景记忆的几何基质。

传统的 3DGS 主要用于静态新视角合成，其图元包含中心位置 $\mathbf{\mu}_i \in \mathbb{R}^3$、协方差矩阵 $\mathbf{\Sigma}_i$、不透明度 $\alpha_i$、基础颜色 $\mathbf{c}_i$ 以及球谐函数特征 $\mathbf{h}_i$。为了承载跨越长时间维度的操作历史，GaussMemory 将每个高斯基元扩充为带有动态生命周期属性的复合体：




{% raw %}$$ G_i = \Big( \underbrace{\mathbf{\mu}_i, \mathbf{\Sigma}_i, \alpha_i, \mathbf{c}_i, \mathbf{h}_i}_{\text{标准 3DGS 属性}}, \; \underbrace{t_i^c, t_i^m, \tau_i, o_i, e_i}_{\text{时序元数据}} \Big) $${% endraw %}



这套时序元数据赋予了高斯球物理与时间层面的身份：$t_i^c$ 和 $t_i^m$ 分别记录了该图元的创建时间戳与最后一次被修改的时间戳；$\tau_i$ 标记其对应的子任务阶段；$o_i \in \{1, \ldots, O\}$ 代表物体实例标签（通过结合 DINO 特征的 Gaussian Grouping 技术自动聚类生成）；$e_i \in (0, 1)$ 则表征该图元当前在场景中的“存在概率”。

为了让后续的 Transformer 架构能够感知这些几何图元随时间演化的状态，系统为每个高斯基元构建了包含多重先验的嵌入特征向量 $\mathbf{g}_i^{\text{mem}}$：




{% raw %}$$ \mathbf{g}_i^{\text{mem}} = \text{MLP}_{\text{enc}}\big([\mathbf{\mu}_i; \text{vec}(\mathbf{\Sigma}_i); \alpha_i; \mathbf{h}_i]\big) + \text{PE}_{3D}(\mathbf{\mu}_i) + \text{PE}_T^{\text{rel}}(t_i^m, t) $${% endraw %}



这里的 $\text{PE}_{3D}(\cdot)$ 为 3D 绝对空间位置编码，而 $\text{PE}_T^{\text{rel}}(t_i^m, t) = \text{PE}_T(t - t_i^m)$ 则巧妙地采用了相对时间正弦编码。这种相对时序设计使得模型无需强行泛化未见过的绝对时长，而是专注于“这个物体多久没有被更新过了”。

<img src="/images/2608.14986v1/crop_archi.webp" alt="系统整体架构图" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 统一记忆注意力（UMA）：将读与写融为一体

GaussMemory 最精妙的理论突破，在于解决了长期以来困扰记忆系统设计的架构割裂问题：**读与写的二元分立**。

在以往的模块化记忆系统中，更新记忆和提取记忆通常是两个完全独立的流程。更新记忆往往依赖手工规则或专用的启发式匹配算法（如 Hungarian 匹配或 Sinkhorn 算法），试图在观测与点云之间建立硬对应；而提取记忆时，则另外套用一个类似 Q-Former 的模块从记忆池中做交叉注意力检索。这种断裂的管线彻底切断了梯度在“任务需求”与“记忆维护”之间的双向流动——动作策略无法告诉更新模块“请重点更新那个抽屉的位置”，更新模块也无法向策略模块自适应地反映“这块区域的信息可能已经过时”。

GaussMemory 提出了**统一记忆注意力（Unified Memory Attention, UMA）**，将记忆的“检索读取”与“匹配写入”统一在单套可微交叉注意力计算中。

<img src="/images/2608.14986v1/archi714.webp" alt="统一记忆注意力机制内部数据流展开图" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

UMA 的计算管线划分为极具逻辑美感的四个阶段：

第一阶段是**构建统一查询（Unified Query）**。系统首先使用预训练的前馈 3DGS 编码器（FF-3DGS）从当前多视角 RGB 图像中快速重建出 $M$ 个实时高斯基元，随后在实例级别进行池化，压缩成 $N_o$ 个物体级观测 Token $\mathbf{Q}^{\text{obs}}$。与此同时，系统根据当前高层语言指令 $\mathcal{L}$ 与子任务进度，生成 $K$ 个任务读取 Query $\mathbf{Q}^{\text{read}}$：




{% raw %}$$ \mathbf{q}_k^{\text{read}} = \mathbf{q}_k^{\text{learn}} + \text{Proj}_L(\text{TextEnc}(\mathcal{L})) + \text{Proj}_\tau(\text{PE}_T(\tau_t)) $${% endraw %}



将 $\mathbf{Q}^{\text{obs}}$ 与 $\mathbf{Q}^{\text{read}}$ 拼接在一起，就形成了进入 UMA 的统一查询张量。

第二阶段是**自注意力交互（Observation-Task Exchange）**。在触碰底层 3D 记忆之前，$\mathbf{Q}^{\text{obs}}$ 与 $\mathbf{Q}^{\text{read}}$ 首先在一组自注意力层中展开深度交互。这一步的物理意义极其关键：任务 Query 会向观测 Token 施加任务意图，引导它筛选出与当前子任务最相关的物体（指令引导感知）；反过来，当前的即时观测 Token 也能将抽象的语言指令锚定到此时此刻的真实场景状态（感知落地任务）。这一步输出的上下文统一 Query，同时具备了**任务感知力**与**物理接地性**。

第三阶段是**时序记忆交叉注意力（Cross-Attention Against Memory）**。经过自注意力强化的 Query 向量组，以记忆库中保存的数万个持久高斯基元 $\mathbf{G}^{\text{mem}}$ 作为 Key 和 Value 展开跨模态注意力。计算过程中引入了基于相对修改时间的偏置矩阵 $\mathbf{B}_T$，让模型天然对新近演化的物体保持敏感：




{% raw %}$$ \mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{W}_Q(\mathbf{G}^{\text{mem}}\mathbf{W}_K)^\top}{\sqrt{d_k}} + \mathbf{B}_T\right) $${% endraw %}



第四阶段是**读写双流分流（Read and Write Streams）**。历经 $L$ 层 UMA 迭代后，输出张量顺理成章地兵分两路：

*   **读取流（Readout Stream）**：抽取出的 $K$ 个时空隐变量 $\mathbf{Z}^{\text{mem}}$，直接作为 3D 场景空间先验注入下游的 VLM 主干网络与扩散动作头（Diffusion Action Head），用于推断高精度的物理操作轨迹；

*   **更新流（Update Stream）**：交叉注意力矩阵中观测 Token 与记忆高斯之间的注意力权重，被直接复用为软匹配矩阵 $\mathbf{A}^{\text{match}}$。系统完全免去了复杂的几何对齐求解器，直接根据注意力分布计算观测对旧记忆的修正量。

在更新旧记忆位置时，系统通过轻量级网络动态预测更新率 $\eta_k \in (0, 1)$：




{% raw %}$$ \eta_k = \sigma\Big(\text{MLP}_\eta\big([\tilde{\mathbf{h}}_k^{\text{obs}}; w_k; \text{PE}_T^{\text{rel}}(t_k^m, t)]\big)\Big) $${% endraw %}






{% raw %}$$ \mathbf{\mu}_k^{\text{new}} = (1 - \eta_k)\mathbf{\mu}_k^{\text{mem}} + \eta_k \tilde{\mathbf{\mu}}_k^{\text{obs}} $${% endraw %}



如果当前视野中出现了与现有记忆都不匹配的新物体，系统会触发新图元插入机制；而那些存在概率 $e_i$ 衰减至阈值以下的高斯，则会被动态剪枝移出记忆库。整个过程彻底打通了下游动作损失 $\mathcal{L}_{\text{action}}$ 回传至更新门控 $\eta_k$ 和匹配矩阵 $\mathbf{A}^{\text{match}}$ 的梯度链路，让操作任务的结果直接“教会”记忆库该如何自我演化。

### 梯度引导下的自发认知：记忆行为的涌现

这种端到端学习框架带来的最震撼结果，体现在模型自主学会的“注意力分配策略”上。研究人员并没有写下任何一行“抽屉关上时必须保存其状态”、“背景墙面无需更新”的硬编码规则，然而整个记忆系统却在动作梯度的雕琢下自发涌现出了令人惊叹的动态取舍能力。

<img src="/images/2608.14986v1/crop_shiyanzhu.webp" alt="LIBERO Long-10 任务中动态更新率 eta 的定性可视化" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在 LIBERO Long-10 的“关闭抽屉、拾起碗并放到盘子里”这一连续复合任务中，更新门控 $\eta_k$ 的热力图展示出了极具启发性的演变模式（参见图 5 与图 9）：

在任务初始阶段，机械臂靠近并推动抽屉，对应抽屉把手与箱体的高斯基元 $\eta_k$ 迅速飙升至红色的高位（高激进更新），实时捕捉抽屉从开到关的位移；一旦抽屉关闭完成，其 $\eta_k$ 瞬间回落至接近零的深蓝状态（冰封保存）。紧接着，当机械臂转头抓取桌面的碗时，代表碗的高斯基元立刻转为深红色，直至碗被放置在盘子上方。与此形成鲜明对比的是，四周的背景墙壁、静止的桌面支撑结构，其 $\eta_k$ 在全长生命周期中始终顽固地维持在 0.06 左右的极低水平。

<img src="/images/2608.14986v1/eta.webp" alt="长时间步操作中抽屉与碗的 eta 动态跳变曲线" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

这充分印证了 GaussMemory 的设计哲学：**记忆不应当是均匀的传感器复刻，而应当是选择性的注意力收缩**。与操作强相关的实体享有高频的刷新带宽与极高的几何敏感度，而环境背景只需以极低功耗维持几何骨架。

从显存与计算容量的角度来看，这种机制同样表现出极高的优雅性。图 10 展示了在全长操作序列中高斯数量的变化轨迹。

<img src="/images/2608.14986v1/memory_size_over_time.webp" alt="高斯记忆容量与插入、剪枝动态演化机制" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

每当场景发生实质性交互或新视角切入时，图元插入机制（绿色曲线）会阶段性脉冲式激活；随着交互推进，失效被遮挡或冗余的图元被自适应剪枝（红色曲线）抵消。整个任务生命周期内，活跃的高斯总数被平稳控制在 32K 预设上限的下半区，彻底摆脱了随步数线性膨胀导致的显存溢出风险。

### 实验实测：基准测试与性能上限

为了保证评估的绝对公正，避免“模型参数量碾压”带来的伪增益，论文在骨干网络选择上极度严苛：GaussMemory 刻意采用了与当前顶级基线 OpenVLA、CogACT 及 MemoryVLA 完全相同的 Prismatic VLM（Llama-2-7B 搭配 DINOv2 与 SigLIP 视觉编码器），并统一连接扩散动作头。这就确保了所有性能的跃升，完全归功于 3D 高斯时空记忆与 UMA 架构本身。

在包含 40 组长短程操作任务的权威基准 **LIBERO** 上，GaussMemory 展现出了全方位的压制力。特别是在极度依赖历史状态检索的 Goal 和 Long-10 子套件中，GaussMemory 取得了对现有 SOTA 方案的决定性超越。



细看图 6 中关于 LIBERO Long-10 的单项拆解，GaussMemory 在全部 10 项复杂长程操作中拿下了 9 项第一。基线方案 MemoryVLA 仅仅在高度偏向纯语义理解的 T7 任务中微弱领先，而在所有涉及高精度空间定位、多阶段避障与遮挡重检索的物理交互任务中，GaussMemory 均拉开了显著的差距。这无可辩驳地证明了 3D 显式高斯表征相比于纯 2D 隐式图像历史的降维打击优势。

进一步在 2025 年最新公布的高难度真实感物理基准 **VLABench** 上，GaussMemory 面对当前工业界最强大的原生动作模型同样表现硬核：

*   在评估基本操作技能获取的 **Track 1** 中，GaussMemory 成功率超越强劲基线 $\pi_0\text{-FAST}$ 达 **+5.2%**；

*   在要求多阶段规划与极端遮挡推理的长程复合任务 **Track 6** 中，GaussMemory 的成功率提升更是达到了 **+6.0%**，进度得分（Progress Score）同步拉开明显差距。



在消融实验部分，研究人员对高斯记忆库的容量上限进行了切除测试（见图 8）。当高斯上限从 4K 提升至 32K 时，机械臂在长程任务中的成功率呈现出几乎陡峭的线性爬升，说明更加丰富的空间基元能够提供更精细的几何操作指导；而当容量跨越 32K 继续增加至 64K 时，性能收益开始收敛放缓。这表明 32K 的基元规模配合 UMA 的动态剪枝机制，已经足以在计算开销与空间物理保真度之间取得最优平衡。

### 总结与未来启示

长期以来，具身智能领域在处理“时间与历史”这一维度时，始终在两种极端之间摇摆：要么寄希望于大语言模型万能的上下文窗口，把连续几百步的操作帧无脑压入多模态序列，导致推理延迟骤增且丢失 3D 几何结构；要么退回到经典 SLAM 的老路，构建极其僵硬的几何点云，却丧失了与高层语义意图沟通的能力。

GaussMemory 提供了一条极具启发性的第三条道路。它证明了：

1.  **3D 几何基元（3DGS）完全可以作为 VLA 大模型的外置时空海马体**，它不仅能保留精确的空间尺度，还能无缝承载物体级语义与时间演化属性；

2.  **读与写的界限可以被彻底打破**。通过统一记忆注意力（UMA），模型读取空间信息的过程，同时就是生成写入门控与空间匹配的过程。动作任务的成功与否，成为了引导记忆动态取舍的唯一指挥棒。

从被动地“看见什么就记下什么”，到主动地“为了完成任务而决定记住谁、更新谁、遗忘谁”，GaussMemory 迈出的这一步，为未来构建能够在非结构化真实物理世界中连续自主作业几小时乃至数天的具身通用智能体，提供了极具说服力的架构演进范式。
