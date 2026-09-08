---
layout: default
title: "World Tokens：只在训练时用世界模型，把具身操作成功率提升至98.2%"
description: "中关村学院等机构提出的 World Tokens 给出了一条极具启发性的破解思路： 将世界模型完全限制在训练阶段，推理部署时彻底剥离 。该方案既保留了世界模型带来的物理时空演进监督，又让在线推理维持在毫秒级的纯 VLA 延迟水平。"
arxiv_id: "2608.09730"
paper_published: "2026-08-10"
published_at: "2026-09-08T13:15:08.162312+08:00"
topics:
  - "具身智能"
  - "模型训练"
tags:
  - "VLA models"
  - "WAMs"
  - "World Adapter"
  - "World Tokens"
  - "closed-loop control"
  - "future-video denoising"
related_tutorials:
  - "a-comprehensive-survey-on-world-models-for-embodied-ai"
  - "atlasvla-persistent-world-ego-state-modeling-for-vision-language-action-models"
  - "scaling-automatic-research-agents-via-world-models"
  - "g05-one-autoregressive-stream-for-robot-reasoning-and-action"
---

<p class="paper-original-title" lang="en">World Tokens: Enhancing Embodied Policies with Training-Time World Modeling</p>

<img src="/images/2608.09730v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在具身智能的控制架构中，视觉-语言-动作（VLA）模型与世界动作模型（WAM）长期处于一种效率与物理理解的拉锯战。传统的 VLA 凭借视觉语言模型（VLM）与动作解码器的轻量堆叠，能够实现高频闭环控制，但仅靠静态图文预训练的模型缺乏对接触、遮挡和物体位移等物理动态的直觉理解；而融合了视频生成能力的 WAM 虽然具备推演未来场景的能力，却必须在推理闭环中运行庞大的扩散网络进行视频去噪或多模态自回归，导致单步控制延迟飙升到数百毫秒，难以满足真实机器人的敏捷控制需求。

> ArXiv URL：https://arxiv.org/abs/2608.09730v1

中关村学院等机构提出的 **World Tokens** 给出了一条极具启发性的破解思路：**将世界模型完全限制在训练阶段，推理部署时彻底剥离**。该方案既保留了世界模型带来的物理时空演进监督，又让在线推理维持在毫秒级的纯 VLA 延迟水平。

<img src="/images/2608.09730v1/Figure1_editable_v2.webp" alt="三种具身控制架构推理与训练模式对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

实验表明，基于 2B 参数基础底座且没有经过大规模具身动作预训练的 World Tokens，在基准测试 LIBERO 上取得了 98.2% 的高成功率，并在 SIMPLER 的 WidowX 与 Google Robot 双平台上均刷新了此前公开的最优均分；在真实的 Galaxea R1 Pro 机械臂实验中，该模型将任务成功率从基准的 59.4% 提升至 76.0%，而动作块推理延迟稳定在 61.85 毫秒。

### 为什么具身策略需要“世界模型”，却又用不起它？

常规 VLA 模型擅长理解指令语义并识别桌面上的目标物体，但当机械臂执行抓取、放置或开合容器时，动作的有效性往往不取决于当前帧的静态外观，而取决于场景在物理交互下的未来演变趋势。如果只依赖少量的人类演示轨迹进行行为克隆（Behavior Cloning），策略很难泛化到微小扰动、遮挡或接触突变。

视频世界模型通过预训练掌握了丰富的时空动态规律，能预测物理实体如何受力运动与变形。以 Motus、Cosmos Policy 等为代表的 WAM 试图将视频去噪骨干网络直接融入控制闭环，但代价相当沉重：在执行每一个控制步骤时，机器人不仅要生成连续动作，还要在线进行多步扩散去噪来生成未来视频片段，或者维持一个数十亿参数的视频 DiT（Diffusion Transformer）作为主干特征提取器。这使得动作推理延迟往往高达 130 毫秒至 600 毫秒以上，在面对真实物理环境的不确定性时极易导致控制震荡。

也有部分工作尝试采用轻量级的自监督目标（如未来隐变量对齐或轻量运动预测），试图在推断期规避视频骨干网络的开销。然而这些方案通常将预测目标作为旁路监督，策略网络仍然直接读取底层 VLM 输出的原始序列。这种松散的并联机制使得动作专家可以直接“绕开”被动态目标塑形过的特征，导致动作学习与世界演进预测在隐空间中相互解耦，难以发挥世界模型的表征增益。

### 核心机制：用 World Adapter 打造物理时空瓶颈

针对上述瓶颈，World Tokens 提出了一种严密的“单向收敛”架构设计。其核心不是简单地加一个辅助损失，而是构建了一个名为 World Adapter 的中间模块，将跨模态语义、物理世界动力学以及动作生成三者强制绑定在一个固定的隐空间表征内。

<img src="/images/2608.09730v1/work-version.webp" alt="World Tokens 训练与部署架构" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

整体系统在训练与部署时采用了完全不同的计算图拓扑。在训练期间，架构由三大部分组成：

1. **输入与语义编码**：多视角 RGB 图像 $o_{t}$ 与文本指令 $\ell$ 输入预训练的 VLM（例如 Qwen3-VL-2B-Instruct），输出变长的图文表征序列 $h_{t}$。

2. **瓶颈聚合与独占路由**：World Adapter 借鉴 Perceiver Resampler 机制，维护 $K = 256$ 个可学习的查询向量（Learned Queries）。通过多层交叉注意力、自注意力和前馈网络，将变长的 $h_{t}$ 压缩投影为固定尺寸的张量 $q_{t} \in \mathbb{R}^{K \times d}$，这 $256$ 个向量即为“World Tokens”。

3. **双重条件监督**：这组 World Tokens 同时输入两条互不干扰的下游分支。第一条分支是基于 Flow Matching 的 DiT 动作专家，它**仅仅依赖** $q_{t}$ 来回归多步动作块 $a_{t}^{H}$；第二条分支则是联合微调的视频扩散模型（采用 Cosmos Predict2.5-2B），$q_{t}$ 经过一个轻量线性层映射为 $q_{t}^{\mathrm{wm}}$，充当视频去噪过程的交叉注意力上下文。

这套结构最精妙的地方在于**独占路由（Exclusive Routing）**。动作专家没有向 VLM 原始特征 $h_{t}$ 的直连快捷通道，机器人决策所需的一切几何定位、指令解析和操作语境，都必须强行穿过这 256 个 World Tokens。当视频分支基于真实演示中的未来视频剪辑 $v_{t}$ 进行去噪反向传播时，视频扩散损失产生的梯度会直接回传并雕刻 $q_{t}$ 的特征空间。此时，负责生成动作的策略网络无法投机取巧，它所消费的每一个 Token，都已经被迫注入了物理演化规律。

### Canny 边缘锚点：防止视频分支“偷懒”

在构建视频世界模型的去噪条件时，作者发现了一个极易被忽视的关键问题：视频扩散模型如何使用第一帧参考图像？

如果完全不给世界模型任何初始帧参考，World Tokens 就必须同时承担从零构建全局静态几何与材质纹理的负担，任务难度过大；但如果直接向视频扩散网络输入完整的第一帧 RGB 潜变量作为条件，强大的视频去噪器往往会直接依赖第一帧的外观和像素持久性来修补未来画面，从而弱化对交叉注意力中 $q_{t}^{\mathrm{wm}}$ 的依赖程度。

为了解决这种表征寄生问题，研究团队设计了**外观抑制的结构锚点（Structural Anchor）**。具体而言，系统使用经典 Canny 算法提取主视角第一帧的边缘图 $\mathrm{Canny}(o_{t}^{1})$，再通过冻结的 VAE 编码器得到隐变量 $c_{t}$：




{% raw %}$$c_{t} = E\big(\mathrm{Canny}(o_{t}^{1})\big)$${% endraw %}



边缘轮廓保留了场景的刚性几何边界和物体布局，但剔除了绝大部分颜色、材质和光影信息。如此一来，世界模型若想在未来帧去噪中准确复原物体材质与运动细节，就必须深入依赖 World Tokens 中传递的语义与动力学信息。这种互补性迫使 World Tokens 真正学会表征“物体将在何处、以何种形式发生状态改变”。

训练阶段的目标函数为动作流匹配与视频扩散去噪损失的加权和：




{% raw %}$$\mathcal{L} = \mathcal{L}_{\mathrm{act}} + \lambda_{w}\mathcal{L}_{\mathrm{vid}}$${% endraw %}



其中 $\lambda_{w} = 0.5$。动作流匹配在 8 步的动作块上进行，视频分支则针对主视角未来 8 帧的连续片段进行去噪。进入推理部署阶段，整条视频分支（包括 VAE 编码器、投影层和视频 DiT）被全数剔除，机械臂只需在每个决策点计算 VLM、World Adapter 和动作专家，仅用 4 步欧拉积分即可输出平滑的轨迹块。

### 模拟与真实世界基准评估

验证这种“训练期监督、推理期剥离”范式的首要指标，是看它是否在砍掉计算负担的同时保留了顶尖控制性能。作者在 LIBERO 仿真基准、SIMPLER 跨环境泛化测试以及真实物理机器人上展开了全面评测。

#### 1. LIBERO 仿真评测：小参数模型反超大模型

在汇总了 40 个任务、总计 2,000 次评测的 LIBERO 基准中，World Tokens 展现出极高的一致性。下表整理了它与主流方案在成功率与动作块延迟上的关键对比：


| 模型分类 | 具体方法 | 模型主干规模 | 具身动作预训练 | 4套任务平均成功率 (%) | 最难Long套件成功率 (%) | 动作块推理延迟 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **经典 VLA** | $\pi_{0.5}$ (无预训练) | ~3B | 否 | 96.9 | 93.8 | 56.32 |
| | $\pi_{0.5}$ (完整预训练) | ~3B | 是 | 97.5 | 95.8 | 56.32 |
| | StarVLA | 4B | 否 | 97.8 | 95.7 | - |
| | World2Act | 3B | 是 | 98.1 | 96.0 | - |
| **在线 WAM** | DiT4DiT | 视频DiT | - | 98.8 | 97.6 | 136.00 |
| | Cosmos Policy | 2B+视频DiT | - | 98.5 | 96.6 | 610.00 |
| **本方法** | **World Tokens** | **2B** | **否** | **98.2** | **97.0** | **61.85** |

这些数据指出了两个极为关键的事实：

首先，在没有经历过海量跨机体具身数据预训练（Embodied Pretraining）的前提下，仅 2B 参数的 World Tokens 平均成功率达到了 98.2%，超越了依赖大规模具身预训练的 3B 方案 $\pi_{0.5}$（97.5%）以及 4B 的 StarVLA（97.8%）。

其次，相比那些在在线推理中必须常驻视频扩散模型的 WAM 方案（如 Cosmos Policy 延迟高达 610 毫秒），World Tokens 在同等硬件（RTX 5090 D）上测得的耗时仅为 61.85 毫秒，与纯 VLA 处于完全相同的响应量级，但长程任务（LIBERO-Long）的成功率依然维持在 97.0% 的极高水准。

#### 2. SIMPLER 跨域虚实迁移评估

SIMPLER 基准专门检验在真实机数据集（BridgeV2 和 Fractal）上训练出的策略，在数字孪生模拟器中的零样本跨域控制能力。在该评测下，World Tokens 在 WidowX 测试集中取得了 71.5% 的平均成功率，并在 Google Robot 体系中拿到了 82.1% 的最优均分。在“抓取可乐罐”（Pick Coke Can）、“开关抽屉”（Open/Close Drawer）以及最为复杂的“打开顶层抽屉并放入苹果”（Open Top Drawer + Place Apple）任务中，World Tokens 均显著压制了包括 $\pi_0$、Octo 在内的对照方法，验证了时空动态特征在未知干扰环境下的抗漂移特性。

### 消融实验：为什么不能开直连通道？

为了彻底证明物理提升来源于精细架构而非参数量堆叠，研究团队在 LIBERO-Long 长程任务上进行了一系列拆解实验。长程任务需要机械臂完成多次连续接触（例如从多个容器中依次拾取并转移罐头），误差极易发生级联放大，对物理状态的追踪要求最为严苛。

消融结果揭示了极具启发的机制依赖：

- **完全去掉世界模型监督（w/o wm）**：将视频分支完全剪掉，仅保留 World Adapter 结构，Long 成功率从 97.0% 下跌至 95.0%。

- **恢复 VLM 直连绕流（w/ VLM bypass）**：如果保留视频世界模型的训练，但允许动作专家同时读取 VLM 的全局特征 $h_{t}$，Long 成功率不仅没有提升，反而剧烈暴跌至 94.1%——这个表现甚至**低于完全不加世界模型的对照组**。这强力佐证了独占路由的必要性：一旦给策略提供捷径，策略解码器就会绕开正在被动力学梯度调整的表征，使预测任务与控制任务产生隐空间内耗。

- **将 Query 架构换成普通 FFN（FFN adapter）**：若将基于可学习查询的交叉注意力机制替换为参数量对齐的前馈全连接层，成功率骤降至 93.4%。这说明从通用视觉语言表征向“物理-动作”双重表征对齐，需要一个独立、富含容量的交互瓶颈，普通逐 Token 映射不足以承载这种多任务语义重塑。

- **改用完整的 RGB 初始帧锚点**：将 Canny 边缘恢复为标准的 RGB 图片输入，成功率掉落至 91.5%。这一显著滑坡直接证实了先前的理论猜想：RGB 纹理泄露诱使世界模型发生懒惰学习，彻底削弱了对 World Tokens 物理表征的梯度打磨。

<img src="/images/2608.09730v1/attn.webp" alt="World Tokens 与无世界模型监督的交叉注意力热力图对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

可视化 World Adapter 在执行复杂长程任务时的交叉注意力权重，可以直观看到表征层面的蜕变。在图 3 的动作序列中，机械臂需要先后将字母汤罐头和番茄酱放进篮子：

在具备世界模型梯度塑造的完整模型中（Ours），空间注意力呈现出高度清晰的**任务阶段依赖性**。在抓取前（$t=1, 4$），注意力精准聚焦于目标罐体的受力边缘；一旦物体被成功抓取抬升（$t=2, 3, 5$），注意力焦点瞬时切换至目标放置篮，展现出极强的动态因果敏感度。

反观缺失视频监督的基准（w/o wm），其注意力自始至终高度发散、在桌面背景与多物体间游移。对主视角注意力图计算空间熵（Entropy，越低代表越聚焦），World Tokens 的平均熵仅为 3.29 bits，而无视频监督基准高达 5.22 bits（完全均匀分布为 6.0 bits）。视频去噪的监督信号，确实把物理动态的演化逻辑刻进了这 256 个 Token 当中。

### 真实机器人实验：Galaxea R1 Pro 落地

在物理实体验证中，研究人员将策略部署在配备单臂的 Galaxea R1 Pro 移动操作机器人上，通过头部安装的鱼眼相机截取工作区画面，直接以 8 维末端关节与夹爪位置进行闭环操作。任务要求机器人在混乱放置的桌面场景中，根据随机指令将香蕉、芒果、草莓或柠檬精准放入收纳篮中。

<img src="/images/2608.09730v1/r1pro.webp" alt="Galaxea R1 Pro 机械臂放置水果定性对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在此项物理试验中，基准模型 Qwen-GR00T（采用同等 Qwen3-VL 骨干与 GR00T 动作专家，但不含 World Tokens 机制）的 96 次测试总成功率为 59.4%。相比之下，World Tokens 将真实操作成功率大幅提升到了 76.0%。

从典型的失败案例分析中可以看出，Qwen-GR00T 频繁出现两类致命物理交互错误：一是在目标被部分遮挡或摆放杂乱时，无法在三维空间中准确形成抓取航向角，导致直接撞偏目标；二是即便指尖接触到了物体，因未能感知物体受压后的位移倾向而导致夹持虚浮、在抬臂中途掉落。World Tokens 则在第一帧就准确定位到了指令物体的交互重心，在抓取闭环中能够自适应根据物体的动态趋势完成稳固抱合与轨迹平滑过渡。

### 总结与未来启示

World Tokens 证明了一件具身智能领域长期探讨的事：**物理世界模型对控制策略的价值，核心在于其对状态表征空间的动态重塑，而非在线推理时必须生成一张肉眼可见的高清未来图片**。

通过精心设计的 World Adapter 接口、严格的独占路由以及抑制外观泄露的 Canny 边缘锚点，该架构成功将一个复杂的视频生成大模型，在训练期“蒸馏”为一组凝聚了高维时空动态的离散 Token。在推理阶段，策略直接丢弃所有沉重的生成构件，以最纯粹的 VLA 结构维持着超 16 Hz 的控制频率。

这项工作的局限性同样鲜明：虽然在线推断十分轻快，但在训练期联合反传多模态 VLM 与高阶视频扩散模型，对显存和算力的开销依旧高昂；此外，手选 Canny 边缘作为结构锚点虽然有效，但仍带有一定的人工启发式色彩。如果未来能将这种几何轮廓约束演进为端到端的可学习隐式表征，并在更海量的跨机体异构机器人数据上铺开预训练，World Tokens 所代表的“轻量控制接口承载前置物理世界知识”范式，将成为下一代具身大模型走向敏捷工业与家庭落地的重要演进方向。
