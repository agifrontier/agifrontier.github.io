---
layout: default
title: "World Action Planner：让机械臂在世界模型中“脑补”试错，破解具身泛化难题"
description: "针对端到端策略这一根本缺陷，论文《World Action Planner: Generalizable Decision-Making with Action-Conditioned World Models》提出了全新的破局思路—— World Action Planner （简称 WAP ）。"
arxiv_id: "2607.27599"
paper_published: "2026-07-30"
published_at: "2026-09-07T13:15:08.344824+08:00"
topics:
  - "具身智能"
tags:
  - "具身智能"
  - "AI论文解读"
related_tutorials:
  - "wcm-a-world-critic-model-for-vision-language-action-reinforcement-learning"
  - "when-replanning-becomes-the-bottleneck-budgeted-replanning-for-embodied-agents"
  - "longhorizon-harness-advancing-long-horizon-agents-for-real-world-tasks"
  - "openforgerl-train-harness-native-agents-in-any-environment"
---

<p class="paper-original-title" lang="en">World Action Planner: Generalizable Decision-Making with Action-Conditioned World Models</p>

在具身智能与机器人控制领域，端到端（End-to-End）模仿学习近年来几乎占据了主导地位。无论是将视觉语言模型与动作输出强行绑定的视觉-语言-动作（VLA）模型，还是近期基于大规模视频生成微调的“世界-动作模型”（World-Action Models, WAM），学术界与工业界都倾注了极高的热情。然而，这类端到端策略在实际落地时普遍面临极其脆弱的泛化瓶颈：一旦环境中的物体位置发生偏移、任务目标被组合拉长，或者遇到未曾见过的全新摆放布局，策略便往往迅速崩溃。机械臂常常机械地抓向训练集中的旧坐标，或者在完成第一阶段动作后彻底陷入停滞。

> ArXiv URL：https://arxiv.org/abs/2607.27599

针对端到端策略这一根本缺陷，论文《World Action Planner: Generalizable Decision-Making with Action-Conditioned World Models》提出了全新的破局思路——**World Action Planner**（简称 **WAP**）。作者团队没有继续堆砌更大规模的专家演示轨迹进行行为克隆，而是重拾经典机器人规划的模块化精髓，将多模态大模型（VLM）的高阶推理能力与高精度的动作条件世界模型（Action-Conditioned World Model）相结合。

<img src="/images/2607.27599/fig6.webp" alt="World Action Planner 框架总览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在该框架下，机器人不再像条件反射般直接吐出控制指令，而是让 VLM 先提出初步动作，再由动作条件世界模型在“脑海”中推演预测未来视频。如果世界模型推演出的画面显示抓手会发生碰撞或偏离目标，VLM 就会基于想象的画面给出修正反馈；在精细操作阶段，系统更会通过局部搜索对多种候选动作进行推演排名。这套“先脑内演练、再物理执行”的模型化规划范式，在组合任务、新桌面布局和零样本场景中展现出极其强大的泛化能力，大幅超越了以 $\pi_{0.5}$ 和 cosmos-policy 为代表的端到端前沿模型。

### 为什么端到端模仿学习越学越脆弱？

要理解 World Action Planner 的价值，必须先正视当前具身大模型面临的核心死结。

现阶段基于端到端模仿学习的策略，本质上是在高维感知输入与低维电机动作之间拟合一条条件分布。然而，机器人收集高质量演示数据的成本极其昂贵，训练数据往往只能覆盖非常狭窄的动作流形。这种机制导致了两个致命缺陷：

其一是虚假关联与空间过拟合。当人类专家演示“抓取桌上的杯子”时，抓手通常都会移动到某一个特定的三维空间范围。端到端模型极易将视觉背景、光影与该局部绝对坐标强行绑定。当测试场景中杯子被移动到另一侧时，VLA 模型常常完全无视新的视觉输入，依旧盲目地伸向训练数据里的旧坐标“抓空气”。

其二是缺乏长程组合与多任务推理机制。在实际场景中，复杂任务往往由多个原子技能组合而成，例如“打开抽屉”然后“拿出里面的积木”。如果模型只在单一子任务的演示数据上训练过，当面对串联的复合任务时，执行完第一阶段动作的机械臂会进入未见过的中间状态分布，进而彻底迷失。

从控制理论与强化学习的角度来看，模仿学习在多任务场景下的性能界甚至存在理论上的劣势。作者在论文中给出了严格的理论证明：在任务数量 $\lvert \mathcal{C} \rvert$ 线性增长的情况下，即便在最基础的表格马尔可夫决策过程（MDP）中，多任务模仿学习的平均次优差距（Suboptimality Gap）至少达到 $\Omega\left(\frac{\lvert \mathcal{C} \rvert}{K}\right)$，即误差会随着任务数的增加而急剧线性累积。而在引入线性函数近似的设置下，单任务最优策略虽可由简单的线性函数表示，但在多任务背景下，最优策略对任务特征的依赖会变成高阶多项式甚至不可实现的极高复杂度。

相反，物理世界底层的动态转移规律（Dynamics）在不同任务间是高度共享通用的。世界模型只负责学习“机械臂怎么动、物体会怎么受力位移”，而具体任务的评价与奖励则可借助预训练的视觉语言模型来判断。论文证明，基于模型的规划算法在面对新任务时，次优差距能稳定维持在 $\tilde{\mathcal{O}}\left(\frac{1}{\sqrt{K}}\right)$ 水平，且不会受到任务数量激增的线性惩罚。这为“通过世界模型规划来摆脱模仿学习泛化魔咒”奠定了坚实的理论基石。

### 姿态骨架图：打造真正可控的高保真机器人世界模型

要让“脑内演练”行之有效，核心前提是拥有一个精准受控且具备跨构型泛化能力的世界模型。如果世界模型想象出的未来画面本身就充满幻觉，或者动作控制根本对不上，后续的所有规划都将沦为空中楼阁。

以往基于扩散模型的视频生成世界模型，在接入机械臂控制动作时，大多采用 AdaLN-Zero 调制或 Cross-Attention 交叉注意力机制，将低维的连续动作向量（如 7 维末端位姿增量）注入模型。但实验证明，这种低维向量在跨场景、跨机器人构型以及遇到训练分布外的激进动作时，极易退化失效，画面中的机械臂要么原地融化，要么完全不遵循控制信号。

<img src="/images/2607.27599/fig1.webp" alt="姿态骨架图条件化世界模型架构" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

World Action Planner 在此提出了一个兼具物理确定性与计算效率的设计：**姿态图像条件化（Pose-Image Conditioning）**。

具体而言，系统并不把低维动作向量直接丢给神经网络去胡乱拟合，而是利用机器人精确的前向运动学与动力学模型，预先计算出机械臂每个关节在未来时间步的三维物理坐标。随后，根据当前相机的外参和视角，将机械臂的骨架连线渲染为直观的骨架图像（Pose Skeleton Image）。这种表示将物理模拟与神经生成优雅地解耦开来：机械臂自身刚体结构的运动完全由物理法则精准约束，而神经网络只需要专注于基于这套骨架渲染真实纹理、预测被抓取物体的形变与位移。

在网络实现上，研究团队将多视角的真实相机画面（包括第三人称主视角和机械臂手腕视角）拼接成图像网格，同时将对应视角的骨架图通过视频模型的变分自编码器（VAE）提取 Token，并在序列维度上与视频 Token 直接拼接。训练采用 Flow-Matching 目标与 Diffusion-Forcing 策略，在保持姿态骨架 Token 无噪纯净的前提下，对历史帧注入随机噪声、对未来帧注入均匀噪声，驱动模型学习出对机械臂动作极其服从的多视角物理推演能力。

实验数据显示，仅在这一项改进上，基于骨架图条件化的世界模型就取得了显著突破：在单机械臂（LIBERO-90）与双手机械臂（DexMimicGen）的分布内预测中，图像质量（PSNR 与 LPIPS 指标）平均超越现有顶尖基线 11.4%；而在零样本迁移到新场景、新机械臂硬件（如从 Franka 迁移到 Sawyer、UR5e、IIWA）时，性能优势更进一步扩大到了 16.8%。甚至在混合了单臂、双臂并行夹爪乃至 24 自由度灵巧手的异构大杂烩训练中，该结构也展现出了极强的通用表征能力。

### World Action Planner 的规划回路：提议、反思与精调

在拥有了可靠的世界模型之后，决策系统究竟如何运转？World Action Planner 将规划流程拆解为三层渐进式的协同机制：

首先是 **动作提议（Agent Action Proposal）**。系统并没有要求多模态大模型直接在像素空间内预测精准的毫米级电机扭矩，而是让大模型负责它最擅长的事情——宏观语义理解与任务分解。面对当前的场景图像与自然语言任务指令，VLM 首先生成高级动作原语序列，例如移动（MOVE）、旋转（ROTATE）、抓取（GRASP）与释放（RELEASE）。针对移动类动作，VLM 在多视角图像中点选出目标位置的二维像素坐标，系统通过双目三角测量自动反投影为三维世界坐标，摆脱了对昂贵外置深度传感器的依赖。

其次是 **基于世界模型推演的全局反馈优化（Global Optimization Guided by Agent Feedback）**。直觉提出的动作序列极可能在物理执行中出现致命缺陷，例如直线位移可能直接撞倒障碍物，或者下降高度过猛导致抓手硬着陆。此时，系统调用动作条件世界模型，将提议的动作序列渲染成未来的多视角推演视频。VLM 作为“观察者”审视这一推演视频：如果发现抓手在运动轨迹上有碰撞隐患，VLM 会生成具体的高层次语义修正建议（如“轨迹过低，抓手需要提高 5 厘米以避开前方的收纳盒”）；如果发现放下物体的落点在目标篮子外侧，VLM 则给出平移修正。底层控制器接收修正建议后更新动作序列，再次交由世界模型验证，直至安全可靠。

<img src="/images/2607.27599/fig2.webp" alt="组合任务泛化推演过程" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

最后是 **局部搜索与候选重排序（Local Search with Agent Ranking）**。VLM 具有强大的语义推理能力，但天生缺乏毫米级的绝对度量直觉，很难单凭单次提示词就准确指定最完美的抓取微调位置。为了攻克诸如“捏住马克杯手柄”或“精密搭积木”这类极度依赖毫米级精度的微操，WAP 巧妙地将“生成任务”转化为“判别选择任务”。系统在目标邻域内通过网格采样生成若干微小的候选动作扰动，并利用世界模型分别推演这些候选动作。随后，VLM 观察这些不同的推演结果，通过排名挑选出物理交互最稳定、最符合任务预期的一组动作。在接近抓取点时，系统甚至还可以将成熟的局部扩散策略（Diffusion Policy）作为即插即用的工具调用，并在世界模型中模拟策略轨迹，确保交接顺畅。

这种把传统策略（Policy as Tools）降级为执行工具、由世界模型提供安全护栏与推演闭环的设计，从根本上打破了端到端系统对海量精细专家数据的重度依赖。

### 实验检验：复杂场景下的全面胜出

为了验证 World Action Planner 的综合表现，研究团队在具身基准 LIBERO 与 Robosuite 上部署了涵盖 12 个长程复合场景的严苛评测，并与当红的端到端具身大模型 $\pi_{0.5}$、世界-动作模型 cosmos-policy 以及未经世界模型优化的基线进行了全方位对比。

<img src="/images/2607.27599/fig4.webp" alt="新布局泛化下的对比表现" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在**长程组合任务泛化（Compositional Task Generalization）**实验中，机器人需要将训练集中学过的两个独立原子技能（如先关闭微波炉门、再将托盘移至特定餐盘）顺次串联。以 $\pi_{0.5}$ 为代表的传统 VLA 模型表现出了典型的后劲不足：由于无法处理两个子任务拼接时的中间盲区，端到端策略往往在第一步结束后便陷入停滞或机械震荡。而 WAP 凭借 VLM 的阶段分解与世界模型的动态试错，成功平滑处理了子任务之间的空间过渡，在复合长任务上的成功率相比基线展现出了断层式的优势。

在新摆放布局（New Layout Generalization）场景下，这种对比显得尤为悬殊。实验中，目标物体与干扰物的空间拓扑被完全打乱。传统 VLA 和 WAM 几乎无一例外地踩中了空间过拟合的陷阱——模型视线扫过了新位置的物体，却依然操控机械臂扑向空无一物的原始训练坐标，陷入徒劳的抓取死循环。

而在 WAP 框架下，VLM 准确识别了目标物体的新空间位置并生成了导航原语；当中间生成的粗糙动作可能导致撞翻干扰物时，世界模型在脑内推演中精准预警，引导 VLM 向上抬升抓手弧度避开障碍。一旦抓手移动到物体附近，局部扩散策略与局部搜索机制迅速接管，高精度完成了抓取闭环。

<img src="/images/2607.27599/fig3.webp" alt="零样本泛化测试中的世界模型介入" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

哪怕是在面对从未见过的全新物体或全新交互目标的零样本泛化（Zero-shot Generalization）设定下，WAP 同样展现出了极佳的自愈能力。由于不需要针对每个新任务从头微调端到端权重，机器人只需依托基础世界模型的物理动力学理解，通过一次次低成本的“脑内试错”，就能动态找出通往目标的可行路径。

### 具身智能的技术演进思考

回顾 World Action Planner 的整体架构，这项工作实际上给当前热火朝天的具身智能与机器人基础模型路线提供了一剂清醒剂。

在过去两年中，整个领域很大程度上被“大语言模型的 Scaling Law 能直接平移到机器人动作控制”的乐观情绪所裹挟。人们试图直接训练一个千亿参数的多模态黑盒，输入图像与文本，单步自回归直接吐出关节电机角度。然而，物理世界对误差的容忍度极低，毫米级的偏差就会导致物体滑落或硬件碰撞损坏，单纯靠行为克隆去盲目拟合连续控制量，其样本效率和分布外泛化能力从数学理论上就被判了极刑。

World Action Planner 的成功表明，**通往通用机器人的正确路径，或许不是让单一网络包揽一切，而是实现高阶推理与物理因果的彻底解耦**。

视觉语言大模型负责处理开放世界的常识、高层规划和目标审视，这是人类智能中偏向系统 2（System 2）慢思考的认知部分；动作条件世界模型负责拟合客观世界的物理规律，充当可微、可推演的低成本脑内沙盘；而底层的低级控制器与局部扩散策略，则退回到最擅长的高频闭环控制角色。这种“VLM 提议 -> 世界模型模拟推演 -> VLM 反思闭环 -> 局部执行”的协同架构，不仅极大地解放了对极其昂贵的真机物理专家演示数据的依赖，更为机器人在未见物理世界中的自主探索与安全规划指明了一条极具可行性的技术路线。
