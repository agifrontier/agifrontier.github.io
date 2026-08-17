---
layout: default
title: "谷歌打通10+款3D游戏：揭秘通用键鼠AI智能体SIMA"
description: "当前的人工智能可以在瞬间写出复杂的底层代码，或者在围棋棋盘上碾压人类世界冠军，却在被要求控制虚拟角色“走到飞船旁边”时显得笨拙无比。这正是莫拉维克悖论在虚拟世界中的真实写照：对AI来说，处理高度抽象的语言和逻辑相对容易，但要在复杂的三维环境中进行基础的感知与行动，却面临着巨大的挑战。"
arxiv_id: "2404.10179"
topics:
  - "多模态"
tags:
  - "3D Simulated Environments"
  - "Embodied AI"
  - "Free-form Instruction Following"
  - "Instructable Agent"
  - "Keyboard-and-Mouse Interface"
  - "Language Grounding"
related_tutorials:
  - "qwen2-vl-enhancing-vision-language-models-perception-of-the-world-at-any-resolut"
  - "process-supervised-reinforcement-learning-for-interactive-multimodal-tool-use-ag"
  - "toward-general-purpose-robots-via-foundation-models-a-survey-and-meta-analysis"
  - "large-language-model-brained-gui-agents-a-survey"
---

<p class="paper-original-title" lang="en" style="font-size:1rem; line-height:1.5; color:var(--global-text-color-light, #6c757d); margin:-0.5rem 0 1.5rem;">Scaling Instructable Agents Across Many Simulated Worlds</p>

当前的人工智能可以在瞬间写出复杂的底层代码，或者在围棋棋盘上碾压人类世界冠军，却在被要求控制虚拟角色“走到飞船旁边”时显得笨拙无比。这正是莫拉维克悖论在虚拟世界中的真实写照：对AI来说，处理高度抽象的语言和逻辑相对容易，但要在复杂的三维环境中进行基础的感知与行动，却面临着巨大的挑战。

> **ArXiv URL**：http://arxiv.org/abs/2404.10179v3

为了跨越语言符号与物理（或虚拟）动作之间的鸿沟，Google DeepMind联合英属哥伦比亚大学推出了一个雄心勃勃的项目——**可扩展指令多世界智能体**（**Scalable, Instructable, Multiworld Agent, SIMA**）。该研究的核心目标非常明确：开发一个能够在其所处的三维模拟环境中，听懂任意自然语言指令，并完成人类所能完成的任何操作的通用智能体。

<img src="/images/2404.10179v3/x1.jpg" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">

### 通用接口范式

过去，许多强化学习研究（如AlphaStar或OpenAI Five）依赖于游戏底层的API接口，智能体能够直接读取精准的坐标或状态数据。然而，这种依赖特定环境接口的路径，注定无法通向真正的**通用人工智能**（**Artificial General Intelligence, AGI**）。

SIMA采取了一种极其克制且彻底“拟人化”的架构设计。该系统摒弃了所有特权信息的输入，智能体与环境交互的接口被精简到了极致：
输入端仅仅是屏幕的视觉图像（Image）和用户的自然语言指令（Language）；输出端则是标准的键盘和鼠标操作（Keyboard-and-mouse actions）。

<img src="/images/2404.10179v3/x4.jpg" alt="Refer to caption" style="width:90%; max-width:700px; margin:auto; display:block;">

这种设计带来了一个显而易见的优势：极强的可扩展性。因为所有的3D游戏和模拟器，最终都是通过向人类展示画面并接收键鼠操作来运行的。SIMA的这套通用接口，使其无需针对新游戏重新设计控制空间，就具备了零样本迁移到全新环境的潜力。

### 核心对齐机制

在没有任何底层状态奖励信号的情况下，如何让智能体学会听从指令？该研究采用了大规模**行为克隆**（**Behavioral Cloning**）的方法，即通过监督学习，让模型直接拟合人类专家在游戏中的“观察-动作”映射。

但是，仅仅模仿人类操作是不够的。人类玩家在玩游戏时，往往有很强的“肌肉记忆”或“默认习惯”（例如无聊时乱跳、随意环顾四周），这会导致智能体在执行特定语言指令时容易分心。为了解决这个问题，我们可以引入一个比喻：假设SIMA的核心网络是一个“通用数字翻译器”，它负责将输入的画面和语言意图翻译成手指的敲击动作。但这个翻译器常常会陷入“自动驾驶”状态，凭直觉瞎逛，而忽略了具体的指令。

为了打破这种“自动驾驶”的惯性，研究团队巧妙地引入了**无分类器引导**（**Classifier-Free Guidance, CFG**）技术。该技术最初在扩散模型中用于增强文本对图像生成的控制力，本文将其创新性地应用于动作策略的计算中。其核心公式如下：




{% raw %}$$ \pi\_{CFG}=\pi\left(\text{image},\text{language}\right)+\lambda\left(\pi\left(\text{image},\text{language}\right)-\pi\left(\text{image},\cdot\right)\right) $${% endraw %}



在这个公式中，$\pi$代表智能体的动作策略网络。$\pi\left(\text{image},\text{language}\right)$是翻译器在听到指令时的动作倾向；而$\pi\left(\text{image},\cdot\right)$则是翻译器在没有指令时的“自动驾驶”动作倾向。

CFG在这里就像是一个“注意力放大器”。它首先计算出“受指令驱动的动作”与“无脑本能动作”之间的差值，然后用超参数$\lambda$将这个差值放大，并叠加回原有策略中。通过这种方式，网络被强行推离了无意义的背景习惯，被死死地锚定在语言指令所要求的确切任务上。

### 复杂数据工程

要支撑起如此庞大的通用模型，数据质量是成败的关键。SIMA的训练数据覆盖了超过10个视觉风格和物理机制截然不同的3D环境，包括《无人深空》、《英灵神殿》等商业大作，以及研究专用的虚拟实验室环境。

<img src="/images/2404.10179v3/x2.jpg" alt="Refer to caption" style="width:90%; max-width:700px; margin:auto; display:block;">

收集这类数据并非易事。研究团队采用了双人“设定-求解”（setter-solver）的数据收集模式：一名玩家负责在特定场景下向另一名玩家下达自然语言指令，后者则负责完成任务，整个过程的屏幕录像和键鼠操作被精确记录。此外，为了防止智能体学到捷径，数据预处理阶段还引入了严格的过滤和加权机制，确保模型面对的不是冗长无聊的跑图，而是高密度的有效交互经验。

语言并非仅仅是任务的标签，它为智能体提供了理解世界的抽象骨架。正如论文所强调的，语言能够促进底层感知的高效学习与泛化，而丰富的环境具身交互反过来又使得系统对语言本身的理解更加系统化。

### 多维评估体系

在缺乏底层代码支持的商业游戏中，如何客观地评价一个AI是否完成了“挖一块碳”或“走到工作台”的指令？

这是本文面临的最棘手的工程挑战之一。由于商业游戏不会为了AI研究专门吐出“任务完成”的日志，研究团队构建了一套多维度的评估体系：

1.  **光学字符识别**（**Optical Character Recognition, OCR**）：对于像《无人深空》这样拥有丰富界面提示的游戏，系统会通过OCR实时读取屏幕上的文本。如果系统检测到“获得碳元素”的文字弹窗，即可自动判定相关采集任务成功。
2.  **人类专家评审**（**Human Evaluation**）：对于无法用文字量化的开放性指令，研究团队聘请了游戏时长超过16小时的人类专家，对智能体的录像进行严格的交叉盲审。即便智能体最终完成了任务，只要在中途进行了无关操作，也会被无情地判定为失败。
3.  **基准环境测试**（**Ground-truth**）：在受控的研究环境中，则直接利用引擎底层的真实状态来进行高精度的能力探针测试，例如打断测试（下达指令后中途更改，看智能体是否能迅速切换目标）。


### 局限与工程启示

尽管SIMA展现出了跨越多种三维环境执行自然语言指令的惊人潜力，但在实际工程落地中依然存在显著的局限性。

首先是**异步环境下的延迟问题**。与传统回合制强化学习环境不同，商业游戏是实时运行的。视觉画面的渲染、网络传输以及大规模神经网络的推理，都会带来不可忽视的时间差（Latency）。本文的工程启示在于，必须在行为克隆阶段就让网络预测“未来”的动作以抵消延迟，并在评估时利用TPU加速器配合精心设计的缓冲机制，才勉强维持住实时控制的连贯性。

其次是**视觉与动作的精细度瓶颈**。虽然智能体能完成导航、简单资源采集等任务，但在需要极高操作精度的复杂工具使用或快速战斗场景中，单纯依靠像素到键鼠的映射依然显得力不从心。这表明，目前的视觉-语言-动作模型在处理高频动态微操时，其特征提取和时序对齐机制仍有待突破。

总体而言，SIMA提供了一个极具启发性的通用具身智能范式。通过在虚拟游戏世界中以最低假设条件（仅依靠屏幕和键鼠）训练AI，这不仅为下一代NPC的进化指明了方向，也为未来真实世界中的通用机器人控制，储备了宝贵的跨环境泛化经验。
