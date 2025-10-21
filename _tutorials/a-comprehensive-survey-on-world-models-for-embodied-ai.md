---
layout: default
title: "A Comprehensive Survey on World Models for Embodied AI"
---

# A Comprehensive Survey on World Models for Embodied AI

- **ArXiv URL**: http://arxiv.org/abs/2510.16732v1

- **作者**: Yun Liu; Le Zhang; Xinqing Li; Xin He

- **发布机构**: Nankai University; Tianjin University of Technology; University of Electronic Science and Technology of China

---

# 面向具身智能的世界模型综合综述

## 核心概念与数学基础

### 核心概念
世界模型（World Models）作为环境动态的内部模拟器，其功能建立在三个核心支柱之上：
*   **想象 (Imagination)**：利用学习到的动态模型生成未来可能发生的场景，允许智能体在不与真实世界交互的情况下，通过“想象”来评估潜在行动的后果。
*   **动态学习 (Dynamics Learning)**：学习编码后的状态如何随时间演变，从而实现时间上一致的推演（rollout）。
*   **空间编码 (Spatial Encoding)**：以适当的保真度对场景几何进行编码，使用诸如隐式Token或神经场等格式，为控制提供上下文信息。

### 数学形式化
本文将智能体与环境的交互形式化为一个部分可观察马尔可夫决策过程（Partially Observable Markov Decision Process, POMDP）。在每个时间步 $$t$$，智能体接收观测 $$o_t$$ 并执行动作 $$a_t$$，而真实状态 $$s_t$$ 保持不可见。世界模型通过一步滤波后验（one-step filtering posterior）推断出一个学习到的隐状态 $$z_t$$，该后验假设前一时刻的隐状态 $$z_{t-1}$$ 已经总结了所有相关的历史信息。最后，使用 $$z_t$$ 来重构当前的观测 $$o_t$$。

模型的关键组件定义如下：


{% raw %}$$
\begin{array}{ll}
\text{Dynamics Prior:}&p_{\theta}(z_{t}\mid z_{t-1},a_{t-1})\\ 
\text{Filtered Posterior:}&q_{\phi}(z_{t}\mid z_{t-1},a_{t-1},o_{t})\\ 
\text{Reconstruction:}&p_{\theta}(o_{t}\mid z_{t})
\end{array}
$${% endraw %}



基于马尔可夫假设，观测和隐状态的联合分布可以分解为：


{% raw %}$$
p_{\theta}(o_{1:T},z_{0:T}\mid a_{0:T-1}) = p_{\theta}(z_{0})\prod_{t=1}^{T}p_{\theta}(z_{t}\mid z_{t-1},a_{t-1})p_{\theta}(o_{t}\mid z_{t})
$${% endraw %}



由于真实后验分布难以计算，本文引入一个时序分解的变分分布 $$q_φ$$ 来近似：


{% raw %}$$
q_{\phi}(z_{0:T}\mid o_{1:T},a_{0:T-1})=q_{\phi}(z_{0}\mid o_{1})\prod_{t=1}^{T}q_{\phi}(z_{t}\mid z_{t-1},a_{t-1},o_{t})
$${% endraw %}



模型的学习目标是最大化观测的对数似然，这通过优化其证据下界（Evidence Lower Bound, ELBO）来实现：


{% raw %}$$
\mathcal{L}(\theta, \phi)=\sum_{t=1}^{T}\mathbb{E}_{q_{\phi}(z_{t})}\!\big[\log p_{\theta}(o_{t}\mid z_{t})\big] -D_{\mathrm{KL}}\!\big(q_{\phi}(z_{0:T}\mid o_{1:T},a_{0:T-1})\,\ \mid \,p_{\theta}(z_{0:T}\mid a_{0:T-1})\big)
$${% endraw %}



这个目标函数分解为两部分：第一项是**重构目标**，鼓励模型忠实地预测观测；第二项是**KL散度正则化**，旨在使滤波后验分布 $$q_φ$$ 与动态先验分布 $$p_θ$$ 保持一致。现代世界模型普遍采用这种**“重构-正则化”**的训练范式。

## 世界模型的三轴分类体系

本文沿着三个核心维度对世界模型进行分类，为后续分析奠定基础。

![论文结构图](https://github.com/Li-Zn-H/AwesomeWorldModels/raw/main/figs/Framework.jpg)

**1. 功能决策耦合 (Decision Coupling)**：区分**决策耦合 (Decision-Coupled)** 和 **通用目的 (General-Purpose)** 模型。
    *   **决策耦合模型**是任务特定的，其学习的动态模型是为了优化某个特定的决策任务。
    *   **通用目的模型**是任务无关的模拟器，专注于广泛的预测能力，从而能泛化到各种下游应用。

**2. 时间推理 (Temporal Reasoning)**：描述了两种不同的预测范式。
    *   **序贯模拟与推断 (Sequential Simulation and Inference)** 以自回归的方式对动态进行建模，一步一步地展开未来状态。
    *   **全局差异预测 (Global Difference Prediction)** 直接并行地估计整个未来状态，效率更高，但可能牺牲时间上的一致性。

**3. 空间表征 (Spatial Representation)**：包含当前研究中用于建模空间状态的四种主要策略。
    *   **全局隐向量 (Global Latent Vector)**：将复杂的世界状态编码为紧凑的向量，适用于物理设备上的高效实时计算。
    *   **Token特征序列 (Token Feature Sequence)**：将世界状态建模为Token序列，专注于捕捉Token之间复杂的空间、时间及跨模态依赖关系。
    *   **空间隐式网格 (Spatial Latent Grid)**：通过利用鸟瞰图（Bird’s-Eye View, BEV）或体素网格等几何先验，将空间归纳偏置融入世界模型。
    *   **解构式渲染表征 (Decomposed Rendering Representation)**：将3D场景分解为一组可学习的图元（如3D高斯溅射或神经辐射场），并通过可微分渲染实现高保真度的新视角合成。

下表应用此分类体系对代表性工作进行了梳理。

#### 表 I：机器人领域代表性世界模型方法分类


| 方法 | 功能 | 时间 | 空间 | 核心技术 | 数据平台数 | 物理机器人 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Ha and Schmidhuber [9] | DC | SSI | GLV | MDN-RNN | 1 | |
| PlaNet [38] | DC | SSI | GLV | RSSM | 6 | |
| Dreamer [10] | DC | SSI | GLV | RSSM | 6 | |
| DreamerV2 [11] | DC | SSI | GLV | RSSM | 8 | |
| **DreamerV3** [12] | **DC** | **SSI** | **GLV** | **RSSM** | **22** | **✓** |
| GLAMOR [39] | DC | GD | GLV | Transformer | 2 | |
| Iso-Dream [40] | DC | SSI | GLV | RSSM | 6 | |
| **MWM** [41] | **DC** | **SSI** | **TFS** | **RSSM / MAE** | **6** | |
| Inner Monologue [42] | DC | SSI | TFS | LLM | 2 | |
| **DayDreamer** [43] | **DC** | **SSI** | **GLV** | **RSSM** | **18** | **✓** |
| IRIS [44] | DC | SSI | TFS | Transformer | 7 | |
| RoboAgent [45] | GP | SSI | TFS | VQ-VAE / Transformer | 14 | |
| Statler [46] | GP | SSI | TFS | LLM | 4 | |
| T-Dreamer [47] | DC | SSI | GLV | Transformer | 6 | |
| **DWL** [48] | **DC** | **SSI** | **GLV** | **RNN** | **2** | **✓** |
| **GAIA-1** [49] | **GP** | **SSI** | **TFS** | **Transformer** | **4** | |
| **V-JEPA 2** [14] | **GP** | **GD** | **SLG** | **ViT / M-JEPA / D-JEPA** | **2** | |
| Drive-WM [50] | GP | SSI | TFS | Transformer | 1 | |
| SIMA [51] | DC | SSI | TFS | Transformer | 9 | ✓ |
| **PreLAR** [52] | **DC** | **SSI** | **GLV** | **MAE-ViT** | **2** | **✓** |
| **ManiGaussian** [53] | **DC** | **GD** | **DRR** | **3DGS** | **1** | **✓** |
| **ECoT** [54] | **DC** | **SSI** | **TFS** | **Foundation Models / LLM** | **2** | **✓** |
| **Genie** [55] | **GP** | **SSI** | **TFS** | **ST-Transformer** | **23** | |
| **Sora** [13] | **GP** | **GD** | **TFS** | **Diffusion Transformer** | | |
| **Drive-Sora** [56] | **GP** | **SSI** | **TFS** | **DiT-Sora** | **1** | |
| GLAM [57] | DC | GD | GLV | Mamba | 3 | |
| **NavCoT** [58] | **GP** | **SSI** | **TFS** | **LLM** | **2** | |
| **MineWorld** [59] | **DC** | **GD** | **TFS** | **VQ-GAN / Transformer** | **1** | |
| DreMa [60] | DC | SSI | DRR | 3DGS | 3 | ✓ |
| V-JEPA [61] | GP | GD | GLV | ViT / I-JEPA | 1 | |
| **UniSim** [62] | **GP** | **SSI** | **SLG** | **Q-Transformer** | **16** | |
| **GAMM** [63] | **DC** | **SSI** | **DRR** | **SDF / D-NeRF** | **2** | **✓** |
| **WorldVLA** [64] | **DC** | **SSI** | **TFS** | **Foundation Models** | **✓** | |
| **NWM** [66] | **GP** | **SSI** | **TFS** | **cDiT** | **3** | |
| **STEVE-2** [68] | **DC** | **SSI** | **TFS** | **VQ-GAN / Transformer** | **1** | **✓** |
| **Dyn-O** [69] | **DC** | **SSI** | **TFS** | **Mamba** | **2** | |
| **DINO-WM** [70] | **DC** | **SSI** | **SLG** | **ViT-DINOv2** | **3** | **✓** |
| **LaVi-Bridge** [72] | **DC** | **SSI** | **TFS** | **LLM / LLaVA** | **1** | **✓** |
| **GAF** [74] | **DC** | **GD** | **DRR** | **3DGS** | **1** | **✓** |
| **WONDER** [76] | **DC** | **SSI** | **TFS** | **VQ-GAN / Transformer** | **4** | **✓** |
| **Control-Sora** [77] | **GP** | **SSI** | **TFS** | **DiT-Sora** | **2** | |
| **MineDreamer** [79] | **DC** | **SSI** | **TFS** | **LLM / Diffusion** | **1** | |
| **ManiGaussian++** [80] | **DC** | **GD** | **DRR** | **3DGS** | **3** | **✓** |

*   **缩写**: DC-决策耦合, GP-通用目的, SSI-序贯模拟与推断, GD-全局差异预测, GLV-全局隐向量, TFS-Token特征序列, SLG-空间隐式网格, DRR-解构式渲染表征。
*   **核心技术**: 代表性的骨干网络或核心技术方法。
*   **数据平台数**: 下划线表示新提出或聚合的数据集。
*   **物理机器人**: ✓ 表示在物理机器人上进行了验证。

#### 表 II：自动驾驶领域代表性世界模型方法分类


| 方法 | 功能 | 时间 | 空间 | 核心技术 | 数据平台数 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MILE** [81] | **DC** | **SSI** | **GLV** | **RSSM** | **2** |
| **GAIA-1** [49] | **GP** | **SSI** | **TFS** | **Transformer** | **4** |
| SEM2 [83] | DC | SSI | GLV | RSSM-SA | 2 |
| **UniAD** [84] | **DC** | **SSI** | **SLG** | **Transformer** | **1** |
| **Occ-AD** [85] | **GP** | **SSI** | **SLG** | **Transformer** | **3** |
| **VAD** [86] | **GP** | **SSI** | **TFS** | **DiT** | **2** |
| **DriveWorld** [87] | **GP** | **SSI** | **SLG** | **RSSM / ViT** | **1** |
| RoboAgent [45] | GP | SSI | TFS | VQ-VAE / Transformer | 14 |
| **DrivingGPT** [23] | **DC** | **SSI** | **TFS** | **LLaMA** | **1** |
| **DriveDreamer** [91] | **GP** | **SSI** | **SLG** | **GRU** | **1** |
| **GenAD** [92] | **GP** | **SSI** | **SLG** | **GRU** | **1** |
| **OccWorld** [93] | **GP** | **SSI** | **SLG** | **Transformer** | **1** |
| **AD-ADAPTER** [95] | **DC** | **SSI** | **TFS** | **LLM/VLM** | **1** |
| **DTT** [98] | **GP** | **SSI** | **DRR** | **Transformer** | **1** |
| **Panacea** [100] | **DC** | **SSI** | **SLG** | **ViT / Transformer** | **1** |
| **FSDrive** [101] | **DC** | **SSI** | **TFS** | **DiT / LLM** | **1** |
| **MuKEA** [103] | **DC** | **SSI** | **SLG** | **ViT / GNN** | **3** |
| **Think-and-Drive** [104] | **DC** | **SSI** | **TFS** | **Foundation Models**| **1** |
| **World-in-the-loop** [105] | **DC** | **SSI** | **SLG** | **ViT-BEV** | **1** |
| **WoTE** [107] | **DC** | **SSI** | **SLG** | **RSSM-BEV** | **1** |
| **MagicDrive** [108] | **GP** | **GD** | **SLG** | **cDiT** | **1** |
| **OccLLaMA** [18] | **GP** | **SSI** | **SLG** | **LLaMA** | **1** |
| **Drive-Sora** [56] | **GP** | **SSI** | **TFS** | **DiT-Sora** | **1** |

### 全局隐向量表征 (Global Latent Vector Representation)

早期的决策耦合世界模型将序贯推理与全局隐状态相结合，主要使用循环神经网络（Recurrent Neural Networks, RNNs）来实现高效的实时和长时程预测。

*   **开创性工作**：Ha和Schmidhuber [9] 首次提出将观测编码到隐空间，并用RNN建模动态以优化策略。PlaNet [38] 在此基础上引入了循环状态空间模型（Recurrent State-Space Model, RSSM），它融合了确定性记忆与随机性组件，实现了稳健的长时程想象。后续的Dreamer、DreamerV2和DreamerV3 [10, 11, 12] 进一步发展了这一框架。

*   **RSSM的演进**：研究者通过修改或去除解码器来更好地捕捉动态，例如Dreaming [110] 使用对比学习缓解状态漂移，DreamerPro [111] 用原型替换解码器以抑制视觉干扰。为了增强鲁棒性，HRSSM [25] 设计了双分支架构对齐隐式观测。在表征迁移性方面，PreLAR [52] 学习隐式动作抽象，连接视频预训练表征与控制微调。

*   **新兴架构**：为捕捉更长期的依赖关系，TransDreamer [28] 引入了Transformer状态空间模型（TSSM），替代了Dreamer中的循环核心。近来，状态空间模型（State Space Models, SSMs），如Mamba，因其线性时间复杂度和长时程建模能力而受到关注。例如，GLAM [57]利用基于Mamba的并行框架提升了保真度和效率。

*   **逆向动态模型 (IDM)**：IDM也是世界模型构建的重要范式，它推断从初始状态到目标状态所需的动作。GLAMOR [39] 训练了一个物体条件的IDM来预测到达指定目标所需的动作。Iso-Dream [40] 利用IDM将世界模型分解为可控和不可控部分，指导策略学习。

### Token特征序列表征 (Token Feature Sequence Representation)

该范式专注于对离散化的Token之间的依赖关系进行建模，支持因果推理、多模态融合及复用大语言模型（Large Language Model, LLM）的能力。

*   **与RSSM结合**：MWM [41] 通过掩码自编码器将视觉Token与基于RSSM的动态解耦，提升了性能和数据效率。TWM [29] 使用Transformer在训练时对齐多模态Token与历史状态。
*   **LLM的集成**：一些方法将LLM与RSSM结合，将长期任务分解为子任务。例如，EvoAgent [131] 使用LLM指导低级动作。NavCoT [58] 将导航任务分解为想象、过滤和预测，实现了高效的域内训练。MineDreamer [79] 提出“想象链”（Chain-of-Imagination, CoI），由多模态LLM想象未来观测来引导扩散模型和动作。
*   **自动驾驶应用**：在自动驾驶中，Token序列被用来建模跨模态交互和时空结构。DrivingGPT [23] 将视觉和动作Token交错排列，把世界建模和轨迹规划统一为下一Token预测问题。
*   **通用智能体**：Token化表示统一了视觉、语言和动作（VLA）等多模态输入，使得像WorldVLA [67] 这样的通用智能体具备了跨域适应性。
*   **与规划的结合**：一些方法采用以物体为中心的方式表示场景。CarFormer [142] 和 Dyn-O [69] 等模型将场景表示为槽（slot）的集合，并自回归地建模槽之间的关系。
*   **基于扩散的方法**：为实现稳定生成和长时程规划，Epona [148] 通过轨迹和视觉扩散Transformer（DiTs）实现长时程多模态生成。SceneDiffuser++ [150] 将其扩展到城市规模的交通模拟。

### 空间隐式网格表征 (Spatial Latent Grid Representation)
该范式通过在与几何对齐的网格上编码特征或引入显式空间先验，保留了局部性，支持高效的卷积或注意力更新。

*   **自动驾驶应用**：许多研究将基于RNN的动态与空间网格（特别是BEV）相结合以指导规划。例如，DriveDreamer [91] 和 GenAD [92] 在网格或实例中心的Token上采用基于GRU的动态模型。DriveWorld [87] 则在BEV Token上实例化RSSM动态。
*   **3D占据预测**：另一主流方向是自回归地预测未来的3D占据（Occupancy）表征，以支持运动规划。OccWorld [93] 将场景离散化为占据Token进行序列预测。Drive-OccWorld [157] 直接预测体素特征。OccLLaMA [18] 将占据、动作和文本统一到单个Token词汇表中，并使用LLaMA进行预测、规划和问答。
*   **机器人应用**：RoboOccWorld [164] 针对室内机器人，通过预测精细的3D占据来支持探索和决策。EnerVerse [34] 应用分块自回归视频扩散生成4D隐式动态。DINO-WM [70] 在DINOv2特征空间中学习动态，以支持零样本规划。

### 解构式渲染表征 (Decomposed Rendering Representation)
该范式使用可渲染的显式图元（如NeRFs和3D高斯溅射）来表示场景，并通过更新这些图元来模拟动态并渲染未来观测。它能提供视角一致的预测和物体级别的组合性。

*   **基于3DGS的方法**：GAF [74] 为每个高斯点增加可学习的运动属性来预测未来状态。ManiGaussian [53] 通过预测每个点的变化来生成未来的高斯场景以用于操作任务。ManiGaussian++ [80] 进一步引入了层级化的“领导者-跟随者”设计，以建模多体和双手协作技能。
*   **与物理仿真结合**：DreMa [60] 将3DGS与物理模拟器结合，构建数字孪生以合成数据。PIN-WM [168] 结合3DGS与可微分物理引擎，从有限观测中估计物理参数，用于零样本的模拟到现实（Sim-to-Real）策略学习。
*   **表征层面**：DTT [98] 采用三平面（triplane）表示，并结合多尺度Transformer来捕捉动态。

## 未来方向与开放挑战

根据本文摘要的提炼，世界模型领域面临以下关键挑战和未来研究方向：

*   **统一的数据集与评估指标**：当前领域缺乏用于训练和评估的统一大规模数据集。此外，评估指标需要从注重像素保真度转向评估物理一致性、几何准确性和长期连贯性，这对于具身智能体的实际应用至关重要。
*   **性能与效率的权衡**：高保真度的世界模型通常计算成本高昂，难以满足物理设备（如机器人、自动驾驶汽车）上实时控制的需求。未来的研究需要在模型性能（如预测精度和长度）与计算效率之间找到更好的平衡点。
*   **长时程时间一致性**：在长时间序列的推演中，误差会随时间累积，导致预测结果与真实世界偏离。这是世界模型的核心建模难题之一。如何设计能够保持长期时间一致性并有效抑制误差累积的模型架构，是一个亟待解决的问题。

## 总结

本文对面向具身智能的世界模型进行了全面综述。通过提出一个包含功能决策耦合、时间推理和空间表征的三轴分类体系，本文系统地梳理了现有方法。该分类法不仅澄清了不同研究分支间的术语和目标差异，还为理解各方法的创新点和适用场景提供了统一的视角。从早期的基于循环网络的全局隐向量模型，到当前融合了大模型、扩散模型和显式3D表征的复杂系统，世界模型在模拟真实世界动态方面取得了显著进展。尽管如此，领域仍面临统一基准、计算效率和长时程一致性等关键挑战，这些挑战将是未来研究的核心方向。