---
layout: default
title: "AppliancePlan：看说明书操作家电，7B端到端模型规划成功率超基线10倍"
description: "为打破这一僵局，来自北京航空航天大学、京东科技与北京大学的研究团队提出了可扩展的数据合成管线 MAGE ，并基于其构建了首个大规模家电操作规划数据集 UseAppliance 。"
arxiv_id: "2608.15863"
paper_published: "2026-08-16"
published_at: "2026-09-06T13:15:07.886648+08:00"
topics:
  - "具身智能"
  - "推理"
tags:
  - "AppliancePlan"
  - "HAG"
  - "MAGE"
  - "RealAppliance-Bench"
  - "UseAppliance"
  - "closed-loop recovery"
related_tutorials:
  - "open-data-synthesis-for-deep-research"
  - "wait-wait-wait-why-do-reasoning-models-loop"
  - "socratic-swe-self-evolving-coding-agents-via-trace-derived-agent-skills"
  - "agentfrontier-expanding-the-capability-frontier-of-llm-agents-with-zpd-guided-da"
---

<p class="paper-original-title" lang="en">Scaling Manual-Grounded Appliance Manipulation with Data Synthesis and Unified Planning</p>

<img src="/images/2608.15863v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

让机器人走进千家万户并承担日常家务，一直是具身智能领域最引人注目的愿景。然而，相比于在实验室里将积木或水果移来移去这种简单的抓取放置任务（Pick-and-Place），真正操作一台现代家用电器要复杂得多。不同的微波炉、空气炸锅或咖啡机，不仅控制面板布局千差万别，其内部运作逻辑更是高度依赖具体状态。机器人必须理解诸如“旋转旋钮 90 度以切换到中火、连续按两次启动键、等待加热 3 分钟后拉开炉门”等包含时序、状态转换与精确几何参数的长时程操作。更棘手的是，一旦中途出现旋钮滑移或按键漏触等外界干扰，机器人若缺乏闭环自愈能力，整个任务就会彻底崩溃。

> ArXiv URL：https://arxiv.org/abs/2608.15863v1

人类面对全新家电时的做法是查阅产品说明书。说明书清晰地将整个认知过程拆解为三个环环相扣的能力：**部件定位与多模态对齐（Part Grounding）**、**开环长时程规划（Open-loop Manipulation Planning）**，以及在发生偏差时的**在线闭环规划调整（Closed-loop Adjustment）**。虽然此前学界推出了包含高保真数字资产与真实说明书的评估基准 RealAppliance-Bench，但现有的多模态大模型（MLLM）在上面的表现近乎瘫痪。即使是目前顶尖的闭源通用大模型，在该基准上的开环任务成功率也仅有可怜的 2.68%。

核心痛点并不在于多模态模型看不懂图文，而在于缺乏针对“根据说明书指导执行家电操作规划”的大规模、高质量结构化数据。为打破这一僵局，来自北京航空航天大学、京东科技与北京大学的研究团队提出了可扩展的数据合成管线 **MAGE**，并基于其构建了首个大规模家电操作规划数据集 **UseAppliance**。在此基础上训练的 7B 端到端统一规划模型 **AppliancePlan**，在 RealAppliance-Bench 上取得了断层式的性能突破，其开环规划任务成功率达到 31.36%，超过最佳基线 10 倍以上；在 6 种真实家电的实机部署中，AppliancePlan 同样以 40.00% 的任务成功率大幅压制了 GPT-5 等多模态巨无霸模型。

<img src="/images/2608.15863v1/applianceplan.webp" alt="MAGE 数据合成管线概览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 从手册到图谱：破解长时程规划数据的生成瓶颈

过去的研究尝试让机械臂操作家电时，往往陷入两极分化：一类方法专注于微波炉开门等单一步骤的机械交互，完全脱离了真实产品逻辑；另一类方法如 CheckManual 或 ApBot 虽引入了说明书，却依赖多阶段零样本级联管道，调用外部视觉检测器、状态推理器和大语言模型进行堆叠。这类级联系统不仅推理延迟高，而且误差极易逐级放大，其动作空间也往往受限于离散步数，难以单步输出精准的连续旋转角度。

造成这种局面的根本原因，是学术界长期缺乏包含精确参数、多样化状态覆盖以及闭环纠错轨迹的专门训练集。真实的人工采集成本极高，不可能为成百上千款家电一一雇佣工人录制长时程操作示范。研究团队设计的 MAGE（Manual-grounded Appliance data GEneration pipeline）给出了一种极为巧妙的解法：**将非结构化的说明书转化为结构化的图谱，再通过图遍历完成组合式的数据合成**。

MAGE 的核心支柱是**层次化家电图（Hierarchical Appliance Graph, HAG）**。一个 HAG 涵盖四层语义结构：

- **文档层与页面层**：负责收录手册文档节点，并标记说明书页面的功能类型（如产品概览、操作指南等）。

- **部件层**：锚定物理操作实体（如旋钮、开关门、触控按键、显示屏）。

- **状态与转换层**：为每个部件建立有向状态转换图。例如，开关门是典型的二值状态图（开/关）；模式选择键是离散多状态图；温度和定时旋钮则按说明书规格离散化为包含精确旋转角度的有向图。

一旦 HAG 构建完毕，数据生成任务就变成了优雅的图遍历问题。通过在状态空间中均匀采样初始状态 $s_0$ 与目标状态 $s^*$，算法能够无偏见地遍历全部状态转移路径，彻底摆脱人类示范总是集中在“热饭两分钟”等常用设置上的分布偏差。

为了给生成的动作序列配上对应的时序视觉观察，MAGE 引入了生成式世界模型（Seedream 4.0）。系统根据家电初始实拍图生成起始帧 $I_0$，并在后续每一步将动作 $a_t$ 转化为预期视觉变化的文本提示，迭代生成观测图像：




{% raw %}$$ I_{t+1}=\mathcal{M}_{\text{gen}}(I_t,\texttt{prompt}(a_t)) $${% endraw %}



配合跨步一致性检查与人工校验，整套管线能够生成高保真且视觉对齐的观测序列 $(I_0, I_1, \dots, I_T)$。

更进一步，为了让模型学会纠错，研究团队在生成的观察序列中间步骤主动注入“受控扰动”。这些扰动包括二值状态突变（如炉门被意外弹开）、按键漏触或多按，以及旋钮旋转角度偏差。扰动注入后，世界模型会生成异常观察帧 $I_t'$，系统随之根据当前状态与最终目标生成纠正动作 $a_{t+1}^{\text{corr}}$，从而大规模合成了极具价值的闭环恢复三元组 $(I_t', \text{context}, a_{t+1}^{\text{corr}})$。

### UseAppliance：首个大规模多模态家电操作规划底座

依托 MAGE 强大的生成与审计能力，研究团队构建了目前业界最具广度与深度的家电操作规划数据集 **UseAppliance**。

<img src="/images/2608.15863v1/applianceplan_vis.webp" alt="UseAppliance 数据集统计分布" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

该数据集横跨 22 种常见家电类别，涵盖了来自全球主流制造商的多样化控制面板与交互界面。其核心统计特征包含：

1. **密集的空间接地数据**：共包含 8.9 万余个部件边界框标注（Part Bounding Boxes）。如图 3(a) 所示，部件标注呈现出健康的长尾分布，既包含了门、标准按键、通用旋钮等高频部件，也覆盖了特殊功能拨杆、异形屏等低频操作区，极大增强了模型应对陌生机械结构的泛化鲁棒性。

2. **丰富的规划与转移轨迹**：包含 5.3 万多条开环操作任务，平均任务步长达 8.15 步。图 3(b) 清晰揭示了不同类别家电截然不同的动作分布模式——旋钮密集型设备（如传统烤箱、搅拌机）由旋转动作主导，而数码面板型设备（如微波炉、电饭煲）则充斥着大量的按压与等待逻辑。

3. **针对性的闭环恢复监督**：数据集包含了超过 3.3 万步闭环纠偏样本，为模型在面对真实物理扰动时提供即时修正的策略支撑。

最重要的是，MAGE 管线的解耦设计赋予了 UseAppliance 强大的扩展性。任何全新的家电产品，只要提供其 PDF 说明书与面板照片，就能以极低的边际成本接入该管线并扩充进数据集。

### AppliancePlan：六大训练目标铸就统一端到端智能体

拥有高质量数据之后，模型该如何设计？过去的级联架构之所以频现故障，是因为各个子模块之间语义割裂。AppliancePlan 摒弃了复杂的流水线，以开源多模态基座 Qwen2.5-VL-7B-Instruct 为骨架，构建了一个端到端的统一规划模型。

<img src="/images/2608.15863v1/model.webp" alt="AppliancePlan 模型架构与六大训练目标" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

模型的输入直接包含说明书页面图像、多视角家电实时观测图像 $\mathcal{I}=\{I_1,\dots,I_t\}$ 以及自然语言任务指令 $L$。视觉编码器将图像投射为视觉 Token $H_v$，与文本 Token $X = \mathrm{Tok}(L)$ 拼接后，自回归地预测操作动作序列：




{% raw %}$$ P_{\theta}(Y\mid H_v,X)=\prod_{n=1}^{N}P_{\theta}\left(y_n\mid y_{<n},H_v,X\right) $${% endraw %}



为了让仅有 7B 参数的模型同时精通多项复杂能力，研究团队将学习任务形式化为 3 个主规划目标与 3 个辅助训练目标，全部统一为自回归 Next-token Prediction 损失函数进行端到端微调。

#### 1. 三大多任务主目标

主损失函数 $\mathcal{L}_{\text{main}} = \mathcal{L}_g + \mathcal{L}_o + \mathcal{L}_c$ 紧密对应具身操作的三大核心阶段：

- **部件定位损失 $\mathcal{L}_g$**：监督模型根据说明书中的指引图示，在真实多视角实物图像中输出部件精确的边界框坐标 $\mathbf{b}$。

- **开环规划损失 $\mathcal{L}_o$**：在给定初始状态和目标描述下，自回归生成包含完整动作类型与连续参数的长时程动作序列 $Y_o = (a_1, a_2, \dots, a_T)$。

- **闭环调整损失 $\mathcal{L}_c$**：在发生意外扰动导致状态偏离预定轨迹时，监督模型根据扰动图像输出一步纠偏动作 $a_{t+1}^*$。

#### 2. 定向赋能的三大辅助目标

如果仅仅依赖宏观的任务监督，模型往往容易对长文本与复杂图像产生虚假相关性。研究团队额外引入了辅助损失函数 $\mathcal{L}_{\text{aux}} = \mathcal{L}_{\text{align}} + \mathcal{L}_{\text{key}} + \mathcal{L}_{\text{state}}$，精准打击空间、参数与状态维度的薄弱环节：

- **说明书-实物双向对齐 $\mathcal{L}_{\text{align}}$**：强制模型学习说明书示意图与实物部件 ID、空间位置的跨域双向映射，大幅改善跨域视觉鸿沟下的部件定位精度。

- **关键步动作预测 $\mathcal{L}_{\text{key}}$**：专门针对旋钮角度偏差敏感、时间参数苛刻的瓶颈动作进行针对性抽样监督，防止动作参数在长序列解码中漂移。

- **部件状态判别 $\mathcal{L}_{\text{state}}$**：要求模型显式预测当前观察中某个具体部件所处的状态标签（如“已旋转 45 度”、“门处于微开状态”），确保纠偏决策不是基于幻觉盲猜，而是源于扎实的状态认知。

最终的联合优化目标为 $\mathcal{L} = \mathcal{L}_{\text{main}} + \mathcal{L}_{\text{aux}}$，所有任务权重设为 1.0，在 96 个 NVIDIA H200 GPU 时内即完成了高效训练。

### 实验评测：不仅超越 235B 巨模，更实现 10 倍性能跃升

为了检验模型的真本领，所有评测都在权威基准 **RealAppliance-Bench** 上展开。需要强调的是，训练集 UseAppliance 与 RealAppliance-Bench 在家电型号与数字资产上**完全零重叠**，所有测试均为严格的分布外（Out-of-Distribution, OOD）泛化评估。

评测对比涵盖了闭源巨型多模态模型（如 GPT-5、Gemini 2.5 Pro）、开源前沿多模态大模型（如 Qwen3-VL-235B-Instruct）以及主流具身规划基线（如 RoboBrain 2.0-7B、ApBot）。

在最为考验综合推理能力的**开环长时程规划**赛道上，传统大模型普遍遭遇滑铁卢：最强的基线模型任务完成率仅为 4.36%，成功率低至 2.68%。这直接证明了单纯依靠通识图文预训练，模型根本无法掌握带有连续旋转角度和时序依赖的精密动作链。而仅有 7B 规模的 AppliancePlan 一举拿下了 **47.86% 的任务完成率**与 **31.36% 的任务成功率**，相较于最佳基线实现了超过 10 倍的飞跃，并且在全部 14 个测试品类中无一例外夺得第一。

在**部件定位**赛道，AppliancePlan 取得了 22.96% IoU 与 22.24% mAP@0.5，比复杂的级联架构 ApBot 分别高出 12.36 和 10.14 个百分点；在**闭环单步纠偏**测试中，模型以 37.12% 的准确率稳稳压过 Gemini 2.5 Pro（31.73%）与 RoboBrain 2.0-7B（31.77%）。

最具现实意义的测试是**序贯规划与动态调整（Sequential Planning and Adjustment）**。在这个完全仿真的闭环场景下，机械臂不仅要按照自己生成的初始计划去执行，一旦执行中出现偏差或机械失误，还必须由模型自行根据即时视觉反馈进行二次修正。这极其考验模型的自洽性。


| 模型类型 | 代表模型 | 序贯任务完成率 (%) | 序贯任务成功率 (%) |
| :--- | :--- | :---: | :---: |
| 闭源前沿模型 | GPT-5 | 6.12 | 2.04 |
| 闭源前沿模型 | Gemini 2.5 Pro | 7.84 | 3.40 |
| 开源旗舰模型 | Qwen3-VL-235B-Instruct | 5.84 | 4.08 |
| 具身规划专用 | RoboBrain 2.0-7B | 6.45 | 2.72 |
| **端到端统一模型** | **AppliancePlan (7B)** | **44.59** | **28.07** |

在这项极其严苛的评测中，参数量高达 235B 的前沿开源模型成功率仅为 4.08%，而 AppliancePlan 达到了 **28.07%**。消融实验进一步证实，去掉说明书对齐损失会导致空间定位崩溃，去掉关键步预测会导致参数预测失准，而去掉状态判别则会使模型在闭环时丧失纠偏依据。六大目标的有机整合缺一不可。

### 实机验证：从虚拟仿真顺利迈向物理厨房

仿真环境里的亮眼数据能否转化为物理世界中机械臂的稳健操作？研究团队搭建了一套真实的机器人验证系统：由一台搭载 RealSense D415 眼在手（Eye-in-Hand）相机与固定机外视角的 Franka Emika Panda 机械臂组成。

<img src="/images/2608.15863v1/case_study.webp" alt="物理机器人操作真实家电实录" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

测试场景涵盖了微波炉、空气炸锅、多士炉、电饭煲、咖啡机和搅拌机 6 种完全真实的物理家电。部署流程清晰明确：AppliancePlan 运行在一台配置单张 RTX 4090 显卡的本地工作站上，接收说明书扫描件与双目相机采集的面板照片，首先输出长时程开环动作规划；底层的 6D 位姿估计算法（FoundationPose）与抓取生成网络（AnyGrasp）将动作参数映射到空间执行轨迹；机械臂在每执行一步后，都会由 AppliancePlan 根据新的实时回传图像重新审视当前状态，一旦发现旋钮未到位或舱门未关严，立即在线触发纠偏动作。

在横跨 6 种家电的 60 组真实任务实测中，GPT-5 的真实任务成功率仅有 3.33%（平均完成率 19.35%），ApBot 同样仅为 3.33%（完成率 9.12%）。这些基于通用大模型的方案虽然常常能勉强蒙对第一步（例如把烤箱门打开），但一旦面对“将旋钮精确旋转到对应刻度再按下确认”这种高精度状态依存操作时，便接二连三地陷入死循环。

与之形成鲜明对比的是，**AppliancePlan 在物理世界中斩获了 49.87% 的平均任务完成率与 40.00% 的真实任务成功率**。如图 5 所示，无论是准确抠开空气炸锅的推拉抽屉、精确将微波炉的机械旋钮转过指定角度，还是在咖啡机按钮漏触时自动补按，AppliancePlan 均展现出了扎实可靠的 Sim-to-Real 迁移能力。

这项工作给具身智能领域带来了一个极具说服力的启示：**长时程规划的短板，无法单靠盲目堆叠通识大模型的参数量来自然弥合**。家电操作这类高度依赖专有规范、带有物理因果律的任务，其核心在于高质量数据合成范式与端到端结构化监督的深度结合。MAGE 管线证明了将非结构化说明书转化为图谱与生成式仿真数据的可行性，而 AppliancePlan 的优异表现则表明，一个精巧设计并经过领域数据充分滋养的 7B 轻量级模型，完全可以在专业具身任务上击溃数百倍于自身体积的通用庞然大物。向通用家庭机器人迈进的道路上，如何让机器学会像人类一样“照着说明书干活”，AppliancePlan 已经给出了极其坚实的第一步。
