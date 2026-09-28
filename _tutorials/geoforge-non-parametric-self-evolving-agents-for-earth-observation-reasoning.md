---
layout: default
title: "GeoForge：中科院等提出免微调自进化框架，遥感Agent准确率达77%"
description: "来自中国科学院与重庆大学的研究团队提出了 GeoForge ，这是一个 免微调（Training-free）、非参数化自进化的地球观测智能体框架 。该研究的核心转变在于：放弃代价高昂且难以持续迭代的模型权重微调，转而在大模型外部构建一套结构化的“三层非参数执行记忆”。"
arxiv_id: "2608.10494"
paper_published: "2026-08-11"
published_at: "2026-09-28T13:15:07.830421+08:00"
topics:
  - "AI Agent"
  - "推理"
tags:
  - "Action-Level Experiences"
  - "Adapted Skill Standard Operating Procedure"
  - "EO"
  - "GeoForge"
  - "Non-parametric execution state"
  - "Safety-gated distillation"
related_tutorials:
  - "agentic-reinforcement-learning-with-observation-calibrated-self-distillation"
  - "latent-learning-episodic-memory-complements-parametric-learning-by-enabling-flex"
  - "deep-self-evolving-reasoning"
  - "self-evolving-embodied-agents-via-skill-harness-evolution"
seo_title: "GeoForge: Non-Parametric Self-Evolving Agents for Earth-Observation Reasoning"
---

<p class="paper-original-title" lang="en">GeoForge: Non-Parametric Self-Evolving Agents for Earth-Observation Reasoning</p>

<img src="/images/2608.10494v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

利用大语言模型（LLM）构建自主智能体（Agent）来处理复杂现实任务，已经是当前人工智能领域的热门方向。但在地球观测（Earth Observation, EO）与遥感科学领域，通用的 Agent 范式往往显得力不从心。遥感分析不仅要求模型理解自然语言，还深度依赖光学、合成孔径雷达（SAR）、多光谱红外等异构传感器数据，具有严格的数据产品依赖链条、时空分辨率兼容性以及精确的数值参数要求。现有智能体在面对海量遥感工具库时，常常陷入无序搜索、参数错配与执行轨迹中断的泥潭。

> ArXiv URL：https://arxiv.org/abs/2608.10494v1

来自中国科学院与重庆大学的研究团队提出了 **GeoForge**，这是一个**免微调（Training-free）、非参数化自进化的地球观测智能体框架**。该研究的核心转变在于：放弃代价高昂且难以持续迭代的模型权重微调，转而在大模型外部构建一套结构化的“三层非参数执行记忆”，涵盖全局工作流图、动作级微观经验与标准化作业程序（SOP）。配合安全门控的轨迹蒸馏机制，智能体在每次执行任务后都能自动提取经验、更新外部记忆。

实验表明，GeoForge 在 Earth-Bench 等基准上全面超越了现有遥感智能体。在 DeepSeek-V3.1 和 GPT-5 骨干模型上分别取得了 **77.09%** 与 **74.33%** 的任务准确率；在对多步工具链要求极高的地学产品（Products）与多光谱（Spectrum）任务中，相对主流基线更是实现了接近 **30 个百分点** 的大幅提升，同时几乎消除了智能体在宏观规划与高层推理层面的逻辑错误。

<img src="/images/2608.10494v1/x1.webp" alt="传统 EO Agent 与 GeoForge 的执行对比" style="width:min(600px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; display:block;">

### 遥感科学决策的特殊泥潭：通用智能体为何频频崩溃？

在通用的软件调用或网页浏览任务中，Agent 即使走了一两步弯路，通常也可以通过反复试错或者单步反馈进行纠偏。但在地球观测与遥感科学分析中，这种“盲目探索”的代价极其高昂，甚至在原理上就无法成立。

遥感科学工作流具有极强的物理与地理语义约束。比如，计算归一化燃烧比（NBR）需要指定前期的近红外波段与短波红外波段数据，随后必须经过影像正射校正、辐射定标、波段运算、阈值分割与空间聚类统计等一系列前后级依赖严密的专业处理。一旦模型调错了工具顺序，或是将光学波段参数输入给了 SAR 算法，后续整个分析链条就会彻底崩溃。

以往的研究尝试过两种路径：一是使用静态的系统提示词（Prompt）或者通用 ReAct 循环，但这无法应对复杂多变的传感器组合与异构任务空间；二是像部分自进化系统那样，将成功的代码或零散的对话经验直接存入向量数据库。然而，未加整理的原始轨迹充斥着大量具体实例的噪音，而过于简略的文字摘要又丢失了波段依赖、数据尺寸与调用顺序等核心科学约束。面对海量专业工具，智能体依然只能在超大的动作空间中碰运气，导致大量冗余调用和死循环。

GeoForge 的破局思路很明确：**任务规划所需要的科学先验知识，不一定非要写进 LLM 的神经元权重里，而应该抽象成外置的、结构化的非参数记忆状态。**

<img src="/images/2608.10494v1/x2.webp" alt="GeoForge 架构与自进化闭环流程" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 解构执行认知：三层非参数记忆体系

为了在宏观工作流顺序与微观参数约束之间取得平衡，GeoForge 构建了三层互补的非参数记忆库。这三层记忆分别对应了人类科研人员在解决遥感问题时的不同思维维度。

第一层是**工作流图记忆（Workflow Graph Memory, WGM）**。它负责把控全局的操作顺序。研究团队没有采用通用的 API 调用转移图，而是将历史成功轨迹抽象为一个有向图结构 $\mathcal{G}=(\mathcal{W},\mathcal{A},\mathcal{U})$。在这个图谱中，节点不仅包含工具信息，还融合了数据探测、输入组织、产品生成、时空聚合到科学判读的标准依赖阶段。更重要的是，节点上记录了正向与负向的使用统计得分。系统通过文本相似度、工具重合度以及任务类别一致性，为新查询检索出最匹配的全局拓扑结构，为后续的推理提供了骨架式的流程指引。

第二层是**动作级经验库（Action-Level Experiences）**。工作流图解决了“先做什么后做什么”，但具体的单步决策依然容易因为参数细节而翻车。动作级经验库以非参数片段的形式，记录了局部微观决策中的“避坑指南”。每一条经验 $e_i$ 都记录了特定的触发上下文、历史采取的动作、带来的执行反馈以及归纳后的改进建议。当智能体在新任务中遇到相似的局部状态时，系统会通过词汇亲和度与任务一致性检索出高精度的矫正策略，有效防止重复踏入相同的参数陷阱。

第三层是**自适应技能标准作业程序（Adapted Skill SOP）**。如果说图记忆是骨架、经验库是纠偏补丁，SOP 则是贯穿任务始终的高级执行规范。它定义了特定任务类型的标准分解步骤、中间产物规格、参数边界与证据汇聚方式。在实际推理前，系统会根据当前的具体问题与动作经验，动态自适应实例化一份专属于该任务的执行指导书，约束智能体的行为不偏离遥感领域的科学规范。

通过这种“图拓扑 + 动作修正 + 规范 SOP”的立体组合，智能体在启动推理的第一步，就建立起了对当前遥感任务的全局认知与局部防御。

### 执行、验证与安全蒸馏的自进化闭环

外部记忆不是一成不变的静态资产，GeoForge 的真正威力在于它的持续自进化能力。整个系统运行在一个严密的闭环流程中：

在**推理执行阶段**，智能体并不直接面对庞大的工具全集。系统首先依据任务输入解析出当前的遥感模态与观测上下文（例如属于 Sentinel-2 光学还是 Sentinel-1 雷达），先验性地对工具空间进行第一道过滤。随后，系统检索三层记忆库并拼接出紧凑的执行上下文 $\mathcal{C}_q$。在推理过程中，模型将这些先验作为“参考建议”，而最终的科学论断依然严格基于工具返回的真实遥感观测数据，从机制上杜绝了将历史旧答案当成当前事实的幻觉问题。

在**任务结束后的自进化阶段**，系统会启动基于安全门控的轨迹蒸馏流程（Safety-Gated Distillation）。研究团队清醒地认识到：不是所有完成的任务轨迹都值得被学习。错误的轨迹会污染记忆库，冗余低效的轨迹会退化规划质量。

因此，GeoForge 设计了一套严格的安全门控判别函数 $\psi(\tau,\hat{y})$。只有同时满足以下条件的轨迹才会被送入蒸馏器：

1. 真实调用了有效遥感工具且输出了有效答案；

2. 轨迹未发生格式溃散或严重幻觉；

3. 总工具调用步数未超出最大上限 $L_{\max}$；

4. 单个工具的重复尝试次数未超过阈值 $R_{\max}$（即排除了死循环尝试的轨迹）；

5. 提取出的新 SOP 没有触及预设的失败模式集合。

一旦通过安全门控，蒸馏器 $D_\Theta$ 就会对调用链条、传参规律、中间数据产物路径进行压缩归纳，分别生成针对工作流图的更新量 $\Delta\mathcal{W}$、针对动作经验的更新量 $\Delta\mathcal{E}$ 以及针对技能库的更新量 $\Delta\mathcal{S}$。这种非参数化的更新完全发生在外部知识库中，完全不需要对大模型本身进行梯度更新，既保证了极高的计算经济性，又避免了深度学习常见的灾难性遗忘。

<img src="/images/2608.10494v1/x4.webp" alt="不同骨干大模型下的错误类型分布演变" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 实验评测：不仅是准确率跃升，更是错误形态的“降维”

为了验证框架的有效性，研究团队在 Earth-Bench、ThinkGeo 和 GeoPlan-Bench 等多个代表性遥感与地学智能体基准上展开了系统性评测。测试涵盖了闭源前沿模型（GPT-5、Gemini-2.5-Flash）以及开源强模型（DeepSeek-V3.1、Qwen3-Max）。

在 Earth-Bench 的综合测试中，GeoForge 展现出对多种骨干 LLM 的普适增强能力。如前所述，搭载 GeoForge 后，DeepSeek-V3.1 准确率达到 **77.09%**，GPT-5 达到 **74.33%**，Gemini-2.5-Flash 达到 **67.91%**，Qwen3-Max 达到 **69.72%**。相比于原生的 Earth-Agent、OpenEarth-Agent 以及同样引入自进化思想的 GeoEvolver，GeoForge 均取得了显著领先。

不仅最终答案的准确率提升，轨迹质量指标也发生了质的飞跃。在衡量工具链顺序忠实度的 Tool-In-Order 和衡量全序列精确匹配的 Tool-Exact-Match 上，GeoForge 带来了普遍提升。以 GPT-5 为例，全序列完全匹配率达到了 **51.18%**，证明智能体不再是盲目地乱撞工具，而是真正跑出了合乎规范的遥感分析流水线。

消融实验进一步厘清了三层记忆各自的不可替代性。测试数据显示：

- 如果完全移除记忆库，智能体的基础准确率仅为 **52.23%**；

- 移除 Skill SOP 导致了断崖式的准确率下跌（从 74.33% 骤降至 52.66%），这表明面向任务的高层次规程约束是保证最终科学结论成立的生命线；

- 移除工作流图记忆（Workflow Graph Memory）后，轨迹指标受创最重，Tool-In-Order 从 70.01% 跌至 61.74%，Tool-Exact-Match 更是跌破 46%，证明了有向依赖图对约束全局调用顺序的决定性价值。

在模态泛化测试中，这种结构化记忆的优势在光谱（Spectrum）与数据产品（Products）分析两类任务中展现得尤为明显。相比强基线 Earth-Agent，GeoForge 在光谱任务上取得了 **77.00%** 的准确率（提升 27.00 个百分点），在产品生成任务上取得了 **71.26%**（提升 29.15 个百分点）。这两类任务步骤极其琐碎、波段要求严苛，正是传统无引导 Agent 最容易崩溃的深水区。

通过分析模型出错分布的演变，研究人员观察到了一个有趣的现象：GeoForge 几乎把所有测试大模型的高层规划错误（Tool-Planning Error）与推理逻辑错误（Reasoning Error）完全消除。残留的少量错误，主要集中在底层具体的图像生成参数与深层执行追踪上。这意味着，GeoForge 成功将原本难以控制的系统性逻辑混乱，“降维”成了局部的底层参数调试问题。

<img src="/images/2608.10494v1/x5.webp" alt="检索技能数量与相似度阈值的敏感性分析" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

从超参数敏感性分析中可以看到，GeoForge 在不同的检索技能数量（Top-$k$）和相似度阈值下均表现出较强的稳定性。当 Top-$k$ 取 3、检索阈值设为 0.6 时，系统在避免上下文冗余和保留充分先验之间找到了最佳平衡点。过多的技能注入反而会引入无关上下文干扰，这印证了“精准高质的非参数先验优于泛化堆砌”的设计哲学。

<img src="/images/2608.10494v1/x3.webp" alt="基线与 GeoForge 在 NBR 指数计算任务上的轨迹对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 典型案例：从混乱死循环到精准四步成图

在本文呈现的归一化燃烧比（NBR）实测对比中，基线 Agent 面对火灾评估任务时，陷入了频繁切换工具、反复试探无效参数的混乱循环，最终因为步数超限而直接报错崩溃。

相比之下，GeoForge 在工作流图和 SOP 的双重护航下，表现得如同一位经验老到的遥感分析员：

1. 第一步，精准调用接口检索匹配的目标区域文件列表；

2. 第二步，直接装配正确的光学与短波红外波段计算 NBR 指数；

3. 第三步，依据指数阈值调用空间聚类算子锁定异常火灾热点区域；

4. 第四步，针对热点区域做最终的空间方位与蔓延分析，输出正确的决策选项。

整个过程仅仅使用了 4 个关键工具步骤，没有任何多余的反复试探，不仅大幅节约了模型推理的 Token 开销与响应延迟，更保障了遥感分析结论的严密可溯源。

### 总结与展望

GeoForge 为专业垂类智能体的发展提供了一条极具说服力的技术路线：面对高度严密、容错率极低的科学计算任务，一味追求增大模型参数量或依赖高成本全量微调并非最优解。相反，**将领域专家经验转化为可检索、可验证、可非参数进化的外部记忆结构**，不仅能大幅抑制大模型的幻觉与规划混乱，更能让普通骨干模型在复杂专业场景中发挥出顶级分析师的作业水准。

这种以工作流拓扑、微观动作经验与标准化 SOP 为核心的三层自进化闭环，不仅在地球观测遥感领域验证了价值，也为生物信息计算、材料分子模拟、工业控制自动化等一系列重流程、重依赖的科学智能体（Scientific Agents）构建，提供了极具参考价值的底层架构范式。
