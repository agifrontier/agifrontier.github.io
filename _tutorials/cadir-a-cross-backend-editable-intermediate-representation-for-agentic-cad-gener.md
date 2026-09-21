---
layout: default
title: "CADIR：跨软件可编辑中间表示，三大CAD环境节点重建率达100%"
description: "特别针对困扰业界的拓扑选择难题，CADIR 提出了结构化的语义选择器语法： {% raw %} {% endraw %} 该选择器通过显式指定目标类型 （点、边、面）、空间或几何参考原点 、邻接遍历算子 、几何谓词过滤条件 以及确定性序关系 ，彻底终结了“在未排序边列表中盲选第一项”的不可控操作。"
arxiv_id: "2608.00891"
paper_published: "2026-08-01"
published_at: "2026-09-21T13:15:08.283101+08:00"
topics:
  - "AI Agent"
tags:
  - "AI Agent"
  - "AI论文解读"
related_tutorials:
  - "a-survey-of-weight-space-learning-understanding-representation-and-generation"
  - "agent-harness-engineering-a-survey"
  - "robobridge-a-modular-framework-for-bridging-policies-to-robust-real-world-roboti"
  - "clawtrack-towards-trace-level-evaluation-and-improvement-of-real-world-autonomou"
---

<p class="paper-original-title" lang="en">CADIR: A Cross-Backend Editable Intermediate Representation for Agentic CAD Generation</p>

在大模型驱动工业设计的浪潮中，自然语言生成三维模型一直面临着一道难以逾越的鸿沟：模型生成的产物究竟是“死几何”还是“活特征”。过去几年里，基于边界表示（B-Rep）或网格的生成方案虽然能在视觉上还原零件形状，却丢失了所有草图拉伸、旋转、倒角和布尔运算等建模历史，工程师拿到后根本无法进行特征级参数修改；而那些尝试直接编写 Python 建模脚本（如 CadQuery 或 FreeCAD 脚本）的 Agent，又往往受困于脆弱的拓扑引用机制与隐式上下文状态，不仅跨软件平台无法迁移，在遇到复杂几何交错时也很容易因拓扑命名歧义而彻底崩溃。

> ArXiv URL：https://arxiv.org/abs/2608.00891

针对这些核心瓶颈，一项名为 **CADIR**（Cross-Backend Editable Intermediate Representation）的研究给出了系统性的破局思路。该方案既不是单纯去逼近最终网格，也不是写一段特定后端的胶水代码，而是设计了一套专为大模型智能体优化、可跨后端无损执行与二次编辑的中间表示系统。该系统以显式解耦的建模操作和图结构语义为纽带，打通了从文本或图像生成到主流工程软件原生特征树的完整链路。评测数据显示，CADIR 不仅在执行成功率上达到 100%，更通过全新的“几何签名匹配”机制，在 FreeCAD、SolidWorks 与 Fusion 360 三大主流商业及开源软件中，实现了 100% 的节点级原生建模历史重建率。

<img src="/images/2608.00891/figure1_cadir_generation.webp" alt="CADIR智能体生成的机械与产品设计CAD模型示例" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 现有CAD代码生成的症结：隐式状态与脆弱的拓扑索引

要理解 CADIR 的价值，首先需要看清当前大模型写 CAD 脚本时的根本死穴。传统的参数化 CAD 建模绝非简单的三维坐标堆砌，它是一个严格依赖拓扑父子关系的构造历史图。当人类工程师在某个表面绘制草图并打孔时，软件底层已经记录了该特征依赖于“哪一个面”以及“哪几条边”。

然而，现存的 CAD 生成方案在工程落地上存在两类明显的断层。一类是直接输出无历史的静态 B-Rep，如 BrepGen 等模型，下游工程师面对导出的 STEP 文件宛如面对一块实心顽石，想把一个轴承位公差从 10mm 改为 10.5mm 必须推倒重来；另一类则是依赖线性脚本的 Agent，例如直接调用 CadQuery 或商业 CAD 的二次开发 API。

以 CadQuery 这类流行的程序化建模库为例，它们的设计初衷是服务于拥有完整空间想象力的人类开发者。代码中普遍存在大量隐式的活动工作平面（Workplane）状态流转与连续链式调用。大型语言模型在长程规划时极难精准追踪当前上下文处于哪一个三维局部坐标系中。更致命的是，代码中常常使用类似 `.faces(">Z").edges().first()` 这种依赖局部几何排序的筛选规则。一旦上游参数发生微调，几何拓扑发生分裂或排序重组，下游的边界面索引就会产生严重漂移，导致后续特征附加在错误的几何图元上，甚至触发内核几何报错。

此外，现有工业界生态极度割裂。某一家企业习惯使用 Dassault 的 SolidWorks，另一家采用 Autodesk 的 Fusion 360，而开源社区则推崇 FreeCAD。如果生成的代码高度绑定在特定软件的专用 API 上，便无法被工程上下游顺畅采纳。智能体生成的 CAD 资产必须脱离单一软件绑架，以一种通用的、结构化的方式在异构系统间无损流转。

### CADIR的架构破局：两阶段生成与解耦中间表示

CADIR 团队将这一挑战抽象为严谨的两阶段映射范式。给定用户的自然语言指令或参考图像 $x$，生成流程首先由多智能体协作网络将其编译为一组可执行、可检验的中间产物集合 $\mathcal{Y}$；随后，特定平台的适配器（Adapter）将中间产物无损翻译为目标 CAD 软件的原生可编辑工程文件 $\mathcal{E}_{\mathrm{CAD}}$：




{% raw %}$$ x\stackrel{{\scriptstyle\mathrm{Agent}}}{{\longrightarrow}}\mathcal{Y}\stackrel{{\scriptstyle\mathrm{Adapter}}}{{\longrightarrow}}\mathcal{E}_{\mathrm{CAD}} $${% endraw %}



中间集合 $\mathcal{Y} = \{p_{\mathrm{CADIR}}, g_{\mathrm{STEP}}, g_{\mathrm{STL}}, G_{\mathrm{CADIR}}\}$ 同时包含了高层建模代码 $p_{\mathrm{CADIR}}$、标准化几何与网格文件，以及整套系统最核心的资产——在执行轨迹中自动沉淀的构造图（Construction Graph）$G_{\mathrm{CADIR}}$。

<img src="/images/2608.00891/figure2_cadir_agent_framework.webp" alt="CADIR系统总体框架与跨后端适配流程" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了让大模型在第一阶段不“掉进坑里”，CADIR 基于工业级几何内核 Open CASCADE Technology（OCCT，通过 OCP 封装）定义了一套专门面向智能体的建模接口。这套接口摒弃了容易混淆的隐式工作平面和过长链式调用，提炼出 115 个原子化的显式建模算子，涵盖草图绘制、基础成型、局部特征修改及装配约束。

特别针对困扰业界的拓扑选择难题，CADIR 提出了结构化的语义选择器语法：




{% raw %}$$ \mathsf{TSel}::={} \texttt{Select}(\tau,\rho)[\texttt{Traverse}(\pi)][\texttt{Where}(\phi)] [\texttt{Order}(o)][\texttt{Take}(k)][\texttt{Card}(c)] $${% endraw %}



该选择器通过显式指定目标类型 $\tau$（点、边、面）、空间或几何参考原点 $\rho$、邻接遍历算子 $\pi$、几何谓词过滤条件 $\phi$ 以及确定性序关系 $o$，彻底终结了“在未排序边列表中盲选第一项”的不可控操作。智能体不仅能清晰表达“选择由上一步拉伸操作生成、且法向偏向正 Z 的圆柱侧面”，还能在执行失败时获得细粒度的诊断回溯，精准定位几何交叠或空引用的确切步序。

### 跨越异构内核：构造图与几何签名匹配（GSM）

在中间表示被成功执行后，系统会捕获底层的所有参数流与拓扑映射，并固化为一张完整的有向无环构造图：




{% raw %}$$ G=(V,E,\mathcal{O},\mathcal{R}) $${% endraw %}



图中的节点记录了建模操作类型、参数字典、语义标签与几何增量上下文，边则明确表征了特征依赖关系。然而，将这张图迁移到其他 CAD 软件（例如将基于 OCCT 内核的图迁移到 Parasolid 内核支撑的 SolidWorks，或是 ShapeManager 内核驱动的 Fusion 360）并非易事。每个软件给面和边分配的内部 ID 截然不同，数值精度也存在微小差异。

<img src="/images/2608.00891/figure3_CCG.webp" alt="跨后端构造图与拓扑实体映射机制" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了让目标软件的原生适配器能够精准认出“上游步骤引用的那条倒角边到底对应当前几何体里的哪一条边”，研究者提出了**几何签名匹配**（Geometric Signature Matching, GSM）机制。

GSM 不依赖任何易变的原生图元索引，而是为几何拓扑实体构建了包含几何类型、特征点空间分布、边界曲率、相对面积及邻接拓扑关系的综合签名向量。当适配器在目标 CAD 软件中按拓扑序逐节点回放操作时，每当需要选择面或边，系统就会比对源端签名 $\sigma_s$ 与当前候选图元签名 $\sigma_c$ 的综合加权距离：




{% raw %}$$ D(\sigma_{s},\sigma_{c})=\sum_{k\in\mathcal{K}}w_{k}E_{k}(\sigma_{s},\sigma_{c}) $${% endraw %}



通过在目标几何体中搜索最小几何距离实体，GSM 绕过了异构软件之间难以直接对齐的命名机制。在这一技术支撑下，FreeCAD、SolidWorks 和 Fusion 360 的适配器能够忠实调动各自的原生 API，复刻出与官方原厂建模完全一致的、层级分明且随时可二次编辑的特征树（Feature Tree）。

### 构造图驱动的双塔检索增强：从整图复用到子图借鉴

在真实的工程场景中，大模型从零开始生成复杂的机械装配体依然面临巨大的空间推理负担。人类工程师做设计时，往往会从历史图纸库中借调局部标准结构（如法兰盘孔组、减速齿轮副或电机固定卡槽）。

受此启发，CADIR 进一步将构造图引入检索增强生成（RAG）领域，构建了一套支持全图与局部子图检索的双塔架构。传统的 CAD 检索大多停留在图像多模态对比或纯文本匹配，检索出来的整段脚本很难拆开复用；而通用的图神经网络又缺乏对参数化建模依赖关系的先验建模。

CADIR 针对文本和图像输入分别训练了双塔匹配网络：




{% raw %}$$ s_{t}(q_{t},G)=z_{t}^{\top}z_{g}^{t},\qquad s_{i}(q_{i},G)=z_{i}^{\top}z_{g}^{i} $${% endraw %}



网络通过对称对比损失函数 $\mathcal{L}_{\mathrm{con}}$ 进行端到端优化，使得查询嵌入与图表征在共享空间中紧密对齐。

<img src="/images/2608.00891/fig-retrieval.webp" alt="基于构造图的全图与子图检索效果对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

更重要的是，CADIR 允许提取图的诱导子图 $G[U]$ 进行索引。这意味着，当用户输入一个复杂装配体需求时，智能体不仅能检索到整体外观相似的既有案例，还能精准定位到某个历史案例中专门用于生成“行星齿轮架”的局部子图结构，并将其剪切、参数自适应后无缝嫁接到当前的生成任务中。

### 实验评测：执行稳定性与跨平台编辑力的飞跃

评测围绕中间表示有效性、检索增益、端到端生成质量以及跨后端可编辑性展开。测试基准严格采样自 DeepCAD 与 Fusion 360 Gallery 数据集，覆盖了从简单零件到拥有上百条操作的高复杂度模型，并使用统一的基础大模型（GPT-5.4）作为骨干。

在中间表示对比实验中，CADIR 与目前主流的脚本与中间格式——包括 CadQuery (CQ)、build123d、HistCAD、ForgeCAD 和 ECIP 展开了正面对决。在仅提供官方 API 文档而不注入额外样例的前提下，由于传统脚本普遍受制于脆弱的拓扑引用和隐式环境状态，执行成功率大多徘徊在 0.60 至 0.85 之间。CADIR 凭借显式组合算子与严格语义选择，实现了 **1.0000** 的执行成功率，且几何重合度交并比（IoU）相较于表现最好的基准方法提高了 4.9%，倒角距离（Chamfer Distance）和豪斯多夫距离（Hausdorff Distance）分别下降了 8.2% 与 5.4%。这证明结构更明确、语义更正交的 API 能大幅降低大语言模型的几何幻觉。

在跨平台重建与编辑这项极具硬核价值的测试中，GSM 的威力展露无遗。研究人员设计了包含 100 个复杂程序、覆盖全部 115 种算子、节点总数达 30,466 个的留存测试集。在目标软件中回放特征树时，若采用简单的实体索引重放，节点重建率（Node Reconstruction Rate, NRR）在 FreeCAD、Fusion 360 和 SolidWorks 中分别暴跌至 0.4490、0.4679 和 0.3175；即便采用此前业内探索过的空间点采样对齐方法，NRR 也仅在 0.36 至 0.53 之间徘徊。而 CADIR 的 GSM 机制在三大 CAD 后端中，全部斩获了 **1.0000**（100%）的节点重建成功率，重建几何与原始 OCCT 几何的 IoU 均达到 0.93 至 0.97 以上。

<img src="/images/2608.00891/example.webp" alt="CADIR生成的复杂装配体与机构案例" style="width:min(600px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; display:block;">

重建出原生模型只是起点，工程意义上的核心检验在于“能否继续编辑”。团队进一步设计了 294 项涵盖参数变更、局部特征替换和拓扑增删的二次编辑任务。实验结果表明，在重建后的原生模型上，FreeCAD、Fusion 360 和 SolidWorks 的编辑成功率（ESR）分别达到了 **1.0000**、**0.9796** 和 **0.9184**，而此前的方法在该指标上几乎全面失效。SolidWorks 和 Fusion 360 未能达到绝对满分，主要受限于个别特定高级曲面特征在外部自动化 API 中缺乏完全等价的可参数化暴露接口，而非表示本身的缺陷。

从图示的复杂生成案例可以看出，CADIR 已经突破了以往 Text-to-CAD 只能生成简易对称小零件的局限。它不仅能够稳定生成多层级行星轮系与集成致动器，还能够在内置的静态碰撞检测算子协助下，对包含旋转、移动、齿轮传动等装配关系的机构在多自由度极限位置下执行干涉校验，验证了生成模型在运动物理层面的自洽性。

### 从单点代码到工程基础设施的演进

CADIR 的意义超越了单一模型指标的涨跌。在此之前，学术界和开源界探索的大模型 CAD 生成，常常被工业界资深工程师诟病为“玩具”——导出的 STL 文件进不了装配图，生成的脚本换台电脑就因版本依赖或拓扑漂移而报错。

这项研究表明，要让大模型真正接入重工业生产力管线，核心不在于去刷多模态模型的几何点云拟合精度，而在于建立一套兼顾机器可执行性、拓扑鲁棒性与系统互操作性的中间表示标准。通过将拓扑命名问题转化为图驱动的几何特征签名匹配，CADIR 证明了大模型生成的成果完全可以无缝无损地沉淀为 Dassault、Autodesk 等工业巨头生态中的原生特征树。

展望后续发展，当这类可编辑中间图结构进一步融合机构运动学仿真与有限元应力分析反馈时，CAD 智能体将从单纯的“按图纸要求画模型”，真正演进为理解工程物理边界的“自主研发协作者”。CADIR 为这一演进铺平了最为关键的语义底座。
