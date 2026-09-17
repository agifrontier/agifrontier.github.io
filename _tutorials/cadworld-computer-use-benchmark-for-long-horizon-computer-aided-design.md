---
layout: default
title: "CADWorld：当Agent进入FreeCAD，最强模型成功率为何跌至17.5%？"
description: "为了全面覆盖机械工程设计流程，CADWorld 系统梳理了机械设计与制造领域的系统知识，构建了涵盖 11 个工作流类别的任务分类学（Taxonomy）。基准包含的 200 个任务覆盖了 183 个具体机械知识点。"
arxiv_id: "2609.16251"
paper_published: "2026-09-14"
published_at: "2026-09-17T13:15:08.246355+08:00"
topics:
  - "AI Agent"
  - "AI评测"
tags:
  - "CADWorld"
  - "FreeCAD"
  - "computer-aided design"
  - "engineering workflow automation"
  - "executable task checks"
  - "geometry manipulation"
related_tutorials:
  - "clinlens-towards-long-horizon-coding-agents-for-longitudinal-multimodal-clinical"
  - "gui-360-a-comprehensive-dataset-and-benchmark-for-computer-using-agents"
  - "contextweave-a-real-world-workflow-benchmark"
  - "handbookmd-a-benchmark-for-long-context-agentic-instruction-following"
---

<p class="paper-original-title" lang="en">CADWorld: Computer-Use Benchmark for Long-Horizon Computer-Aided Design</p>

<img src="/images/2609.16251/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

大语言模型与多模态计算机使用智能体（Computer-Use Agents, CUAs）在操作系统自动化、网页浏览和日常办公软件中展现出越来越强的操作能力。然而，一旦离开表单填写、文件整理或简单的问答界面，进入极其依赖几何连续性、状态依赖与严密参数约束的专业工程领域，现有的前沿模型立刻遭遇了严峻的滑铁卢。

> ArXiv URL：https://arxiv.org/abs/2609.16251

来自 CAMEL-AI、佐治亚理工学院、新加坡国立大学等机构的研究人员推出了专注于机械工程领域的长程计算机使用基准 **CADWorld**。该基准构建于开源参数化三维建模软件 FreeCAD 之上，涵盖草图绘制、实体建模、装配、计算机辅助制造（CAM）、有限元分析（FEM）、尺寸测量与工程图纸等 11 个机械工程工作流类别，共包含 200 个长程任务与 183 个细分知识点。与过去仅依赖文本比对或渲染图像外观相似度的基准不同，CADWorld 强调**产物级可执行评测**（Artifact-Grounded Evaluation），直接检测模型最终生成的工程原生文件与仿真状态。

测试结果令人警醒：在涵盖完整工作流的测试中，人类专家的基准通过率达到了 87.0%，而当前业内顶尖的计算机使用 Agent 在完整基准上的最高成功率仅为 17.5%（在预算受控的 60 任务子集上最高仅 25.0%）。这一巨大鸿沟清晰表明，通用的图形界面感知与点击定位能力，并不等价于可靠执行具有持久性、高精度几何约束和严格拓扑结构的工程任务。

<img src="/images/2609.16251/x1.webp" alt="FreeCAD 中典型的长程参数化建模工作流" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 机械 CAD 成为智能体长程交互的极限压力测试

在过去几年中，针对 CAD 领域的 AI 研究大多集中在“代码生成”或“命令序列预测”路径上，例如直接预测 OpenCASCADE 或 Python 脚本来构建三维几何体。然而在实际工业生产中，工程师极少通过编写纯脚本完成复杂的机械设计。工程师依赖的是图形用户界面（GUI）：在空间中布置几何元素、捕捉几何约束（重合、平行、相切等）、实时观察三维视图、调整特征树（Feature Tree）、设定刀路或施加网格约束。

图形交互并非简单包裹在底层 API 外面的“皮囊”，而是工程状态被表达、理解和验证的核心通道。一个操作的工程语义高度依赖于当前激活的视图、已选中的边或面、坐标系定位以及几何体之间的相对位置。因此，让计算机使用智能体直接面对真正的 CAD 桌面界面，是迈向工业实用的必经之路。

这一设定将智能体推向了多重复杂度的交汇处：

1. **跨模态空间推理与几何意图恢复**：Agent 必须从二维桌面截图中推断出复杂的三维空间关系，不仅要理解零件投影，还要感知实体之间的装配自由度、材料去除量以及特征拓扑关系。

2. **误差传递与容错极低**：在 CAD 软件中，一个微小的像素级点选偏移、拖拽错位或参数输入错误，就会导致后续草图无法闭合、布尔运算失败或约束求解器报出欠约束/过约束警告，进而导致整个历史特征树崩溃。

3. **极长的动作依赖链**：一个中等复杂度的机械零件，往往需要跨越几十甚至上百步连续操作，涉及新建文档、切换工作台、绘制基础草图、标注尺寸约束、拉伸实体、倒角、镜像阵列、进入装配环境施加副连接等，要求模型在漫长的交互过程中始终维持一致的目标状态表征。

<img src="/images/2609.16251/x2.webp" alt="CADWorld 单个任务的结构组成" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### CADWorld 的任务设计与工程知识图谱

为了全面覆盖机械工程设计流程，CADWorld 系统梳理了机械设计与制造领域的系统知识，构建了涵盖 11 个工作流类别的任务分类学（Taxonomy）。基准包含的 200 个任务覆盖了 183 个具体机械知识点。其中，190 个任务专注于检验深层次的工程概念组合，另外设置了 10 个轻量级任务，用于在更低复杂度下检验基础操作能力。

整个任务体系不再局限于学术界常用的简单长方体或圆柱体堆叠，而是引入了工程实践中极易出错且此前鲜少被系统评测的特征，例如复杂的放样、扫掠、倒角（Fillets）、倒圆、抽壳，以及下游的 CAM 刀轨生成、有限元网格划分和材料力学载荷定义。

基准中的每一个任务均以标准化的四元组进行打包：

- **自然语言指令**：包含设计要求与设计意图，平均长度为 63 个词，最复杂的任务长达 178 个词；

- **初始运行环境配置**：包含初始空白项目或任务前置文件（Preconditions），如半成品几何体、参考点云、网格或初始装配部件；

- **可选参考资产**：包括工程二维图纸、装配示意图等图形参考资产；

- **任务专属的可执行评估器**：在无人工干预的情况下，自动挂载并验证最终成果。

在执行流程中，虚拟机采用 Ubuntu 配合 Docker/QEMU 环境，Agent 仅能接收桌面截图、历史动作轨迹以及自然语言指令，通过模拟真实的鼠标点击、拖拽、键盘输入以及有限的等待与终止控制标记完成操作。智能体无法绕过 GUI 直接访问底层的 FreeCAD 运行脚本或文件系统，这保证了评测环境与人类工程师日常面对的环境高度对齐。

<img src="/images/2609.16251/x3.webp" alt="CADWorld 任务流程与人类/模型输出对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 拒绝“以貌取人”：产物级可验证性评测

在传统的视觉或多模态生成评测中，研究者经常依赖图像相似度（如 PSNR、SSIM 或特征余弦相似度）甚至视觉语言模型评分（LLM-as-a-Judge）来评判生成结果。但在精密工程制造中，这种“视觉相似”往往带来致命的误导。

CADWorld 的关键设计哲学在于**产物级可验证性**（Artifact-Grounded Evaluation）。如图所示，两个在渲染画面上看起来极其相似的三棱柱零件，若拆解其内部工程结构，模型的成果往往是完全不可用的：可能缺少底层的草图约束，可能使用了错误的特征拉伸方式，也可能未构建正确的参数依赖关系。在工业场景下，下游工程师如果无法打开历史树去修改某一个关键尺寸，或者在后续加工时因为拓扑缺陷导致刀具碰撞，该设计就是彻底失败的废品。

因此，CADWorld 的评估器完全在宿主机侧针对保存的 `.FCStd` 文件和衍生导出的辅助数据（如 NC 代码、有限元应力 CSV、工程图导出文件）展开无死角的可执行分析：

- **草图与几何维度**：解析底层文件归档，提取草图中的点、线、圆弧图元，核验几何约束类型、关键尺寸数值、轮廓封闭性、截面面积与质心坐标；

- **特征树与拓扑结构**：检查 FreeCAD 模型中的特征对象类型、层级树关系、体积、表面积、三维包围盒（Bounding Box）交并比（IoU）；

- **装配与物理机构**：检验固定副（Grounded）、齿轮齿条副（Rack and Pinion）、滑动副（Slider）等运动副元数据的有效性；

- **制造与仿真状态**：在 CAM 任务中比对毛坯到目标的实际切除体积、刀具路径完整性、过切与欠切比例；在 FEM 任务中核查材料参数、网格单元分布、载荷与固定边界条件、求解器配置及导出应力位移极值。

CADWorld 对成功的定义极其严苛：只有当一项任务预设的所有检查规则全部通过时，该任务才被判定为“成功（Success）”，不给予任何模糊的折中部分分。这种纯客观、无外部大模型主观评判的机制，保证了评测的高重现性与工业严肃性。

<img src="/images/2609.16251/model_performance_heatmap.webp" alt="主流 Agent 模型在 CADWorld 上的性能与失败原因热力图" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 实验结果：错误没有消失，只是“向后移动”

在对 7 种代表性前沿计算机使用智能体的综合评估中，CADWorld 揭示了现有模型与专业工程应用之间的巨大断层。即便是在表现最好的模型组合中，任务成功率也仅停留在 17.5% 至 25.0% 的低位，远落后于人类专家的 87.0%。

从失败分布的热力图可以看出一个极富启示性的趋势：**随着模型底层感知与基础指令跟随能力的进化，错误并没有直接消失，而是呈现“向后迁移”（Shift Downstream）的规律。**

对于性能较弱的基线模型，绝大多数失败集中在“早期执行崩溃”阶段：模型无法正确定位 FreeCAD 复杂的工具栏图标，频繁出现由于误操作导致的弹窗卡死、死循环拖拽，甚至无法正常保存出合规的 `.FCStd` 归档文件。

相比之下，代表当前最强水平的多模态 Agent（如测试中的 GPT 系列与 Claude Opus 系列高阶变体）已经基本攻克了界面的初始导航难题。它们能够顺畅地点击菜单、切换工作台、调出草图环境并保存项目。然而，这些顶尖模型的主要失败原因，迅速转向了更为隐蔽和致命的**工程语义缺陷**：

- **错误的文档拓扑结构（Wrong Document Structure）**：模型在不属于活动实体的层级中创建几何体，或者打断了特征之间的父子引用依赖；

- **几何与尺寸漂移（Geometry / Constraint Failures）**：在草图绘制中，模型虽然“画”出了一条封闭线框，但关键端点没有施加重合约束，导致在实体拉伸时生成非流形（Non-manifold）几何或无法解算的病态特征；

- **建造逻辑违规（Modeling Process Errors）**：颠倒了特征生成的次序，例如在未生成基体前尝试构建依赖于特定面的倒角，导致几何内核抛出无拓扑实体的底层错误。

这意味着，当前的通用多模态 GUI 智能体在“认出按钮”和“点击元素”层面已经取得了显著进展，但在“理解专业几何约束求解器状态”与“维持长程因果一致性”方面，依然处于非常初级的阶段。

<img src="/images/2609.16251/terminal_ablation_failure_heatmap.webp" alt="终端代码模式与 GUI 交互模式在失败原因上的消融对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 关键消融：为什么命令行写脚本取代不了真实的 GUI？

在 CAD 自动化领域，一直存在一种直觉性假设：既然大模型擅长写代码，为什么不直接让它在后台生成 Python 脚本控制 CAD 内核，而偏要通过“慢且容易点错”的图形界面（GUI）来操作？

针对这一关键疑问，研究团队在涵盖全部分布的 50 个任务子集上进行了深入的消融对比实验（Terminal-based vs. GUI Computer-Use），结果直击要害。

实验表明，让 Agent 在终端中直接运行 Python 代码生成 CAD 模型，整体成功率并未取得预期中的显著优势，反而引发了大量的“文档结构严重错误（Wrong Document Structure）”。当模型通过纯代码构建零件时，倾向于使用底层几何布尔运算（如生成若干个离散实体后做 Union 并集），而不是像工程界面那样建立具有清晰历史记录的“草图-参数-拉伸-特征”树。

这种生成的最终结果，往往只是一个退化了的、类似扁平 STEP 文件的死实体。后续工程师在 FreeCAD 中重新打开这个项目时，根本无法找到任何可以双击修改的草图尺寸，也无法编辑任何特征参数。这种产物在现代数字化制造与协同工程中几乎没有实际流转价值。

这项消融实验深刻证明了研究的核心论断：**对于参数化机械 CAD 而言，界面的交互式建模历史正是产物不可分割的一部分。** 能够感知上下文状态、能够即时响应几何视图反馈的计算机使用智能体，才是通往工业数字化助理的正确架构。

### 总结与展望

CADWorld 的诞生为大模型在工业工程领域的评测树立了全新的标杆。它不仅将评价标准从“生成代码能否运行”、“生成的图片像不像”拉回到了严肃的“工程成果是否可测量、可编辑、可制造”，更用详尽的实验数据指出了通用计算机使用智能体在进入工业软件时所面临的核心技术瓶颈。

这项研究表明，要让 Agent 真正成为合格的“数字工程师”，未来的研究必须跳出简单的网页抓取或日常办公交互范式，必须深入探索能够理解三维拓扑变化、能够对几何求解器反馈进行闭环反思，并在数十步高密度约束下保持目标不漂移的全新交互推理架构。在智能体真正能够熟练驾驭 CAD 软件之前，距离其在高端制造与实体工程中实现全面落地的愿景，依然有着关键的一大步需要跨越。
