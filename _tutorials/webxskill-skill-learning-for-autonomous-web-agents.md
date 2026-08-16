---
layout: default
title: "微软提出WebXSkill：赋予网页Agent“双模”技能，成功率最高涨12.9%"
description: "当前，由大语言模型（LargeLanguageModels,LLMs）驱动的自主网页智能体正试图接管复杂的浏览器工作流。但它们常常因为无法留存和复用交互知识，被迫在每次遇到熟悉任务时从零开始推理，导致极高的错误率。"
arxiv_id: "2604.13318"
topics:
  - "AI Agent"
tags:
  - "URL-based skill graph"
  - "WebXSkill"
  - "autonomous web agents"
  - "executable skills"
  - "grounded mode"
  - "guided mode"
related_tutorials:
  - "training-task-reasoning-llm-agents-for-multi-turn-task-planning-via-single-turn-"
  - "learning-on-the-job-an-experience-driven-self-evolving-agent-for-long-horizon-ta"
  - "skillos-learning-skill-curation-for-self-evolving-agents"
  - "skillcomposer-learning-to-evolve-agent-skills-for-specification-and-generalization"
---

<p class="paper-original-title" lang="en" style="font-size:1rem; line-height:1.5; color:var(--global-text-color-light, #6c757d); margin:-0.5rem 0 1.5rem;">WebXSkill: Skill Learning for Autonomous Web Agents</p>

当前，由**大语言模型**（**Large Language Models, LLMs**）驱动的自主网页智能体正试图接管复杂的浏览器工作流。但它们常常因为无法留存和复用交互知识，被迫在每次遇到熟悉任务时从零开始推理，导致极高的错误率。

> **ArXiv URL**：http://arxiv.org/abs/2604.13318v1

为了打破这一瓶颈，研究者曾引入“技能”概念，但这陷入了所谓的“执行鸿沟”：文本类技能提供自然语言指导却无法直接执行；代码类技能虽能执行，但对智能体而言是无法理解的“黑盒”，中途出错便彻底宕机。

近期，来自微软和北卡罗来纳大学的研究团队提出了全新框架 WebXSkill。该研究通过引入带有步骤级自然语言指导的“可执行技能”，彻底弥合了这一鸿沟。在 WebArena 和 WebVoyager 测试中，WebXSkill 将任务成功率分别提升了最高9.8和12.9个百分点。

### 弥合执行鸿沟：WebXSkill的核心架构

给定用户任务 $q$，网页智能体需要在每一步接收包含网页文本和截图的观察 $o_{t}$，并从原始动作空间 $\mathcal{A}_{\text{prim}}$ 中生成动作 $a_{t}$。

为了避免智能体每次都从头规划，WebXSkill 构建了一个三阶段的系统流水线，为其配备了结构化的可执行知识库。

<img src="/images/2604.13318v1/x2.jpg" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">

#### 第一阶段：轨迹抽象与技能精炼

与依赖昂贵在线探索的方法不同，WebXSkill 直接从低成本的合成智能体轨迹中挖掘可复用的动作序列。

系统利用大语言模型识别出连贯的操作，并将具体的动作值抽象为带类型的参数。例如，将特定的搜索动作替换为带有 `query: str` 参数的通用搜索技能。

更为关键的是，每个动作步骤都被标注了自然语言的意图指导。在入库前，系统会结合规则与嵌入向量进行在线去重，并在真实环境中验证其可执行性，剔除无效技能，保证技能库的高效与紧凑。

#### 第二阶段：基于URL的上下文图谱组织

提取出的海量技能需要被精准检索。传统的扁平化API库会在每一步将所有技能暴露给智能体，增加决策噪音。

WebXSkill 则创新性地构建了基于页面结构的技能图谱。图谱节点由泛化的URL模式（如 `shopping/catalogsearch/*`）构成，因为URL比极易变动的HTML节点更稳定。

推理时，智能体会根据当前页面的URL匹配对应的图谱节点，并进一步校验目标UI元素是否存在。这种严谨的上下文感知检索确保了最终呈现的技能绝对与当前页面状态高度相关且可执行。

### 核心创新：双模部署的“导航仪”机制

现有的代码级技能大多是“黑盒”调用。为了理解 WebXSkill 的突破，我们可以引入一个“智能导航仪”的比喻。

以往的代码技能就像是一辆车窗全黑的自动驾驶汽车，智能体输入目的地后只能被动等待。一旦前方网页结构改变导致“封路”，汽车停摆，智能体完全无法接管。

WebXSkill 则提供了一个透明且带有逐向播报的导航仪，并开放了两种互补的部署模式：

#### Grounded 模式：全自动的高效执行

在此模式下，技能作为原子工具扩展到智能体的动作空间中。伴随的自然语言指导仍作为规划辅助保留。

当智能体调用技能（如 `search_product(query="laptop")`）时，底层运行环境会自动匹配DOM元素并顺序分发动作。

这就好比开启了高级自动驾驶，将多步繁琐操作压缩为单次工具调用，大幅提升了操作效率。但这种模式对智能体应对突发中断的推理能力要求极高。

#### Guided 模式：保留自主性的逐步引导

这种模式将技能转化为高级的步骤指导，智能体仍使用原生的动作空间，一步步跟随指导操作。

如果某一步因为页面更新而失效，智能体能利用其对整体意图的理解，主动规划替代动作。

这就像你亲自开车，导航仪在旁提示“前方点击搜索框”。即便搜索框换了位置，你凭借肉眼观察依然能找到它。这种模式以牺牲少许效率为代价，换取了极大的容错率。

### 深度实验与性能表现

研究团队在极具挑战的 WebArena 以及基于真实网站的 WebVoyager 基准上进行了广泛评估。

<img src="/images/2604.13318v1/x3.jpg" alt="Refer to caption" style="width:90%; max-width:700px; margin:auto; display:block;">

首先，从技能库质量来看（如图3所示），WebXSkill 提取的技能覆盖了全部十个功能类别且分布均衡。相比之下，SkillWeaver 超60%的技能局限在信息检索上，而 WALT 仅提供了极其稀疏的数十个技能。

在效率与使用率方面，Grounded 模式的技能调用率（16.5%）和使用率（70.8%）几乎是对比基线的两倍，且平均仅需9.3步即可完成任务，显著低于基线系统。

实验揭示了一个重要的工程现象：最佳部署模式高度依赖于底层模型能力。
对于超强模型 GPT-5，Grounded 模式略占优势，因为它足以应对自动执行偶发的失败。
但对于开源模型 Qwen，Guided 模式（53.9%）明显优于 Grounded 模式（48.7%），证明了步骤级引导对较弱模型的巨大保护作用。

此外，在技能迁移测试中，仅使用 WebArena 提取的技能去跑真实的 WebVoyager 环境，Guided 模式依然取得了极佳的成绩。这突显了步骤级指令在跨域环境下的强大泛化能力。

### 局限性诊断与工程启示

为了探究系统瓶颈，研究团队对失败案例进行了深度归因诊断。

<img src="/images/2604.13318v1/x4.jpg" alt="Refer to caption" style="width:90%; max-width:700px; margin:auto; display:block;">

数据揭示了令人意外的现象：绝大多数失败并非源于技能本身的崩溃。
1. **技能后推理失败占比最大（38%）**：技能已经完美执行并到达了目标页面，但智能体在最后一步提取答案时报错（例如数错商品数量）。
2. **智能体无视技能（36%）**：智能体完全跳过了系统提供的有效技能，强行使用原生动作瞎试导致失败。
3. **“良性”中断**：在 CMS 站点中，导致技能执行失败的主要原因仅仅是“无法关闭非交互性的导航弹窗”，而80%遇到此问题的任务最终其实是成功的。

**工程启示**：
这些诊断证明，当前的技能抽象与执行框架已经足够健壮。真正制约任务成功率的瓶颈，已经从“操作执行”转移到了智能体的高级决策层。

未来，提升大语言模型对复杂多模态上下文的理解能力，以及优化最终的答案提取逻辑，将是让网页智能体真正落地生产环境的关键所在。WebXSkill 为这一愿景提供了一个坚实的基础范式。
