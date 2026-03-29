---
layout: default
title: "MemEvolve: Meta-Evolution of Agent Memory Systems"
---

## 让Agent自己“进化”大脑？MemEvolve实现17%性能跃升与跨模型泛化

<img src="/images/2512.18746v1/A__title.jpg" alt="" style="width:90%; max-width:700px; margin:auto; display:block;">

**如果说现在的AI Agent像是一个勤奋的学生，那么它的“记忆系统”就是它的笔记本。**

> ArXiv URL：http://arxiv.org/abs/2512.18746v1

目前的Agent虽然能把做过的题（轨迹）、学到的技巧（经验）记在笔记本上，但无论遇到数学题还是语文题，它都死板地用同一种格式记笔记。这显然不够聪明——真正的学霸，会根据科目不同，调整记笔记的策略：数学题整理公式模板，语文题摘抄金句。

今天要解读的这篇论文 **MemEvolve**，就是要解决这个问题。它提出了一种让Agent不仅能积累经验，还能**自动进化“记笔记方式”（内存架构）**的元进化框架。

### 核心痛点：静态记忆的局限性

目前的自主进化Agent（Self-evolving Agents）大多依赖**人工设计的静态记忆架构**。

- **早期做法**：简单存储原始轨迹（Raw Trajectories），做Few-shot Prompting。

- **进阶做法**：抽象成文本形式的Tips、Shortcuts，或者结构化的工具库（APIs）。

**问题在于**：没有一种通用的记忆架构能通吃所有任务。

- 适合Web浏览的记忆系统（提取API），可能完全不适合数学推理。

- 适合推理的反思型记忆（Self-critique），在写代码时可能效率极低。

如果记忆系统本身是静态的，Agent就无法真正适应多变的任务环境。这就像一个学生虽然在不断刷题，但他永远只会死记硬背，而不会总结归纳方法论。

### MemEvolve：双重进化引擎

为了打破这一僵局，作者提出了 **MemEvolve**，这是一个**元进化（Meta-Evolutionary）框架**。它的核心思想是**双层优化（Bilevel Optimization）**，就像给Agent装了两个进化的轮子：

1.  **内层循环（第一阶进化）**：

    Agent在固定的记忆系统下，通过与环境交互，不断填充和更新具体的经验（Experience）。这是传统的“积累知识”。

2.  **外层循环（第二阶进化 - 核心创新）**：

    系统会根据Agent的表现，**自动修改记忆系统的架构本身**。这是“优化学习方法”。

为了让这种“修改架构”变得可行，作者将记忆系统解耦为一个模块化的设计空间：

- **♣ Encode（编码）**：如何感知和格式化经验？

- **♦ Store（存储）**：如何保存信息？

- **♥ Retrieve（检索）**：如何根据上下文召回记忆？

- **♠ Manage（管理）**：如何合并新旧知识或遗忘无用信息？

MemEvolve通过**“诊断-设计”（Diagnose-and-Design）**机制，利用大模型（如GPT-5-Mini）作为元优化器，根据Agent在任务中的表现（成功率、成本、延迟），自动重写上述四个模块的代码，从而进化出更强的记忆系统。

### EvolveLab：统一的实验场

为了验证这一想法，作者还开源了 **EvolveLab**。这是一个标准化的代码库，复现了12种代表性的记忆系统（如ExpeL, Agent Workflow Memory等），并将它们统一到上述的模块化空间中。这不仅为MemEvolve提供了进化的“基因库”，也为社区提供了一个公平的竞技场。

### 实验结果：惊人的泛化能力

MemEvolve的效果如何？作者在GAIA、WebWalkerQA等四个高难度Benchmark上进行了测试，结果非常亮眼：

1.  **性能显著提升**：

    在集成到SmolAgent和Flash-Searcher等框架后，性能提升高达 **17.06%**。

2.  **强大的跨任务与跨模型泛化**：

    这是最令人惊讶的一点。

    - **跨任务**：在TaskCraft任务上进化出来的记忆架构，直接拿去跑完全没见过的WebWalkerQA任务，依然能带来性能提升。

    - **跨模型**：用GPT-5-Mini进化出来的架构，直接套用到 **DeepSeek V3.2** 和 **Kimi K2** 上，依然有效！例如，Kimi K2在WebWalkerQA上的表现提升了17%以上。

    - **跨框架**：在一种Agent框架上进化出的记忆，迁移到另一种截然不同的框架上同样有效。

![MemEvolve Performance](images/page_1_Figure_0.jpg)

*图：MemEvolve与几种流行的自进化Agent记忆系统在不同基准上的对比。可以看到MemEvolve（红色）在各项指标上均处于领先地位。*

### 进化出的记忆长什么样？

MemEvolve自动进化出的记忆系统（如文中提到的 *Lightweight*, *Riva*, *Cerebra*）展现出了人类设计的特征：

- **分层抽象**：在任务规划阶段提供宏观指导，在执行阶段提供具体的工具使用建议。

- **预判能力**：甚至能预测目标信息可能出现在图片中，指导Agent去截图。

### 总结

MemEvolve 告诉我们要从“授人以鱼”（给Agent具体的经验）转向“授人以渔”（教Agent如何构建适合自己的记忆系统）。这种**元进化**的思路，让Agent不再是被动地记录者，而是成为了主动优化自身认知架构的智能体。

对于开发者而言，EvolveLab的开源也是一大福音，以后我们在设计Agent记忆模块时，或许可以直接让AI帮我们“写”一个最适合当前任务的架构了。