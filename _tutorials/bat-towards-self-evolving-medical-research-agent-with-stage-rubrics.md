---
layout: default
title: "NVIDIA提出BaT：以评测为师分阶段强化，9B模型超越Claude Opus"
description: "针对这一困境，来自 NVIDIA 与加利福尼亚大学圣克鲁兹分校（UC Santa Cruz）的研究团队提出了 BaT（Benchmark-as-Teacher） 。该方法颠覆了传统基准测试只作为静态考卷的固有定位，将其转变为指导智能体自我进化的动态基础设施。"
arxiv_id: "2608.16211"
paper_published: "2026-08-17"
published_at: "2026-09-07T13:15:08.344824+08:00"
topics:
  - "AI Agent"
  - "AI评测"
tags:
  - "AutoMedBench-Lite"
  - "BaT"
  - "BiCuRL"
  - "GRPO"
  - "Stage Bank"
  - "content-isolated training states"
related_tutorials:
  - "dr-tulu-reinforcement-learning-with-evolving-rubrics-for-deep-research"
  - "beyond-two-stage-training-cooperative-sft-and-rl-for-llm-reasoning"
  - "deep-self-evolving-reasoning"
  - "the-two-stage-decision-sampling-hypothesis-understanding-the-emergence-of-self-r"
---

<p class="paper-original-title" lang="en">BaT: Towards Self-Evolving Medical Research Agent with Stage Rubrics</p>

<img src="/images/2608.16211v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在严肃的专业工作流中，构建长程智能体（Long-horizon Agent）始终面临一个尴尬的死结：任务链条极长，中间涉及规划、环境配置、数据校验、模型推理到最终成果交付等多个连续阶段，任何一环的微小失误都会导致全盘溃败。与此同时，真实环境中的专家轨迹不仅极度稀缺，而且受制于隐私合规难以公开共享。更令人沮丧的是，现有的强化学习后训练方案往往把整段数十轮的交互压缩成一个末端标量奖励，智能体即便失败了，也根本不知道自己究竟“死”在哪个步骤。

> ArXiv URL：https://arxiv.org/abs/2608.16211v1

针对这一困境，来自 NVIDIA 与加利福尼亚大学圣克鲁兹分校（UC Santa Cruz）的研究团队提出了 **BaT（Benchmark-as-Teacher）**。该方法颠覆了传统基准测试只作为静态考卷的固有定位，将其转变为指导智能体自我进化的动态基础设施。依托包含环境沙盒生成的离线数据管道 **Stage Bank**，以及内外双层课程强化学习方法 **BiCuRL（Bilevel Curriculum Reinforcement Learning）**，BaT 在绝对隔离评测集具体内容的前提下，仅凭评测反馈的阶段诊断得分驱动模型靶向强化。在极具挑战的医疗科研智能体基准 AutoMedBench-Lite 上，参数量仅为 9B 的 BaT-9B Agent 取得了 79.6 的 Overall 得分，一举超越了由 Claude Code 驱动的闭源旗舰 Claude Opus 4.6（77.5 分）。这一结果不仅展现了结构化反馈在长程强化学习中的威力，更为敏感领域小参数模型的本地化落地开辟了全新路径。

<img src="/images/2608.16211v1/fig_method.webp" alt="BaT 系统概览与 BiCuRL 循环机制" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 长程医疗任务的信用分配困局与评测浪费

医疗影像与科研分析智能体与普通的对话式 AI 截然不同。一个标准的科研任务通常要求智能体在复杂的交互环境中连续操作数十步：首先要通读需求并完成规划（Plan），接着配置计算环境与依赖工具（Setup），随后加载并清洗多模态数据（Validate），之后编写脚本执行推理并分析误差（Inference），最后打包交付符合医学规范的标准文件（Submit）。在 AutoMedBench-Lite 等基准中，智能体单次运行平均需要进行 33 轮交互。

若按照标准的多轮强化学习方案（如直接套用针对单一最终结果的 GRPO），环境通常只能在最终交互结束时给出一个“成功”或“失败”的二元奖励。当一个智能体在第 2 步的环境依赖配置中漏装了某个包，导致后续第 28 步的推理脚本报错崩溃时，单标量奖励算法会把整条长达 30 多轮的轨迹全部打上低分负反馈。这种极其粗糙的信用分配（Credit Assignment）使得模型极难学到底层归因，训练方差急剧放大，收敛异常缓慢。

讽刺的是，当前的结构化评估基准（例如 MedAgentBench、HealthAgentBench 与 AutoMedBench）本身就具备精细的阶段诊断能力。这些基准完全知晓智能体在规划、设置、验证、推理还是提交阶段丢了分，甚至拥有细粒度的阶段规则（Rubrics）。然而在以往的标准研发流程中，这部分极具价值的诊断信号在评估结束的瞬间就被直接丢弃了。研究团队正是抓住这个盲区，提出核心假设：能否在完全不泄露具体测试题目的前提下，利用结构化基准的阶段诊断信号，将其转化为指导智能体递归自进化的专属导师？

### 解耦数据与策略：Stage Bank 的沙盒生成工厂

将基准测试引入训练循环，最致命的隐患就是数据泄露与过拟合。为了彻底阻断评测内容对训练空间的污染，BaT 将整个体系严格拆分为外部异步数据管道与内部策略优化循环。负责数据生产的组件被称为 Stage Bank，它完全独立于策略更新循环之外运行。

Stage Bank 的核心在于从公开的工作流描述与阶段契约中抽象出任务模式，由高质量的教师模型批量合成虚构的医疗影像分析任务。合成过程并不凭空捏造文本，而是严格按照 Self-Instruct 模式生成完整的多轮交互轨迹，并构建出真实可运行的环境沙盒。为了保证绝对的内容隔离，Stage Bank 引入了一套严密的防泄漏预检机制（Leakage Preflight），一旦合成轨迹中出现保留评测集的任务标识符、本地系统路径、报告文本、特定追踪记录或测试集答案，该样本就会在进入训练池之前被硬性剔除。

经检验合格后，Stage Bank 会把可执行的数据状态分流为三种具有不同功能的训练沙盒池：

1. **定向阶段沙盒（$S_{\text{target}}$）**：专门从当前智能体表现最薄弱的阶段边界切入，配备该阶段特有的目标描述、恢复步骤与评分细则，用于高强度的靶向攻坚训练。

2. **混合阶段沙盒（$S_{\text{mix}}$）**：随机采样其他各个非靶向阶段的环境状态，用于在强化弱项的同时维持智能体在其余阶段的操作记忆。

3. **端到端沙盒（$E2E$）**：将五个阶段完整串联的中小型工作流沙盒，支持多轮推演，保障全局长程协同感。

此外，在进行强化学习之前，Stage Bank 还会将合格的教师轨迹切分为单步切片，保留完整的历史上下文与工具观测，仅对选定的专家动作计算损失，以此完成初始策略模型 $\theta_0$ 的监督微调（SFT）冷启动。

<img src="/images/2608.16211v1/fig_data_state_bank.webp" alt="Stage Bank 数据沙盒构建与三种混合池" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 双层课程强化学习：BiCuRL 的内外循环闭环

有了隔离的内容沙盒，如何根据评测诊断自适应调整训练方向？这就是 BiCuRL（Bilevel Curriculum Reinforcement Learning）的核心职责。BiCuRL 形式化为一个双层优化问题：




{% raw %}$$ \max_{\mathbf{q}\in\mathcal{Q}_{\mathrm{SB}}} \mathcal{M}\left(\hat{\theta}(\mathbf{q})\right), \quad \text{s.t.} \quad \hat{\theta}(\mathbf{q})\in\arg\max_{\theta}\mathcal{J}_{\mathrm{GRPO}}(\theta;\mathbf{q}) $${% endraw %}



其中外层负责在 Stage Bank 的混合分布族 $\mathcal{Q}_{\mathrm{SB}}$ 中寻找能最大化验证集整体性能 $\mathcal{M}$ 的课程配置 $\mathbf{q}$；内层则在给定的沙盒配比下，通过强化学习优化策略参数 $\theta$。

在内层循环中，智能体在当前采样的沙盒状态 $p_i$ 中展开 $K$ 次采样 Rollout，生成轨迹 $y_{i,k}$ 及产出文件 $x_{i,k}$。这里 BiCuRL 摒弃了容易出现评分漂移的自由式大模型打分，而是采用严格的二元规则验证器（LLM Rubric Verifier）。验证器逐项核对预设的阶段规则 $\ell \in c_i$，判定其是否达成：$v_{i,k,\ell} \in \{0, 1\}$。同时，验证器还会输出一个证据完备度系数 $\eta_{i,k} \in [0, 1]$，用来惩罚那些“宣称任务成功但并未生成实体证据”的幻觉行为。单次 Rollout 的复合奖励定义为规则通过率与完备度的乘积：




{% raw %}$$ r_{i,k} = \eta_{i,k} \frac{1}{\lvert c_i \rvert} \sum_{\ell \in c_i} v_{i,k,\ell} $${% endraw %}



随后，该奖励在共享相同环境状态与规则契约的 $K$ 个组内候选中进行标准化，计算相对优势 $A_{i,k}$，并通过 GRPO（Group-Relative Policy Optimization）更新策略，从而在无需额外训练 Critic 价值网络的前提下实现稳定的梯度反向传播。

而在外层循环中，系统根据评测端返回的阶段性诊断得分，自动定位得分最低的弱项阶段 $s_r^\star$，并动态生成本轮的训练配比：




{% raw %}$$ q_r(z) = \rho_{\mathrm{target}}\,q_{\mathrm{target}}(z\mid s_r^\star) + \rho_{\mathrm{mix}}\,q_{\mathrm{mix}}(z\mid s_r^\star) + \rho_{\mathrm{E2E}}\,q_{\mathrm{E2E}}(z) $${% endraw %}



为了防止模型在多轮强化学习中出现策略崩溃或灾难性遗忘，BiCuRL 引入了严格的检查点回退保护逻辑。如果候选模型连续多轮未能取得性能突破，或者新策略与历史最优策略 $\theta^\star$ 的 KL 散度超出了预设阈值 $\tau$，系统将强制回退到历史最优权重重新调整。这种“外层控方向、设防线，内层抓执行、精核销”的双层闭环，使得训练过程既保持了强烈的目标导向，又规避了多轮长程强化学习极其脆弱的崩盘风险。

<img src="/images/2608.16211v1/fig_teaser_figure.webp" alt="三类医疗基准上不同体系的综合对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 从翻倍基线到硬刚旗舰：AutoMedBench 上的性能跃迁

BaT 的实际进化能力在实验中得到了极其亮眼的验证。实验选用开源社区广泛采用的 Qwen3.5-4B 与 Qwen3.5-9B 作为基础 Instruct 骨干，并在相同的 OpenHands 执行沙盒框架下进行了严格评测。为了消除长程交互随机波动带来的统计误差，评测覆盖了 7 条长程任务赛道，每条赛道重复运行 10 次，总计 70 次独立实验，取两次平均的统计值。

<img src="/images/2608.16211v1/fig_baseline_comparison.webp" alt="基础模型在不同后训练阶段下的得分跃迁" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

纯策略层面的后训练结果展现出阶段强化所带来的爆发力。在 AutoMedBench-Lite 上，原始的 Qwen3.5-4B Instruct 基线 Overall 评分仅有 6.1 分，经过 BaT 循环强化后，BaT-4B 直接跃升至 22.9 分；而对于 9B 规模的模型，原始基线得分仅为 19.9 分，常规的端到端单奖励 GRPO 训练将其推升至 31.9 分，而完整的 BaT-9B 则一举飙升到了 53.4 分，相比初始基线提升了近三倍，相比常规 GRPO 带来了 21.5 个百分点的显著净增益。

当策略模型搭配固化的 OpenHands 运行环境组成完整的智能体系统后，BaT-9B Agent 在 AutoMedBench-Lite 上的 Overall 得分最终锁定了 79.6 分的高位。作为横向参照，配备 Claude Code 环境的 Claude Opus 4.6 最终得分为 77.5 分，这意味着仅仅 9B 规模的开源小模型在经过高质量的阶段强化后，在垂直长程科研赛道上正面击败了参数量远超自身的世界顶尖商用模型。

在另外两项极具代表性的医学与放射科评测集上，BaT 同样展现出不俗的攻坚能力：在考察放射学影像调阅、标注与报告生成的 ABRA 平台（包含 655 个 DICOM 处理任务）中，BaT-9B 取得了 70.6 分，仅比 GPT-5.5 + Codex 组合（79.9 分）落后 9.3 分；在考察跨 17 个专科纯文本医学推理的 MedXpertQA-Text 上，BaT-9B 达到 50.2 分，虽然与超大模型 Gemini 3.1 Pro（65.0 分）仍有差距，但已全方位大幅超越未经特定阶段优化的通用开源模型。

### 拆解 Stage Bank：为什么单靠端到端不行？

为了探究系统各个模块的必要性，研究团队进行了一系列消融实验。针对 Stage Bank 中 $S_{\text{target}}$、$S_{\text{mix}}$ 与 E2E 三个数据池的配比拆解，揭示了长程智能体训练的深刻机理。

<img src="/images/2608.16211v1/fig_ablation_study.webp" alt="Stage Bank 不同数据池组合消融实验" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

实验结果清晰地表明，任何单一或残缺的数据配比都会导致性能出现断崖式下跌。在 9B 模型的消融对比中：

- 采用全配置的三池混合策略时，模型斩获了最高的 53.4 分。

- 若完全移除阶段切片，仅依靠完整的端到端长轨迹沙盒（E2E alone）进行训练，最终得分仅有 31.9 分，整整暴跌了 21.5 分。

- 若采取“三缺一”的剥离实验，无论剔除哪一个沙盒池，模型的得分点估计相较全集都会至少下滑 26 分。

这组对比深刻说明：在长程任务中，单纯依靠从头跑到尾的端到端仿真，模型在错误分配与探索效率上面临巨大阻力；而如果只进行孤立的阶段切片靶向特训，智能体又会丧失在长程上下文中的跨阶段连贯性。只有把靶向弱项突击（$S_{\text{target}}$）、全流程串联（E2E）与其余阶段的能力保持（$S_{\text{mix}}$）以有机比例结合，才能形成真正稳固的智能体进化动力。

<img src="/images/2608.16211v1/fig_local_llm_comparison.webp" alt="小参数量本地开源模型在医疗智能体评测中的分布对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 本地化部署价值与跨领域能力保留的尺度效应

在涉及患者敏感隐私数据、严苛伦理合规的医疗场景中，将数据打包上传给第三方闭源 API 往往不可行。具备高安全性、支持离线部署的小参数本地模型（Tiny Local LLM，通常定义在 12B 参数以下）是医疗机构最现实的落地形态。在相同 OpenHands 环境下对 12B 以下主流开源模型进行的系统性对比显示，BaT-9B 凭借 53.4 分牢牢占据榜首，其性能甚至是排在第二位的 Gemma 12B 的两倍以上；而 BaT-4B 虽仅有 4B 参数，其实际表现却超越了绝大多数 7B 至 9B 级别的通用竞品。

然而，高度垂直的智能体强化是否会导致模型在通用推理上出现灾难性遗忘？研究团队在八个外部通用基准上检验了迁移效果，观察到了非常明确的“尺度依赖（Scale-Dependent）”特征。

对于 4B 这样较小容量的模型，专注垂直医疗阶段强化的代价十分显著，其在所有 8 个外部通用任务（包括数学推理与常规代码基准）上的得分均低于原始基线，表明小模型在吸收高强度领域知识时极易发生能力挤压。

而在 9B 参数规模上，情况发生了戏剧性的转折：BaT-9B 在通用短程推理测试中仅微弱下滑了 3.4 到 5.8 分，但在长程软件与系统任务中却表现出意想不到的反哺效应。例如在真实软件工程评测 SWE-bench Verified、命令行环境操作 Terminal Bench 2.0 以及多轮交互基准 $\tau^2$-Bench（提升 5.4 分）中，BaT-9B 的性能全面反超了原始的通用 Instruct 基线。这表明，当模型参数容量达到一定阈值后，通过阶段沙盒与严密细则培养出的严谨环境探索、工具调试以及代码容错习惯，具备向其他长程通用代理场景泛化迁移的通用价值。

### 局限性与智能体自我进化路线的未来演进

尽管 BaT 展现了令人振奋的自进化潜力，但在审视这一工作时，依然需要客观看待其当前的边界：

首先，BaT 实际上重塑了基准测试的角色定位。由于 AutoMedBench-Lite 的聚合诊断信号直接参与了训练课程的调整，该基准在严格意义上已从“纯净的第三方盲测终点线”转变成了一个“闭环诊断教练”。虽然实验通过沙盒生成、阶段技能抽象与预检拦截实现了严格的内容隔离（没有测试题直接参与微调），但在严格的科学评估范式下，未来仍需在一个完全独立、从未用于反馈循环的盲测集上进一步实施语义泄露审计。

其次，BaT 的成功高度依赖于任务本身是否具有“良定义的阶段边界（Stage Contracts）”与“客观可验证的输出证据”。在医疗影像分析、生物信息流水线或标准软件开发等结构化场景中，阶段与规则是天然存在的；但若要将其推向更为模糊、主观性强的开放式长程决策任务，如何低成本生成保真的阶段沙盒与二元规则核验器，将成为下一阶段必须攻克的工程难题。

不可否认，BaT 为业界提供了一种极具启发性的思路：在数据飞轮与模型演进逐渐陷入轨迹匮乏的瓶颈期，传统意义上单向打分的评测基准不应只是模型迭代的终点。通过合理的解耦与规则化设计，让评测扮演“教练”，在沙盒中对智能体进行分阶段、抓凭证的靶向强化，不仅能够大幅摊薄长程强化学习的试错成本，更能让原本受制于参数规模的小型开源模型，在极其严苛的专业长程赛道上爆发出匹敌商用巨无霸的惊人能量。
