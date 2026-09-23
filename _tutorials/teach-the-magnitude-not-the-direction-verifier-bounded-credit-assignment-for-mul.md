---
layout: default
title: "CrEST：只教幅度不教方向！分层信用分配突破多轮 Agent 训练瓶颈"
description: "来自浙江大学、蚂蚁集团（Inclusion AI / AWorld 团队）、上海创新研究院、西湖大学与南京大学的研究团队提出了全新框架 CrEST （Hierarchical Credit Assignment via Entropy-Gated Self-Teacher）。"
arxiv_id: "2608.13179"
paper_published: "2026-08-13"
published_at: "2026-09-23T13:15:08.398930+08:00"
topics:
  - "AI Agent"
  - "模型训练"
tags:
  - "BFCL V3"
  - "CrEST"
  - "RLVR"
  - "WildToolBench"
  - "entropy-gated self-teacher"
  - "multi-turn tool-use agents"
related_tutorials:
  - "dual-lora-enhancing-lora-with-magnitude-and-direction-updates"
  - "skillrise-agentic-reinforcement-learning-for-cross-task-skill-evolution"
  - "when-history-lies-evaluating-and-improving-tool-use-under-misleading-multi-turn-"
  - "music-multi-step-instruction-contrast-for-multi-turn-reward-models"
---

<p class="paper-original-title" lang="en">Teach the Magnitude, Not the Direction: Verifier-Bounded Credit Assignment for Multi-Turn Multi-step LLM Agents</p>

<img src="/images/2608.13179v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在让大模型学会使用工具、自主完成复杂任务的过程中，强化学习（RL）尤其是基于可验证奖励的强化学习（RLVR），已经成为主流的后训练手段。然而，当应用场景从单轮问答延伸至真实的多轮、多步交互（Multi-Turn Multi-step）环境时，现有的后训练机制迅速暴露出致命缺陷。在多轮交互中，模型需要在长达数轮的会话里持续调用外部工具并根据反馈调整策略，不同轮次间的结果往往悲喜交加：第一轮精准命中，第二轮却逻辑错乱导致调用崩溃。

> ArXiv URL：https://arxiv.org/abs/2608.13179v1

标准的强化学习算法（如 GRPO）通常只将整条轨迹的最终结果打包成单一的标量奖励，粗暴地反向广播给整条序列中的每一个 Token。这种“一荣俱荣、一损俱损”的做法，使得失败轮次中的优质推理被无辜打压，而成功轮次之后胡乱生成的废话却享受了正向激励，导致跨轮次的信用分配（Credit Assignment）严重失真。在线策略蒸馏（On-policy Distillation, OPD）或引入特权信息的自蒸馏（OPSD）虽然能提供密集的 Token 级监督，但往往受制于教师模型本身的性能上限（Teacher-bounded），且极易因为梯度过度集中在少数低信息量的格式 Token 上而陷入训练坍塌。

来自浙江大学、蚂蚁集团（Inclusion AI / AWorld 团队）、上海创新研究院、西湖大学与南京大学的研究团队提出了全新框架 **CrEST**（Hierarchical Credit Assignment via Entropy-Gated Self-Teacher）。这项工作的核心哲学非常鲜明：**教师模型不应决定策略更新的方向，而应只负责微调更新的幅度**。通过将更新方向牢牢锚定在环境可验证奖励上，同时利用熵门控的自教师信号细化 Token 级的贡献权重，CrEST 在保持强化学习理论性能上限的同时，彻底解决了多轮多步 Agent 的细粒度信用分配难题。在真实工具调用基准 BFCL V3 和 WildToolBench 上的评测显示，CrEST 在多个模型尺度下均大幅超越现有的强化学习与蒸馏基线，将 Qwen3-4B-Instruct 的平均任务准确率从基座的 22.12% 提升至 52.00%，并在最严苛的多轮会话指标上实现了显著飞跃。

### 多轮多步 Agent 的两层信用分配危机

要理解 CrEST 的设计，首先需要看清多轮交互场景下信用分配究竟断裂在何处。在单轮任务中，模型的所有动作都服务于同一个用户意图，此时整条轨迹的单一奖励虽然带有噪声，但大体逻辑一致。但在多轮会话中，每一轮包含独立的用户指令，每一轮内部又由多步思考、工具调用与环境观察交织而成。这就构成了一个自然的层级结构：粗粒度的“跨轮次”（Inter-turn）与细粒度的“轮次内”（Intra-turn）。

在粗粒度层面，轨迹级奖励造成了严重的跨轮次稀释。假设一个包含两轮的对话，第一轮模型准确调用航班查询工具并给出正确回复，第二轮在预订酒店时因参数缺失而报错。若采用整体验证，整条轨迹往往直接被判为 0 分，第一轮极具价值的工具调用决策被赋予负向优势；若按部分得分进行轨迹平均，第二轮的错误行为又会被动享受正向加权。在业界广泛测试的 WildToolBench 基准上，主流开源模型的会话级准确率长期徘徊在 15% 以下，且随着交互轮数的增加性能发生断崖式下跌，其核心症结就在于此。

在细粒度层面，即单轮内部，即使某一轮次被明确判定为成功，内部每个 Token 的贡献也千差万别。一个包含几十甚至上百个 Token 的工具调用输出中，真正决定成败的往往是函数名（如 `get_flight`）和核心实参（如 `"JFK"`），其余的大量内容则是语法闭包、标点符号或样板模板。常规强化学习将该轮的优势值等额分摊给每个 Token，完全忽视了决策的关键程度。

自蒸馏方案（OPSD）试图通过给学生模型喂入标准答案或特权信息作为“自教师”（Privileged Self-Teacher），利用教师与学生之间的对数几率差（Divergence）来提供密集 Token 监督。但先前的理论与实验均表明，自蒸馏极其脆弱：在不加严苛裁剪的情况下，训练往往在 100 步以内就会因“梯度集中”（Gradient Concentration）而崩溃。更关键的是，纯蒸馏的终点永远受限于教师模型的特权分布，无法突破环境验证器的边界。面对这种“RL 有上限但信号稀疏、蒸馏信号密集但上限被锁死且易崩溃”的两难困境，如何将两者的优势安全地融为一体，构成了该研究最本质的切入点。

### 核心机制：解耦更新方向与幅度

CrEST 的核心突破在于建立了一个结构化的统一策略梯度目标函数。给定提示词 $x$ 与模型采样的轨迹 $y \sim \pi_\theta(\cdot \mid x)$，其优化目标形式如下：




{% raw %}$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)} \left[ \sum_{t} A_t \cdot \log \pi_\theta(y_t \mid y_{<t}, x) \right]$${% endraw %}



与标准方法不同的是，CrEST 将每个 Token 的最终优势值 $A_t$ 正式分解为两项的乘积：




{% raw %}$$A_t = A_{\text{turn}[t]} \cdot \phi_t$${% endraw %}



这里的 $A_{\text{turn}[t]}$ 是轮次级优势值，负责在粗粒度上回答“究竟是哪一轮导致了成败”；而 $\phi_t$ 则是轮内调制因子，负责在细粒度上回答“该轮内部哪些 Token 承担了核心责任”。

#### 轮次级切分：杜绝跨轮次污染

对于一条包含 $K$ 个轮次的交互轨迹，CrEST 并不直接汇总整个会话的最终得分，而是依靠环境验证器针对每一个轮次 $k$ 分别返回验证奖励 $R_k$。在包含 $G$ 个候选采样的组内（以 GRPO 的组相对机制为基础），轮次 $k$ 的优势值计算完全独立进行：




{% raw %}$$A^{(i)}_k = \frac{R^{(i)}_k - \text{mean}(\{R^{(j)}_k\}_{j=1}^G)}{\text{std}(\{R^{(j)}_k\}_{j=1}^G) + \epsilon}$${% endraw %}



该轮次内的所有 Token $t \in \text{turn}_k$ 共同继承这一基础优势值 $A^{(i)}_k$。这意味着，即便某一轨迹在第一轮表现优异、第二轮翻车，系统也能精确地在第一轮赋予正优势、在第二轮赋予负优势，从物理切分上彻底阻断了不同轮次之间的信号干扰。

#### Token 级调制：特权自教师与熵门控的协同

获得了正确的轮次级方向后，下一步是如何调节轮内的 Token 权重。CrEST 让同一个模型在拼接了 Ground Truth 的特权上下文下扮演自教师 $\pi_T$，计算学生分布与教师分布在每个位置的平滑发散度：




{% raw %}$$\Delta_t = \frac{\log \pi_T(y_t \mid h_t^T) - \log \pi_\theta(y_t \mid h_t)}{\tau}$${% endraw %}



其中 $\tau$ 是平滑温度超参数。为了使该信号能安全地缩放梯度，研究者构造了一个带有符号对齐的原始权重：




{% raw %}$$w_t = \text{clip}\left(\exp\left(\text{sign}(A_{\text{turn}[t]}) \cdot \Delta_t\right), 1 - \epsilon, 1 + \epsilon\right)$${% endraw %}



这一构造极其精妙：当轮次优势为正且教师认同该 Token 时，$w_t > 1$；当轮次优势为负且教师同样认为该 Token 糟糕时，同样会放大负向惩罚的力度。

为了彻底消除传统蒸馏的梯度坍塌问题，CrEST 引入了组合双门控机制。首先是**方向门（Direction Gate）**：




{% raw %}$$g^{\text{dir}}_t = \mathbf{1}\left[\text{sign}(A_{\text{turn}[t]}) \cdot \Delta_t > 0\right]$${% endraw %}



一旦教师的倾向与环境验证器的正负方向发生冲突，方向门瞬间闭合（$g^{\text{dir}}_t = 0$），强制关闭任何蒸馏调制，直接退化为纯验证器优势。这就从数学上确立了第一核心性质：**梯度更新的符号完全且永远由环境验证器决定，教师无权篡改优化方向**。这也正是模型能突破教师天花板、保持 Verifier-bounded 的关键所在。

其次是**熵门控（Entropy Gate）**。以往自蒸馏容易崩溃，是因为在生成诸如括号、冒号、空格等低熵格式 Token 时，概率的微小变动会产生极大的比值，导致梯度被毫无意义的语法结构吸干。CrEST 采用学生自身的惊异度（Surprisal）$u_t = -\log \pi_\theta(y_t \mid h_t)$ 作为不确定性的代理指标，经过组内 Z-score 标准化与 Sigmoid 映射：




{% raw %}$$g^{\text{ent}}_t = \sigma\left(\frac{u_t - \mathbb{E}[u]}{\text{std}(u) + \epsilon}\right), \quad m^{\text{ent}}_t = 1 + \rho (2 g^{\text{ent}}_t - 1)$${% endraw %}



在低不确定性的格式 Token 上，$m^{\text{ent}}_t < 1$，大幅削弱其调制力度；在真正需要决策的高不确定性内容 Token（如关键参数）上，$m^{\text{ent}}_t > 1$，梯度权重被有效放大。

最终的有效调节系数由两道门控与全局缩放因子 $\lambda$ 复合而成：




{% raw %}$$\lambda^{\text{eff}}_t = \text{clip}\left(\lambda \cdot g^{\text{dir}}_t \cdot m^{\text{ent}}_t, 0, \lambda\right), \quad \phi_t = 1 + \lambda^{\text{eff}}_t (w_t - 1)$${% endraw %}



由此，$\phi_t$ 被严格限定在 $[1, 1 + \lambda\epsilon]$ 区间内。整套方案中，除了唯一的控制力度参数 $\lambda = 0.3$ 外，其余如温度 $\tau = 2.0$、裁剪边界 $\epsilon = 0.28$、熵范围 $\rho = 0.5$ 均为固定的常数配置，极具工程工程落地友好性。

### 评测结果与长轨迹优势

为了验证分层信用分配在复杂环境中的真实效能，作者在两个极具代表性的多轮多步工具基准上展开了全面评测：一个是学界标准的 Berkeley Function-Calling Leaderboard (BFCL) V3，涵盖基础轮次、缺失函数、缺失参数及长上下文等细分维度；另一个是更贴近工业级真实复杂交互的 WildToolBench，其中包含大量隐式指代、意图漂移及闲聊与任务切换交织的高难度会话。

基础模型选用了参数规模极具代表性的 Qwen3-4B-Instruct 以及具备原生思考能力的 Qwen3-8B。对比基线不仅涵盖了标准轨迹级强化学习 GRPO、步级强化学习 MT-GRPO、基于细粒度过程奖励的 EnvTuning，还包括同族大模型监督的在线蒸馏 OPD 以及特权自蒸馏 OPSD。

在 BFCL V3 多轮测试中，CrEST 展现出了极强的统治力。在 Qwen3-4B-Instruct 上，CrEST 取得了 52.00% 的平均准确率，相比直接使用 GRPO（43.63%）实现了大幅跃升，甚至超越了引入更细粒度奖励的 MT-GRPO（49.25%）和 EnvTuning（47.25%）。在纯蒸馏路线中，即便是用 235B 巨型教师模型在线指导的 OPD，最终也只停留在 44.50%，而容易产生梯度坍塌的 OPSD 更是跌至 38.75%，直接被强化学习基准甩在身后。这直接在实验层面证实了论文的立论：纯蒸馏的上限被牢牢锁死在教师能力之内，而缺乏 Token 区分度的普通 RL 又无法有效消化多步信用。

更具说服力的表现在高难度子集与会话级指标上。在 BFCL 的 Long Context（长上下文）切分中，交互序列极长，信用稀释尤为致命。CrEST 在 4B 模型上取得了 60.00% 的成绩，相比最强基准大幅领先了 7.0 个百分点；在 8B 模型上也达到 47.00%，保持领先。而在评价标准极为严苛的 WildToolBench 上——只要会话中任意一轮出现偏差整条即算失败——8B 模型的会话准确率通常极低，GRPO 仅有 5.47%，而 CrEST 直接将该指标拉升至 9.38%，在综合任务准确率上也达到了 52.34%。

### 训练动态与消融：为何两者缺一不可？

训练过程中的收敛曲线与梯度分布，揭示了 CrEST 之所以稳健且高效的内在几何机理。

在收敛速度与上限方面，BFCL V3 上的训练曲线显示，CrEST 仅用大约 20 个训练步就达到了 60% 的采样准确率，并最终收敛在 70% 左右；反观标准 GRPO，在经历了极其缓慢的爬升后，到 160 步时仍停留在 57% 附近。而 OPSD 则在 49% 处早早遭遇了平台期，随后因过拟合或梯度异常开始衰退。CrEST 在第 20 步便轻松突破了 OPSD 的平台上限并持续走高，证明了方向门确实起到了“保留强化学习探索自由度、不被教师固有偏见束缚”的关键作用。

在梯度几何分布上，研究者统计了按优势幅度排序后 Top-p% 的 Token 所占用的梯度总量。数据显示：

- **OPSD 表现出极端的病态集中**：前 1% 的 Token 占据了约 42% 的总梯度信号，前 5% 的 Token 更是掳走了超过 77% 的梯度预算。这意味着极少数 Token（通常只是频繁出现的格式符号）主导了模型参数更新，极易导致策略坍塌。

- **GRPO 则走向了另一个极端**：由于每个 Token 被均摊相同的数值，前 10% 的 Token 仅占约 31% 的梯度，整体过于扁平，缺乏重点。

- **CrEST 恰到好处地居于黄金分割点**：其前 10% 的 Token 占据了约 57% 的梯度份额。这表明，它既成功借助自教师的先验拉开了关键决策 Token 与无用 Token 之间的差距，又依靠熵门控遏制了极端梯度的膨胀，保证了全局训练的平稳演进。

消融实验进一步厘清了各模块的职责边界。在剔除轮次切分、仅保留 Token 级调制的“Intra-turn only”配置下，4B 模型在 BFCL 上的均分从 52.00% 回落至 48.75%；而在仅保留轮次切分、不做内部 Token 差异化的“Inter-turn only”配置下，得分为 47.88%。二者各自虽然都能比基线 GRPO 带来 4-5 个点的收益，但唯有双剑合璧才能彻底释放潜能达到 52.00%。

对于门控机制的单独剔除同样致命。一旦移除方向门（w/o Direction gate），模型丧失了方向校验，均分跌至 46.75%，再度被困在教师天花板内；而如果移除熵门控（w/o Entropy gate），均分骤降至 46.25%，尤其在需要精准填参的 Missing Parameter 子集上跌幅巨大，直接证实了格式 Token 抢占梯度会对核心参数生成逻辑造成严重损害。

### 技术启示与未来边界

长期以来，在强化学习与知识蒸馏的结合探索中，学界习惯于将教师模型作为全知全能的指路标，无论是在损失函数中直接添加 KL 散度约束，还是交替进行策略梯度与监督微调。然而在工具调用和复杂推理等长链条任务中，这种范式忽视了强化学习的核心价值——**在未知空间中通过环境真实验证进行自由探索**。

CrEST 给出了一种范式上的重新思考：**教师模型（即使是拼接了标准答案的自教师）最擅长的不是在全局探索中替代环境发号施令，而是在局部语境下帮助模型识别轻重缓急**。将“更新走向何方”（Direction）的决定权完全交还给环境验证器，把“更新用力多少”（Magnitude）的裁量权交给带有不确定性约束的教师网络，这种解耦机制既规避了复杂外部大教师的依赖，又优雅化解了信用分配的颗粒度矛盾。

当然，该方法在工程实践中也存在明确的适用边界。目前的轮次优势切分高度依赖于环境中各轮次任务的相对独立性与局部可验证性（Local Verifiability）。若遇到极端端到端、中途完全黑盒且跨轮状态高度纠缠的非马尔可夫系统，单纯的按轮切分仍需更复杂的时间差分（TD）机制进行补充。但无论如何，对于当下蓬勃发展的多轮对话 Agent、自动化工作流编排以及软件工程自动化场景，CrEST 展示出了一条极具普适性且无需庞大额外算力的性能提升路径。
