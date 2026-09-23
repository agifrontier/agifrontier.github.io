---
layout: default
title: "InSight-doc：把分辨率当推理资源，港科大与华为让长文档延迟降68%"
description: "针对这一核心痛点，香港科技大学与华为的研究团队提出了 InSight-doc 。该工作颠覆了以往默认必须把全量高分辨率图像一次性灌入模型的做法，将“视觉分辨率”重新定义为一种在推理阶段可被模型主动调度的计算资源。"
arxiv_id: "2608.10628"
paper_published: "2026-08-11"
published_at: "2026-09-23T13:15:08.398930+08:00"
topics:
  - "AI Agent"
  - "推理"
tags:
  - "InSight-doc"
  - "RL"
  - "SFT"
  - "active-perception corpus"
  - "adaptive-resolution perception"
  - "agentic visual perception"
related_tutorials:
  - "qwen2-vl-enhancing-vision-language-models-perception-of-the-world-at-any-resolut"
  - "slideagent-hierarchical-agentic-framework-for-multi-page-visual-document-underst"
  - "the-illusion-of-insight-in-reasoning-models"
  - "cogflow-bridging-perception-and-reasoning-through-knowledge-internalization-for-"
---

<p class="paper-original-title" lang="en">InSight-doc: Agentic Visual Perception for Long-Document Understanding</p>

<img src="/images/2608.10628v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在多模态大模型（MLLM）处理数十甚至上百页的长文档时，工程师和研究者常常陷入两难境地：如果为了保留印章、小字排版、密集表格和图表微小标注而将每一页都以高分辨率切片输入，Token 数量会迅速突破数万甚至十几万，不仅导致显存与 KV 缓存暴涨、推理延迟激增，还会触发严重的“上下文腐烂”（Context Rot）现象，让模型的注意力在冗长序列中被严重稀释；如果退而求其次采用 OCR 提取加外部文本检索，又极易在复杂版面分析或图表结构上发生错误级联。

> ArXiv URL：https://arxiv.org/abs/2608.10628v1

针对这一核心痛点，香港科技大学与华为的研究团队提出了 **InSight-doc**。该工作颠覆了以往默认必须把全量高分辨率图像一次性灌入模型的做法，将“视觉分辨率”重新定义为一种在推理阶段可被模型主动调度的计算资源。整个框架不依赖任何外部检索器，而是让模型从极低分辨率的全局概览出发，在多模态思维链（Chain-of-Thought, CoT）的驱动下，自主决定何时放大、放大何处，以区域级（Region-level）局部高分辨率裁剪替代传统的整页拉取。

实验结果显示，以开源的 Qwen3-VL-8B 为底座，在引入包含 1.79 万条多跳放大轨迹的 SFT 数据以及 1.92 万条强化学习数据进行联合训练后，InSight-doc-8B 在多个中长文档基准上的准确率提升了 4.3 至 16.4 个百分点。在超长文档场景下，该方案不仅将幻觉率抑制了 40% 以上，更直接砍掉了 41% 至 68% 的端到端推理延迟，显著刷新了长文档理解的精度与效率前沿。

<img src="/images/2608.10628v1/x1.webp" alt="性能与效率总览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 从整页硬塞到主动观察：视觉感知的动态推理循环

现有的多模态长文档理解范式主要可分为三类：第一类是解析流水线，利用版面分析、OCR 与阅读顺序还原重构文本，遇到复杂版面极易崩溃；第二类是端到端视觉大模型，直接输入多页图像，但在长文档下面临二次方计算开销与注意力稀释；第三类则是近年兴起的由粗到细（Coarse-to-fine）方法，先低分辨率扫读再高分辨率调取特定页面。然而，现存的粗到细方案如 Doc-V* 往往重度依赖外部检索模块，且只能粗粒度地搬运整页图像，依然引入了可观的冗余视觉 Token。

InSight-doc 则彻底转向了类似人类阅读长篇财报或论文时的真实习惯：先快速扫视低清全局版面建立索引感，只有在确认某处存在关键证据时，视线才会聚焦放大特定段落或图表。如图 2 所示，InSight-doc 的感知过程与推理思考高度交织，形成了一个完全内生的主动感知闭环。

<img src="/images/2608.10628v1/x2.webp" alt="InSight-doc 推理过程示意图" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在形式化定义上，系统将初始视觉上下文空间 $\mathcal{I}_{\text{ctx}}^{(0)}$ 设置为全文档所有页面的降采样视图。设原始图像集合为 $\{I_{k}\}_{k=1}^{N}$，初始输入通过缩放因子 $r \leq 1$ 得到下采样图像 $\tilde{I}^{(0)}_{k} = \texttt{resize}(I_{k}, r \cdot \texttt{size}(I_{k}))$。在最激进的设置下，DPI 可降至 50（$r=0.25$），单页分辨率仅约 $425 \times 550$ 像素，能够极低成本地容纳几十页甚至上百页文档。

当模型在思考标签 `<think>` 中梳理线索并推测答案可能藏于某处时，它会输出一个结构化的工具调用标签 `<tool_call>`，其参数包含目标页面索引、区域语义描述以及精细的边界框坐标（Bounding Box, bbox）。执行环境根据这些坐标在原始高分辨率底图中截取对应局部，并根据放大系数 $c > 1$ 生成新的区域级高分辨率图像：




{% raw %}$$ \tilde{I}^{(t)}_{\text{crop}} = \texttt{resize}(I^{(t)}_{\text{crop}}, c \cdot r \cdot \texttt{size}(I^{(t)}_{\text{crop}})) $${% endraw %}



这一高分辨率局部随后被动态追加到多模态上下文空间 $\mathcal{I}_{\text{ctx}}^{(t)} = \mathcal{I}_{\text{ctx}}^{(t-1)} \cup \{\tilde{I}^{(t)}_{\text{crop}}\}$ 中。模型随后观察这一新注入的高清证据，决定继续追问其他线索、进一步放大，还是已经具备充分信心输出最终答案 `<answer>`。这种设计允许模型在不借助任何外置 Retriever 的情况下，原生支持多跳跨页推理与多轮自我修正。

### 为什么区域级放大会在数学上带来巨大延迟收益？

直觉上，引入多轮工具交互与逐步思维链会增加自回归解码步数，是否会抵消掉降分辨率带来的收益？为了定量解释这一权衡，研究者在论文中给出了严格的理论推导。

设基线单轮直接输入的 Prompt 视觉 Token 量为 $P_0$，生成响应 Token 量为 $R_0$，基线的总计算耗时由 Prefill 与 Decoding 两部分构成，可建模为 $T(P, R) = \alpha P^2 + \beta R(2P + R)$。在 InSight-doc 中，输入上下文长度被大幅压缩，其等效 Token 开销可表示为缩放比率与工具调用轮数的函数 $x(r) = r^2 + \delta n(r)$，而生成端开销则扩大为 $y(r) = 1 + \lambda n(r)$，其中 $\delta$ 和 $\lambda$ 分别表示单次工具调用带来的额外输入与输出代价。

本文给出了两个核心命题。首先是关于相对序列长度的命题 1（Proposition 1）：




{% raw %}$$ S_r / S_0 \leq x(r) + \kappa^{-1} y(r) $${% endraw %}



其中 $\kappa = P_0 / R_0$ 为基线下的 Prompt 与生成长度比值。在长文档 VQA 中，$\kappa$ 通常在 50 到 200 之间，图像 Prompt Token 占据压倒性体量。代入典型参数（缩放因子 $r \in [0.25, 0.50]$，放大调用次数 $n(r) \in [1, 3]$），InSight-doc 的最终上下文总长度仅为基线硬塞方案的 $7.8\%$ 至 $45.0\%$。

其次是结合 Prefix Caching 后的相对推理延迟命题 2（Proposition 2）：




{% raw %}$$ T_r / T_0 \leq w_{\mathrm{p}} x(r)^2 + w_{\mathrm{c}} x(r) y(r) + w_{\mathrm{g}} y(r)^2 $${% endraw %}



其中权重系数由基线参数的二次项严密归一化。在长文档场景中，由于长文本自注意力 Prefill 的二次方计算瓶颈和超长上下文对解码带宽的挤压，减少初始视觉 Token 带来的时间红利具有绝对的支配地位。理论推导预测，在 1 到 2 次工具调用的合理区间内，端到端延迟可以压缩至基线的 12.4% 至 33.6%。这种数学层面的显著优势，为后续实验中展现的极端性能-效率帕累托提升奠定了坚实基础。

### 数据工程闭环：从双 Agent 协作到扁平单模型模仿

能够让 8B 参数模型精准掌握“何处需要看细节、何处直接略过”的策略，其核心瓶颈不在于网络架构，而在于高价值多模态交互轨迹数据的匮乏。如果直接人工标注数万条跨页多跳的局部放大路径，成本几乎难以承受；而如果仅使用规则随机生成，模型又无法习得深层次的推理意图。

为此，团队构建了一套严密的三阶段数据合成与强化学习筛选流水线，将数据源拓宽至 arXiv 论文、DUDE、DocVQA、InfographicVQA、Paper2Poster 和 MapTab 六个具有高度视觉版面多样性的基准。

<img src="/images/2608.10628v1/x3.webp" alt="数据合成与过滤流程" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

数据流水线的上半部分执行三阶段级联清洗：

1. **第一阶段：无文档可答性过滤**。直接把问题丢给没有文档上下文的语言模型，凡是仅凭先验常识就能回答的问题一律丢弃，确保所有留存数据都强依赖文档细节。

2. **第二阶段：低分辨率可答性过滤**。将文档压至低分辨率（如 50 DPI），直接让基座模型尝试解答；如果模型无需放大就能答对，说明该题目不需要主动感知能力，直接剔除。这使得训练集高度聚焦在那些“低清能看到线索轮廓、但看清数值必须放大”的深层感知题目上。

3. **第三阶段：CoT 轨迹合成与分流**。对剩余的复杂问题，采用基于 InSight-o3 架构的双 Agent 机制生成放大轨迹。最终能成功推导出正确答案的高质量轨迹送入监督微调（SFT）集合，而未能在规定步数内做对的难题则被剔除答案标签，归入强化学习（RL）题目池。

在下半部分的双 Agent 协作阶段，团队解耦了“逻辑推理”与“空间定位”两项能力。由更擅长逻辑调度的 GPT-5-mini 担任推理大脑 **vReasoner**，负责维护全局思维链并输出形如 $\langle\text{PageID}_{i} : \text{Desc}_{i}\rangle$ 的目标意图；由经过坐标微调的 Qwen3-VL-8B 担任视觉搜索眼 **vSearcher**，负责在指定页面上精准框出对应实体的边界框 $\langle\text{Box}_{i}\rangle$。

数据生成完成后，研究团队做出了一个至关重要的工程处理：**将双 Agent 的对话流压平（Flatten）为单一序列**。在生成的训练目标中，单模型自身需要先后产生思考、目标描述、自预测边界框并接收环境回传的裁剪块，最终给出回答。这种做法摆脱了多模型协作系统的部署臃肿，让单一参数模型在模仿中同时学会了“何时放大、去哪放大、如何利用新证据”。

最终构建的语料库包含 3.71 万个样本，其中 1.79 万条高质量 SFT 轨迹，平均文档长度为 17.75 页，平均包含 2.61 轮交互；RL 池则收纳了 1.92 万个高难度未解决样本。

在训练阶段，模型基于 Qwen3-VL-8B-Instruct 展开，SFT 之后采用基于开源框架 verl 的 GRPO（Group Relative Policy Optimization）进行强化学习。值得注意的是，团队在强化学习过程中**仅使用了二元准确率奖励（Binary Accuracy Reward）**，未对格式或步数施加复杂的人工规则塑形。模型完全在自探索与二元胜负反馈中，自发强化了自我纠错、多步回溯以及精准框选目标区域的策略分布。

### 实验评测：全方位重塑长文档的帕累托前沿

在验证环节，InSight-doc 接受了中长篇文档（DUDE、MP-DocVQA、MMLongBench-Doc、LongDocURL）与通用高分辨率图像基准（MME-RealWorld-Lite、O3-Bench）的严苛检验。基准测试中的 PDF 文档均在 200 DPI 下光栅化作为原始高分辨率图像，并按 $r \in \{0.25, 0.35, 0.5, 0.7\}$ 下采样模拟不同初始分辨率。

在以 50 DPI（$r=0.25$）作为初始输入的严苛条件下，基座模型 Qwen3-VL-8B 毫无招架之力，平均准确率仅有 50.5%。而经过完整 SFT+RL 训练的 InSight-doc-8B 平均准确率飙升至 66.9%，整整取得了 16.4 个百分点的巨大增益。细分到各个数据集，DUDE 提升 17.2 分，MP-DocVQA 提升 18.3 分，MMLongBench-Doc 提升 17.1 分，LongDocURL 提升 12.8 分。

即便将初始分辨率提升至 100 DPI（$r=0.5$），InSight-doc 依然比基线高出 4.3 个百分点，达到 72.6% 的高准确率。更值得关注的是强化学习的增量价值：在 $r=0.25$ 下，仅做 SFT 的模型准确率为 56.6%，而加上 GRPO 强化学习后进一步激增了 10.3 个百分点，充分印证了大规模探索训练在激活多步定位感知能力上的不可替代性。

<img src="/images/2608.10628v1/x4.webp" alt="准确率与效率 Pareto 前沿" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

图 4 的帕累托前沿图直观揭示了该方法的工程优势。在传统的直接输入方案中，想要获得更高的准确率，唯一的途径就是提高输入 DPI，这会导致 Token 消耗与处理耗时呈抛物线式攀升。而 InSight-doc 展现出了向左上方显著迁移的帕累托支配效应：

- **Token 消耗锐减**：在 50 DPI 下，InSight-doc 以 66.9% 的准确率逼近了 100 DPI 基线的 68.3%，而其占用的总 Token 数量减少了 58%；在最长文档的子集分布中，InSight-doc 在 70 DPI 下仅耗费 4.24 万个 Token 就取得了 56.2% 的准确率，相比 140 DPI 基线耗费的 13.68 万个 Token 缩减了足足 69%，大幅解放了服务端 KV 缓存压力。

- **端到端延迟对半腰斩**：在实际延迟对比中，InSight-doc 在 70 DPI 下的综合表现不仅比 140 DPI 的基线高出 1.2 个百分点，平均延迟更是缩减了 54%。在 MMLongBench-Doc 上，基线单次推理需要 21.2 秒，InSight-doc 仅需 9.3 秒；而在极长文档测试子集上，面对基线长达 39.3 秒的缓慢 Prefill 与解码，InSight-doc 仅用 11.2 秒即完成了多轮交互与推导，效率提升幅度达 71%。

### 幻觉抑制与不可答问题的识别能力

长文档问答在工业界落地面临的另一大梦魇是“无中生有”的幻觉，尤其是在海量上下文中寻找根本不存在的信息时，过度拉长的注意力往往会导致模型倾向于编造看似合理的断言。

研究团队针对性地在 DUDE 与 MMLongBench-Doc 的不可答（Unanswerable）子集上评估了模型的拒答 F1 分数。在 50 DPI 极低清晰度下，基座模型在两大数据集上的拒答 F1 仅有 44.5 和 48.5，极易产生虚假关联；而 InSight-doc 将此指标分别推高至 69.1 和 74.4，实现了超过 20 个百分点的显著提升。

这背后的机制在于，传统模型在模糊的全文长上下文里进行单轮贪婪预测，很难判断某个词是“自己没看清”还是“文中根本没有”；而 InSight-doc 赋予了模型主动检验的能力。当全局概览产生怀疑时，模型可以定向发起局部的放大探查。一旦多轮放大均未能发现支撑证据，模型在思维链中确认“不存在该证据”的先验置信度便大幅提升，从而自然、果断地选择拒答。

### 走向自适应推理时代的多模态范式

InSight-doc 的工作提供了一个非常清晰的技术风向标：**处理超长视觉上下文，并不意味着必须无节制地拓宽上下文窗口并堆砌昂贵的自注意力算力。**

从技术演进脉络来看，模型处理信息的方式正在经历从“被动填充（Passive Context Stuffing）”到“主动获取（Active Perception）”的范式迁移。InSight-doc 证明了，只要赋予多模态大模型合适的光学交互工具，辅以高质量的多跳交互轨迹 SFT 和基于终局奖励的强化学习，即便是 8B 级别的开源端侧模型，也能展现出类似人类专家阅读长篇报告时的敏锐、克制与高效。

这种把图像分辨率从静态输入参数降维为“推理时按需分配资源”的思路，不仅有效缓解了上下文腐烂这一 Transformer 固有顽疾，也为边缘端与高并发服务端部署超长多模态文档应用，开辟了一条兼顾超高精度与极致吞吐的可行通路。
