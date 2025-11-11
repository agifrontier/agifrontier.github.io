---
layout: default
title: "Sampling and Loss Weights in Multi-Domain Training"
---

# Sampling and Loss Weights in Multi-Domain Training

- **ArXiv URL**: http://arxiv.org/abs/2511.06913v1

- **作者**: Meisam Razaviyayn; Mahdi Salmani; Pratik Worah

- **发布机构**: Google Research; University of Southern California

---

# TL;DR
本文提出，在多域训练中，应将传统的单一领域权重分解为两个独立的角色：旨在通过降低噪声域影响来改善泛化的**损失权重 (loss weights)**，以及旨在通过平衡梯度方差来加速优化的**采样权重 (sampling weights)**，并证明这两种权重能带来互补的性能提升。

# 关键定义
本文的核心是区分并定义了两种在多域学习中扮演不同角色的权重：

1.  **损失权重 (Loss Weights)**: 在计算总的经验风险时，每个领域经验风险项的缩放因子。其数学表示为 $w\_i$，作用于经验风险目标函数 $\hat{\mathcal{L}}\_{\mathcal{S},\pi,w}(\theta)\;=\;\sum\_{i=1}^{K}\pi\_{i}w\_{i}\,\hat{\mathcal{L}}\_{\mathcal{S}\_{i}}(\theta)$。其核心目标是提升模型的泛化能力，通过降低噪声更大或更不可靠领域的贡献来实现。

2.  **采样权重 (Sampling Weights)**: 在随机梯度下降等优化过程中，决定从每个领域抽取样本的频率或数量。其数学表示为 $b\_i$（批次中来自领域 $i$ 的样本数）。其核心目标是减少随机梯度估计的方差，从而提高优化过程的效率和稳定性。

3.  **域加权总体风险 (Domain-weighted Population Risk)**: 一个理论上的目标函数，定义为 $\mathcal{L}\_{\pi}(\theta)\;=\;\sum\_{i=1}^{K}\pi\_{i}\,\mathcal{L}\_{i}(\theta)$。这里的 $\pi\_i$ 代表每个领域的“固有”重要性，反映了在最终模型评估时我们对不同领域的重视程度。本文假设 $\pi\_i$ 是给定的，专注于优化损失权重和采样权重。

4.  **域加权经验风险 (Domain-weighted Empirical Risk)**: 实际训练中最小化的目标函数，定义为 $\hat{\mathcal{L}}\_{\mathcal{S},\pi,w}(\theta)\;=\;\sum\_{i=1}^{K}\pi\_{i}w\_{i}\,\hat{\mathcal{L}}\_{\mathcal{S}\_{i}}(\theta)$。它结合了领域固有重要性 $\pi\_i$ 和本文重点研究的损失权重 $w\_i$。

# 相关工作
当前，在处理来自多个异构领域（如 Wikipedia、GitHub）的大规模数据训练时，普遍的做法是为每个领域分配一个单一的标量权重。这些权重通常依据启发式规则（如按数据集大小比例分配）或手动调参设定。

这种“单一权重”的视角存在一个关键问题：它混淆了领域权重在学习过程中的两个根本不同的角色。一个领域对最终目标的贡献度（影响泛化）和在优化过程中被采样的频率（影响效率）被同一个参数所控制。这导致了次优的训练策略，因为它无法同时精细地解决由领域异质性带来的两个独立挑战：
1.  **泛化差距 (Generalization Gap)**：不同领域的噪声水平和数据质量不同，导致其经验风险与真实风险的差距各异。
2.  **优化效率 (Optimization Efficiency)**：不同领域的数据分布差异导致梯度方差不同，影响随机优化的收敛速度。

本文旨在解决这一问题，通过明确区分并分别优化损失权重和采样权重，从而建立一个更精细、更高效的多域训练框架。

# 本文方法
本文的核心思想是将领域权重解耦为**损失权重**和**采样权重**，并为它们分别设计了动态估计算法。

### 损失权重：提升泛化能力
损失权重的目标是降低噪声高、不可靠领域对模型训练的影响，从而减小泛化差距。

#### 理论基础：线性回归与广义最小二乘法 (GLS)
在线性回归模型 $y=\theta\_{\text{gt}}^{\top}\mathbf{x}+\epsilon$ 中，如果不同领域的噪声方差 $\sigma\_i^2$ 不同（即存在异方差性），根据 Aitken 定理，最优的线性无偏估计器是广义最小二乘 (Generalized Least Squares, GLS) 估计器。这等价于在经验风险最小化中设置损失权重 $w\_i^{\star} \propto 1/\sigma\_i^2$。这意味着，噪声方差越大的领域，其损失权重应该越小。

#### 创新点1：One-shot FGLS
传统的可行广义最小二乘法 (Feasible GLS, FGLS) 需要先训练一个模型，计算残差来估计 $\sigma\_i^2$，然后再用估计出的权重重新训练模型，过程繁琐。为解决此问题，本文提出了 **One-shot FGLS** (算法1)。该方法在训练过程中，使用一个与训练样本独立的验证集（从训练数据中划分）来动态估计每个领域的误差，并据此平滑地更新损失权重 $w\_i$。这避免了多次完整训练的开销。

<img src="images/2511.06913v1/x4.png" alt="Algorithm 1" style="width:100%;" />

#### 创新点2：ERMA (Empirical Risk Minimization with Adaptation)
为了将此思想推广到线性回归之外的通用模型，本文基于方差泛化界（variance-based generalization bounds）推导了一个通用的权重更新规则，称为 **ERMA**。该方法旨在最小化泛化差距的一个上界，其更新公式为：


{% raw %}$$ w_{i}^{(t+1)}\propto w_{i}^{(t)}\exp\left(\gamma_{1}\,\pi_{i}G(t)\,\mathcal{L}_{i}(\theta_{t})-\gamma_{2}\,\pi_{i}w_{i}^{(t)}\,\operatorname{Var}_{i}(\theta_{t})\right) $${% endraw %}


此规则根据每个领域在当前模型 $\theta\_t$ 下的损失 $\mathcal{L}\_{i}(\theta\_{t})$ 和损失的方差 $\operatorname{Var}\_{i}(\theta\_{t})$ 来动态调整权重 $w\_i$。

### 采样权重：加速与稳定优化
采样权重的目标是降低每个batch梯度估计的方差，从而加速模型收敛并提高优化稳定性。

#### 理论基础：最小化梯度方差
在随机优化中，梯度的方差会影响收敛速度，如下表所示。


| 条件 | 步长 | 收敛率 (非凸) |
| --- | --- | --- |
| 光滑 | $\eta\_{t}\sim 1/\sqrt{t}$ | $\mathcal{O}\!\left(\tfrac{LR^{2}}{T}+\tfrac{\sigma R}{\sqrt{T}}\right)$ |
| 光滑 & 强凸 | $\eta\_{t}\sim 1/(\mu t)$ | $\tilde{\mathcal{O}}\!\left(\tfrac{\sigma^{2}}{\mu T}\right)$ |

本文的目标是最小化批梯度 $g\_t$ 相对于全量经验梯度 $\nabla\_{\theta}\hat{\mathcal{L}}\_{\mathcal{S}}(\theta\_t)$ 的方差：


{% raw %}$$ \min_{\mathbf{b}}\;\mathbb{E}\Bigl[\,\bigl\ \mid g_{t}-\nabla_{\theta}\hat{\mathcal{L}}\_{\mathcal{S}}(\theta\_{t})\bigr\ \mid ^{2}\,\Bigr] \quad \text{s.t.} \sum_{i=1}^{K}b_{i}=B $${% endraw %}


其中 $b\_i$ 是从领域 $i$ 采样的数量，总批大小为 $B$。该方差可以表示为 $\sum\_{i=1}^{K}\frac{\pi\_{i}^{2}w\_{i}^{2}}{b\_{i}}\,v\_{i}^{2}$，其中 $v\_i^2$ 是领域内梯度方差。

#### 创新点3：VA (Variance-Aware Sampling)
通过拉格朗日乘子法求解上述优化问题，可得到最优的采样数量分配策略：


{% raw %}$$ b_{i} \propto \pi_{i}w_{i}v_{i} $${% endraw %}


这意味着，从一个领域采样的样本数应与其**固有重要性** $\pi\_i$、**损失权重** $w\_i$ 以及**领域内梯度方差** $v\_i$ 的乘积成正比。直观上，梯度变化越剧烈的领域，需要抽取更多样本来获得一个稳定的梯度估计。本文提出了 **VA** 算法 (算法2)，在训练中动态估计 $v\_i$ 并调整采样比例。

<img src="images/2511.06913v1/x5.png" alt="Algorithm 2" style="width:100%;" />

# 实验结论

本文通过在线性回归、逻辑回归和神经网络上的实验，验证了所提方法的有效性。

### 主要发现
*   **独立有效性**：无论是在线性回归还是逻辑回归任务中，单独使用损失权重调整方法（One-shot FGLS 或 ERMA）或单独使用采样权重调整方法（VA）都能比基线（均匀权重）取得更好的性能，表现为更快的收敛速度和更低的最终误差。

<img src="images/2511.06913v1/x1.png" alt="Figure 1: Linear Regression Results" style="width:100%;" />
*(上图) 线性回归实验。当领域2数据更优 ($(C\_1,C\_2)=(1,100)$)时，组合方法(One-shot FGLS+VA)在收敛速度和最终误差上表现最佳。*

*   **互补性**：将损失权重和采样权重结合使用时，通常能获得最佳性能，尤其是在领域间差异显著的情况下。这证明了两种权重分别解决了训练过程中的不同问题（泛化与优化），其效益是互补的。

<img src="images/2511.06913v1/x2.png" alt="Figure 2: Logistic Regression Results" style="width:100%;" />
*(上图) 逻辑回归实验。无论是数据质量相似还是差异较大，组合方法(ERMA+VA)均取得了最好的或接近最好的结果。*

*   **场景依赖性**：在神经网络的 MNIST 实验中，ERMA（损失权重）表现出色，因为它能有效降低带标签噪声领域的影响。然而，VA（采样权重）几乎没有效果。作者分析，这是因为在该任务中，干净领域和噪声领域的输入数据高度相似，导致它们的领域内梯度方差差异不显著，因此调整采样权重带来的收益很小。这表明不同权重策略的重要性取决于具体的任务和数据特性。

<img src="images/2511.06913v1/x3.png" alt="Figure 3: Neural Net Results" style="width:100%;" />
*(上图) MNIST实验，ERMA表现优异，而VA效果不佳，证明了VA的有效性依赖于领域间梯度方差的差异。*

### 最终结论
实验结果有力地支持了本文的核心论点：领域加权是一个二维问题，而非一维问题。通过解耦并分别优化损失权重（关注泛化）和采样权重（关注优化），可以实现比传统单一权重方法更优越的性能。这种双重权重框架为理解和实践多域学习提供了更清晰的理论视角和更有效的实操工具。