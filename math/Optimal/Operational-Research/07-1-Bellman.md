---
title: Bellman 最优性原理
date: 2026-03-02
tags:
course: math
status: draft
---
# Bellman 最优性原理
[TOC]

Bellman 最优性原理（Bellman Principle of Optimality）将整体优化问题递归地拆分为""，是动态规划（DP）、最优控制和强化学习的共同理论基础。

<!--more-->

## Takeaway

当前最优决策 + 未来最优子问题。bellman=optimality recursion

---

## Core Mechanism

离散 DP、随机 MDP、连续 HJB为同一原理的不同表现形式

### 离散确定性形式

Discrete deterministic DP

设离散时间系统：
- $s_t \in \mathcal{S}$：状态
- $a_t \in \mathcal{A}(s_t)$：动作
- $s_{t+1} = T(s_t, a_t)$：确定性状态转移
- $c(s_t, a_t)$：单步代价
- $g(s_T)$：终端代价

目标为最小化总代价：
$$
J(s_0) = \min_{a_0,\dots,a_{T-1}} \sum_{t=0}^{T-1} c(s_t, a_t) + g(s_T)
$$

直接搜索全部动作序列是指数级的。Bellman 的关键操作是定义 **value function**：
$$
V_t(s) = \text{从时间 } t \text{、状态 } s \text{ 出发，到终点所需的最小未来代价}
$$

从状态 $s$ 出发，第一步选动作 $a$ 后产生：
1. 当前代价 $c(s, a)$
2. 下一状态 $T(s, a)$
3. 该状态的 future optimal cost $V_{t+1}(T(s, a))$

总成本为 $c(s, a) + V_{t+1}(T(s, a))$，对所有动作取最小即得 Bellman recursion：
$$
V_t(s) = \min_{a \in \mathcal{A}(s)} \left[ c(s, a) + V_{t+1}(T(s, a)) \right]
$$

终端条件：$V_T(s) = g(s)$。这就是 finite-horizon deterministic DP 的标准形式。

### 离散随机形式

Discrete stochastic DP/MDP(markov decision process)

- 状态转移是随机的
  $$
  P(s_{t+1}=s'\mid s_t=s,a_t=a)
  =
  P(s'\mid s,a)
  $$
  那么你选择动作 $a$ 后，不能确定进入哪个 $s'$，只能得到一个 distribution over next states。

  所以 deterministic Bellman 里的

  $$
  V_{t+1}(T(s,a))
  $$
  要变成 expectation：

  $$
  \mathbb E_{s'\sim P(\cdot\mid s,a)} \left[ V_{t+1}(s') \right]
  $$

于是 Bellman recursion 变成：

$$
V_t(s) = \min_{a} \left[ c(s,a) + \sum_{s'}P(s'\mid s,a)V_{t+1}(s') \right]
$$
若状态转移是随机的 $P(s' \mid s, a)$，奖励为 $r(s, a)$，折扣因子 $\gamma \in [0, 1)$：

**Bellman optimality equation**（最优 value function）：
$$
V^*(s) = \max_{a \in \mathcal{A}(s)} \left[ r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \, V^*(s') \right]
$$

**Bellman expectation equation**（给定策略 $\pi(a \mid s)$ 下的 value function）：
$$
V^\pi(s) = \sum_a \pi(a \mid s) \left[ r(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \, V^\pi(s') \right]
$$

### 连续确定性形式

对于连续时间系统，Bellman 原理通过极限过程导出 **Hamilton-Jacobi-Bellman (HJB)** 偏微分方程：
$$
-\partial_t V(t, x) = \min_{u \in \mathcal{U}} \left[ \ell(x, u) + \nabla_x V(t, x)^\top f(x, u) \right]
$$

> 连续形式的 Bellman equation 在数学结构上属于 **Hamilton-Jacobi PDE**，而 Bellman 把它和 dynamic programming / optimal control 结合起来了

#### 推导

考虑连续控制系统：
- $\dot{x}(t) = f(x(t), u(t))$：dynamics
- $x(t) \in \mathbb{R}^n$：状态
- $u(t) \in \mathcal{U}$：控制/动作
- $\ell(x, u)$：running cost
- $g(x(T))$：terminal cost

目标：
$$
J(t, x; u) = \int_t^T \ell(x(\tau), u(\tau)) \, d\tau + g(x(T))
$$

定义 value function：
$$
V(t, x) = \inf_{u(\cdot)} \left[ \int_t^T \ell(x(\tau), u(\tau)) \, d\tau + g(x(T)) \right]
$$

Bellman 原理将 $(t, x)$ 出发的最优控制拆为"现在极短时间 $[t, t+\Delta t]$ + 之后从 $x(t+\Delta t)$ 出发的最优子问题"：
$$
V(t, x) = \min_u \left[ \ell(x, u)\Delta t + V(t+\Delta t, \, x + \Delta t \cdot f(x, u)) \right]
$$

对 $V(t+\Delta t, x + \Delta t f)$ 做 Taylor 展开：
$$
V(t+\Delta t, x + \Delta t f) = V(t, x) + \partial_t V \cdot \Delta t + \nabla_x V^\top f \cdot \Delta t + o(\Delta t)
$$

代回并消去 $V(t, x)$，除以 $\Delta t$，取 $\Delta t \to 0$：
$$
0 = \min_u \left[ \ell(x, u) + \partial_t V + \nabla_x V^\top f \right]
$$

即Hamilton-Jacobi-Bellman equation：
$$
-\partial_t V(t, x) = \min_u \left[ \ell(x, u) + \nabla_x V(t, x)^\top f(x, u) \right]
$$

终端条件：$V(T, x) = g(x)$。

### 连续随机形式

如果系统有随机扰动，动力学通常写成 SDE：

$$
dx_t = f(x_t,u_t)\,dt + \sigma(x_t,u_t)\,dW_t
$$
其中：

*   $f(x,u)$：drift；

*   $\sigma(x,u)$：diffusion coefficient；

*   $W_t$：Brownian motion；

*   $dW_t$：随机增量。

目标仍然是：

$$
V(t,x) = \inf_{u(\cdot)} \mathbb E \left[ \int_t^T \ell(x_\tau,u_\tau)\,d\tau + g(x_T) \right]
$$
此时 Bellman principle 仍然成立，但 Taylor expansion 要换成 **Itô formula**。

于是 HJB 变成：

$$
-\partial_t V(t,x) = \min_u \left[ \ell(x,u) + \nabla_x V(t,x)^\top f(x,u) + \frac{1}{2} \operatorname{Tr} \left( \sigma\sigma^\top \nabla_x^2 V(t,x) \right) \right]
$$
比 deterministic HJB 多了一个二阶项：

$$
\frac{1}{2} \operatorname{Tr} \left( \sigma\sigma^\top \nabla_x^2 V \right)
$$
这个项来自 Brownian motion 的二阶变差：$(dW_t)^2 = dt$

所以随机连续控制的 HJB 是二阶 PDE

## Pros

- **计算降维**：将 $|\mathcal{A}|^T$ 的指数级搜索降至 $T \cdot |\mathcal{S}| \cdot |\mathcal{A}|$ 的多项式复杂度

## Cons

- **维度灾难（Curse of Dimensionality）**：状态空间 $\mathcal{S}$ 维度增长时，value function 的存储与计算量指数级膨胀
- **需要完整模型**：必须已知 $T(s, a)$ / $P(s' \mid s, a)$ / $f(x, u)$，不能直接从数据学习
- **HJB 可解性有限**：连续 HJB 在高维时几乎无法解析或数值求解

---

## References
- Bellman, R. (1957). *Dynamic Programming.* Princeton University Press.
- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction.* MIT Press.
- Bertsekas, D. P. (2012). *Dynamic Programming and Optimal Control.* Athena Scientific.
