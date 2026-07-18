---
title: 00-Intro
date: 2026-03-02
tags:
course: math
status: draft
---
# Intro
[TOC]

研究生课，何衍《优化方法及应用》

<!--more-->

## 考评方法

成绩构成：50%大作业 + 30%期末开卷测试（30min，第八周） + 20%平时作业。
大作业单独完成，按质量与规范性打分。测试考察：I. 基本概念与分析方法；II. 基础理论与典型算法。
奖励分：课堂发言、课后讨论、教学反馈、大作业分享。

## 应用软件

- **求解器调用**：Matlab Optimization Toolbox、IBM CPLEX…
- **建模语言**：
  1. **CVX**（Matlab/Julia/Python）：免费求解器 SDPT3、SeDuMi；商业 MOSEK（稀疏）、Gurobi（MIP）
  2. **YALMIP**：支持多数求解器，电网、ML 等领域广泛
  3. **AMPL**：大规模优化，支持大多数求解器
- **代码生成器**：**CVXGEN** — C 代码生成，110 变量 111 约束组合投资问题，4 核 Xeon 0.4ms（SDPT3 需 350ms）

## 学习路线：只需 Boyd 书约 20%

工科研究生所需凸优化核心：

> 少量凸优化基础会明显提升非线性系统学习效率。

**必要**（理解非线性系统 & 控制系统设计）：
1. 凸集合、凸函数定义（第2章）
2. 最优化问题基本结构（3.1）
3. KKT 条件（工程直观解释，第5章前半）
4. LMI、SDP 是凸优化问题（4.6）
5. 用 CVX/CVXPy/YALMIP 求 LMI（实战）

**不必要**（除非优化方向）：Fenchel 对偶、共轭函数、强对偶深度理论、二阶锥复杂结构、一般锥优化理论、凸分析数学证明。

## Norm 范数

### 定义

$\|\cdot\|: \mathbb{R}^n\to \mathbb{R}$ 为范数，需满足：
1. **正定性**：$\|x\|\ge 0,\ \|x\|=0 \iff x=0$
2. **齐次性**：$\|t x\| = |t|\,\|x\|,\ \forall t\in\mathbb{R}$
3. **三角不等式**：$\|x+y\| \le \|x\| + \|y\|$

---

### 常见向量范数

$\ell_p$ 范数：
$$
\|x\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p},\quad p\ge 1
$$
特例：
- $\ell_1$：稀疏性
- $\ell_2$：欧氏距离
- $\ell_\infty$：$\|x\|_\infty = \max_i |x_i|$

---

### 内积与 $\ell_2$ 范数

向量内积：$\langle x,y\rangle = x^T y = \sum_i x_i y_i$
Cauchy–Schwarz：$|\langle x,y\rangle| \le \|x\|_2\|y\|_2$
$\ell_2$ 范数：$\|x\|_2 = \sqrt{\langle x,x\rangle}$

Frobenius 内积：$\langle X,Y\rangle = \text{tr}(X^T Y) = \sum_{i,j} X_{ij}Y_{ij}$
Frobenius 范数：$\|X\|_F = \sqrt{\langle X,X\rangle}$

---

#### $\ell_2$ 几何意义

距离：$\text{dist}(x,y)=\|x-y\|_2$
角度：$\theta = \arccos\bigl(\langle x,y\rangle/(\|x\|_2\|y\|_2)\bigr)$
正交：$\langle x,y\rangle = 0$

**P-二次范数**（$P\succ 0$ 对称正定）：
$$
\|x\|_P = \sqrt{x^T P x}
$$
对应椭球几何。

---

### 对偶范数

给定 $\|\cdot\|$，对偶范数定义：
$$
\|x\|_* = \sup_{\|v\|\le 1} v^T x
$$

| 范数 | 对偶范数 |
| ---- | -------- |
| $\ell_1$ | $\ell_\infty$ |
| $\ell_\infty$ | $\ell_1$ |
| $\ell_2$ | $\ell_2$ |
| $\ell_p$ | $\ell_q$，$1/p+1/q=1$ |

例：$\|x\|_\infty = \sup_{\|v\|_1\le 1} v^T x$

---

### 算子范数

矩阵 $X\in\mathbb{R}^{m\times n}$，范数 $\|\cdot\|_a,\|\cdot\|_b$ 分别定义在 $\mathbb{R}^m,\mathbb{R}^n$ 上：
$$
\|X\|_{a\to b} = \sup_{\|u\|_a\le 1} \|Xu\|_b
$$

典型算子范数：
- **谱范数**：$\|X\|_2 = \sqrt{\lambda_{\max}(X^T X)} = \sigma_{\max}(X)$
- **最大行和**（$\ell_\infty$ 导出）：$\|X\|_\infty = \max_i\sum_j|X_{ij}|$
- **最大列和**（$\ell_1$ 导出）：$\|X\|_1 = \max_j\sum_i|X_{ij}|$

---

#### 算子范数的性质

$$
\|Ax\|_a \le \|A\|_{a\to b}\,\|x\|_b
$$

- $\|Ax\|_2 \le \|A\|_2\|x\|_2$
- $\|Ax\|_1 \le \|A\|_1\|x\|_1$
- $\|Ax\|_\infty \le \|A\|_\infty\|x\|_\infty$
- Frobenius：$\|Ax\|_2 \le \|A\|_F\|x\|_2$

---

### 范数的等价性

有限维 $\mathbb{R}^n$ 中所有范数等价。$\exists\,\alpha,\beta>0$ 使 $\forall x$：
$$
\alpha\|x\|_a \le \|x\|_b \le \beta\|x\|_a
$$
则 $\|\cdot\|_a$ 与 $\|\cdot\|_b$ 等价。

**结论**：一种范数下的收敛性可推广到任意范数。优化中范数选择不影响可解性——仅影响几何形状，不改变拓扑性质（梯度法、连续性分析、可行域紧性等不受影响）。
