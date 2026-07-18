---
title: Linear Systems and Gaussian Elimination
date: 2026-06-16
tags:
  - linear-algebra
  - mathematics
course: math
status: draft
---
# Linear Systems and Gaussian Elimination

[TOC]

本篇围绕线性方程组展开，核心是理解 $Ax=b$ 的可解性、解空间结构与 Gaussian elimination 的代数意义。阅读时应把行变换、pivot、rank、nullspace 与 affine solution set 联系起来，而不是只记消元步骤。

- **线性方程组的结构**.

  - Takeaway: 方程 $Ax=b$ 的解集由一个 particular solution 加上 homogeneous solution space 构成。可解性由 $b$ 是否落在 column space 中决定。

  - Core Mechanism:

    - **Consistency**:

      $$
      Ax=b \text{ solvable}\iff b\in\operatorname{Col}(A)
      $$

    - **Solution Set**: 若 $x_p$ 是一个特解，则全部解为

      $$
      x=x_p+z,\qquad z\in\ker A
      $$

      因此解集是 affine subspace，而齐次解集是 linear subspace。

    - **Four Fundamental Subspaces**:

      $$
      \operatorname{Col}(A),\quad \operatorname{Null}(A),\quad
      \operatorname{Row}(A)=\operatorname{Col}(A^T),\quad
      \operatorname{Null}(A^T)
      $$

      它们将 domain 与 codomain 分解为可解释的正交结构：

      $$
      \operatorname{Null}(A)=\operatorname{Row}(A)^\perp,\qquad
      \operatorname{Null}(A^T)=\operatorname{Col}(A)^\perp
      $$

  - Pipeline:

    1. 对 augmented matrix $[A\mid b]$ 做 row reduction。
    2. 判断是否出现 $[0\ \cdots\ 0\mid c]$ 且 $c\neq 0$。
    3. 找 pivot variables 与 free variables。
    4. 写出特解与 nullspace basis。

---

- **Gaussian Elimination and LU Factorization**.

  - Takeaway: Gaussian elimination 用 elementary row operations 把线性系统化为三角系统；矩阵分解视角下，它对应 $PA=LU$，是直接法求解线性系统的基础。

  - Motivation: 直接求逆矩阵通常不稳定也不高效；实际计算更偏向 factorize once, solve many times。

  - Core Mechanism:

    - **LU Factorization**:

      $$
      PA=LU
      $$

      其中 $P$ 是 permutation matrix，$L$ 是 lower triangular matrix，$U$ 是 upper triangular matrix。$P$ 来自 pivoting，用于改善数值稳定性并避免零主元。

    - **Solve Flow**:

      $$
      Ax=b
      \quad\Rightarrow\quad
      LUx=Pb
      \quad\Rightarrow\quad
      Ly=Pb,\ Ux=y
      $$

    - **Computational Cost**: dense $n\times n$ LU 的主成本为 $O(n^3)$，之后每个 right-hand side 的 triangular solve 为 $O(n^2)$。

  - Pipeline:

    1. 选主元并必要时交换行。
    2. 用 multiplier 消去主元下方元素。
    3. 记录 multiplier 到 $L$，消去结果形成 $U$。
    4. 对 $b$ 做同样 permutation。
    5. forward substitution + backward substitution。

  - Cons:

    - 对 ill-conditioned matrix，精确算法结构不能保证高精度结果。
    - 无 pivoting 的 LU 对很多矩阵不稳定或不存在。

---
