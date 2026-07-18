---
title: Quadratic Forms and Positive Definiteness
date: 2026-06-16
tags:
  - linear-algebra
  - mathematics
course: math
status: draft
---
# Quadratic Forms and Positive Definiteness

[TOC]

本篇整理 quadratic forms 与 positive definiteness。重点是把 $x^TAx$ 同时看作能量函数、曲率表达和椭球几何，并理解 symmetric positive definite matrix 为什么在优化、统计和数值计算中反复出现。

- **Quadratic Forms**.

  - Takeaway: quadratic form $x^TAx$ 描述二次能量、曲率与椭球几何；对称部分决定其值。它是 optimization、statistics、PDE discretization 与 geometry 的共同语言。

  - Core Mechanism:

    - **Symmetric Reduction**:

      $$
      x^TAx=x^T\left(\frac{A+A^T}{2}\right)x
      $$

      因此实二次型只由 symmetric part 决定。

    - **Principal Axes**: 若 $A=A^T$，由 spectral theorem

      $$
      A=Q\Lambda Q^T,\qquad
      x^TAx=y^T\Lambda y=\sum_i\lambda_i y_i^2
      $$

      其中 $y=Q^Tx$。正交换基把二次型化成主轴形式。

    - **Definiteness**:

      $$
      A\succ 0 \iff x^TAx>0\quad \forall x\neq 0
      $$

      对 symmetric matrix，$A\succ0$ 当且仅当所有 eigenvalues 为正。

  - Pros:

    - 把几何曲率、优化凸性与矩阵谱性质统一起来。

---

- **Positive Definite Matrices and Cholesky**.

  - Takeaway: symmetric positive definite matrix 是内积、能量与凸二次函数的矩阵表示；Cholesky factorization 是利用正定结构的高效稳定分解。

  - Core Mechanism:

    - **Equivalent Characterizations**: 对实 symmetric matrix $A$，以下条件等价：
      1. $A\succ0$；
      2. 所有 eigenvalues 为正；
      3. 所有 leading principal minors 为正；
      4. 存在 nonsingular $R$ 使 $A=R^TR$；
      5. 存在唯一 lower triangular $L$ 且对角元为正，使 $A=LL^T$。

    - **Cholesky Factorization**:

      $$
      A=LL^T
      $$

      它比一般 LU 更省成本和存储，常用于 covariance matrix、normal equation 与 elliptic PDE 离散系统。

    - **Energy Norm**:

      $$
      \|x\|_A=\sqrt{x^TAx}
      $$

      当 $A\succ0$ 时这是合法 norm。

  - Cons:

    - 正定性对 round-off 与建模误差敏感；半正定或近奇异矩阵需要 pivoted Cholesky、regularization 或 eigenvalue thresholding。

---
