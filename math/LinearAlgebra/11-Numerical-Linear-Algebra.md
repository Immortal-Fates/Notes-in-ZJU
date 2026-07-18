---
title: Numerical Linear Algebra
date: 2026-06-16
tags:
  - linear-algebra
  - mathematics
course: math
status: draft
---
# Numerical Linear Algebra

[TOC]

本篇从实际计算角度重新审视线性代数。重点是区分 conditioning 与 algorithmic stability，并理解为什么同一个数学问题在 floating-point arithmetic 下需要选择合适的分解、误差指标和预处理方法。

- **Conditioning and Stability**.

  - Takeaway: conditioning 描述问题本身对输入扰动的敏感性，stability 描述算法是否额外放大误差。博士阶段必须区分“问题难”与“算法差”。

  - Motivation: 精确数学定理默认 exact arithmetic；实际计算在 floating-point arithmetic 下会出现 round-off、cancellation 与 loss of orthogonality。

  - Core Mechanism:

    - **Condition Number for Linear Solve**:

      $$
      \kappa_2(A)=\|A\|_2\|A^{-1}\|_2=\frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}
      $$

      若 $\kappa(A)$ 大，则 $Ax=b$ 的解对 $A,b$ 的小扰动高度敏感。

    - **Forward vs Backward Error**: forward error 比较计算解与真实解的距离；backward error 问“计算解是否精确解了一个轻微扰动后的问题”。

    - **Stable Algorithm Principle**: 好算法通常有小 backward error；若问题本身 ill-conditioned，即使 backward stable 也可能有较大 forward error。

  - Pipeline:

    1. 估计 condition number 或 singular values。
    2. 选择结构适配的算法：SPD 用 Cholesky，least squares 用 QR/SVD，eigenproblem 用 Schur。
    3. 评估 residual 与 backward error。
    4. 对 ill-conditioned case 使用 scaling、regularization 或 higher precision。

  - Cons:

    - 小 residual 不一定意味着小 solution error，尤其当 $A$ ill-conditioned 时。

---

- **Schur Decomposition**.

  - Takeaway: Schur decomposition 是实际 eigenvalue computation 的核心形式；它用 unitary similarity 把矩阵化为 triangular form，比 Jordan form 数值稳定。

  - Core Mechanism:

    - **Complex Schur Form**:

      $$
      A=QTQ^*
      $$

      其中 $Q$ unitary，$T$ upper triangular，$T$ 的对角元是 $A$ 的 eigenvalues。

    - **Why It Matters**: Schur form 保留 unitary transformation 的稳定性，同时避免 Jordan form 对扰动的极端敏感。

    - **Normal Matrix Special Case**: 若 $A$ normal，则 Schur form 中 $T$ 实际为 diagonal，这正是 spectral theorem。

  - Pros:

    - 是 QR algorithm、matrix functions 与稳定谱分析的基础。

---

- **Krylov Subspace Methods**.

  - Takeaway: Krylov methods 利用矩阵-向量乘法在低维子空间中逼近大规模线性系统或特征值问题，适合 sparse 或 matrix-free settings。

  - Motivation: 当矩阵规模巨大时，显式分解 $A$ 的 $O(n^3)$ 成本不可接受；但计算 $Av$ 可能很便宜。

  - Core Mechanism:

    - **Krylov Subspace**:

      $$
      \mathcal K_k(A,b)=\operatorname{span}\{b,Ab,A^2b,\dots,A^{k-1}b\}
      $$

    - **Approximation Principle**: 在 $\mathcal K_k$ 中寻找近似解或 Ritz eigenpairs，把高维问题投影成小规模问题。

    - **Representative Algorithms**:
      1. Conjugate Gradient: 用于 SPD linear systems；
      2. GMRES: 用于一般 nonsymmetric linear systems；
      3. Lanczos: 用于 Hermitian eigenproblems；
      4. Arnoldi: 用于一般 eigenproblems。

  - Cons:

    - 收敛速度强依赖 spectrum distribution 与 preconditioner quality。
    - 有限精度下可能丢失正交性，需要 restart 或 reorthogonalization。

---
