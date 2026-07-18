---
title: Eigenvalues, Diagonalization, and Spectral Theory
date: 2026-06-16
tags:
  - linear-algebra
  - mathematics
course: math
status: draft
---
# Eigenvalues, Diagonalization, and Spectral Theory

[TOC]

本篇讨论谱理论的核心内容，包括 eigenvalues、diagonalization、spectral theorem 与 Jordan form。学习重点是区分一般矩阵、normal matrix 与 defective matrix 的结构差异，并理解谱分解如何控制线性动力系统和矩阵函数。

- **Eigenvalues and Eigenvectors**.

  - Takeaway: eigenvector 是在线性变换下只发生尺度变化的方向；eigenvalue 描述该方向的伸缩因子。谱信息揭示线性动力系统、微分方程、Markov chains 与 quadratic forms 的长期行为。

  - Core Mechanism:

    - **Definition**:

      $$
      Av=\lambda v,\qquad v\neq 0
      $$

      等价于 $(A-\lambda I)v=0$ 有非零解。

    - **Characteristic Polynomial**:

      $$
      p_A(\lambda)=\det(\lambda I-A)
      $$

      其根是 $A$ 的 eigenvalues。

    - **Algebraic vs Geometric Multiplicity**: algebraic multiplicity 是 $\lambda$ 作为 characteristic polynomial 根的重数；geometric multiplicity 是 $\dim\ker(A-\lambda I)$。

      $$
      1\leq \dim\ker(A-\lambda I)\leq \text{algebraic multiplicity of }\lambda
      $$

  - Cons:

    - 非 normal 矩阵的 eigenvectors 可高度病态，eigenvalue analysis 可能对扰动非常敏感。

---

- **Diagonalization**.

  - Takeaway: 矩阵可对角化当且仅当存在一组 eigenvector basis。对角化将复杂的线性算子分解为互不耦合的一维尺度变换，是计算矩阵幂、指数与动力系统的核心工具。

  - Core Mechanism:

    - **Similarity Form**:

      $$
      A=PDP^{-1}
      $$

      $P$ 的列是 eigenvectors，$D$ 的对角元是对应 eigenvalues。

    - **Criterion**: $A\in\mathbb C^{n\times n}$ 可对角化当且仅当所有 eigenspaces 的维数之和为 $n$。

    - **Matrix Powers**:

      $$
      A^k=PD^kP^{-1}
      $$

      因此长期行为由 $|\lambda_i|$ 最大的谱成分主导。

  - Pros:

    - 极大简化函数演算 $f(A)=Pf(D)P^{-1}$。

  - Cons:

    - 重特征值不保证足够 eigenvectors；defective matrix 需要 Jordan form。

---

- **Spectral Theorem for Normal Operators**.

  - Takeaway: 复内积空间中 normal operator 可被 unitary diagonalization 完全分解。它是线性代数中最强、最稳定的结构定理之一。

  - Core Mechanism:

    - **Normal Matrix**:

      $$
      A^*A=AA^*
      $$

      Hermitian、skew-Hermitian、unitary 都是 normal matrix 的重要特例。

    - **Spectral Theorem**:

      $$
      A=U\Lambda U^*
      $$

      其中 $U$ unitary，$\Lambda$ diagonal。对 Hermitian matrix，$\Lambda$ 的元素全为实数。

    - **Orthogonal Spectral Decomposition**:

      $$
      A=\sum_{i=1}^{n}\lambda_i u_i u_i^*
      $$

      这表示 $A$ 沿一组 orthonormal eigenbasis 分别缩放。

  - Pros:

    - unitary transformation 保持 norm，数值稳定性优于一般 similarity。
    - 直接支撑 PCA、quantum observables、positive definite analysis。

  - Cons:

    - 一般非 normal 矩阵不满足 orthonormal eigenbasis，不能强行套用该定理。

---

- **Jordan Canonical Form**.

  - Takeaway: Jordan form 描述不可对角化矩阵的最细代数结构，把算子分解为 eigenvalue 加 nilpotent coupling。它揭示了 repeated eigenvalues 下的 generalized eigenvectors 与 polynomial growth。

  - Motivation: 当矩阵缺少足够 eigenvectors 时，对角化失败；Jordan form 是复数域上 similarity classification 的 canonical answer。

  - Core Mechanism:

    - **Jordan Block**:

      $$
      J_k(\lambda)=
      \begin{bmatrix}
      \lambda & 1 &        & 0\\
              & \lambda & \ddots & \\
              &         & \ddots & 1\\
      0       &         &        & \lambda
      \end{bmatrix}
      =\lambda I+N
      $$

      其中 $N$ nilpotent，满足 $N^k=0$。

    - **Generalized Eigenvectors**:

      $$
      (A-\lambda I)^m v=0
      $$

      Jordan chain 记录 eigenvector 之外的广义方向。

    - **Power Behavior**:

      $$
      J_k(\lambda)^t=(\lambda I+N)^t
      =\sum_{j=0}^{k-1}\binom{t}{j}\lambda^{t-j}N^j
      $$

      因此 defective part 会引入 polynomial factor。

  - Cons:

    - Jordan form 数值上非常不稳定；实际计算更常用 Schur decomposition。

---
