---
title: Matrix Functions and Linear Dynamical Systems
date: 2026-06-16
tags:
  - linear-algebra
  - mathematics
course: math
status: draft
---
# Matrix Functions and Linear Dynamical Systems

[TOC]

本篇讨论如何把标量函数推广到矩阵，以及这种推广在线性动力系统中的作用。重点是理解 $e^{tA}$、$A^t$ 与 spectrum 的关系，同时注意 non-normal matrix 中 eigenvalues 不能完全解释短期增长。

- **Matrix Functions**.

  - Takeaway: matrix function $f(A)$ 把标量函数推广到线性算子，是解 ODE、Markov processes、graph diffusion 与 quantum evolution 的基础。定义方式应与 spectrum 和 canonical form 匹配。

  - Core Mechanism:

    - **Diagonalizable Case**:

      $$
      A=PDP^{-1}\quad\Rightarrow\quad f(A)=Pf(D)P^{-1}
      $$

      其中 $f(D)$ 对每个 diagonal entry 逐点作用。

    - **Power Series Definition**:

      $$
      f(A)=\sum_{k=0}^{\infty}c_kA^k
      $$

      对 matrix exponential 尤其重要：

      $$
      e^{A}=\sum_{k=0}^{\infty}\frac{A^k}{k!}
      $$

    - **ODE Solution**:

      $$
      \frac{dx}{dt}=Ax,\qquad x(t)=e^{tA}x(0)
      $$

      长期行为由 spectral abscissa $\max_i \operatorname{Re}\lambda_i$ 与 Jordan block size 共同决定。

  - Cons:

    - 对非 normal 矩阵，eigenvalues 不能完整解释 transient growth，需关注 pseudospectrum 或 singular values。

---

- **Linear Dynamical Systems**.

  - Takeaway: 离散系统 $x_{t+1}=Ax_t$ 与连续系统 $\dot x=Ax$ 的稳定性主要由 spectrum 控制，但非 normality 会导致短期放大。

  - Core Mechanism:

    - **Discrete Stability**:

      $$
      x_t=A^tx_0,\qquad A^t\to0 \iff \rho(A)<1
      $$

      这里 $\rho(A)=\max_i|\lambda_i|$ 是 spectral radius。

    - **Continuous Stability**:

      $$
      e^{tA}\to0 \iff \max_i\operatorname{Re}(\lambda_i)<0
      $$

      对 defective matrix，还要考虑 Jordan block 引入的 polynomial factor，但指数衰减主导长期行为。

    - **Normal vs Non-normal**: normal matrix 的增长可由 eigenvalues 直接控制；non-normal matrix 即使所有 eigenvalues 稳定，也可能出现 transient growth。

  - Pipeline:

    1. 求 spectrum 或估计 spectral radius/spectral abscissa。
    2. 判断 diagonalizable、normality 与 condition of eigenbasis。
    3. 分析长期稳定性。
    4. 对 non-normal case，补充检查 $\|A^t\|$ 或 $\|e^{tA}\|$。

---
