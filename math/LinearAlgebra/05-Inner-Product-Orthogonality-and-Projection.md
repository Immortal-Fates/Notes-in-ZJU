---
title: Inner Product, Orthogonality, and Projection
date: 2026-06-16
tags:
  - linear-algebra
  - mathematics
course: math
status: draft
---
# Inner Product, Orthogonality, and Projection

[TOC]

本篇为向量空间加入 inner product 结构，从而引入长度、角度、正交分解与投影。重点是理解 least squares 的几何本质：在无法精确求解时，最佳近似来自对 column space 的 orthogonal projection。

- **Inner Product Space**.

  - Takeaway: inner product 给向量空间加入长度、角度与正交性，使代数对象具有几何结构。许多最优化、投影、谱分解与 Fourier-type 展开都依赖 inner product。

  - Core Mechanism:

    - **Definition**: 复向量空间上 inner product $\langle x,y\rangle$ 通常对第一个变量 conjugate-linear，对第二个变量 linear，并满足 positive definiteness。

    - **Norm Induced by Inner Product**:

      $$
      \|x\|=\sqrt{\langle x,x\rangle}
      $$

    - **Cauchy-Schwarz Inequality**:

      $$
      |\langle x,y\rangle|\leq \|x\|\|y\|
      $$

      等号成立当且仅当 $x,y$ 线性相关。

    - **Orthogonal Decomposition**: 若 $W$ 是有限维 inner product space 的子空间，则

      $$
      V=W\oplus W^\perp
      $$

  - Pros:

    - 把 approximation 问题转化为 orthogonal projection。
    - 让 self-adjoint、unitary、normal operators 具有强 canonical structure。

---

- **Orthogonal Projection and Least Squares**.

  - Takeaway: least squares 的本质是把 $b$ 投影到 $\operatorname{Col}(A)$ 上，使 residual 与 column space 正交。它是 overdetermined systems、regression 与 Hilbert space approximation 的共同核心。

  - Motivation: 当 $Ax=b$ 无精确解时，目标变为寻找使 $\|Ax-b\|_2$ 最小的 $x$。

  - Core Mechanism:

    - **Normal Equation**:

      $$
      A^T A\hat x=A^T b
      $$

      它来自 residual $r=b-A\hat x$ 与 $\operatorname{Col}(A)$ 正交，即 $A^Tr=0$。

    - **Projection Matrix**: 若 $A$ 列满秩，则投影到 $\operatorname{Col}(A)$ 的矩阵为

      $$
      P=A(A^TA)^{-1}A^T
      $$

      它满足 $P^2=P$ 与 $P^T=P$。

    - **QR Version**: 若 $A=QR$ 且 $Q$ 列正交归一，则

      $$
      \hat x=R^{-1}Q^Tb,\qquad P=QQ^T
      $$

      QR 通常比显式 normal equation 更稳定，因为 $A^TA$ 会平方 condition number。

  - Pipeline:

    1. 判断 $A$ 是否列满秩。
    2. 优先用 QR 或 SVD 求 least squares。
    3. 计算 $\hat b=A\hat x$ 与 residual $r=b-\hat b$。
    4. 检查 $A^Tr\approx 0$ 作为几何验证。

  - Cons:

    - normal equation 在 ill-conditioned 问题上可能显著损失精度。

---

- **Gram-Schmidt and QR Factorization**.

  - Takeaway: Gram-Schmidt 把一组线性无关向量转化为 orthonormal basis；矩阵形式就是 $A=QR$。它连接 basis construction、least squares 与 numerical orthogonalization。

  - Core Mechanism:

    - **Classical Gram-Schmidt**:

      $$
      u_k=a_k-\sum_{j=1}^{k-1}\langle a_k,q_j\rangle q_j,\qquad
      q_k=\frac{u_k}{\|u_k\|}
      $$

      每一步从当前向量中减去已有正交方向上的投影。

    - **QR Factorization**:

      $$
      A=QR
      $$

      $Q$ 的列为 orthonormal vectors，$R$ 为 upper triangular matrix。

    - **Numerical Note**: Modified Gram-Schmidt、Householder reflection 与 Givens rotation 通常比 classical Gram-Schmidt 更稳定。

  - Pros:

    - 正交变换保持 2-norm，适合稳定算法设计。

  - Cons:

    - Classical Gram-Schmidt 在列向量接近线性相关时容易丢失正交性。

---
