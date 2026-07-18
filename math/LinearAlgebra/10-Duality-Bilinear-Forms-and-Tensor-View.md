---
title: Duality, Bilinear Forms, and Tensor View
date: 2026-06-16
tags:
  - linear-algebra
  - mathematics
course: math
status: draft
---
# Duality, Bilinear Forms, and Tensor View

[TOC]

本篇引入 dual space、linear functionals、bilinear forms 与 adjoint 的视角。重点是区分 vector 与 covector，并理解 inner product 如何把二者联系起来；这对后续学习优化、微分几何、泛函分析和物理记号都很重要。

- **Dual Space and Linear Functionals**.

  - Takeaway: dual space $V^*$ 由所有线性泛函组成，刻画“测量向量”的线性方式。dual viewpoint 是理解 covectors、gradients、adjoints 与 tensor notation 的入口。

  - Core Mechanism:

    - **Dual Space**:

      $$
      V^*=\{\varphi:V\to\mathbb F\mid \varphi \text{ linear}\}
      $$

    - **Dual Basis**: 若 $\mathcal B=(v_1,\dots,v_n)$，则 dual basis $(v^1,\dots,v^n)$ 满足

      $$
      v^i(v_j)=\delta^i_j
      $$

    - **Coordinate Meaning**: 向量坐标是 basis expansion 的系数；covector 坐标描述对 basis vectors 的测量值。换基时二者变换方向相反。

    - **Riesz Representation**: 在有限维 inner product space 中，每个 linear functional 都可唯一写成

      $$
      \varphi(x)=\langle y,x\rangle
      $$

      这里 inner product 提供了 $V$ 与 $V^*$ 的同构，但该同构依赖 inner product。

  - Pros:

    - 明确区分 vector 与 covector，避免在微分几何、优化和物理记号中混淆。

---

- **Adjoint Operators**.

  - Takeaway: adjoint 是在 inner product 下把算子从一个参数“移到另一个参数”的结构。它是 normality、self-adjointness、least squares 与 variational methods 的基础。

  - Core Mechanism:

    - **Definition**:

      $$
      \langle Tx,y\rangle_W=\langle x,T^*y\rangle_V
      $$

      在标准 Euclidean inner product 下，矩阵的 adjoint 是 conjugate transpose $A^*$。

    - **Self-adjoint Operator**:

      $$
      T=T^*
      $$

      有限维复内积空间中 self-adjoint operator 有实特征值与 orthonormal eigenbasis。

    - **Normal Equation Revisited**:

      $$
      A^*(A\hat x-b)=0
      $$

      这是 least squares 在任意 inner product 结构下的自然形式。

  - Pros:

    - 让矩阵转置不再只是符号操作，而是与 inner product 和 dual pairing 绑定的几何对象。

---
