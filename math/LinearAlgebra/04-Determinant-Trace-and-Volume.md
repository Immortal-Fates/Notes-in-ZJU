---
title: Determinant, Trace, and Volume
date: 2026-06-16
tags:
  - linear-algebra
  - mathematics
course: math
status: draft
---
# Determinant, Trace, and Volume

[TOC]

本篇整理 determinant 与 trace 这两个最基本的矩阵不变量。重点是同时掌握它们的代数定义、几何含义与谱解释：determinant 描述有向体积缩放，trace 描述特征值总和与线性算子的整体一阶效应。

- **Determinant**.

  - Takeaway: determinant 是方阵线性变换对 oriented volume 的缩放因子，也是 invertibility、eigenvalue product 与 characteristic polynomial 的核心 invariant。

  - Core Mechanism:

    - **Algebraic Characterization**: determinant 是列向量的 alternating multilinear form，并满足 $\det I=1$。

    - **Geometric Meaning**:

      $$
      |\det A|=\frac{\text{volume of }A\text{-transformed parallelepiped}}{\text{original volume}}
      $$

      符号记录 orientation 是否翻转。

    - **Invertibility Criterion**:

      $$
      A\text{ invertible}\iff \det A\neq 0
      $$

    - **Multiplicativity**:

      $$
      \det(AB)=\det A\det B
      $$

      这说明体积缩放在复合变换下相乘。

    - **Eigenvalue Relation**: 若 $A\in\mathbb C^{n\times n}$ 的特征值按 algebraic multiplicity 计为 $\lambda_1,\dots,\lambda_n$，则

      $$
      \det A=\prod_{i=1}^{n}\lambda_i
      $$

  - Cons:

    - determinant 对高维矩阵的数值大小常极端，不适合直接作为 condition 或 volume 的唯一诊断。

---

- **Trace**.

  - Takeaway: trace 是线性算子对 infinitesimal volume scaling 的一阶刻画，也是特征值之和、矩阵李代数与统计协方差分析中的基本 invariant。

  - Core Mechanism:

    - **Definition**:

      $$
      \operatorname{tr}(A)=\sum_{i=1}^{n}a_{ii}
      $$

    - **Cyclic Property**:

      $$
      \operatorname{tr}(AB)=\operatorname{tr}(BA)
      $$

      更一般地，尺寸匹配时 $\operatorname{tr}(ABC)=\operatorname{tr}(BCA)=\operatorname{tr}(CAB)$。

    - **Similarity Invariance**:

      $$
      \operatorname{tr}(P^{-1}AP)=\operatorname{tr}(A)
      $$

    - **Eigenvalue Relation**:

      $$
      \operatorname{tr}(A)=\sum_{i=1}^{n}\lambda_i
      $$

  - Pros:

    - 在 matrix calculus 中极其方便，常把标量目标写为 trace form。

---
