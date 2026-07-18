---
title: Review Checklist
date: 2026-06-16
tags:
  - linear-algebra
  - mathematics
course: math
status: draft
---
# Review Checklist

[TOC]

本篇是整套线性代数笔记的复习检查表，用来判断理解是否达到研究阶段的可迁移标准。复习时应要求自己能在抽象定义、矩阵表示、几何图像、证明直觉与数值算法之间自由切换。

- **博士级复习检查表**.

  - Takeaway: 判断自己是否真正掌握线性代数，关键不是会算多少题，而是能否在 abstract、geometric、algorithmic、numerical 四个层面来回切换。

  - Core Mechanism:

    - **Abstract Layer**:
      1. 能区分 vector、covector、linear map、matrix representation。
      2. 能解释 quotient space、dual space、direct sum 与 invariant subspace。
      3. 能说明不同等价关系下的 complete/partial invariants。

    - **Spectral Layer**:
      1. 能判断 diagonalizable、normal、self-adjoint、positive definite。
      2. 能解释 algebraic/geometric multiplicity 与 Jordan chain。
      3. 能说明 spectral theorem、SVD、Schur decomposition 的适用范围与差异。

    - **Geometric Layer**:
      1. 能用 projection 解释 least squares。
      2. 能用 ellipsoid 解释 SVD。
      3. 能用 principal axes 解释 quadratic forms。

    - **Computational Layer**:
      1. 能根据矩阵结构选择 LU、QR、Cholesky、SVD、Schur 或 Krylov methods。
      2. 能区分 conditioning、stability、residual、forward error、backward error。
      3. 能解释为什么显式求逆通常不是求解线性系统的好方法。

  - Pipeline:

    1. 对每个定理写出 precise statement。
    2. 给出至少一个 proof intuition。
    3. 写出一个坐标无关版本和一个矩阵版本。
    4. 给出一个几何解释和一个数值计算注意点。
    5. 用一个应用场景检验理解是否可迁移。
