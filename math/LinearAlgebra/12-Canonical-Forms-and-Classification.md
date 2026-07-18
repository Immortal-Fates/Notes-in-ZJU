---
title: Canonical Forms and Classification
date: 2026-06-16
tags:
  - linear-algebra
  - mathematics
course: math
status: draft
---
# Canonical Forms and Classification

[TOC]

本篇关注矩阵与线性结构的分类问题。重点是区分 equivalence、similarity 与 congruence 这三种不同等价关系，因为它们对应不同数学对象，也保留不同的不变量。

- **Equivalence, Similarity, and Congruence**.

  - Takeaway: 不同矩阵等价关系对应不同数学对象：equivalence 分类线性映射，similarity 分类线性算子，congruence 分类 bilinear/quadratic forms。混淆等价关系会导致错误的 invariant。

  - Core Mechanism:

    - **Matrix Equivalence**:

      $$
      B=Q^{-1}AP
      $$

      对应 domain 与 codomain 同时换基；rank 是 complete invariant。

    - **Similarity**:

      $$
      B=P^{-1}AP
      $$

      对应同一空间上线性算子的换基；eigenvalues、trace、determinant、Jordan structure 是 invariant。

    - **Congruence**:

      $$
      B=P^TAP
      $$

      对应 bilinear/quadratic form 的换基；对实 symmetric forms，signature 是关键 invariant。

  - Pros:

    - 明确“什么可以被变换消去，什么必须保留”。

---

- **Rank Factorization and Fundamental Decomposition**.

  - Takeaway: rank-$r$ 矩阵可以分解为两个满秩矩阵的乘积，表达其 action 只通过 $r$ 维有效通道传递。它是 low-rank modeling、matrix completion 与压缩表示的基础。

  - Core Mechanism:

    - **Rank Factorization**:

      $$
      A=CR
      $$

      其中 $C\in\mathbb F^{m\times r}$ 列满秩，$R\in\mathbb F^{r\times n}$ 行满秩。

    - **Mapping View**:

      $$
      \mathbb F^n \xrightarrow{R} \mathbb F^r \xrightarrow{C} \mathbb F^m
      $$

      $R$ 提取 $r$ 个有效坐标，$C$ 将这些坐标嵌入 output space。

    - **SVD as Orthogonal Rank Factorization**:

      $$
      A=U_r\Sigma_r V_r^*
      $$

      这是带 orthogonality 与 ordered singular values 的规范低秩分解。

  - Pros:

    - 直接揭示 rank bottleneck 与 intrinsic dimension。

---
