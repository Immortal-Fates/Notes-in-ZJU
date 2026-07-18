---
title: Vector Spaces and Linear Maps
date: 2026-06-16
tags:
  - linear-algebra
  - mathematics
course: math
status: draft
---
# Vector Spaces and Linear Maps

[TOC]

本篇讨论线性代数最基础也最抽象的对象：向量空间、子空间、基、维数与线性映射。学习重点是把“矩阵运算”提升为“线性映射在不同基下的表示”，从而理解哪些性质依赖坐标，哪些性质是内在不变量。

- **向量空间与子空间**.

  - Takeaway: 向量空间是允许线性组合的抽象结构；子空间是保持线性组合封闭的集合。所有有限维线性代数结论最终都依赖 span、linear independence、basis 与 dimension 的相互关系。

  - Core Mechanism:

    - **Vector Space**: 域 $\mathbb F$ 上的向量空间 $V$ 是带有加法和标量乘法的集合，满足线性运算公理。常见域为 $\mathbb R$、$\mathbb C$，研究 spectral theory 时 $\mathbb C$ 通常更自然。

    - **Span and Independence**:

      $$
      \operatorname{span}\{v_1,\dots,v_k\}
      =\left\{\sum_{i=1}^{k}a_i v_i:a_i\in\mathbb F\right\}
      $$

      若 $\sum_i a_i v_i=0$ 只推出所有 $a_i=0$，则 $\{v_i\}$ 线性无关。

    - **Basis and Dimension**: basis 同时满足 spanning 与 independence。有限维空间中，任意两个 basis 的元素个数相同，称为 $\dim V$。

    - **Subspace Lattice**: 子空间可用 sum、intersection、direct sum 组织：

      $$
      U+W=\{u+w:u\in U,w\in W\},\qquad
      U\oplus W \iff U\cap W=\{0\}
      $$

  - Pros:

    - 坐标无关，适合描述函数空间、多项式空间、解空间等非矩阵对象。
    - dimension 提供强约束，使很多存在性问题可转化为维数计数。

---

- **线性映射与矩阵表示**.

  - Takeaway: 线性映射保留线性组合，是线性代数的核心对象；矩阵是线性映射在给定基下的表示。理解换基与相似变换，是理解特征值、对角化和 canonical form 的前提。

  - Core Mechanism:

    - **Linear Map**: 映射 $T:V\to W$ 线性当且仅当

      $$
      T(\alpha u+\beta v)=\alpha T(u)+\beta T(v)
      $$

    - **Matrix Representation**: 若 $\mathcal B=(v_1,\dots,v_n)$ 是 $V$ 的基，$\mathcal C=(w_1,\dots,w_m)$ 是 $W$ 的基，则矩阵的第 $j$ 列是 $T(v_j)$ 在 $\mathcal C$ 下的坐标。

    - **Change of Basis**: 若 $P$ 把新坐标变成旧坐标，则同一个算子在新基下的矩阵为

      $$
      A_{\text{new}}=P^{-1}A_{\text{old}}P
      $$

      因此 similarity 是“同一个线性算子在不同基下”的矩阵等价。

    - **Kernel and Image**:

      $$
      \ker T=\{v:T(v)=0\},\qquad
      \operatorname{im}T=\{T(v):v\in V\}
      $$

      它们分别描述被压缩掉的方向与可到达的方向。

  - Pipeline:

    1. 选择 domain/codomain 的 basis。
    2. 计算每个 basis vector 的像。
    3. 把像写成 codomain basis 下的坐标。
    4. 将坐标列拼成矩阵。
    5. 若换基，用 $P^{-1}AP$ 或 $Q^{-1}AP$ 处理 domain/codomain 不同基的情况。

---

- **Rank-Nullity Theorem**.

  - Takeaway: 对有限维线性映射 $T:V\to W$，domain 的维数被精确分解为 kernel 的自由度与 image 的有效自由度。这是解线性方程、判断 injective/surjective 与理解 dimension counting 的基础。

  - Core Mechanism:

    - **Theorem**:

      $$
      \dim V=\dim\ker T+\dim\operatorname{im}T
      $$

    - **Proof Intuition**: 取 $\ker T$ 的一组基，并扩张成 $V$ 的基。扩张出来的向量在 $T$ 下的像构成 $\operatorname{im}T$ 的基，因此 domain 的基被分成“映到 0 的方向”和“生成 image 的方向”。

    - **Matrix Form**: 对 $A\in\mathbb F^{m\times n}$，

      $$
      n=\operatorname{nullity}(A)+\operatorname{rank}(A)
      $$

      其中 rank 是 column space 的维数，也是 row space 的维数。

  - Pros:

    - 用一个维数公式统一解释自由变量数量、解空间维数与约束独立性。

---
