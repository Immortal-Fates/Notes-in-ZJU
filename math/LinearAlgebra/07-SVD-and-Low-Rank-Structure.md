---
title: Singular Value Decomposition and Low-Rank Structure
date: 2026-06-16
tags:
  - linear-algebra
  - mathematics
course: math
status: draft
---
# Singular Value Decomposition and Low-Rank Structure

[TOC]

本篇围绕 SVD 与低秩结构展开。SVD 是任意矩阵最稳定、最通用的结构分解，应重点理解它如何统一解释 rank、四个基本子空间、condition number、pseudoinverse 与最优低秩近似。

- **Singular Value Decomposition**.

  - Takeaway: SVD 把任意矩阵分解为两个正交坐标变换与一个非负对角伸缩，是最通用、最稳定的矩阵结构定理。它同时刻画 rank、range、nullspace、condition number 与 best low-rank approximation。

  - Motivation: eigen-decomposition 只适合方阵且对非 normal 矩阵不总是稳定；SVD 适用于任意 $m\times n$ 矩阵。

  - Core Mechanism:

    - **Decomposition**:

      $$
      A=U\Sigma V^*
      $$

      $U,V$ unitary，$\Sigma$ diagonal-like 且奇异值 $\sigma_1\geq\dots\geq\sigma_r>0$。

    - **Operator Geometry**:

      $V^*$ 先旋转输入坐标，$\Sigma$ 沿坐标轴缩放，$U$ 再旋转输出坐标。单位球在 $A$ 作用下变为 ellipsoid，半轴长度为 singular values。

    - **Relation to Eigenvalues**:

      $$
      A^*A=V\Sigma^2V^*,\qquad
      AA^*=U\Sigma^2U^*
      $$

      right singular vectors 是 $A^*A$ 的 eigenvectors，left singular vectors 是 $AA^*$ 的 eigenvectors。

    - **Rank and Norms**:

      $$
      \operatorname{rank}(A)=\#\{i:\sigma_i>0\},\qquad
      \|A\|_2=\sigma_1,\qquad
      \|A\|_F=\sqrt{\sum_i\sigma_i^2}
      $$

  - Pros:

    - 对任意矩阵存在，数值稳定，解释力强。
    - 统一处理 least squares、pseudoinverse、PCA、matrix compression。

  - Cons:

    - full SVD 对大规模 dense matrix 成本较高，常需要 randomized 或 iterative variants。

---

- **Best Low-Rank Approximation**.

  - Takeaway: 截断 SVD 给出 Frobenius norm 与 spectral norm 下的最优 rank-$k$ 近似。这是 PCA、denoising、compression 与 latent factor model 的数学基础。

  - Core Mechanism:

    - **Truncated SVD**:

      $$
      A_k=\sum_{i=1}^{k}\sigma_i u_i v_i^*
      $$

    - **Eckart-Young-Mirsky Theorem**:

      $$
      \min_{\operatorname{rank}(B)\leq k}\|A-B\|_2=\sigma_{k+1}
      $$

      $$
      \min_{\operatorname{rank}(B)\leq k}\|A-B\|_F
      =
      \sqrt{\sum_{i>k}\sigma_i^2}
      $$

      误差完全由被丢弃的 singular values 控制。

  - Pipeline:

    1. 计算或近似前 $k$ 个 singular triplets。
    2. 构造 $A_k=U_k\Sigma_kV_k^*$。
    3. 用 spectral gap 与 tail energy 判断近似质量。

  - Cons:

    - 最优性依赖 chosen norm；对结构化约束如 sparsity、nonnegativity 不直接适用。

---

- **Moore-Penrose Pseudoinverse**.

  - Takeaway: pseudoinverse $A^+$ 给出 least squares 与 minimum-norm solution 的统一表达，是 rank-deficient systems 的标准广义逆。

  - Core Mechanism:

    - **SVD Definition**: 若 $A=U\Sigma V^*$，则

      $$
      A^+=V\Sigma^+U^*
      $$

      其中 $\Sigma^+$ 把非零奇异值取倒数并转置形状。

    - **Least Squares Solution**:

      $$
      \hat x=A^+b
      $$

      当解不唯一时，它给出所有 least squares 解中 norm 最小者。

    - **Projection Relations**:

      $$
      AA^+=P_{\operatorname{Col}(A)},\qquad
      A^+A=P_{\operatorname{Row}(A)}
      $$

  - Pros:

    - 对 rank-deficient 与 rectangular matrix 都适用。

  - Cons:

    - 小 singular values 的倒数会放大噪声，实际需要 thresholding 或 regularization。

---
