# Intro

研究生课，何衍《优化方法及应用》

<!--more-->

## 考评方法

期末成绩=50%大作业成绩+30%期末课内测试成绩+20%平时作业成绩说明：

1. 大作业：单独完成，按质量和规范性打分。
2. 期末课内测试：30分钟开卷测试（第八周），具体考察：
   - I.对基本概念、分析方法的掌握程度；
   - II.对基础理论、典型算法的理解程度。
3. 奖励分：课堂发言、课后讨论、教学反馈、大作业分享。

## 应用软件

- **直接求解器调用**
  Matlab Optimization Toolbox、IBM CPLEX…

- **优化建模语言编程**
  1) **CVX**: matlab\Julia\Python
     - 免费求解器：SDPT3、SeDuMi
     - 商业求解器：MOSEK（稀疏问题）、Gurobi（MIP）

  2) **YALMIP** (Yet Another LMI Matlab Toolbox & Interior-Point algorithm)：
     支持多数求解器，在电网、机器学习等领域应用广泛。

  3) **AMPL**（A Mathematical Programming Language）：
     大规模优化问题，支持大多数求解器。

- **代码生成器**
  **CVXGEN**：c代码生成器，4核Xeon仅用0.4ms求解110个变量、111个约束的组合投资问题（SDPT3需350ms）。

## 从学习路线来说：你需要的凸优化知识只有 20% 的 Boyd 书

如果你是工科研究生，这里是你真正需要学的“凸优化内容”：

> 但学一点“够用的凸优化基础”会明显提升你学非线性系统的效率。**

### 必要（理解非线性系统 & 控制系统设计必备）

1. **凸集合、凸函数的定义**（Boyd 第 2 章）
2. **最优化问题基本结构**（3.1）
3. **KKT 条件（只需工程版直观解释）**（第 5 章前半）
4. **LMI、SDP 是凸优化问题**（第 4.6 节）
5. **如何使用 CVX/CVXPy/YALMIP 求 LMI**（实战）

### 不必要（除非你学优化方向）

- Fenchel 对偶、共轭函数
- 强对偶的深度理论
- 二阶锥的复杂结构
- 一般锥优化理论
- 凸分析的数学证明

## Norm范数

### Definition

在向量空间 $\mathbb{R}^n$ 中，一个函数 $\|\cdot\|: \mathbb{R}^n \to \mathbb{R}$ 被称为范数，需要满足以下三个性质：

1. **正定性（Positive Definiteness）**
   $$
   \|x\|\ge 0,\quad \|x\|=0 \iff x=0
   $$

2. **齐次性（Homogeneity）**: 对任意标量 $t\in\mathbb{R}$：
   $$
   \|t x\| = |t| \cdot \|x\|
   $$

3. **三角不等式（Triangle Inequality）**
   $$
   \|x + y\| \le \|x\| + \|y\|
   $$

------

### 常见向量范数（Vector Norms）

$\ell_p$ 范数
$$
\|x\|_p = \left( \sum_{i=1}^n |x_i|^p \right)^{1/p},\qquad p\ge 1
$$
常见特例：

- **$\ell_1$**：稀疏性相关

- **$\ell_2$**：欧氏距离

- **$\ell_\infty$**（极限情况）：
  $$
  \|x\|_\infty = \max_i |x_i|
  $$

------

### 内积与 $\ell_2$ 范数

向量内积（Dot Product）
$$
\langle x, y\rangle = x^T y = \sum_{i=1}^n x_i y_i
$$
满足著名的 **Cauchy–Schwartz 不等式**：
$$
|\langle x,y\rangle| \le \|x\|_2\,\|y\|_2
$$
由内积导出的 $\ell_2$ 范数
$$
\|x\|_2 = \sqrt{\langle x,x\rangle}
$$
实矩阵内积（Frobenius 内积）
$$
\langle X,Y\rangle = \text{tr}(X^T Y) = \sum_{i,j} X_{ij} Y_{ij}
$$
导出 **Frobenius 范数**：
$$
\|X\|_F = \sqrt{\langle X,X\rangle}
$$

------

#### $\ell_2$ 范数的几何意义

距离
$$
\text{dist}(x,y)=\|x-y\|_2
$$
角度
$$
\theta = \arccos\left( \frac{\langle x,y\rangle}{\|x\|_2 \|y\|_2} \right)
$$
正交（Orthogonality）
$$
\langle x, y\rangle = 0
$$
P-二次范数

给定 $P\succ 0$（对称正定矩阵）：
$$
\|x\|_{P} = \sqrt{x^T P x}
$$
对应椭球几何。

------

### 对偶范数（Dual Norm）

给定一个范数 $\|\cdot\|$，其对偶范数定义为：
$$
\|x\|_* = \sup_{\|v\|\le 1} v^T x , \quad \text{单位球的支撑超平面}
$$
常见对应关系：

| 范数          | 对偶范数                             |
| ------------- | ------------------------------------ |
| $\ell_1$      | $\ell_\infty$                        |
| $\ell_\infty$ | $\ell_1$                             |
| $\ell_2$      | $\ell_2$                             |
| $\ell_p$      | $\ell_q$，满足 $1/p+1/q=1$，共轭条件 |

例如：
$$
\|x\|_\infty = \sup_{\|v\|_1\le 1} v^Tx
$$

------

### 算子范数（Operator Norms）

将矩阵视为线性变换 $X: \mathbb{R}^n\to \mathbb{R}^m$。设||.||a和||.||b分别为定义在$R^m$和$R^n$上的范数

实矩阵$X\in R^{m\times n}$的导出范数，算子范数定义为：
$$
\|X\|_{a\to b} = \sup_{\|u\|_a\le 1} \|Xu\|_b
$$
典型算子范数：

- 谱范数（Spectral Norm）
  $$
  \|X\|_2 = \sqrt{\lambda_{\max}(X^T X)} = \sigma_{\max}(X)
  $$

- 最大行和范数（Induced by $\ell_\infty$）
  $$
  \|X\|_\infty = \max_i \sum_j |X_{ij}|
  $$

- 最大列和范数（Induced by $\ell_1$）
  $$
  \|X\|_1 = \max_j \sum_i |X_{ij}|
  $$

------

#### 算子范数的性质

对于向量 $x$：
$$
\|Ax\|_a \le \|A\|_{a\to b}\,\|x\|_b
$$
例如：

- $\|Ax\|_2 \le \|A\|_2 \|x\|_2$

- $\|Ax\|_1 \le \|A\|_1 \|x\|_1$

- $\|Ax\|_\infty \le \|A\|_\infty \|x\|_\infty$

- Frobenius 范数满足：
  $$
  \|Ax\|_2 \le \|A\|_F \|x\|_2
  $$

------

### 范数的等价性

在有限维空间 $\mathbb{R}^n$ 中，所有范数都是等价的。形式化定义如下：

若存在常数 $\alpha,\beta>0$，使得对任意向量 $x$：
$$
\alpha \|x\|_a \le \|x\|_b \le \beta \|x\|_a
$$
则称范数 $\|\cdot\|_a$ 与 $\|\cdot\|_b$ 等价。

Conclusion：一种范数下的收敛性可推广到另一种范数。如果 $x_k \to x$ 在某个范数下成立，那么在任何范数下都成立。

这个结论非常重要，例如：优化中“范数选择不影响可解性”，目标函数中使用 $\ell_1$、$\ell_2$、$\ell_\infty$ 只会影响几何形状，但不会改变拓扑性质。例如，梯度法、连续性分析、可行域紧性等都不受范数选择影响。因此证明算法收敛的自由度更大。
