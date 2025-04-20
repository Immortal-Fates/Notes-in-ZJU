# Main Takeaway

配套CMU-16-745 Optimal Control and Reinforcement Learning食用



<!--more-->



# ZJU-Optimization & Optimal Control

- Lec 5-8：做轨迹规划

## Lec 5 Constrained Optimization

这节主要介绍以下几个方面

- Focus on constrained optimizations
- QP Problem,EQP,EIQP
- $NLP\to SQP$

### Problem Statement

$$
Minimize\quad f(x)\\

subject~~ to\quad
g_j(x) = 0, j = 1, \ldots, m_1 \\
\quad g_j(x) \leq 0, j = m_1 + 1, \ldots, m_1 + m_2
$$

Such problems are called **nonlinear programming problems or mathematical programming problems.**

**Definitions**:
- A point $x \in \mathbb{R}^n$ is **feasible** if it satisfies all constraints.

- The **feasible region** $\mathcal{X}$ is the set of all feasible points.

- The $j$-th inequality constraint is **active** at $x$ if $g_j(x) = 0$.

  > 在不等式边界上才称为active

- The **active set** at $x$ is:
  $$J(x) = \left\{ j \in \{ m_1 + 1, \ldots, m_1 + m_2 \} : g_j(x) = 0 \right\} = \\ indices~~ of ~~active~~ inequality~~ constraints$$



### Problem Solving

#### KKT Conditions

推导：[Karush-Kuhn-Tucker (KKT)条件](https://zhuanlan.zhihu.com/p/38163970)

下面是KKT条件的完整叙述：

KKT条件是确认候选点是否为严格局部极小值的**一阶必要条件**

Suppose that $x^* \in \mathcal{X}$ is a local minimizer of the constrained optimization problem.

If $x^*$ is a regular point (i.e., the LICQ holds), then there exists a vector of Lagrange multipliers $\lambda^* = [\lambda_1^*, \ldots, \lambda_{m_1+m_2}^*]^\top$ 
$$
L(x^*, \lambda^*) = f(x)+\sum_{i=1}^{m_1}\lambda_i g_i(x)+\sum_{i=m_1+1}^{m_1+m_2}\lambda_i g_i(x)
$$
such that the following conditions are satisfied:
$$
\frac{\partial L(x^*, \lambda^*)}{\partial x} = 0
$$

primal feasibility（原始可行性）:
$$
g_j(x^*) = 0, \quad j = 1, \ldots, m_1
$$

$$
g_j(x^*) \leq 0, \quad j = m_1 + 1, \ldots, m_1 + m_2
$$

complementary feasibility（互补松弛条件）:
$$
\lambda_j^* g_j(x^*) = 0, \quad j = 1, \ldots, m_1 + m_2
$$

dual feasibility（对偶可行性）:
$$
\lambda_j^* \geq 0, \quad j = m_1 + 1, \ldots, m_1 + m_2
$$
The conditions $\lambda_j^* g_j(x^*) = 0$ are called **complementarity conditions**.

The complementarity conditions imply that either constraint $j$ is active or the corresponding Lagrange multiplier $\lambda_j^* = 0$ (or possibly both).

Given a local minimizer $x^* \in \mathcal{X}$ and a corresponding Lagrange multiplier vector $\lambda^*$ satisfying the KKT conditions, we say that the **strict complementarity condition** holds if exactly one of $\lambda_j^*$ or $g_j(x^*)$ is zero for each index $j \in \{m_1 + 1, \ldots, m_1 + m_2\}$.

##### 对偶性和正则化

然后我们来介绍一下对偶性和正则化

考虑仅含等式约束的优化问题：

$$
\min_{x \in \mathbb{R}^n} f(x) \quad \text{s.t.} \quad h(x) = 0
$$
上述问题可以写成
$$
\begin{aligned}
&\min_{x} f(x) + P_{\infty}(h(x)) \\
&\text{where } P_{\infty}(x) = 
\begin{cases} 
0, & x = 0 \\
+\infty, & x \neq 0
\end{cases}
\end{aligned}
$$
等价地我们可以转化为如下形式
$$
d(\lambda) := \min_x \left[ f(x) + \lambda^\top h(x) \right]
$$

$$
\min_x \max_\lambda \left[ 
  f(x) + \lambda^\top h(x) 
\right]
$$

在$h(x)$不等于0时，内层的max一定会把问题最大话成无穷大，所以当约束不满足时这个最优化问题肯定取不到最小值

对于**大部分的**凸优化问题，可以证明将min和max的符号切换后结果不变，这个结果也叫做强对偶

若我们用KKT条件解优化问题，得到的解是关于$x,\lambda$的**驻点**，假如最终得到的点x和λ确实是最小化的解，那么说明(3)式是成立的，则说明得到的x是极小值点而λ是极大值点。因此，对于**带约束的问题的正则化**（目的是**保证优化方向一直向着最小化方向**走），则其Hessian阵应该有dim(x)个正的特征值，有dim(λ)个负的特征值。这样的KKT系统叫做"**Quasi-definite''系统**.

> 这种特征值分布使得KKT系统具有“准对角”性质，即通过添加适当的正则项，可以调整Hessian矩阵的特征值，使其满足上述条件。这在实际应用中非常有用，尤其是在使用牛顿法等迭代优化方法时，通过确保Hessian矩阵的“quasi-definiteness”，可以有效避免迭代过程中可能出现的不收敛或收敛到非极小值点的问题

下面给出一个带约束问题的regularization的实例

在牛顿法进行迭代时在某些初始点不能收敛到极小值，因此我们要对其添加正则化

```
function regularized_newton_step(x,λ)
    H = ∇2f(x) + ForwardDiff.jacobian(xn -> ∂c(xn)'*λ, x)
    C = ∂c(x)
    K = [H C'; C 0]
    e = eigvals(K)  # 这种判断Hessian是不是"quasi-definite"的方式是非常expensive的
    while !(sum(e .> 0) == length(x) && sum(e .< 0) == length(λ))   # 不断添加正则项，直到满足要求
        K = K + Diagonal([ones(length(x)); -ones(length(λ))])
        e = eigvals(K)
    end
    Δz = K\[-∇f(x)-C'*λ; -c(x)]
    Δx = Δz[1:2]
    Δλ = Δz[3]
    return Δx, Δλ
end
```

> 解决了收敛于极值小值点的问题，但是仍然存在超调现象

KKT条件只能找到驻点，无法区分极小、极大或鞍点。

接着我们介绍确认候选点是否为严格局部极小值的**二阶充分条件**：二阶条件通过曲率分析，确保目标函数在可行方向上“向上弯曲”——排除鞍点

#### **Second-order Sufficient Conditions**

Consider the Hessian (with respect to $x$) of the Lagrangian function $L(x, \lambda)$:
$$
\nabla_x^2 L(x, \lambda) = 
\begin{bmatrix}
\frac{\partial^2 L}{\partial x_1^2} & \frac{\partial^2 L}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 L}{\partial x_1 \partial x_n} \\
\frac{\partial^2 L}{\partial x_2 \partial x_1} & \frac{\partial^2 L}{\partial x_2^2} & \cdots & \frac{\partial^2 L}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 L}{\partial x_n \partial x_1} & \frac{\partial^2 L}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 L}{\partial x_n^2}
\end{bmatrix}
= \frac{\partial}{\partial x} \left( \frac{\partial L}{\partial x} \right)^\top
$$
Suppose that a feasible point $x^* \in \mathcal{X}$ satisfies the KKT conditions for some Lagrange multiplier vector $\lambda^* \in \mathbb{R}^{m_1 + m_2}$. If the following condition holds:
$$
p^\top \nabla_x^2 L(x^*, \lambda^*) p > 0, \quad \forall p \in M^* \setminus \{0\}
$$
where $M^*$ is the set of vectors $w \in \mathbb{R}^n$ satisfying:

$$
\begin{aligned}
w^\top \nabla g_j(x^*) &= 0, && j = 1, \ldots, m_1 \quad \text{(等式约束)} \\
w^\top \nabla g_j(x^*) &= 0, && j \in \left\{ i \in J(x^*) : \lambda_i^* > 0 \right\} \quad \text{(活跃且乘子正的不等式约束)} \\
w^\top \nabla g_j(x^*) &\geq 0, && j \in \left\{ i \in J(x^*) : \lambda_i^* = 0 \right\} \quad \text{(活跃但乘子零的不等式约束)}
\end{aligned}
$$
then $x^*$ is a **strict local minimum** of the constrained optimization problem

这里$M^*$为在可行域$x^*$定义的切锥

> 因此不能直接看Hessian矩阵是否正定，只要$p$在定义域内满足即可（直接看Hessian可能不一定满足）

- **理论分析**：证明优化算法的收敛性（如SQP方法）。
- **工程控制**：在最优控制问题中验证Bang-Bang控制的严格最优性。
- **非凸优化**：深度学习中的损失函数可能存在多个鞍点，二阶条件可辅助设计逃离鞍点的算法。

| **条件类型**        | **要求**                          | **作用**           |
| ------------------- | --------------------------------- | ------------------ |
| 一阶必要条件（KKT） | 梯度平稳性、可行性、互补松弛性    | 筛选候选点         |
| 二阶必要条件        | Hessian在**临界子空间**上半正定   | 排除明显非最优驻点 |
| 二阶充分条件        | Hessian在**临界子空间**上严格正定 | 确认严格局部最优   |



#### Penalty Methods

罚函数法分为两种：

- 内罚法（Barrier Method）：目标函数中引入一个障碍函数（如倒数障碍、对数障碍），仅适用于不等式约束

  算法步骤：给出罚因子$\rho$求解无约束问题，然后不断减小$\rho$，使得解逐渐收敛到原问题的最优点

  ![image-20250402134529194](markdown-img/ZJU-OP-Lec5-8.assets/image-20250402134529194.png)

  内罚函数的优点：每个近似最优解都是可行解（因为迭代点始终处于可行域内部）

  缺陷：

  障碍因子$\rho$r不断减小也会导致海森阵趋于病态（梯度悬崖），在数值求解过程中造成很大麻烦

- 外罚法（Exterior Penalty Method）：添加惩罚项，$\min f(x)+\lambda\times P(x),\lambda$逐渐增大至无穷

  - L1罚函数（精确罚函数）：
    $$
    F_\rho(x) = f(x) + \rho( \sum_{j=1}^{m_1} |g_j(x)| + \sum_{j=m_1+1}^{m_1+m_2} \max\{g_j(x), 0\})
    $$
    对违反约束的点施加线性惩罚（绝对值项），惩罚强度与偏离程度成比例

    有限大小的$\mu$即可保证解是原问题的**精确解**，能够快速收敛![image-20250402142754627](markdown-img/ZJU-OP-Lec5-8.assets/image-20250402142754627.png)

    > 注意墙角的非光滑性

    缺陷：需要处理非光滑性
    
  - L2罚函数（二次罚函数）：
    $$
    F_\rho(x) = f(x) +\frac{ \rho}{2} (\sum_{j=1}^{m_1} g_j(x)^2 +  \sum_{j=m_1+1}^{m_1+m_2} \max\{g_j(x), 0\}^2)
    $$
    
  
       - $\rho > 0$：罚参数，控制惩罚强度
       - **等式约束惩罚**：$g_j(x)^2$（强制$g_j(x) \to 0$）
       - **不等式约束惩罚**：$\max(g_j(x),0)^2$（仅在违反约束时激活）

    要求问题光滑，数值稳定性有限。求解过程如下：

    ![image-20250402134202549](markdown-img/ZJU-OP-Lec5-8.assets/image-20250402134202549.png)

    缺陷：
  
    罚因子$\rho$增大时，解逐渐逼近原问题的最优解。但$\rho$过大会导致Hessian矩阵病态。对于无约束优化问题的 数值方法拟牛顿法与共轭梯度法存在数值困难，且需要多次迭代求解子问题
    
    这里只给出等式约束的罚函数的Hessian矩阵：
    $$
    \nabla_{xx}^2F(x,\rho) \approx \nabla_{xx}^2L(x,\lambda^*)+\sigma\nabla c(x)\nabla c(x)^T
    $$
    右边为一个定值矩阵和一个最大特征值趋于正无穷的矩阵，这导致$\nabla_{xx}^2F(x,\rho)$条件数越来越大，求解子问题的难度也会相应地增加

    ![image-20250402134011637](markdown-img/ZJU-OP-Lec5-8.assets/image-20250402134011637.png)
  
    > 可以看到求解的区域越来越狭长
    
    - 对于存在不等式约束的function可能不存在二次可微性质，光滑性降低
    - 不精确，与原问题最优解存在距离



![image-20250402134507798](markdown-img/ZJU-OP-Lec5-8.assets/image-20250402134507798.png)

**Quadratic Penalty Method**

Algorithm:

1. **Choose an initial penalty parameter** $ \rho_0 > 0 $ and an initial starting point $ \hat{x} $
2. **Set $ 0 \to k $**
3. **Starting from $ \hat{x} $**, use an unconstrained optimization method to find a minimizer of the penalty function $ F_{\rho_k}(x) $. Let $ x^{k,*} $ denote the minimizer obtained
4. **Choose a new penalty parameter** $ \rho_{k+1} > \rho_k $
5. **Set $ k + 1 \to k $** and $ x^{k,*} \to \hat{x} $ and return to step 3

Theoretical Results:
$$
 x^{k,*} \to \text{global solution of constrained problem}
$$

- This convergence assumes exact global minimization at each iteration (theoretical ideal)
- Belongs to **inexact penalty methods** (feasibility attained as $ \rho \to \infty $)
- Contrast with **exact penalty methods** (finite $ \rho $ solutions)





对比其他方法：

| **方法**         | **优点**                     | **缺点**                 |
| ---------------- | ---------------------------- | ------------------------ |
| 二次罚函数法     | 简单易实现，可处理非线性约束 | 需大$\rho$导致数值不稳定 |
| 障碍函数法       | 保证内点解                   | 无法处理等式约束         |
| 增广Lagrangian法 | 允许有限$\rho$收敛           | 需估计Lagrange乘子       |



### QP Problem

Quadratic Programming Problem

#### Problem Statement

A quadratic programming problem (QP) is an optimization problem with quadratic objective function and linear constraints.
$$
\min f(x) = \frac{1}{2} x^\top Q x + c^\top x
$$
Subject to:
$$
a_j^\top x = b_j, \quad j = 1, \ldots, m_1
$$

$$
a_j^\top x \leq b_j, \quad j = m_1+1, \ldots, m_1+m_2
$$

Parameters:

- $  m_1  $: Number of equality constraints
- $  m_2  $: Number of inequality constraints
- $  Q \in \mathbb{R}^{n \times n}  $: Symmetric matrix
- $  c \in \mathbb{R}^n  $: Linear term vector
- $  a_j \in \mathbb{R}^n  $: Constraint vectors ($  j = 1,\ldots,m_1+m_2  $)
- $  b_j \in \mathbb{R}  $: Constraint constants ($  j = 1,\ldots,m_1+m_2  $)

QP problems appear as subproblems in:
- Sequential Quadratic Programming (SQP) methods for nonlinear optimization



#### Quadratic Programming with Equality Constraints
Minimize:

$$
\min f(x) = \frac{1}{2} x^\top Q x + c^\top x
$$

Subject to:
$$
g(x) = Ax - b = 0
$$

---

Parameters:

- $  Q \in \mathbb{R}^{n \times n}  $: Symmetric matrix
- $  c \in \mathbb{R}^n  $: Linear term vector
- $  A \in \mathbb{R}^{m \times n}  $: Constraint matrix
- $  b \in \mathbb{R}^m  $: Equality constraint constants

Assumptions:

- Number of equality constraints $  m < n  $
- $  \text{rank}(A) = m  $ (full row rank)

---

Lagrangian Function:
$$
L(x, \lambda) = \frac{1}{2} x^\top Q x + c^\top x + \lambda^\top (Ax - b)
$$
where:  $  \lambda \in \mathbb{R}^m  $: Lagrange multiplier vector

---


$$
\frac{\partial L(x^*, \lambda^*)}{\partial x} = (x^*)^\top Q + c^\top + (\lambda^*)^\top A = 0
$$
$$
Ax^* = b
$$
$$
\begin{bmatrix}
Q & A^\top \\
A & 0
\end{bmatrix}
\begin{bmatrix}
x^* \\
\lambda^*
\end{bmatrix}
=
\begin{bmatrix}
-c \\
b
\end{bmatrix}
$$
当 $  Q  $ 正定且 $  A  $ 满秩时，分块矩阵的逆矩阵表达式为：

$$
\begin{bmatrix}
Q & A^\top \\
A & 0
\end{bmatrix}^{-1}
=
\begin{bmatrix}
H & P^\top \\
P & S
\end{bmatrix}
$$

其中各子矩阵：
- $  H = Q^{-1} - Q^{-1}A^\top(AQ^{-1}A^\top)^{-1}AQ^{-1}  $
- $  P = (AQ^{-1}A^\top)^{-1}AQ^{-1}  $
- $  S = -(AQ^{-1}A^\top)^{-1}  $

$$
\begin{bmatrix}
x^* \\
\lambda^*
\end{bmatrix}
=
\begin{bmatrix}
Q & A^\top \\
A & 0
\end{bmatrix}^{-1}
\begin{bmatrix}
-c \\
b
\end{bmatrix}
$$
$$
x^* = -Hc + P^\top b \\
= -Q^{-1}c + Q^{-1}A^\top(AQ^{-1}A^\top)^{-1}(AQ^{-1}c + b)
$$
$$
\lambda^* = -Pc + Sb \\
= -(AQ^{-1}A^\top)^{-1}(AQ^{-1}c + b)
$$
---

> $AQ^{-1}A^\top$将原始空间的梯度投影到约束空间时，根据$Q^{-1}$权重进行缩放



#### Active Set Strategy for Quadratic Programming

针对QP问题和KKT条件，使用以下方法

The active set method works as follows:

1. At each iteration,an approximation of the active set is made.

   将复杂的带不等式约束的QP问题转化为一系列等式约束的QP子问题，来逐步逼近最优解

2. 等式约束都可以用Quadratic Programming with Equality Constraints来求解，那些不等式约束如果在active set中则视为等式约束，如果不在则不考虑

3. If all Lagrange multipliers corresponding to active inequality constraints are
   non-negative,then the solution of the equality-constrained subproblem is optimal for the original QP.Otherwise,the active set needs to be updated.

   如果所有的都不是负定的则为最优解，否则则需要更新active set



#### Sequential Quadratic Programming(SQP)

针对带等式和不等式约束的问题

SQP involves modeling the optimization problem at the current point $x^k$ by a QP subproblem, which can be solved using quadratic programming techniques. The solution of this subproblem is used to define the next point $x^{k+1}$.

Let $x^k \in \mathbb{R}^n$ be the current point and let $\lambda^k \in \mathbb{R}^n$ be the corresponding Lagrange multiplier vector.

Quadratic approximation of the objective:
$$
f(x^k + p) \approx f(x^k) + \nabla f(x^k)^T p + \frac{1}{2} p^T \nabla_x^2 L(x^k, \lambda^k) p
$$
Linear approximation of the constraints:
$$
g_j(x^k + p) \approx g_j(x^k) + \nabla g_j(x^k)^T p = 0
$$
**QP subproblem**:

Minimize 
$$
f(x^k) + \nabla f(x^k)^T p + \frac{1}{2} p^T \nabla_x^2 L(x^k, \lambda^k) p
$$
subject to 

$g_j(x^k) + \nabla g_j(x^k)^T p = 0$, $j = 1, \ldots, m_1$
$g_j(x^k) + \nabla g_j(x^k)^T p \leq 0$, $j = m_1 + 1, \ldots, m_1 + m_2$

where $L$ is the Lagrangian defined by
$$
L(x, \lambda) = f(x) + \sum_{j=1}^{m_1+m_2} \lambda_j g_j(x)
$$
Lagrangian Hessian Matrix
$$
\nabla_x^2 L(x, \lambda) = \begin{bmatrix}
\frac{\partial^2 L}{\partial x_1^2} & \cdots & \frac{\partial^2 L}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 L}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 L}{\partial x_n^2}
\end{bmatrix} 
= \frac{\partial}{\partial x} \left[ \frac{\partial L(x, \lambda)}{\partial x} \right]^\top
$$
---

用Lagrangian的Hessian矩阵而不是用f的Hessian矩阵,是为了将约束的曲率也考虑进去

实际中我们常用BFGS来更新Lagrangian Hessian Matrix

---

Fortran SQP Implementations

Notable Packages

1. **NLPQLP**
   Nonlinear Programming using Quadratic Lagrangian (with Powell stabilization)

2. **FFSQP**
   Fast Feasible Sequential Quadratic Programming

3. **NPSQP**
   Nonlinear Programming Sequential Quadratic Programming

---



#### PHR Augmented Lagrangian Method

增广拉格朗日法

> 做轨迹规划用这个做得很多？

核心思想：融合拉格朗日法+惩罚法，通过引入二次惩罚项增强目标函数的光滑性，从而避免传统外罚函数法中因惩罚系数过大导致的数值不稳定性（增广目标函数越来越病态）。

##### For Equality-Constrained Cases

Problem Statement：

考虑仅含等式约束的优化问题：

$$
\min_{x \in \mathbb{R}^n} f(x) \quad \text{s.t.} \quad h(x) = 0
$$
下面我们先介绍一下Uzawa方法

方法原理：交替优化原始变量$x$和对偶变量$\lambda$，通过在对偶函数上执行对偶梯度上升：
$$
\begin{aligned}
&\min_{x} f(x) + P_{\infty}(h(x)) \\
&\text{where } P_{\infty}(x) = 
\begin{cases} 
0, & x = 0 \\
+\infty, & x \neq 0
\end{cases}
\end{aligned}
$$
等价地我们可以转化为如下形式
$$
d(\lambda) := \min_x \left[ f(x) + \lambda^\top h(x) \right]
$$
$$
\min_x \max_\lambda \left[ 
  f(x) + \lambda^\top h(x) 
\right]
$$

在$h(x)$不等于0时，内层的max一定会把问题最大话成无穷大，所以当约束不满足时这个最优化问题肯定取不到最小值

对于**大部分的**凸优化问题，可以证明将min和max的符号切换后结果不变，这个结果也叫做强对偶

但是这种方法存在一些问题：

- 若拉格朗日函数关于$ x $非严格凸，则对偶函数非光滑：
  $$
  \nabla d(\lambda) \ \text{可能不存在!}
  $$

  > 非严格凸时，可能存在多个x使$ L(x,λ)$ 达到极小值。此时，对偶函数在参数变化时可能发生“跳跃”或“拐角”，导致其导数不连续或不存在（非光滑）。

- 对偶梯度上升（Dual Gradient Ascent）因梯度不存在而失效。

因此在PHR方法中，通过引入**邻近点(proximal point)正则化项**平滑极小化极大（Minimax）问题，同时避免交换min和max顺序的复杂性
$$
\min_x \max_\lambda \left[ 
  f(x) + \lambda^\top h(x) - \frac{1}{2\rho} \|\lambda - \bar{\lambda}\|^2 
\right]
$$

- $ \rho > 0 $：正则化参数
- $ \bar{\lambda} $：先验拉格朗日乘子估计值，如上一轮迭代结果

**优势**：
1. 避免对偶梯度上升的不可行性
2. 直接优化原始极小化极大问题，不需要进行Uzawa的交替优化
3. 正则化项$ \frac{1}{2\rho}\|\lambda - \bar{\lambda}\|^2 $迫使乘子更新不过度偏离历史值，提升数值稳定性

下面我们对问题进行推导求解：

先进行内层问题求解
$$
\max_\lambda \left[ 
f(x) + \lambda^\top h(x) - \frac{1}{2\rho} \|\lambda - \bar{\lambda}\|^2 
\right]
$$
通过求导可得闭式解：

$$
\lambda^*(\bar{\lambda}) = \bar{\lambda} + \rho h(x)
$$
外层问题转换：将解析解代入原问题，推导无约束优化形式：
$$
\begin{align*}
\min_x \max_\lambda \left[ \cdot \right] 
&= \min_x \left[ 
f(x) + (\bar{\lambda} + \rho h(x))^\top h(x) 
- \frac{\rho}{2} \|h(x)\|^2 
\right] \\
&= \min_x \left[ 
f(x) + \bar{\lambda}^\top h(x) + \frac{\rho}{2} \|h(x)\|^2 
\right]
\end{align*}
$$
这样我们就得到原问题的近似解，如何提高精度？：

- 通过减小近似权重 $\frac{1}{\rho}$ 提升精度
- 迭代更新先验值$\bar{\lambda} \leftarrow \lambda^*(\bar{\lambda})$

实际算法流程：

1. **初始化**：设置初始乘子 $\bar{\lambda}_0$ 和惩罚参数 $\rho_0$

2. **迭代步骤**：
   - 求解无约束子问题：
     $$
     x_k = \arg\min_x \left[ 
          f(x) + \bar{\lambda}_k^\top h(x) + \frac{\rho_k}{2} \|h(x)\|^2 
          \right]
     $$
   
   - 更新乘子：

$$
     \bar{\lambda}_{k+1} = \bar{\lambda}_k + \rho_k h(x_k)
$$

3. **精度控制**：当 $\|h(x_k)\| < \epsilon$ 时终止



现在我们重新来看$x_k$的计算公式，相当于求解
$$
\min_{x \in \mathbb{R}^n} f(x)+\frac{\rho_k}{2} \|h(x)\|^2  \quad \text{s.t.} \quad h(x) = 0
$$
与原始问题相等，它的Lagrangian也被称为Augmented Lagrangian of the original problem
$$
L(x,\lambda,\rho) =f(x) + \bar{\lambda}_k^\top h(x) + \frac{\rho_k}{2} \|h(x)\|^2
$$
因此被称为融合拉格朗日法+惩罚法



##### For Inequality-Constrained Cases

考虑带非凸不等式约束的优化问题：

$$
\min_{x \in \mathbb{R}^n} f(x) \quad \text{s.t.} \quad g(x) \leq 0
$$
通过引入松弛变量 $s \in \mathbb{R}^m$，将不等式约束转化为等式约束：

$$
\begin{cases}
\min_{x, s} f(x) \\
\text{s.t.} \quad g(x) + [s]^2 = 0 \\
\quad \ \ s \geq 0 
\end{cases}
$$

> 其中，$[\cdot]^2$ 表示逐元素平方运算。松弛变量s也不知道，也被当做一个决策变量进行优化

定义含松弛变量和惩罚项的增广拉格朗日函数：

$$
\mathcal{L}_\rho(x, \mu) = f(x) + \frac{\rho}{2} \left\| \max\left( g(x) + \frac{\lambda}{\mu},\ 0 \right) \right\|^2 - \frac{\lambda}{2\rho} \|\mu\|^2
$$

对偶上升算法乘子更新规则：
$$
\mu^{k+1} = \max\left( \mu^k + \rho g(x^{k+1}) ,\ 0 \right)
$$

通过松弛变量$s$在可行域（$s \geq 0$）和约束边界（$g(x)+s=0$）之间建立缓冲带，使非凸约束的优化轨迹更平滑。惩罚项$\|\max(g(x)+s,0)\|^2$的引入，使得解在远离可行域时受到更强惩罚。

## Lec 6 控制论思想下的约束问题

### 控制论思想下的约束问题

- Proximal Gradient Dynamics and Feedback Control for Equality-Constrained Composite Optimization  

  解决等式约束的复合凸优化问题：
  $$
  \min f(x)+g(x),s.t.\quad h(x)=0
  $$
  提出PI-PGD方法，将优化问题建模为连续时间动力系统，并通过反馈控制调节拉格朗日乘子（视为控制输入），驱动系统收敛至可行解

  ![image-20250401164130590](markdown-img/ZJU-OP-Lec5-8.assets/image-20250401164130590.png)

- Control-Barrier-Function-Based Design of Gradient Flows for Constrained Nonlinear Programming  

  在带约束的非线性优化问题中，传统梯度流方法难以保证解的**可行性**（始终满足约束）和**稳定性**（收敛到可行集的局部/全局最优）。论文提出一种结合**控制屏障函数（CBF）**的梯度流方法，解决以下问题：

  - 确保可行集的前向不变性（任何初始可行点保持可行）；
  - 实现可行集的渐近稳定性（不可行初始点收敛到可行集）；
  - 设计连续、光滑的动力学系统，避免传统投影梯度流的不连续性。

  方法：**Safe Gradient Flow**，核心思想：将优化问题视为闭环控制系统





### HW

![e2474e7b99bcf003186710282cbbf34](markdown-img/ZJU-OP-Lec5-8.assets/e2474e7b99bcf003186710282cbbf34.jpg)

对比$l_1$和$l_2$罚函数法，关注“墙角”的过度平滑性

罚函数如何让增广函数越来越病态？画图看看





## Lec 7 连续最优问题

今天的主要任务：

- 复习无约束变分问题
- 测地线方程－力学观点
- 测地线方程－几何观点
- 拉格朗日力学基础
- 几个算例



### Review

泛函极值问题分为两种：unconstrained/constranied ~

在介绍连续最优问题之前，我们先来复习一下经典的无约束变分问题

> 直接看Lec 1部分的变分法及拉格朗日推导吧

这里仅明确一下概念：

泛函（functional）通常是指定义域为函数集，而值域为实数或者复数的映射，输入为函数，而输出为标量。

> 算子是一个函数到另一个函数的映射，它是从**向量空间到向量空间的映射**，泛函是从**向量空间到数域的映射**，函数是从**数域到数域的映射**

#### example

简单介绍一个泛函运用的具体例子：两点之间直线最短

在二维平面中，寻找从点 $ (0,0) $ 到 $ (a,b) $ 的曲线 $  y = y(x)  $，使得曲线的弧长 $  s  $ 最小。

弧长的微分形式为：

$$
ds = \sqrt{1 + \left( \frac{dy}{dx} \right)^2} \, dx
$$

总弧长为：

$$
s = \int_{0}^{a} \sqrt{1 + y'(x)^2} \, dx
$$

变分法求解，目标是最小化泛函 $  s = \int_{0}^{a} \sqrt{1 + y'(x)^2} \, dx  $，应用 **欧拉-拉格朗日方程**：

$$
\frac{d}{dx} \left( \frac{\partial F}{\partial y'} \right) - \frac{\partial F}{\partial y} = 0 \quad \text{其中} \quad F = \sqrt{1 + y'^2}
$$

- $ \frac{\partial F}{\partial y} = 0 $（因 $  F  $ 不显含 $  y  $）
- $ \frac{\partial F}{\partial y'} = \frac{y'}{\sqrt{1 + y'^2}} $

代入方程得：

$$
\frac{d}{dx} \left( \frac{y'}{\sqrt{1 + y'^2}} \right) = 0 \quad \Rightarrow \quad \frac{y'}{\sqrt{1 + y'^2}} = C \quad (\text{常数})
$$

化简方程：

$$
y'^2 = \frac{C^2}{1 - C^2} \quad \Rightarrow \quad y' = m \quad (\text{常数})
$$

积分得直线方程：

$$
y(x) = m x + c
$$

代入边界条件 $  y(0) = 0  $ 和 $  y(a) = b  $，得：

$$
c = 0, \quad m = \frac{b}{a}
$$

因此最短路径曲线为直线：
$$
y(x) = \frac{b}{a} x
$$


### Geodesic

测地线（Geodesic）是微分几何中的核心概念，表示流形上两点之间的“最短路径”

在介绍测地线之前，我们先介绍一下哈密顿原理（Hamilton's principle）

哈密顿原理（Hamilton's principle）是经典力学中的核心变分原理，揭示了力学系统的真实运动轨迹遵循作用量取极值的规律

- 定义：在完整保守系统中，系统从初始时刻 t1 到终了时刻 t2 的真实运动轨迹，使得作用量（Action）取驻值（极值或鞍点）——作用量极值决定真实轨迹
  $$
  \delta S = \delta \int_{t_1}^{t_2} L(q,\dot q,t)dt = 0 
  $$
  
- 平面上的测地线：表现为直线

  - 几何观点
    $$
    F = \sqrt{1 + \dot{f}^2(x)}
    $$

    代入欧拉-拉格朗日方程
    $$
    \frac{\partial F}{\partial f} = 0
    $$


    $$
    \frac{\partial F}{\partial \dot{f}(x)} = \frac{\dot{f}(x)}{\sqrt{1 + \dot{f}^2(x)}}
    $$
    
    得到
    $$
    0 - \frac{d}{dx} \left( \frac{\dot{f}(x)}{\sqrt{1 + \dot{f}^2(x)}} \right) = 0
    $$
    
    因此
    $$
    - \frac{\dot{f}(x)}{\sqrt{1 + \dot{f}^2(x)}} = c
    $$


    $$
    \dot{f}(x) = \frac{c}{\sqrt{1 - c^2}} = c_1
    $$


    $$
    f(x) = c_1 x + c_2
    $$
    
    证明为直线

  - 力学观点

    > 力学与微分几何的内在联系

    定义拉格朗日量为动能形式$L = \frac{1}{2}g_{ij}\dot{x}^i\dot{x}^j  = \frac{1}{2}(\frac{ds}{dt})^2$， 其中 $ g_{ij} $ 是流形的度规张量，$ \dot{x}^i = dx^i/dt $ 为坐标对参数 $ t $ 的导数。

    > 上述L是in metric terms (Euclidean 3D non-relativistic fat space)
    >
    > 单位质量时$T = \frac{1}{2}v^2$，然后在任意坐标系中弧长微分$ds$由度规张量$g_{ij}$定义：
    >
    > $ds^2 = g_{ij}dx^i dx^j$
    >
    > **用一句最土最直白的话来说，度量张量就是用来把斜角坐标的读数转换成直角坐标读数的，度量张量的本质就是坐标变换**——当然这里是粗浅的理解，不管是什么坐标系，是弯曲时空还是平直时空，有了度规和任意一个向量在这组基下的分量可以计算长度和角度
    >
    > 有了度规张量就可以推广到任何空间，例如笛卡尔坐标系：per unit mass so $T = \frac{1}{2}[(dx/dt)^2+(dy/dt)^2+(dz/dt)^2]$

    代入拉格朗日方程，展开后可得
    $$
    \frac{d}{dt} \left( g_{ik} \dot{x}^k \right) - \frac{1}{2} \partial_i g_{jk} \dot{x}^j \dot{x}^k = 0
    $$

    通过整理导数项，定义克氏符：

    $$
    \Gamma^i_{jk} = \frac{1}{2} g^{il} \left( 
    \partial_j g_{lk} + \partial_k g_{jl} - \partial_l g_{jk} 
    \right)
    $$

    最终得到测地线方程：

    $$
    \ddot{x}^i + \Gamma^i_{jk} \dot{x}^j \dot{x}^k = 0
    $$
    
- 球面上的测地线：球的表示$r,\theta,\phi$，这里令$r=1$

  - 几何观点

    球面路径长度积分：
    $$
      L = \int_A^B |d\mathbf{r}| = \int_{\theta_A}^{\theta_B} \sqrt{1 + \sin^2 \theta \, (\phi')^2} \, d\theta
    
    $$
    
    代入E-L方程
    $$
    \frac{d}{d\theta} \left( \frac{\partial}{\partial \phi'} \sqrt{1 + \sin^2 \theta \, (\phi')^2} \right) =  \frac{\partial}{\partial \phi} \sqrt{1 + \sin^2 \theta \, (\phi')^2} 
    = 0
    $$
    
      展开后得到：
    
    $$
    \frac{d}{d\theta} \left( \frac{\sin^2 \theta \, \phi'}{\sqrt{1 + \sin^2 \theta \, (\phi')^2}} \right) = 0
    $$
    守恒量恒等式：
    $$
      \frac{\sin^2 \theta \, \phi'}{\sqrt{1 + \sin^2 \theta \, (\phi')^2}} = c
    
    $$
    
    导数关系推导：
    $$
    \phi' = \frac{c}{\sin \theta \sqrt{\sin^2 \theta - c^2}}
    $$
    
    问题转化为对$\phi \prime$进行积分

    变量替换：
    $$
    u = \cot \theta \quad \Rightarrow \quad du = -\csc^2 \theta \, d\theta
    $$
    
    $$
      \phi = \int \frac{-c \, du}{\sqrt{1 - c^2 \csc^2 \theta}} = \cos^{-1}\left(\frac{u}{a}\right) + \phi_0
    
    $$
    
      其中 $ a = \frac{\sqrt{1 - c^2}}{c} $，$ \phi_0 $ 为积分常数。
    
    得到测地线方程：
    $$
      \cot \theta = a \cos(\phi - \phi_0)
    
    $$
    
    a great circle path
    
  - 力学观点

    同样的，在球面上就是$ds^2 = a^2\dot\theta^2+a^2\sin^2\theta \dot\phi^2$，则
    $$
    L=\frac{1}{2}(a^2\dot\theta^2+a^2\sin^2\theta \dot\phi^2)
    $$
    然后对$\theta,\phi$分别应用拉格朗日函数
  
    先对$\theta$应用E-L
    $$
    \frac{d(a^2 \dot{\theta})}{ds} = a^2 \dot{\phi}^2 \sin \theta \cos \theta \quad \Rightarrow \quad \ddot{\theta} = \sin \theta \cos \theta \dot{\phi}^2
    $$
    然后对$\phi$应用E-L
    $$
    \frac{d}{ds} (a^2 \sin^2 \theta \dot{\phi}) = 0 \quad \Rightarrow \quad \ddot{\phi} + 2 \cot \theta \dot{\theta} \dot{\phi} = 0
    $$
  
    

### some example

A sliding point mass on a sliding wedge

![image-20250414214107932](markdown-img/ZJU-OP-Lec5-8.assets/image-20250414214107932.png)

质点系总动能包含三个部分：

$$
T = \frac{1}{2} M \dot{X}^2 + \frac{1}{2} m (\dot{X} + \dot{x})^2 + \frac{1}{2} m \dot{y}^2
$$

通过几何约束关系 $  y = x \tan\alpha  $ 化简为：

$$
T = \frac{1}{2} (M + m) \dot{X}^2 + m \dot{X} \dot{x} + \frac{1}{2} m (1 + \tan^2 \alpha) \dot{x}^2
$$

仅考虑重力势能：

$$
U = mgy = mgx \tan\alpha
$$

$$
L = T - U = \frac{1}{2} (M + m) \dot{X}^2 + m \dot{X} \dot{x} + \frac{1}{2} m (1 + \tan^2 \alpha) \dot{x}^2 - mgx \tan\alpha
$$

应用欧拉-拉格朗日方程：

广义坐标 X 方程：

$$
\frac{d}{dt} \left( \frac{\partial L}{\partial \dot{X}} \right) - \frac{\partial L}{\partial X} = 0 \quad \Rightarrow \quad (M + m) \ddot{X} + m \ddot{x} = 0
$$


广义坐标 x 方程：

$$
\frac{d}{dt} \left( \frac{\partial L}{\partial \dot{x}} \right) - \frac{\partial L}{\partial x} = 0 \quad \Rightarrow \quad m \ddot{x} + m (1 + \tan^2 \alpha) \ddot{x} = -mg \tan\alpha
$$

联立求解：
$$
\ddot{x} = -\frac{(M + m)g \sin\alpha \cos\alpha}{M + m \sin^2\alpha}
$$


$$
\ddot{X} = \frac{mg \sin\alpha \cos\alpha}{M + m \sin^2\alpha}
$$



### HW

1. 推导Snell`s law in optics，费马原理

   > Snell`s law：$\frac{\sin \theta_1}{\sin \theta_2} = \frac{n_2}{n_1} = \frac{v_1}{v_2}$
   >
   > 费马原理：光传播的路径是光程取极值的路径

2. 球面上的测地线推导与求解

3. 三角块与小球系统建模及matlab/simscope仿真

![lQDPJwMj13pGsPfNA8DNBQCwjqqIMQElyiEH3PS4sZbuAA_1280_960](markdown-img/ZJU-OP-Lec5-8.assets/lQDPJwMj13pGsPfNA8DNBQCwjqqIMQElyiEH3PS4sZbuAA_1280_960.jpg)



## Lec 8 Continuous-Time Optimal Control

里面有非常详细的变分法对变动端点的变分问题的推导和应用：[数字人生](https://www.zhihu.com/column/c_1703789122005610496)

### warm up

下面两个问题是起始和终端状态完全固定，因此可以直接变分，代入E-L方程即可

- Isoperimetric problem
- catenary problem——看hw8.md，里面有详细推导

其实在求解E-L方程时，常遇到二阶微分方程然后降维为一阶微分方程，实际上我们可以推导得
$$
f-y\prime \cdot \frac{\partial f}{\partial y\prime} =J(第一积分，运动常数)
$$

### Continuous-Time Optimal Control

现在我们来考虑terminal is free的情况

> process cost and terminal cost

#### Lagrangian Formulation

目标函数
$$
J = S(t_f, \mathbf{x}_{t_f}) + \int_{t_0}^{t_f} \left[ L(t, \mathbf{x}(t), \mathbf{u}(t)) + \lambda^T \left( \mathbf{f}(t, \mathbf{x}(t), \mathbf{u}(t)) - \dot{\mathbf{x}} \right) \right] dt
$$

terminal cost积分表示
$$
S(t_f, \mathbf{x}_{t_f}) = \int_{t_0}^{t_f} \frac{d}{dt} S(t, \mathbf{x}(t)) \, dt + S(t_0, \mathbf{x}_{t_0})(为常数，可扔掉)
$$

目标函数重写
$$
J = S(t_0, \mathbf{x}_{t_0}) + \int_{t_0}^{t_f} \left[ \frac{d}{dt} S(t, \mathbf{x}(t)) + L(t, \mathbf{x}(t), \mathbf{u}(t)) + \lambda^T \left( \mathbf{f}(t, \mathbf{x}(t), \mathbf{u}(t)) - \dot{\mathbf{x}} \right) \right] dt
$$
Augmented Lagrangian:
$$
L^a = \frac{\partial}{\partial x} S(t, \mathbf{x}(t))\dot x+ \frac{\partial}{\partial t} S(t, \mathbf{x}(t)) + L(t, \mathbf{x}(t), \mathbf{u}(t)) + \lambda^T \left( \mathbf{f}(t, \mathbf{x}(t), \mathbf{u}(t)) - \dot{\mathbf{x}} \right)
$$
然后通过一系列推导，可得

When the cost is optimum,the increment $\delta J=0$.Hence,the necessary constrain for optimal control are as follows: * refers to the optimum.

最优控制问题的必要条件
$$
\begin{aligned}
\left( \frac{\partial L^a}{\partial \mathbf{x}} \right)_* - \frac{d}{dt} \left( \frac{\partial L^a}{\partial \dot{\mathbf{x}}} \right)_* &= 0 \quad \text{co-states}\\
\left( \frac{\partial L^a}{\partial \lambda} \right)_* - \frac{d}{dt} \left( \frac{\partial L^a}{\partial \dot{\lambda}} \right)_* &= 0 \quad \Rightarrow \quad \frac{\partial L^a}{\partial \lambda} = 0 
\quad \text{the state dynamics}
\\
\left( \frac{\partial L^a}{\partial \mathbf{u}} \right)_* &= 0 
\quad \text{optimal control}
\\
\left( \frac{\partial L^a}{\partial \mathbf{x}} \right)_*^T \delta \mathbf{x}(t_f) + L^a |_{t_f} \delta t_f &= 0
\quad \text{boundary conditions}
\end{aligned}
$$

因为终止条件is free所以我们来探究一下boundary conditions

![image-20250415110909013](markdown-img/ZJU-OP-Lec5-8.assets/image-20250415110909013.png)

boundary conditions
$$
\left( \frac{\partial L^a}{\partial \dot{\mathbf{x}}} \right)_*^T \delta \mathbf{x}(t_f) + L^a |_{t_f} \delta t_f = 0
$$

代入$\delta x_f = \delta x(t_f)+ \dot{x}(t_f)\delta_{t_f}$，可得
$$
\left( \frac{\partial L^a}{\partial \mathbf{x}} \right)_*^T\Bigg|_{t_f} \{\delta \mathbf{x}_f - \dot{\mathbf{x}}(t_f) \delta t_f\} + L^a |_{t_f} \delta t_f = 0
$$

$$
\left[ L^a - \left( \frac{\partial L^a}{\partial \dot{\mathbf{x}}} \right)^T \dot{\mathbf{x}} \right]_{t_f} \delta t_f + \left( \frac{\partial L^a}{\partial \dot{\mathbf{x}}} \right)_{*t_f}^T \delta \mathbf{x}_f = 0
$$

下面举一个具体的栗子：

Exercise: Free Final Time Optimal Control，这里是一个两端状态均固定但是终端时间不固定的栗子

Design the optimal control for the following plant:
$$
\dot{x}(t) = \begin{bmatrix} 0 & 1 \\ 0 & -2 \end{bmatrix} x(t) + \begin{bmatrix} 0 \\ 1 \end{bmatrix} u(t)
$$

$$
= \begin{bmatrix} x_2(t) \\ -2x_2(t) + u(t) \end{bmatrix}
$$
to reach a final state $(3, 5)^T$ from an initial state $(0, 1)^T$ while minimizing the cost
$$
J = \left[ x_1(t_f) - 3 \right]^2 + \frac{1}{2} \left[ x_2(t_f) - 5 \right]^2 + \frac{1}{2} \int_0^{t_f} \left[ x_1^2(t) + 3x_2^2(t) + 2u^2(t) \right] dt
$$
for the given information :
$$
L = \frac{1}{2} \left[ x_1^2(t) + 3x_2^2(t) + 2u^2(t) \right]
$$

终端成本 $  S  $
$$
S = \left[ x_1(t) - 3 \right]^2 + \frac{1}{2} \left[ x_2(t) - 5 \right]^2
$$

外力向量 $  \mathbf{f}  $
$$
\mathbf{f} = \begin{bmatrix} 
x_2(t) \\ 
-2x_2(t) + u(t) 
\end{bmatrix}
$$

拉格朗日乘子向量 $  \lambda  $
$$
\lambda = \begin{bmatrix} 
\lambda_1(t) \\ 
\lambda_2(t) 
\end{bmatrix}
$$
将上述公式代入重写目标函数的形式得到

Augmented Lagrangian:
$$
L^a = \frac{1}{2} \left[ x_1^2(t) + 3x_2^2(t) + 2u^2(t) \right] + 2(x_1(t) - 3)\dot{x}_1(t) + (x_2(t) - 5)\dot{x}_2(t) \\ 
+ \lambda_1(t)[x_2(t) - \dot{x}_1(t)] + \lambda_2(t)[-2x_2(t) + u(t) - \dot{x}_2(t)]
$$
Necessary Conditions:

1. For $  x_1(t)  $:

$$
   x_1(t) + 2\dot{x}_1(t) - \frac{d}{dt}[2(x_1(t) - 3) - \lambda_1(t)] = 0 \\
   \quad \Rightarrow \quad \dot{\lambda}_1^*(t) = -x_1^*(t)
$$
2. For $  x_2(t)  $:

$$
3x_2(t) + \dot{x}_2(t) + \lambda_1(t) - 2\lambda_2(t) - \frac{d}{dt}(x_2(t) - 5 - \lambda_2(t)) = 0 \\
   \quad \Rightarrow \quad \dot{\lambda}_2^*(t) = -3x_2^*(t) - \lambda_1^*(t) + 2\lambda_2^*(t)
$$

**Applying the necessary condition for optimal control:**
$$
2u^*(t) + \lambda_2(t) = 0 \quad \Rightarrow \quad u^*(t) = -0.5\lambda_2^*(t)
$$
**Necessary condition at final time:**
$$
\left[ L^a - \left( \frac{\partial L^a}{\partial \dot{x}} \right)^T \dot{x} \right]_{t_f}^* = 0
$$
**Expanded form:**
$$
\frac{1}{2} \left[ x_1^2(t) + 3x_2^2(t) + 2u^2(t) \right] + 2[x_1(t) - 3]\dot{x}_1(t) + [x_2(t) - 5]\dot{x}_2(t) \\ 
+ \lambda_1(t)[x_2(t) - \dot{x}_1(t)] + \lambda_2(t)[-2x_2(t) + u(t) - \dot{x}_2(t)] \\
- 2[x_1(t) - 3 - \lambda_1(t)]\dot{x}_1(t) - [x_2(t) - 5 - \lambda_2(t)]\dot{x}_2(t) = 0
$$
**Simplified result:**
$$
\frac{1}{2} \left[ x_1^2(t) + 3x_2^2(t) + 2u^2(t) \right] + \lambda_1(t)x_2(t) + \lambda_2(t)[-2x_2(t) + u(t)] = 0
$$
然后写代码求解

#### Pontryagin-Hamiltonian Formulation

主要是使用Hamiltonian method来进行简化，哈密顿函数的定义及其优势可从数学结构与物理意义两个维度解析，其核心在于将复杂动态系统的演化规律转化为更对称、简洁的形式，并揭示能量守恒与对称性等深层性质

**Hamiltonian Method Formulation:**
$$
H(t, \mathbf{x}, \mathbf{u}, \mathbf{\lambda}) = L(t, \mathbf{x}, \mathbf{u}) + \mathbf{\lambda}^T \mathbf{f}
$$
where

$$
L(t, \mathbf{x}, \mathbf{u}) = H - \mathbf{\lambda}^T \mathbf{f} \quad \text{(Note: Contains notation conflict)}
$$
**Augmented Lagrangian:**
$$
L^a = \frac{\partial}{\partial \mathbf{x}} S(t, \mathbf{x}) \dot{\mathbf{x}} + \frac{\partial}{\partial t} S(t, \mathbf{x}) + H - \mathbf{\lambda}^T \dot{\mathbf{x}}
$$
将上述$L^a$代入最优控制问题的必要条件可得
$$
(\frac{\partial H}{\partial x})_* +\dot \lambda =0
$$

$$
(\frac{\partial H}{\partial \lambda})_* - \dot x =0
$$

$$
(\frac{\partial H}{\partial u})_*=0
$$

再由上面的boundary conditions可得
$$
(\frac{\partial S}{\partial x}+H)_{t_f} \delta_{t_f} + (\frac{\partial S}{\partial x}-\lambda)^T_{t_f} \delta x_{f} =0
$$
上面这四个方程就是Pontryagin-Hamiltonian Formulation相比Lagrangian Formulation更加紧凑

TBVP，shooting method



### hw

1. catenary problem求解



# Term Project

大作业有几个选题：

1. 熟悉PHR-ALM方法

   关于小车运动轨迹规划的MPC问题求解





# References

- https://zhuanlan.zhihu.com/p/629131647
- [2023-2024春夏许超老师最优化与最优控制课程分享 - CC98论坛](https://www.cc98.org/topic/5923253)

- [Tutorial — Ceres Solver (ceres-solver.org)](http://ceres-solver.org/tutorial.html)