---
title: 04-Algorithm
date: 2026-03-02
tags:
course: math
status: draft
---
# Optimal Algorithm
[TOC]

## Overview

优化算法层次：二次规划（KKT系统精确解）→ 牛顿法（转化为QP）→ 内点法（处理不等式约束）

## Unconstrained minimization

$$
\min_{x\in\mathbb{R}^n} f(x)
$$
$f$ 可微或二次可微。

**最优性条件**
1. 一阶：$\nabla f(x^*)=0$
2. 二阶：$\nabla^2 f(x^*)\succeq 0$；充分条件 $\nabla f(x^*)=0,\ \nabla^2 f(x^*)\succ 0$
3. 若 $f$ 凸，$\nabla f(x^*)=0$ 即全局最优

**迭代算法**：生成序列 $x^{(0)}, x^{(1)}, \dots$ 使 $f(x^{(k)})\to p^*$。初始水平集 $S=\{x\in\mathrm{dom}\,f\mid f(x)\le f(x^{(0)})\}$ 须闭。

**强凸性**：$\nabla^2 f(x)\succeq m I$ 保证解唯一，给出次优性界
$$
f(x)-p^*\le \frac{1}{2m}\|\nabla f(x)\|_2^2
$$

### Descent methods

更新：$x^{(k+1)}=x^{(k)}+t^{(k)}\Delta x^{(k)}$，满足 $\nabla f(x)^{\mathsf T}\Delta x<0$

**线搜索**
- 精确：沿方向最小化 $t$
- 回溯：参数 $\alpha\in(0,0.5),\ \beta\in(0,1)$，逐步缩小步长

Armijo 条件：
$$
f(x+tp)\le f(x)+c_1 t\nabla f(x)^\top p,\quad c_1\in(0,1)
$$

Wolfe 条件：Armijo + 曲率条件
$$
\nabla f(x+tp)^\top p \ge c_2\,\nabla f(x)^\top p,\quad 0<c_1<c_2<1
$$

### Gradient descent method

$p^{(k)}=-\nabla f(x^{(k)})$

- $\nabla f$ $L$-Lipschitz，固定步长 $t\in(0,2/L)$ 可保证下降
- $f$ $m$-强凸且 $L$-光滑，$t=1/L$ 时线性收敛
  $$
  f(x^{(k)})-f^*\le \bigl(1-\tfrac{m}{L}\bigr)^k\bigl(f(x^{(0)})-f^*\bigr)
  $$
- 条件数大时之字形，可通过预条件/拟牛顿/LM改善

### Steepest descent method

规范化方向：$p_{nsd}=\arg\min_{\|v\|\le1}\nabla f(x)^\top v=-\nabla f(x)/\|\nabla f(x)\|_*$

$l_2$ 范数下与梯度一致。$P$-二次范数 $\|v\|_P^2=v^\top P v$：
$$
p_{sd}=-P^{-1}\nabla f(x)
$$

### Newton's method

二阶模型：$m_k(p)=f(x^{(k)})+\nabla f(x^{(k)})^\top p+\frac12 p^\top \nabla^2 f(x^{(k)})p$

牛顿方向：
$$
p_N^{(k)}=-[\nabla^2 f(x^{(k)})]^{-1}\nabla f(x^{(k)})
$$

性质：严格凸二次函数一次终止；Hessian正定时局部二次收敛；计算代价高

牛顿减量：
$$
\lambda(x)=(\nabla f(x)^{\mathsf T}\nabla^2 f(x)^{-1}\nabla f(x))^{1/2}
$$

仿射不变性，阻尼牛顿阶段（线性收敛）→ 纯牛顿阶段（二次收敛）

**全局化与修正**
1. 阻尼牛顿：$x_{k+1}=x_k+\alpha_k p_N^{(k)}$，Armijo/Wolfe
2. LM修正：$p_{LM}^{(k)}=-[\nabla^2 f(x^{(k)})+\mu_k I]^{-1}\nabla f(x^{(k)})$，$\mu_k\to0$ 趋近牛顿，$\mu_k\to\infty$ 趋近梯度
3. 拟牛顿：$B_{k+1}s_k=y_k$，常见 DFP、BFGS、L-BFGS

### Self-concordance

Nesterov & Nemirovski 框架，控制三阶导数：
$$
|D^3 f(x)[h,h,h]|\le 2\,(h^\top\nabla^2 f(x)h)^{3/2}
$$

对数障碍函数典型自共轭。此框架下阻尼牛顿全局收敛，内点法复杂度分析基础。

## Equality constrained minimization

**消除法**：$Ax=b$，$A\in\mathbb{R}^{m\times n}$ 满行秩。取可行基点 $x_0$ 及零空间基 $N$，令 $x=x_0+Nz$，化为无约束 $\min_z f(x_0+Nz)$，梯度/ Hessian 投影到零空间。

**等式约束牛顿法**：保持可行，求解 KKT 系统
$$
\begin{bmatrix}H&A^\top\\A&0\end{bmatrix}\begin{bmatrix}p\\\lambda\end{bmatrix}=\begin{bmatrix}-\nabla f(x)\\0\end{bmatrix}
$$
$H$ 在切空间正定时为下降方向，配合可行线搜索。

**不可行起点牛顿法**：允许初始 $Ax\neq b$，设残差 $r_p=Ax-b,\ r_d=\nabla f(x)+A^\top\lambda$，系统
$$
\begin{bmatrix}H&A^\top\\A&0\end{bmatrix}\begin{bmatrix}\Delta x\\\Delta\lambda\end{bmatrix}=-\begin{bmatrix}r_d\\r_p\end{bmatrix}
$$
更新 $x^+=x+\alpha\Delta x,\ \lambda^+=\lambda+\alpha\Delta\lambda$，线搜索确保下降与残差缩小。

## Interior-point methods

求解带不等式约束凸优化。

**对数障碍函数**：$\phi(x)=-\sum\log(-f_i(x))$。障碍参数 $t$ 控制近似精度。

**中心路径**：$x^*(t)=\arg\min_x tf_0(x)+\phi(x)$，一阶条件
$$
t\nabla f_0(x)+\sum_i\frac{1}{-f_i(x)}\nabla f_i(x)=0
$$
$t\to\infty$ 时收敛于原问题最优解。

**障碍方法**：外迭代增大 $t$（乘 $\mu$），内迭代用牛顿法。停止准则 $m/t\le\varepsilon$。

**Phase I**：不可行时求解 $\min_{x,s} s\ \text{s.t.}\ f_i(x)\le s$，若 $s^*\le0$ 得可行点。

**原始-对偶内点法**：同时更新原始/对偶变量，超线性收敛。残差
$$
r_d=\nabla f_0(x)+\sum\lambda_i\nabla f_i(x),\quad r_c=-\lambda\circ f(x)
$$
中心化参数推动 $\lambda_i f_i(x)\approx\mu$。

## 收敛性分析

### 预备知识

**收敛性与收敛速度**
- oracle complexity / arithmetical complexity
- 线性收敛：$f(x^{(k)})-f(x^*)\le c^k(f(x^{(0)})-f(x^*))$，$0<c<1$，$O(\log(1/\varepsilon))$
- 二次收敛：$\|x^{(k+1)}-x^*\|\le C\|x^{(k)}-x^*\|^2$，$O(\log\log(1/\varepsilon))$
- 次线性收敛：$f(x^{(k)})-f(x^*)\le K/k^\alpha$，$O((1/\varepsilon)^{1/\alpha})$

**强凸性**：$\nabla^2 f(x)\succeq m I$ ⇒ 严格凸。二次下界
$$
f(y)\ge f(x)+\nabla f(x)^\top(y-x)+\frac{m}{2}\|y-x\|^2
$$

若 $m I\preceq\nabla^2 f(x)\preceq M I$，则
$$
f(y)\ge f(x)+\nabla f(x)^\top(y-x)+\frac{m}{2}\|y-x\|^2
$$
$$
f(y)\le f(x)+\nabla f(x)^\top(y-x)+\frac{M}{2}\|y-x\|^2
$$

$m=\inf\lambda_{\min}(\nabla^2 f(x))$，$M=\sup\lambda_{\max}(\nabla^2 f(x))$

**梯度 Lipschitz 连续**：$\|\nabla f(y)-\nabla f(x)\|\le L\|y-x\|$

若 $\|\nabla^2 f(x)\|\le L$ 则 $\nabla f$ Lipschitz。二次上界
$$
f(y)\le f(x)+\nabla f(x)^\top(y-x)+\frac{L}{2}\|y-x\|^2
$$

最优值间隙下界：$f(x)-f(x^*)\ge\frac{1}{2L}\|\nabla f(x)\|^2$

**寻优间隙上下界**（强凸 + Lipschitz）
$$
\frac{1}{M}\|\nabla f(x)\|\le\|x-x^*\|\le\frac{1}{m}\|\nabla f(x)\|
$$
$$
\frac{1}{2M}\|\nabla f(x)\|^2\le f(x)-f(x^*)\le\frac{1}{2m}\|\nabla f(x)\|^2
$$

### 下降法

强凸条件下精确/回溯线搜索均线性收敛，速度与条件数 $\kappa=M/m$ 相关。

精确线搜索：$f(x^+)-f(x^*)\le(1-m/M)(f(x)-f(x^*))$，复杂度 $O(\kappa\log(1/\varepsilon))$

回溯线搜索：Armijo $f(x+td)\le f(x)+\alpha t\nabla f(x)^\top d$，$\alpha\in(0,0.5]$，$t\leftarrow\beta t$。
$d=-\nabla f(x)$ 时 $t_f\ge\min(1,\beta/M)$，
$$
f(x^+)-f(x^*)\le(1-2m\alpha t_f)(f(x)-f(x^*))
$$

弱条件：非凸+Lipschitz→驻点 $O(1/\varepsilon)$；凸+Lipschitz→次优 $O(1/\varepsilon)$；加速 $O(1/\sqrt{\varepsilon})$

### 牛顿法

**经典分析法**

纯牛顿全局收敛性差，需阻尼/线搜索。阻尼阶段线性收敛→邻近最优二次收敛。

若 $f$ 强凸且 Hessian Lipschitz：$\|\nabla^2 f(y)-\nabla^2 f(x)\|_2\le L_H\|y-x\|_2$
当 $\frac{L_H}{2m}\|x^{(0)}-x^*\|_2<1$ 时迭代二次收敛。

牛顿减量 $\lambda(x)=\sqrt{\nabla f(x)^T\nabla^2 f(x)^{-1}\nabla f(x)}$
性质：$\lambda(x)^2=-\nabla f(x)^T\Delta x_{nt}$；$m\|\Delta x_{nt}\|_2^2\le\lambda(x)^2\le M\|\Delta x_{nt}\|_2^2$

**自和谐函数法**：阻尼牛顿全局收敛，邻域内全步长快速收敛。

## Review

1. **梯度下降方向**：$p^{(k)}=-\nabla f(x^{(k)})$。等高线狭长时之字形摆动。

2. **$P$-二次范数最速下降**：$p=-P^{-1}\nabla f(x)$

3. **精确线搜索最优步长**：$t_k^*=-\frac{\nabla f(x^{(k)})^T p^{(k)}}{p^{(k)T}\nabla^2 f(x^{(k)})p^{(k)}}$

4. **Armijo**：$f(x^{(k)}+t_k p^{(k)})\le f(x^{(k)})+\alpha t_k\nabla f(x^{(k)})^T p^{(k)}$。回溯：$t\leftarrow\beta t$

5. **牛顿方向**：$p_N^{(k)}=-[\nabla^2 f(x^{(k)})]^{-1}\nabla f(x^{(k)})$。优点：局部二次收敛，严格凸二次一步到位。缺点：Hessian 代价高，非正定时不一定下降。

6. **LM修正**：$p_{LM}^{(k)}=-[\nabla^2 f(x^{(k)})+\mu_k I]^{-1}\nabla f(x^{(k)})$，保证正定。

7. **梯度下降收敛速率**：$m$-强凸 + $L$-Lipschitz，固定步长 $t\in(0,2/L)$，$f(x^{(k)})-f^*\le(1-m/L)^k(f(x^{(0)})-f^*)$

8. **$f(x)=x^T\mathrm{diag}(1,4)x$**：$\nabla^2 f=2\cdot\mathrm{diag}(1,4)$，$m=2,\ L=8,\ \kappa=4$。$t=1/8$ 时收敛因子 $3/4$；精确线搜索因子 $((\kappa-1)/(\kappa+1))^2=(3/5)^2$。

   $P$-二次范数最速下降：
   $$
   \Delta x_{sd}=-\frac{P^{-1}\nabla f(x)}{\sqrt{\nabla f(x)^\top P^{-1}\nabla f(x)}}
   $$

9. **原始-对偶内点法**：迭代点保持不等式严格可行（$g_j(x)<0$）。不可行内点法允许暂时违反约束，但松弛/对偶变量严格正，仍在锥内部。

10. **Zoutendijk法**：解LP得$\eta$，$\eta<0$有可行下降方向，$\eta=0$终止。

11. **阻尼牛顿**：牛顿方向+回溯线搜索，全局收敛+局部二次收敛。牛顿减量 $\lambda(x)=\sqrt{\nabla f(x)^T\nabla^2 f(x)^{-1}\nabla f(x)}$ 作停止准则。

12. **非凸问题KKT算法**：SQP、增广拉格朗日法、罚函数/内点法/信赖域法。局部极小判别：KKT且在临界锥上
    $$
    p^T\nabla_{xx}^2 L(x^*,\lambda^*,\nu^*)p>0,\quad \forall p\in C\setminus\{0\}
    $$

**SQP**：每步对拉格朗日二阶近似+约束线性化，解QP得方向，线搜索/信赖域更新。

**ALM**：乘子+二次罚项并入目标，交替最小化增广拉格朗日函数与更新乘子/罚参数。
