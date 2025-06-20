# Main Takeaway
无约束最优问题的求解

- [ ] 掌握多元函数的梯度与Hessian计算
- [ ] 理解方向导数理论与几何意义
- [ ] 熟练掌握变分法与Euler-Lagrange方程
- [ ] 掌握经典优化算法：梯度下降、牛顿法、拟牛顿法
- [ ] 理解递推最小二乘与Sherman-Morrison-Woodbury公式

<!--more-->

# Mathematical Fundamentals

## Hessian and Jacobi

> 对多元情况先做一元，二元找规律

这里不再多做介绍，直接给出一个实际例子

设 $A  \in R^{n\times n}$为对称矩阵,$b\in R^Pn,c\in R$ ，求：

1. 求线性函数 $  f(\mathbf{x}) = \mathbf{b}^\top \mathbf{x}  $ 的梯度和Hessian矩阵。

2. 给定二次函数：
   $$
   f(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x} + \mathbf{b}^\top \mathbf{x} + c
   $$

   求其梯度和Hessian矩阵。

### Problem1 

Gradient Hessian
$$
\frac{\partial f}{\partial x_k} = b_k \quad \Rightarrow \quad \nabla f(\mathbf{x}) = \mathbf{b}
$$

所有二阶偏导数为零：

$$
\nabla^2 f(\mathbf{x}) = 
\begin{bmatrix}
0 & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0
\end{bmatrix}
= \mathbf{0}
$$

### Problem2

Hessian：

展开函数：
$$
   f(\mathbf{x}) = \sum_{i=1}^n \sum_{j=1}^n a_{ij} x_i x_j + \sum_{k=1}^n b_k x_k + c
 
$$

我们先来看$f(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x}$这个部分
$$
f_1(\mathbf{x}) = \sum_{i=1}^n \sum_{j=1}^n a_{ij} x_i x_j
$$

单独对 $  x_k  $ 求偏导：
$$
\frac{\partial f_1}{\partial x_k} = \sum_{\substack{i=1 \\ i \neq k}}^n a_{ik} x_i + \sum_{\substack{j=1 \\ j \neq k}}^n a_{kj} x_j + 2a_{kk} x_k
\\
=\sum_{i=1}^{n}a_{ik}x_i +\sum_{j=1}^n a_{kj}x_j,~k = 1,2,...,n
$$

所以
$$
\nabla f_1(\mathbf{x}) =A^Tx+Ax =  2A\mathbf{x}(A~is~sysmetric)
$$

$$
\nabla^2 f_1(\mathbf{x})=2A
$$

对总体来说
$$
\frac{\partial f}{\partial x_k} = 2\sum_{i=1}^n a_{ki} x_i + b_k
$$

$$
\nabla f(\mathbf{x}) = 2A\mathbf{x} + \mathbf{b}
$$

二阶偏导数为常数：

$$
\frac{\partial^2 f}{\partial x_i \partial x_j} = 2a_{ij} \quad \Rightarrow \quad \nabla^2 f(\mathbf{x}) = 2A
$$

Jacobi：

计算向量值函数 $\mathbf{F}(\mathbf{x}) = \begin{pmatrix} f_1(\mathbf{x}) \\ f_2(\mathbf{x}) \end{pmatrix}$ 在点 $\mathbf{x} = (1, 0, \pi)^\top$ 处的雅可比矩阵，其中：

- $f_1(\mathbf{x}) = 3x_1 + e^{x_2} x_3$
- $f_2(\mathbf{x}) = {x_1^3 + x_2 \sin x_3}$

- 对 $f_1$ 求偏导：


$$
     \begin{cases}
     \dfrac{\partial f_1}{\partial x_1} = 3 \\
     \dfrac{\partial f_1}{\partial x_2} = e^{x_2}x_3 \\
     \dfrac{\partial f_1}{\partial x_3} = e^{x_2}
     \end{cases}
   
$$

   - 对 $f_2$ 求偏导：



$$
\begin{cases}
     \dfrac{\partial f_2}{\partial x_1} = 3x^1 \\
     \dfrac{\partial f_2}{\partial x_2} = 2x_2\sin{x_3} \\
     \dfrac{\partial f_2}{\partial x_3} = x_2^2\cos{x_3}
     \end{cases}
$$

代入点 $\mathbf{x} = (1, 0, \pi)$：
$$
\mathbf{J_F}(1, 0, \pi) = \begin{pmatrix}
3 & \pi & 1 \\
3 & 0 & 0
\end{pmatrix}
$$

## 方向导数


> “形式”很重要：先有鸡还是先有蛋？先接受这个形式再最优化求解

方向导数定义：方向导数就是函数值在某个“**方向**”上的变化率。

函数$f(x)$在x点关于方向d的方向导数，设 $\Phi(a) = f(x + ad)$，令$u=x+ad$
$$
u=(x_1+ad_1,...,x_n+ad_n)^T = (u_1,...u_n)^T
$$

$$
\Phi\prime(a) = \frac{\partial f(u)}{\partial u_1}\frac{du_1}{da}+...+\frac{\partial f(u)}{\partial u_n}\frac{du_n}{da}\\
 = [\nabla f(u)]^Td = [\nabla f(x+ad)]^Td = d^T[\nabla f(x+ad)]
$$

> $<a,b> = <b,a>$

则一阶方向导数：
$$
\Phi'(0) = \lim_{a \to 0} \frac{f(x+ad) - f(x)}{a} = \nabla f(x) \cdot d
$$

$$
\nabla f(x) \cdot d = \sum_{i=1}^n \frac{\partial f(x)}{\partial x_i} d_i = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \cdots & \frac{\partial f}{\partial x_n} \end{bmatrix}
\begin{bmatrix} d_1 \\ \vdots \\ d_n \end{bmatrix}
$$

二阶方向导数可通过继续求导得到：
$$
\Phi''(a) = \sum_{i=1}^n \left[ \sum_{j=1}^n \frac{\partial^2 f(u)}{\partial u_i \partial u_j} d_j \right] d_i
$$

$$
\Phi''(a) = \begin{bmatrix} d_1 & \cdots & d_n \end{bmatrix}
\begin{bmatrix}
\frac{\partial^2 f}{\partial u_1^2} & \cdots & \frac{\partial^2 f}{\partial u_1 \partial u_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial u_n \partial u_1} & \cdots & \frac{\partial^2 f}{\partial u_n^2}
\end{bmatrix}
\begin{bmatrix} d_1 \\ \vdots \\ d_n \end{bmatrix}\\
 = d^\top \nabla^2 f(x+a) \ d
$$

$$
\Phi''(0) = d^\top \nabla^2 f(x) \ d
$$

对于任意给定的$d\ne 0$，若极限$\lim_{a \to 0^+} \frac{f(\mathbf{\bar x} + ad) - f(\mathbf{\bar x})}{a||d||}$记为$\frac{\partial}{\partial d}f(\bar x)$，即
$$
\frac{\partial}{\partial d}f(\bar x)=\lim_{a \to 0^+} \frac{f(\mathbf{\bar x} + ad) - f(\mathbf{\bar x})}{a||d||}
$$
**定理一**：若函数$f(x)$具有连续一阶偏导数，则它在$\bar x$处沿方向d的一阶方向导数为
$$
\frac{\partial}{\partial d}f(\bar x) =<\nabla f,\frac{d}{||d||}>=\frac{1}{\|\mathbf{d}\|}\mathbf{d}^T \nabla f
$$
证明如下：

对于任意非零方向向量 $\mathbf{d} = (d_1, d_2, \ldots, d_n)^\top$，函数 $f(\mathbf{x})$ 在点 $\mathbf{x} = (x_1, x_2, \ldots, x_n)^\top$ 处的方向导数定义为：

$$
D_{\mathbf{d}} f(\mathbf{x}) = \lim_{a \to 0^+} \frac{f(\mathbf{x} + a\mathbf{d}) - f(\mathbf{x})}{a \|\mathbf{d}\|}
$$

1. 定义辅助函数：

    $$
      \phi(a) = f(\mathbf{x} + a\mathbf{d})
    $$

   其中 $\phi(a)$ 是关于标量 $a$ 的一元函数。

2. 对 $\phi(a)$ 求导：

    $$
    \phi'(a) = \frac{d}{da} f(\mathbf{x} + a\mathbf{d}) = \mathbf{d}^\top \nabla f(\mathbf{x} + a\mathbf{d})
    $$

3. 计算 $a=0$ 处的导数：

    $$
      \phi'(0) = \mathbf{d}^\top \nabla f(\mathbf{x})
    $$

4. 方向导数表达式推导：

    $$
    D_{\mathbf{d}} f(\mathbf{x}) = \frac{1}{\|\mathbf{d}\|} \lim_{a\to0^+} \frac{\phi(a)-\phi(0)}{a}
    =
    \frac{1}{\|\mathbf{d}\|} \phi'(0) = \frac{1}{\|\mathbf{d}\|} \mathbf{d}^\top \nabla f(\mathbf{x})
    $$

    由Cauchy-Sehwarz不等式可得
    $$
    -||\nabla f(\bar x)||\le \frac{\partial}{\partial d}f(\bar x)\le ||\nabla f(\bar x)||
    $$
    $d=-||\nabla f(\bar x)$为最速下降方向

**定理二**：$f(x)$在x沿方向d的二阶方向导数为：
$$
\frac{\partial}{\partial d}f(\bar x)=\lim_{a \to 0^+} \frac{\frac{\partial}{\partial d}f(\mathbf{\bar x} + ad) - \frac{\partial}{\partial d}f(\mathbf{\bar x})}{a||d||}=\frac{1}{||d||^2}d^T \nabla^2f(\bar x)d
$$

## 鞍点与焦点

- 焦点（focus）：在动力系统的相平面分析中，**焦点**是一类**平衡点**（即系统状态变化率为零的点），其周围轨迹呈现螺旋状收敛或发散的特征。焦点分为**稳定焦点**和**不稳定焦点**两种类型
- 鞍点（saddle point）



# 变分法

这一节主要是为了介绍变分法并得到E-L方程这个大杀器

## 核心理论

通过对泛函极值问题:
$$
J[y] = \int_{x_0}^{x_1} L(x, y(x), y'(x)) dx
$$
的推导可以得到传说中的欧拉-拉格朗日方程（E-L equation）

> 直接对泛函极值做变分求最优路径

$$
\frac{\partial L}{\partial y}-\frac{\partial}{\partial x}\frac{\partial L}{\partial y\prime} = 0
$$
当$L$的表达式中不显含$x$时有$L-y\prime \frac{\partial L}{\partial y\prime}=C$，为什么呢？：
假设拉格朗日函数 $  L = L(y, y')  $ 不显含自变量 $  x  $，即满足：
$$
\frac{\partial L}{\partial x} = 0
$$
对 $  L  $ 关于 $  x  $ 求全导数：
$$
\frac{\mathrm{d}L}{\mathrm{d}x} = \frac{\partial L}{\partial y} \cdot \frac{\mathrm{d}y}{\mathrm{d}x} + \frac{\partial L}{\partial y'} \cdot \frac{\mathrm{d}y'}{\mathrm{d}x} = \frac{\partial L}{\partial y} \cdot y' + \frac{\partial L}{\partial y'} \cdot y''
$$
根据欧拉-拉格朗日方程：
$$
\frac{\partial L}{\partial y} = \frac{\mathrm{d}}{\mathrm{d}x} \left( \frac{\partial L}{\partial y'} \right)
$$
将其代入全导数表达式：
$$
\frac{\mathrm{d}L}{\mathrm{d}x} = \left[ \frac{\mathrm{d}}{\mathrm{d}x} \left( \frac{\partial L}{\partial y'} \right) \right] \cdot y' + \frac{\partial L}{\partial y'} \cdot y''
$$
观察右侧表达式，可改写为：
$$
\frac{\mathrm{d}}{\mathrm{d}x} \left( y' \cdot \frac{\partial L}{\partial y'} \right) = y'' \cdot \frac{\partial L}{\partial y'} + y' \cdot \frac{\mathrm{d}}{\mathrm{d}x} \left( \frac{\partial L}{\partial y'} \right)
$$
因此有：
$$
\frac{\mathrm{d}L}{\mathrm{d}x} = \frac{\mathrm{d}}{\mathrm{d}x} \left( y' \cdot \frac{\partial L}{\partial y'} \right)
$$
移项后得到守恒方程：
$$
\frac{\mathrm{d}}{\mathrm{d}x} \left( L - y' \cdot \frac{\partial L}{\partial y'} \right) = 0
$$
积分后得到守恒量：
$$
L - y' \cdot \frac{\partial L}{\partial y'} = C \quad (\text{常数})
$$

- 推广：[变分法笔记(2)——Euler-Lagrange方程的基础推广](https://zhuanlan.zhihu.com/p/358115697)

  - Lagrange函数推广到关于y的高阶导数、y是一元向量值函数的情形
  - 经典力学的数学基础
  - 推广到y是多元函数的情形



## 经典变分问题

- 最速下降线[什么是最速降线？它又有何奇妙的性质呢？](https://zhuanlan.zhihu.com/p/68140784)

  经过建模得到系统方程：
  $$
  L(x,y,y\prime)=\sqrt{\frac{1+(\frac{dy}{dx})^2}{2gy}}
  $$
  ![v2-0f20826aa3a5fcbd4bcaea8f843ce764_b](markdown-img/OP-无约束最优问题.assets/v2-0f20826aa3a5fcbd4bcaea8f843ce764_b.webp)

- 平面两点直线距离最短[如何只通过计算证明“两点之间，线段最短”?](https://www.zhihu.com/question/355602892)

# Unconstrained Optimization

介绍一些常见的优化算法



## 梯度下降
梯度下降：gradient flow梯度流的介绍：[梯度流：探索通向最小值之路](https://kexue.fm/archives/9660)

梯度流是将我们在用梯度下降法中寻找最小值的过程中的各个点连接起来，形成一条随（虚拟的）时间变化的轨迹，这条轨迹便被称作“梯度流”。梯度流核心思想就是将**离散的优化算法（如梯度下降）转化为连续的动力学系统**，通过微分方程描述参数随时间的演化轨迹
$$
\frac{dx(t)}{dt}=-\nabla f(x(t))
$$
最速方向：为什么“梯度的负方向是局部下降最快的方向”，实际上是在在欧氏空间中且有约束条件$||x-x_t||=\epsilon$
$$
x_{t+1}=x_t -\gamma \nabla f(x_t)
$$
所以这是一个带约束的优化，常将其转化为：
$$
x_{t+1}=\arg \min_x \frac{||x-x_t||^2}{2\alpha}+f(x)
$$
将约束当成惩罚项加入到优化目标，这样就不用考虑求解约束，也容易推广

> 根据不同的正则项，可以行程不同的梯度下降方案

搜索方向$d_t=-\nabla f(x_t)$，若采用精确线性搜索即$\gamma_k = \arg \min_{\gamma>0}f(x_{t}+\gamma d_t)$

则有：
$$
\frac{df(x_t+\gamma d_t)}{d\gamma}|_{\gamma=\gamma_t} =(d_t)^T\nabla f(x_{t+1})=0
$$
这表明相邻两次的搜索方向$d_t$ and $d_{t+1}$是正交的

<img src="markdown-img/OP-无约束最优问题.assets/image-20250311140727165.png" alt="image-20250311140727165" style="zoom:50%;" />

## Newton`s Method

在当前迭代点对目标函数$f(x)$进行二阶泰勒展开，构造一个近似二次函数，并求解这个二次函数的极值点作为下一步的迭代点
$$
x_{t+1} =x_t +step = x_t - \nabla^2 f(x_t)^{-1} \nabla f(x_t)
$$

## 高斯-牛顿法

牛顿法的变体，在非线性最小二乘中通过忽略Hessian矩阵的二阶项将Hessian矩阵近似为$J(x)^T J(x)$

## Sherman-Morrison-Woodbury Formula

先介绍Sherman-Morrison-Woodbury Formula

是**矩阵逆计算**的重要工具，适用于对可逆矩阵进行低秩修正的场景。

核心思想：通过低秩修正项快速更新逆矩阵，避免直接计算大规模矩阵的逆
$$
(A + U V^\top)^{-1} = A^{-1} - A^{-1} U (I_k + V^\top A^{-1} U)^{-1} V^\top A^{-1}
$$
当秩为1时：
$$
(A + U V^\top)^{-1} = A^{-1} - \frac{A^{-1} U  V^\top A^{-1}}{1 + V^\top A^{-1} U}
$$

## 拟牛顿法

Quasi-Newton Method拟牛顿法

Hessian矩阵求逆太难了，找了个$H_K$来替代，对$H_K$的要求：

- 仅需要迭代点处的梯度信息
- $H_{K+1}$在迭代过程中始终保持正定
- 方法具有较快的收敛速度

> $B_K$是Hessian矩阵的近似，$H_K$是Hessian矩阵逆的近似

在拟牛顿法中，初始Hessian近似矩阵$B_0$常取$B_0=I$，然后每次更新$B_{K+1}(H_{K+1})$都是在$B_{K}(H_{K})$的基础上增加一个修正项$\Delta B_{K}(\Delta H_{K})$，即$B_{K+1}(H_{K+1}) = B_{K}(H_{K})+\Delta B_{K}(\Delta H_{K})$

**拟牛顿条件**：在迭代优化中，希望近似Hessian逆矩阵 $  H_{k+1}  $ 满足：
$$
H_{k+1} y_k = s_k \quad \text{其中} \quad 
\begin{cases} 
s_k = x_{k+1} - x_k \\
y_k = \nabla f(x_{k+1}) - \nabla f(x_k)
\end{cases}
$$

> 注意其中的拟牛顿条件是推导得到$H_K$需要满足的

下面介绍三种拟牛顿法：

### SR1 对称秩1

由于我们希望$H_k \approx \nabla^2 f(x_k)^{-1}, \quad H_{k+1} \approx \nabla^2 f(x_{k+1})^{-1}$，所以有
$$
\Delta H_k \approx \nabla^2 f(x_{k+1})^{-1} - \nabla^2 f(x_k)^{-1}
$$
因为$$\nabla^2 f(x_{k+1})^{-1} \in S^{n \times n}, \quad \nabla^2 f(x_k)^{-1} \in S^{n \times n}$$都是对称矩阵，所以$$\Delta H_k \in S^{n \times n}$$也应该是一个对称矩阵

因此在SR1算法中我们设更新矩阵为：
$$
\Delta H_k = \beta u u^T \quad (\beta \in \mathbb{R},\ u \in \mathbb{R}^n)
$$

则迭代公式为：

$$
H_{k+1} = H_k + \beta u u^T
$$
左乘向量 $y_k$ 并代入拟牛顿条件 $H_{k+1} y_k = s_k$：

$$
\begin{aligned}
H_{k+1} y_k &= H_k y_k + \beta u u^T y_k = s_k \\
\Rightarrow s_k - H_k y_k &= \beta (u^T y_k) u
\end{aligned}
$$
因为$u^T y_k$为实数，因此$u与s_k-H_ky_k$共线，存在标量 $\gamma$，使得：
$$
u = \gamma(s_k - H_k y_k)
$$

将 $u$ 代入约束方程：
$$
\begin{aligned}
   s_k - H_k y_k &= \beta \gamma^2 (s_k - H_k y_k)^T y_k (s_k - H_k y_k) \\
  \quad \Rightarrow \beta \gamma^2 (s_k - H_k y_k)^T y_k &= 1
   \end{aligned}
$$
解得标量系数：$$ \beta^2 = \frac{1}{(s_k - H_k y_k)^T y_k} $$

将 $u,\beta$ 回代可得 SR1 更新：

$$
H_{k+1}^{SR1} = H_k + \frac{(s_k - H_k y_k)(s_k - H_k y_k)^T}{(s_k - H_k y_k)^T y_k}
$$

### DFP Method（Davidon-Fletcher-Powell）

目标：

拟牛顿条件：在迭代优化中，希望近似Hessian逆矩阵 $  H_{k+1}  $ 满足：
$$
H_{k+1} \Delta g_k = \Delta x_k \quad \text{其中} \quad 
\begin{cases} 
\Delta x_k = x_{k+1} - x_k \\
\Delta g_k = \nabla f(x_{k+1}) - \nabla f(x_k)
\end{cases}
$$
DFP更新公式：
$$
H_{k+1} = H_k + \frac{\Delta x_k \Delta x_k^\top}{\Delta x_k^\top \Delta g_k} - \frac{H_k \Delta g_k \Delta g_k^\top H_k}{\Delta g_k^\top H_k \Delta g_k}
$$
**正向推导**：

假设Hessian逆的更新形式为秩2修正：

$$
H_{k+1} = H_k + \beta u u^\top + \gamma v v^\top
$$

其中 $  u, v \in \mathbb{R}^n  $，$  \beta, \gamma \in \mathbb{R}  $。

将修正形式代入 $  H_{k+1} \Delta g_k = \Delta x_k  $：

$$
H_k \Delta g_k + \beta u (u^\top \Delta g_k) + \gamma v (v^\top \Delta g_k) = \Delta x_k
$$
为了简化方程，选择基向量与物理量直接相关：

- 方向1：参数变化方向 $  u = \Delta x_k  $
- 方向2：梯度变化方向 $  v = H_k \Delta g_k  $

代入后得到：

$$
\beta (\Delta x_k^\top \Delta g_k) \Delta x_k + \gamma (\Delta g_k^\top H_k \Delta g_k) H_k \Delta g_k = \Delta x_k - H_k \Delta g_k
$$
令两边系数相等：

$$
\beta (\Delta x_k^\top \Delta g_k) = 1 \quad \Rightarrow \quad \beta = \frac{1}{\Delta x_k^\top \Delta g_k}
$$

$$
\gamma (\Delta g_k^\top H_k \Delta g_k) = -1 \quad \Rightarrow \quad \gamma = -\frac{1}{\Delta g_k^\top H_k \Delta g_k}
$$

将 $  \beta, \gamma, u, v  $ 代入秩2修正假设：

$$
H_{k+1} = H_k + \frac{\Delta x_k \Delta x_k^\top}{\Delta x_k^\top \Delta g_k} - \frac{H_k \Delta g_k \Delta g_k^\top H_k}{\Delta g_k^\top H_k \Delta g_k}
$$

**逆向验证（使用SMW）**：

目标是通过SMW公式推导其逆矩阵 $  B_{k+1} = H_{k+1}^{-1}  $。

将DFP公式分解为两个秩1修正项：
$$
H_{k+1} = H_k + \underbrace{\frac{\Delta x_k \Delta x_k^\top}{\Delta x_k^\top \Delta g_k}}_{\text{秩1项}} - \underbrace{\frac{H_k \Delta g_k \Delta g_k^\top H_k}{\Delta g_k^\top H_k \Delta g_k}}_{\text{秩1项}}
 
对每个秩1修正项分别应用Sherman-Morrison公式（SMW的秩1特例）。
$$
第一项修正（正项）：
$$
     H^{(1)} = H_k + \frac{\Delta x_k \Delta x_k^\top}{\Delta x_k^\top \Delta g_k}
   
$$

 设 $  u = \Delta x_k  $, $  v = \frac{\Delta x_k}{\Delta x_k^\top \Delta g_k}  $，则：
$$
     H^{(1)} = H_k + u v^\top
   
$$

 应用SMW逆公式：
$$
\left( H^{(1)} \right)^{-1} = H_k^{-1} - \frac{H_k^{-1} \Delta x_k \Delta x_k^\top H_k^{-1}}{\Delta x_k^\top \Delta g_k + \Delta x_k^\top H_k^{-1} \Delta x_k}
$$

第二项修正（负项）：
$$
         H_{k+1} = H^{(1)} - \frac{H_k \Delta g_k \Delta g_k^\top H_k}{\Delta g_k^\top H_k \Delta g_k}
       
$$

 设 $  u = -H_k \Delta g_k  $, $  v = \frac{H_k \Delta g_k}{\Delta g_k^\top H_k \Delta g_k}  $，则：
$$
         H_{k+1} = H^{(1)} + u v^\top
       
$$
再次应用SMW公式得到最终DFP的$B_{k+1}$更新公式
$$
    B_{k+1} = \left( I - \frac{\Delta x_k \Delta g_k^\top}{\Delta g_k^\top \Delta x_k} \right) B_k \left( I - \frac{\Delta g_k \Delta x_k^\top}{\Delta g_k^\top \Delta x_k} \right) + \frac{\Delta x_k \Delta x_k^\top}{\Delta g_k^\top \Delta x_k}
$$
下面验证其满足 $  H_{k+1} \Delta g_k = \Delta x_k  $：
将 $  H_{k+1}  $ 代入拟牛顿条件 $  H_{k+1} \Delta g_k = \Delta x_k  $：
$$
       \begin{aligned}
       H_{k+1} \Delta g_k &= \left( H_k + \frac{\Delta x_k \Delta x_k^\top}{\Delta x_k^\top \Delta g_k} - \frac{H_k \Delta g_k \Delta g_k^\top H_k}{\Delta g_k^\top H_k \Delta g_k} \right) \Delta g_k \\
       &= H_k \Delta g_k + \Delta x_k - H_k \Delta g_k \\
       &= \Delta x_k
       \end{aligned}
     
$$

 结论：公式满足拟牛顿条件。

### BFGS Method

目标：

拟牛顿条件：在迭代优化中，希望近似Hessian矩阵的逆 $  H_{k+1}  $ 满足：
$$
H_{k+1} \Delta g_k = \Delta x_k \quad \text{其中} \quad 
\begin{cases} 
\Delta x_k = x_{k+1} - x_k \\
\Delta g_k = \nabla f(x_{k+1}) - \nabla f(x_k)
\end{cases}
$$
BFGS更新公式（Hessian逆矩阵形式）：

$$
H_{k+1} = \left( I - \frac{\Delta x_k \Delta g_k^\top}{\Delta g_k^\top \Delta x_k} \right) H_k \left( I - \frac{\Delta g_k \Delta x_k^\top}{\Delta g_k^\top \Delta x_k} \right) + \frac{\Delta x_k \Delta x_k^\top}{\Delta g_k^\top \Delta x_k}
$$

BFGS直接更新Hessian矩阵 $  B_k  $，其更新规则为：

$$
B_{k+1} = B_k + \frac{\Delta g_k \Delta g_k^\top}{\Delta g_k^\top \Delta x_k} - \frac{B_k \Delta x_k \Delta x_k^\top B_k}{\Delta x_k^\top B_k \Delta x_k}
$$

目标是通过SMW公式推导其逆矩阵 $  H_{k+1} = B_{k+1}^{-1}  $。

将BFGS更新公式视为对 $  B_k  $ 的秩2修正：

$$
B_{k+1} = B_k + U V^\top,\\ U = \begin{bmatrix} \frac{\Delta g_k}{\sqrt{\Delta g_k^\top \Delta x_k}} & -\frac{B_k \Delta x_k}{\sqrt{\Delta x_k^\top B_k \Delta x_k}} \end{bmatrix}, \quad V = \begin{bmatrix} \frac{\Delta g_k}{\sqrt{\Delta g_k^\top \Delta x_k}} & \frac{B_k \Delta x_k}{\sqrt{\Delta x_k^\top B_k \Delta x_k}} \end{bmatrix}
$$

根据Woodbury公式（SMW的秩k推广）：

$$
H_{k+1} = B_{k}^{-1} - B_{k}^{-1} U (I + V^\top B_{k}^{-1} U)^{-1} V^\top B_{k}^{-1}
$$

代入 $  H_k = B_{k}^{-1}  $，并化简后得到：

$$
H_{k+1} = H_k + \frac{\Delta x_k \Delta x_k^\top}{\Delta g_k^\top \Delta x_k} - \frac{H_k \Delta g_k \Delta g_k^\top H_k}{\Delta g_k^\top H_k \Delta g_k} + \frac{H_k \Delta g_k \Delta x_k^\top + \Delta x_k \Delta g_k^\top H_k}{\Delta g_k^\top \Delta x_k}
$$
将交叉项合并为对称形式：

$$
H_{k+1} = \left( I - \frac{\Delta x_k \Delta g_k^\top}{\Delta g_k^\top \Delta x_k} \right) H_k \left( I - \frac{\Delta g_k \Delta x_k^\top}{\Delta g_k^\top \Delta x_k} \right) + \frac{\Delta x_k \Delta x_k^\top}{\Delta g_k^\top \Delta x_k}
$$
将 $  H_{k+1}  $ 代入 $  H_{k+1} \Delta g_k = \Delta x_k  $：

$$
\begin{aligned}
H_{k+1} \Delta g_k &= \left( I - \frac{\Delta x_k \Delta g_k^\top}{\Delta g_k^\top \Delta x_k} \right) H_k \left( I - \frac{\Delta g_k \Delta x_k^\top}{\Delta g_k^\top \Delta x_k} \right) \Delta g_k + \frac{\Delta x_k \Delta x_k^\top}{\Delta g_k^\top \Delta x_k} \Delta g_k \\
&= \left( I - \frac{\Delta x_k \Delta g_k^\top}{\Delta g_k^\top \Delta x_k} \right) H_k \left( \Delta g_k - \Delta g_k \right) + \Delta x_k \\
&= \Delta x_k
\end{aligned}
$$

验证通过：BFGS更新公式满足拟牛顿条件。

### Broyden族

既然DFP和BFGS是互为对偶的，那用哪一个比较好呢？你当然可以通过若干组实验来测试哪个的性能的更优，或者对其收敛一通验证。但是一个比较的朴素的做法就是“我都要”，也就是取DFP迭代式和BFGS迭代式的正加权组合



  1. 问题定义

      目标：求解线性最小二乘问题 $  \min_x \|Ax - b\|^2  $，并在新增数据点时**增量更新解** $  x  $，避免重新计算逆矩阵。

      符号定义：

      - $  A_k \in \mathbb{R}^{k \times n}  $: 前 $  k  $ 个样本的设计矩阵
      - $  b_k \in \mathbb{R}^k  $: 前 $  k  $ 个样本的观测向量
      - $  P_k = (A_k^\top A_k)^{-1}  $: 信息矩阵的逆（协方差矩阵）
      - $  x_k = P_k A_k^\top b_k  $: 第 $  k  $ 步的最小二乘解


  2. 增量更新推导

      当新增一个样本 $  (a_{k+1}, b_{k+1})  $ 时，设计矩阵和观测向量扩展为：
      $$
      A_{k+1} = \begin{bmatrix} A_k \\ a_{k+1}^\top \end{bmatrix}, \quad b_{k+1} = \begin{bmatrix} b_k \\ b_{k+1} \end{bmatrix}
      $$

      1.   更新信息矩阵逆 $  P_{k+1}  $

         定义$  P_k = (A_k^\top A_k)^{-1}  $: 信息矩阵的逆（协方差矩阵），因为
         $$
             A_{k+1}^\top A_{k+1} = A_k^\top A_k +a_{k+1} a_{k+1}^\top
         $$
           根据Sherman-Morrison公式（秩1修正）：

         $$
             P_{k+1} = \left( A_k^\top A_k + a_{k+1} a_{k+1}^\top \right)^{-1} = P_k - \frac{P_k a_{k+1} a_{k+1}^\top P_k}{1 + a_{k+1}^\top P_k a_{k+1}}
         $$

      2. 更新参数估计 $  x_{k+1}  $
         $$
             x_{k+1} = P_{k+1} A_{k+1}^\top b_{k+1} = P_{k+1} \left( A_k^\top b_k + a_{k+1} b_{k+1} \right)
         $$
           代入 $  P_{k+1}  $，得到参数更新公式：
         $$
         x_{k+1} = x_k + \frac{P_k a_{k+1}}{1 + a_{k+1}^\top P_k a_{k+1}} (b_{k+1} - a_{k+1}^\top x_k)
         $$

  3. 算法步骤

  输入：初始解 $  x_0  $, 初始逆矩阵 $  P_0 = \lambda^{-1} I  $（正则化项）
  迭代流程（对每个新样本 $  (a_{k+1}, b_{k+1})  $）：

  1. 计算预测残差：
     $$
     e_{k+1} = b_{k+1} - a_{k+1}^\top x_k
     $$

  2. 计算增益向量：
     $$
     K_{k+1} = \frac{P_k a_{k+1}}{1 + a_{k+1}^\top P_k a_{k+1}}
     $$

  3. 更新逆矩阵：
     $$
     P_{k+1} = P_k - K_{k+1} a_{k+1}^\top P_k
     $$

  4. 更新参数估计：
     $$
     x_{k+1} = x_k + K_{k+1} e_{k+1}
     $$
     

#### 扩展：Woodbury公式批量更新

若一次新增 $  m  $ 个样本 $  \{a_{k+1}^{(i)}, b_{k+1}^{(i)}\}_{i=1}^m  $，设：

$$
    U = V = \begin{bmatrix} a_{k+1}^{(1)} & \cdots & a_{k+1}^{(m)} \end{bmatrix} \in \mathbb{R}^{n \times m}
$$

则Woodbury公式给出：

$$
    P_{k+1} = P_k - P_k U (I + V^\top P_k U)^{-1} V^\top P_k
$$

适用于高吞吐量场景（如传感器网络）。

#### 数值稳定性

  - 正则化：初始 $  P_0 = \lambda^{-1} I  $ 避免 $  A^\top A  $ 奇异。

- 数值误差控制：定期重置 $  P_k  $ 或使用Cholesky分解更新。



## HW

- 复习最小二乘

- 整理+推导iterative least squares

  要求使用sherman-Morrison-Woodbarry

- BFGS，DFP公式证明（SMW推导）








# References

- [变分法简介Part 1.（Calculus of Variations)](https://zhuanlan.zhihu.com/p/20718489)
- [变分法笔记(2)——Euler-Lagrange方程的基础推广](https://zhuanlan.zhihu.com/p/358115697)

- [如何只通过计算证明“两点之间，线段最短”?](https://www.zhihu.com/question/355602892)
- [什么是最速降线？它又有何奇妙的性质呢？](https://zhuanlan.zhihu.com/p/68140784)
- 代码可以直接运行：[11. 优化算法 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh-v2.d2l.ai/chapter_optimization/index.html)
- MIT优化课件（前两节）[Lecture Notes | Principles of Optimal Control | Aeronautics and Astronautics | MIT OpenCourseWare](https://ocw.mit.edu/courses/16-323-principles-of-optimal-control-spring-2008/pages/lecture-notes/)
- [梯度流：探索通向最小值之路](https://kexue.fm/archives/9660)
- 最小二乘（超级经典问题）[Solving Non-linear Least Squares — Ceres Solver (ceres-solver.org)](http://ceres-solver.org/nnls_solving.html)

- Iteratively reweighted least squares (IRLS)

- [拟牛顿法与SR1,DFP,BFGS三种拟牛顿算法](https://zhuanlan.zhihu.com/p/306635632)