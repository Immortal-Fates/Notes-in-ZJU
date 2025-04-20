# Main Takeaway

配套CMU-16-745 Optimal Control and Reinforcement Learning食用



<!--more-->



# ZJU-Optimization & Optimal Control

- Lec 1-4：做神经网络优化

## Lec 1 intro

这一节主要是介绍了控制论角度和E-L方程

- 深度学习的控制论视角——PID加速优化训练框架（PIDAO）

  深度神经网络-DNN，本质上是，面向信息加工的过程系统设计与控制

  深度神经网络反馈机制缺失

  本质上是一个高纬度的参数优化问题

  多用连续求解（方法多）再离散化

- 瓦特改进[蒸汽机离心调速器原理与应用](https://blog.csdn.net/u013414501/article/details/82428094)——相当于加了反馈

nature-inspired computing



主要介绍两个部分

- 优化optimization——finite 

  - unconstrained opt——PID
  - constrained opt
  - linear programming
  - nonlinear programming——NLP

- 最优控制optimal control——infinite dimensioned optimization

  - 变分法

  - 分析力学建模

  - PMP[module3.pdf (nd.edu)](https://www3.nd.edu/~lemmon/courses/ee565/lectures/module3.pdf)

    [https://www.cimat.mx/~murrieta/HJBandPMP.pdf](https://www.cimat.mx/~murrieta/HJBandPMP.pdf#:~:text=PMP expresses conditions along the optimal trajectory%2C as,optimal control is function of (t) %3D rV(x(t)).)

  - $DP\to RL$



根据[变分法简介Part 1.（Calculus of Variations)](https://zhuanlan.zhihu.com/p/20718489)，得到传说中的欧拉-拉格朗日方程（E-L equation）
$$
\frac{\partial L}{\partial y}-\frac{\partial}{\partial x}\frac{\partial L}{\partial y\prime} = 0
$$
当$L$的表达式中不显含$x$时有$L-y\prime \frac{\partial L}{\partial y\prime}=C$

> 为什么当$L$的表达式中不显含$x$时有这样的形式：
>
> **1. 数学推导：守恒量的形式**
>
> 假设拉格朗日函数 $  L = L(y, y')  $ 不显含自变量 $  x  $，即满足：
>
> $$
> \frac{\partial L}{\partial x} = 0
> $$
> 对 $  L  $ 关于 $  x  $ 求全导数：
>
> $$
> \frac{\mathrm{d}L}{\mathrm{d}x} = \frac{\partial L}{\partial y} \cdot \frac{\mathrm{d}y}{\mathrm{d}x} + \frac{\partial L}{\partial y'} \cdot \frac{\mathrm{d}y'}{\mathrm{d}x} = \frac{\partial L}{\partial y} \cdot y' + \frac{\partial L}{\partial y'} \cdot y''
> $$
> 根据欧拉-拉格朗日方程：
>
> $$
> \frac{\partial L}{\partial y} = \frac{\mathrm{d}}{\mathrm{d}x} \left( \frac{\partial L}{\partial y'} \right)
> $$
>
> 将其代入全导数表达式：
>
> $$
> \frac{\mathrm{d}L}{\mathrm{d}x} = \left[ \frac{\mathrm{d}}{\mathrm{d}x} \left( \frac{\partial L}{\partial y'} \right) \right] \cdot y' + \frac{\partial L}{\partial y'} \cdot y''
> $$
> 观察右侧表达式，可改写为：
>
> $$
> \frac{\mathrm{d}}{\mathrm{d}x} \left( y' \cdot \frac{\partial L}{\partial y'} \right) = y'' \cdot \frac{\partial L}{\partial y'} + y' \cdot \frac{\mathrm{d}}{\mathrm{d}x} \left( \frac{\partial L}{\partial y'} \right)
> $$
>
> 因此有：
>
> $$
> \frac{\mathrm{d}L}{\mathrm{d}x} = \frac{\mathrm{d}}{\mathrm{d}x} \left( y' \cdot \frac{\partial L}{\partial y'} \right)
> $$
> 移项后得到守恒方程：
>
> $$
> \frac{\mathrm{d}}{\mathrm{d}x} \left( L - y' \cdot \frac{\partial L}{\partial y'} \right) = 0
> $$
>
> 积分后得到守恒量：
>
> $$
> L - y' \cdot \frac{\partial L}{\partial y'} = C \quad (\text{常数})
> $$
> ---
>
> **2. 关键公式总结**
>
> | 步骤       | 公式                                                         |
> | ---------- | ------------------------------------------------------------ |
> | 全导数     | $ \frac{\mathrm{d}L}{\mathrm{d}x} = \frac{\partial L}{\partial y} y' + \frac{\partial L}{\partial y'} y'' $ |
> | 守恒量形式 | $ L - y' \cdot \frac{\partial L}{\partial y'} = C $          |

- 推广：[变分法笔记(2)——Euler-Lagrange方程的基础推广](https://zhuanlan.zhihu.com/p/358115697)

  > - Lagrange函数推广到关于y的高阶导数、y是一元向量值函数的情形
  > - 经典力学的数学基础
  > - 推广到y是多元函数的情形

两个使用E-L方程的例子：

- 最速下降线[什么是最速降线？它又有何奇妙的性质呢？](https://zhuanlan.zhihu.com/p/68140784)

  建模后：$L(x,y,y\prime)=\sqrt{\frac{1+(\frac{dy}{dx})^2}{2gy}}$

  ![v2-0f20826aa3a5fcbd4bcaea8f843ce764_b](markdown-img/最优化与最优控制.assets/v2-0f20826aa3a5fcbd4bcaea8f843ce764_b.webp)

- 平面两点直线距离最短[(21 封私信 / 80 条消息) 如何只通过计算证明“两点之间，线段最短”?](https://www.zhihu.com/question/355602892)

## Lec 2 Unconstrained Optimization

这节主要介绍一些常见的优化算法

- 代码可以直接运行：[11. 优化算法 — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh-v2.d2l.ai/chapter_optimization/index.html)

- MIT优化课件（前两节）[Lecture Notes | Principles of Optimal Control | Aeronautics and Astronautics | MIT OpenCourseWare](https://ocw.mit.edu/courses/16-323-principles-of-optimal-control-spring-2008/pages/lecture-notes/)

  > 课件见courseware



### Mathematical Fundamentals

- 矩阵条件数的看法：[矩阵的条件数](https://zhuanlan.zhihu.com/p/91393594)

  条件数同时描述了矩阵 A 对向量的拉伸能力和压缩能力，换句话说，令向量发生形变的能力。条件数越大，向量在变换后越可能变化得越多。

  减小病态矩阵的影响——加正则项



Gradient, Hessian, Jacobian

The gradient of a scalar function $f : \mathbb{R}^n \rightarrow \mathbb{R}$ is defined by
$$
\nabla f(\mathbf{x}) = 
\begin{bmatrix}
\frac{\partial f(\mathbf{x})}{\partial x_1} \\
\frac{\partial f(\mathbf{x})}{\partial x_2} \\
\vdots \\
\frac{\partial f(\mathbf{x})}{\partial x_n}
\end{bmatrix}
= 
\left[ \frac{\partial f(\mathbf{x})}{\partial x_1}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n} \right]^\top
= \left[ \frac{\partial f(\mathbf{x})}{\partial \mathbf{x}} \right]^\top
$$
The Hessian of a scalar function $f : \mathbb{R}^n \rightarrow \mathbb{R}$ is defined by
$$
\nabla^2 f(\mathbf{x}) = 
\begin{bmatrix}
\frac{\partial^2 f(\mathbf{x})}{\partial x_1^2} & \frac{\partial^2 f(\mathbf{x})}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f(\mathbf{x})}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f(\mathbf{x})}{\partial x_2 \partial x_1} & \frac{\partial^2 f(\mathbf{x})}{\partial x_2^2} & \cdots & \frac{\partial^2 f(\mathbf{x})}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f(\mathbf{x})}{\partial x_n \partial x_1} & \frac{\partial^2 f(\mathbf{x})}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f(\mathbf{x})}{\partial x_n^2}
\end{bmatrix}
= \frac{\partial}{\partial \mathbf{x}} [\nabla f(\mathbf{x})]
$$
The Jacobian of a vector-valued function $f : \mathbb{R}^n \rightarrow \mathbb{R}^m$ is defined by
$$
\frac{\partial \mathbf{f}(\mathbf{x})}{\partial \mathbf{x}} = 
\begin{bmatrix}
\frac{\partial f_1(\mathbf{x})}{\partial x_1} & \frac{\partial f_1(\mathbf{x})}{\partial x_2} & \cdots & \frac{\partial f_1(\mathbf{x})}{\partial x_n} \\
\frac{\partial f_2(\mathbf{x})}{\partial x_1} & \frac{\partial f_2(\mathbf{x})}{\partial x_2} & \cdots & \frac{\partial f_2(\mathbf{x})}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m(\mathbf{x})}{\partial x_1} & \frac{\partial f_m(\mathbf{x})}{\partial x_2} & \cdots & \frac{\partial f_m(\mathbf{x})}{\partial x_n}
\end{bmatrix}
$$
Mean Value Theorem for Vector-Valued Functions

Let $\mathbf{f} : \mathbb{R}^n \rightarrow \mathbb{R}^m$ be a vector-valued function. The **mean value theorem** states:
$$
\mathbf{f}(\mathbf{x} + \mathbf{p}) - \mathbf{f}(\mathbf{x}) = \int_0^1 \frac{\partial \mathbf{f}(\mathbf{x} + \eta \mathbf{p})}{\partial \mathbf{x}} \mathbf{p} \, d\eta
$$
where:
- $\frac{\partial \mathbf{f}(\mathbf{x})}{\partial \mathbf{x}}$ is the Jacobian matrix of $\mathbf{f}$ at $\mathbf{x}$
- $\eta \in [0, 1]$ parametrizes the integration path between $\mathbf{x}$ and $\mathbf{x}+\mathbf{p}$
- The integral represents an averaged linear approximation of $\mathbf{f}$ over the line segment connecting $\mathbf{x}$ and $\mathbf{x}+\mathbf{p}$



若函数 $f(\mathbf{x})$ 对其所有变量的二阶偏导数连续，则有：
$$
\frac{\partial^2 f(\mathbf{x})}{\partial x_i \partial x_j} = \frac{\partial^2 f(\mathbf{x})}{\partial x_j \partial x_i}
$$
此时函数的Hessian矩阵为对称矩阵，矩阵满足 $\nabla^2 f = (\nabla^2 f)^\top$。

应用意义

- **优化理论**：Hessian矩阵的对称性是判断极值点（凸性/凹性）的基础。
- **泰勒展开**：二阶泰勒展开式 $f(\mathbf{x}+\mathbf{h}) \approx f(\mathbf{x}) + \nabla f \cdot \mathbf{h} + \frac{1}{2} \mathbf{h}^\top \nabla^2 f \mathbf{h}$ 依赖此对称性。
- **物理学方程**：连续介质力学、电磁学中的场方程需满足此条件以保证解的存在性。



### optimation

**下面介绍一些优化算法：**

- 梯度下降：gradient flow梯度流的介绍：[梯度流：探索通向最小值之路](https://kexue.fm/archives/9660)

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

  <img src="markdown-img/最优化与最优控制.assets/image-20250311140727165.png" alt="image-20250311140727165" style="zoom:50%;" />

- Newton`s Method
  
  在当前迭代点对目标函数$f(x)$进行二阶泰勒展开，构造一个近似二次函数，并求解这个二次函数的极值点作为下一步的迭代点
  $$
  x_{t+1} =x_t +step = x_t - \nabla^2 f(x_t)^{-1} \nabla f(x_t)
  $$
  
- 高斯-牛顿法

  牛顿法的变体，在非线性最小二乘中通过忽略Hessian矩阵的二阶项将Hessian矩阵近似为$J(x)^T J(x)$

  
  
- 先介绍**Sherman-Morrison-Woodbury Formula**

  是**矩阵逆计算**的重要工具，适用于对可逆矩阵进行低秩修正的场景。

  **核心思想**：通过低秩修正项快速更新逆矩阵，避免直接计算大规模矩阵的逆
  $$
  (A + U V^\top)^{-1} = A^{-1} - A^{-1} U (I_k + V^\top A^{-1} U)^{-1} V^\top A^{-1}
  $$
  当秩为1时：
  $$
  (A + U V^\top)^{-1} = A^{-1} - \frac{A^{-1} U  V^\top A^{-1}}{1 + V^\top A^{-1} U}
  $$

- Quasi-Newton Method拟牛顿法[拟牛顿法与SR1,DFP,BFGS三种拟牛顿算法](https://zhuanlan.zhihu.com/p/306635632)

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
  
  - SR1 对称秩1
  
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
  
  - DFP Method（Davidon-Fletcher-Powell）
  
    **目标**：
  
    **拟牛顿条件**：在迭代优化中，希望近似Hessian逆矩阵 $  H_{k+1}  $ 满足：
    $$
    H_{k+1} \Delta g_k = \Delta x_k \quad \text{其中} \quad 
    \begin{cases} 
    \Delta x_k = x_{k+1} - x_k \\
    \Delta g_k = \nabla f(x_{k+1}) - \nabla f(x_k)
    \end{cases}
    $$
    **DFP更新公式**：
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
  
    - **方向1**：参数变化方向 $  u = \Delta x_k  $
    - **方向2**：梯度变化方向 $  v = H_k \Delta g_k  $
  
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
  
    ---
  
    **逆向验证（使用SMW）**：
  
    目标是通过SMW公式推导其逆矩阵 $  B_{k+1} = H_{k+1}^{-1}  $。
  
    将DFP公式分解为两个秩1修正项：
    $$
    H_{k+1} = H_k + \underbrace{\frac{\Delta x_k \Delta x_k^\top}{\Delta x_k^\top \Delta g_k}}_{\text{秩1项}} - \underbrace{\frac{H_k \Delta g_k \Delta g_k^\top H_k}{\Delta g_k^\top H_k \Delta g_k}}_{\text{秩1项}}
     
    对每个秩1修正项分别应用Sherman-Morrison公式（SMW的秩1特例）。
    $$
    **第一项修正**（正项）：
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

  
    **第二项修正**（负项）：
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
  
   **结论**：公式满足拟牛顿条件。
  
  - BFGS Method
  
    **目标：**
  
    **拟牛顿条件**：在迭代优化中，希望近似Hessian矩阵的逆 $  H_{k+1}  $ 满足：
    $$
    H_{k+1} \Delta g_k = \Delta x_k \quad \text{其中} \quad 
    \begin{cases} 
    \Delta x_k = x_{k+1} - x_k \\
    \Delta g_k = \nabla f(x_{k+1}) - \nabla f(x_k)
    \end{cases}
    $$
    **BFGS更新公式**（Hessian逆矩阵形式）：
  
    $$
    H_{k+1} = \left( I - \frac{\Delta x_k \Delta g_k^\top}{\Delta g_k^\top \Delta x_k} \right) H_k \left( I - \frac{\Delta g_k \Delta x_k^\top}{\Delta g_k^\top \Delta x_k} \right) + \frac{\Delta x_k \Delta x_k^\top}{\Delta g_k^\top \Delta x_k}
    $$
  
    ## 
  
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
  
    **验证通过**：BFGS更新公式满足拟牛顿条件。
  
  - Broyden族
  
    既然DFP和BFGS是互为对偶的，那用哪一个比较好呢？你当然可以通过若干组实验来测试哪个的性能的更优，或者对其收敛一通验证。但是一个比较的朴素的做法就是“我都要”，也就是取DFP迭代式和BFGS迭代式的正加权组合


- 最小二乘（超级经典问题）[Solving Non-linear Least Squares — Ceres Solver (ceres-solver.org)](http://ceres-solver.org/nnls_solving.html)

- Iteratively reweighted least squares (IRLS)

  **1. 问题定义**

  **目标**：求解线性最小二乘问题 $  \min_x \|Ax - b\|^2  $，并在新增数据点时**增量更新解** $  x  $，避免重新计算逆矩阵。

  **符号定义**：
  - $  A_k \in \mathbb{R}^{k \times n}  $: 前 $  k  $ 个样本的设计矩阵
  - $  b_k \in \mathbb{R}^k  $: 前 $  k  $ 个样本的观测向量
  - $  P_k = (A_k^\top A_k)^{-1}  $: 信息矩阵的逆（协方差矩阵）
  - $  x_k = P_k A_k^\top b_k  $: 第 $  k  $ 步的最小二乘解

  ---

  **2. 增量更新推导**

  当新增一个样本 $  (a_{k+1}, b_{k+1})  $ 时，设计矩阵和观测向量扩展为：

  $$
  A_{k+1} = \begin{bmatrix} A_k \\ a_{k+1}^\top \end{bmatrix}, \quad b_{k+1} = \begin{bmatrix} b_k \\ b_{k+1} \end{bmatrix}
  $$
  **2.1 更新信息矩阵逆 $  P_{k+1}  $**

  定义$  P_k = (A_k^\top A_k)^{-1}  $: 信息矩阵的逆（协方差矩阵），因为
  $$
  A_{k+1}^\top A_{k+1} = A_k^\top A_k +a_{k+1} a_{k+1}^\top
  $$
  根据Sherman-Morrison公式（秩1修正）：

  $$
  P_{k+1} = \left( A_k^\top A_k + a_{k+1} a_{k+1}^\top \right)^{-1} = P_k - \frac{P_k a_{k+1} a_{k+1}^\top P_k}{1 + a_{k+1}^\top P_k a_{k+1}}
  $$
  **2.2 更新参数估计 $  x_{k+1}  $**
  $$
  x_{k+1} = P_{k+1} A_{k+1}^\top b_{k+1} = P_{k+1} \left( A_k^\top b_k + a_{k+1} b_{k+1} \right)
  $$
  代入 $  P_{k+1}  $，得到参数更新公式：
  $$
  x_{k+1} = x_k + \frac{P_k a_{k+1}}{1 + a_{k+1}^\top P_k a_{k+1}} (b_{k+1} - a_{k+1}^\top x_k)
  $$
  ---

  **3. 算法步骤**

  **输入**：初始解 $  x_0  $, 初始逆矩阵 $  P_0 = \lambda^{-1} I  $（正则化项）
  **迭代流程**（对每个新样本 $  (a_{k+1}, b_{k+1})  $）：
  1. **计算预测残差**：

  $$
     e_{k+1} = b_{k+1} - a_{k+1}^\top x_k
   
  $$

  2. **计算增益向量**：

  $$
     K_{k+1} = \frac{P_k a_{k+1}}{1 + a_{k+1}^\top P_k a_{k+1}}
   
  $$

  3. **更新逆矩阵**：

  $$
     P_{k+1} = P_k - K_{k+1} a_{k+1}^\top P_k
   
  $$

  4. **更新参数估计**：

  $$
     x_{k+1} = x_k + K_{k+1} e_{k+1}
   
  $$
  ---

  扩展：Woodbury公式批量更新

  若一次新增 $  m  $ 个样本 $  \{a_{k+1}^{(i)}, b_{k+1}^{(i)}\}_{i=1}^m  $，设：
  
  $$
  U = V = \begin{bmatrix} a_{k+1}^{(1)} & \cdots & a_{k+1}^{(m)} \end{bmatrix} \in \mathbb{R}^{n \times m}
  $$

  则Woodbury公式给出：

  $$
  P_{k+1} = P_k - P_k U (I + V^\top P_k U)^{-1} V^\top P_k
  $$

  适用于高吞吐量场景（如传感器网络）。

  **6. 数值稳定性**
  
  - **正则化**：初始 $  P_0 = \lambda^{-1} I  $ 避免 $  A^\top A  $ 奇异。
- **数值误差控制**：定期重置 $  P_k  $ 或使用Cholesky分解更新。




hw：

- 复习最小二乘

- 整理+推导iterative least squares

  要求使用sherman-Morrison-Woodbarry

- BFGS，DFP公式证明（SMW推导）



## Lec 3 Review

这节主要介绍以下几个方面

- Go over a little bit of the last lecture
- Focus on classical numerical optimizations
- References
  - MlT,theprincipleofoptimal control,thefirsttwolectures
  - Dive into deep learning, prepare for the later advanced optimizations in deeplearning
  - Lecturesforthe classtoday-Ryan的课程讲义，pdf文件另发



### review

- 对多元情况先做一元，二元找规律

****

设 $  A  \in R^{n\times n}为对称矩阵,b\in R^Pn,c\in R$ ，求：

1. 求线性函数 $  f(\mathbf{x}) = \mathbf{b}^\top \mathbf{x}  $ 的梯度和Hessian矩阵。

2. 给定二次函数：
   $$
   f(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x} + \mathbf{b}^\top \mathbf{x} + c
   $$

   求其梯度和Hessian矩阵。

**Problem1** Gradient Hessian
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

**Problem2**

1. **展开函数**：

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

**Problem2** Jacobi

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

**代入点 $\mathbf{x} = (1, 0, \pi)$**：
$$
\mathbf{J_F}(1, 0, \pi) = \begin{pmatrix}
3 & \pi & 1 \\
3 & 0 & 0
\end{pmatrix}
$$

### Mathematical Fundamentals

- “形式”很重要：先有鸡还是先有蛋？先接受这个形式再最优化求解

**方向导数定义**：方向导数就是函数值在某个“**方向**”上的变化率。

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

----



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
1. **定义辅助函数**：

$$
   \phi(a) = f(\mathbf{x} + a\mathbf{d})
$$

   其中 $\phi(a)$ 是关于标量 $a$ 的一元函数。

2. **对 $\phi(a)$ 求导**：

$$
\phi'(a) = \frac{d}{da} f(\mathbf{x} + a\mathbf{d}) = \mathbf{d}^\top \nabla f(\mathbf{x} + a\mathbf{d})
$$
3. **计算 $a=0$ 处的导数**：

$$
   \phi'(0) = \mathbf{d}^\top \nabla f(\mathbf{x})
$$
4. **方向导数表达式推导**：

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

定理二：f(x)在x沿方向d的二阶方向导数为：
$$
\frac{\partial}{\partial d}f(\bar x)=\lim_{a \to 0^+} \frac{\frac{\partial}{\partial d}f(\mathbf{\bar x} + ad) - \frac{\partial}{\partial d}f(\mathbf{\bar x})}{a||d||}=\frac{1}{||d||^2}d^T \nabla^2f(\bar x)d
$$

----

**鞍点与焦点**

- 焦点（focus）：在动力系统的相平面分析中，**焦点**是一类**平衡点**（即系统状态变化率为零的点），其周围轨迹呈现螺旋状收敛或发散的特征。焦点分为**稳定焦点**和**不稳定焦点**两种类型
- 鞍点（saddle point）



## Lec 4 NN and  Gradient descent

这节主要介绍以下几个方面：

- 递归最小而成方法
- 拟牛顿法
- 梯度算法的变体

> 高斯白噪声频域是全频段都有

### 神经网络的基本原理

凸优化问题有助于分析算法的特点。 毕竟对大多数非凸问题来说，获得有意义的理论保证很难，但是直觉和洞察往往会延续。所以我们研究一个优化算法，常常将其运用于如下问题：
$$
\min f(x)=\frac{1}{2}x^TQx+c^Tx+b
$$

对于损失函数，两种方法：

- 梯度下降
- 反向传播







### 梯度下降方法的变体

在统计学和机器学习中，许多问题的核心可归结为以下优化问题：
$$
\min_{x} f(x) \equiv \frac{1}{m} \sum_{i=1}^{m} f_i(x)
$$
$m$表示样本量（即子函数个数）

计算目标函数的全梯度在m非常大时可能需要巨大的计算量，甚至在m为无穷时无法实现。**因此，常用随机子函数的梯度来估计目标函数的全梯度，这类方法称为随机方法**。在实际应用中，这类方法通常比确定性的梯度下降方法更快。

- 随机梯度下降SGD（stochastic gradient descent）

  - 核心思想：每次迭代中随机选择一个样本i，然后利用该样本的梯度来更新参数

    为什么SGD能有效果？我们强调随机梯度$\nabla f_i(x)$是对完整梯度$\nabla f(x)$的无偏估计：
    $$
    E_i\nabla f_i(x)=\frac{1}{n}\sum \nabla f_i(x) = \nabla f(x)
    $$

  - 算法步骤
    
    1. 在每次迭代中随机选择一个样本 $i \sim \text{Uniform}(1,m)$
    
    2. 计算单个样本的梯度$\nabla f_i(x_t)$
    
    3. 参数更新$x_{t+1} = x_t - \eta \nabla f_i(x_t)$，其中 $\eta$ 为学习率 (learning rate)

- 特性分析

  ✅ **优点**
  - 单次迭代计算量低，适合大规模数据集
  - 能逃离局部极小点 (得益于随机性)

  ⛔ **缺点**
  - 收敛路径存在明显震荡
  - 需要仔细调节学习率 $\eta$
  - 不保证严格单调收敛

- 小批量梯度下降Mini-batch Gradient Descent

  - 核心思想：每次迭代中随机选择一个小批量样本，然后利用这些样本的梯度平均值来更新参数——**计算效率**

  - 特性分析：小批量梯度下降在SGD和批量梯度下降之间取得平衡，既减少了计算量，又降低了更新的波动性。

    好的副作用：使用平均梯度减小了方差


- 动量算法：

  - 核心思想：在迭代过程中引入动量项，利用历史梯度信息来加速收敛并减少震荡
    $$
    v_t = \beta v_{t-1}+g_{t,t-1},~\beta \in(0,1)
    $$
    

- Nesterov加速梯度方法（Nesterov Accelerated Gradient, NAG）

  - 核心思想：改进的动量优化算法，其核心在于**前瞻性梯度计算**。与传统动量法（Momentum）不同，NAG 在更新参数时先沿当前动量方向进行一步预测，然后在预测点计算梯度，从而更准确地调整参数方向

  - 算法步骤：

    1. 生成辅助变量$v_k$：$v_k=x_k+\gamma(x_k-x_{k-1})$

       其中，$\gamma$是动量参数，常取0.9

    2. 梯度下降：
       $$
       x_{k+1}=v_k-\alpha_k \nabla f(v_k)
       $$
       其中，$\alpha_k$是步长

  - 优缺点：

    | **优点**                                                                                                                      | **缺点**                                                 |
    | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
    | ✅ **更快的收敛速度**：通过前瞻性梯度修正方向，减少震荡，加速收敛至局部最优（理论收敛速率 O(1/t2)，优于标准梯度下降的 O(1/t)） | ⚠️ **超参数敏感**：需仔细调节学习率 η 和动量系数 γ        |
    | ✅ **高曲率适应性强**：在病态条件或高曲率区域表现优于传统动量法                                                                | ⚠️ **非凸问题局限**：对非凸目标函数可能陷入鞍点或局部极小 |
    | ✅ **减少震荡**：前瞻梯度计算使更新方向更贴近实际下降路径                                                                      | ⚠️ **计算复杂度略高**：需额外计算预测点梯度               |

    特别在处理大规模数据集和高维参数空间时表现出色。

- Adagrad（Adaptive Gradient）是一种随机梯度下降的方法[11.7. AdaGrad算法](https://zh-v2.d2l.ai/chapter_optimization/adagrad.html)

  - 引言：稀疏特征，鉴于学习率下降，我们可能最终会面临这样的情况：常见特征的参数相当迅速地收敛到最佳值，而对于不常见的特征，我们仍缺乏足够的观测以确定其最佳值。 换句话说，学习率要么对于常见特征而言降低太慢，要么对于不常见特征而言降低太快。
    
    解决此问题的一个方法就是记录我们看到特定特征的次数，然后将其用作调整学习率：$\eta_i = \frac{\eta_0}{\sqrt{s(i,t)+c}},~s(i,t)$这里计下了我们截至$t$观察到功能$i$的次数

  - 核心思想：为每个参数动态调整学习率，是一种**自适应学习率优化算法**
    
    通过将上面粗略的计数器$s(i,t)$替换为先前观察所得梯度的平方之和来解决这个问题
    $$
    s(i,t+1)=s(i,t)+(\partial_if(x))^2
    $$
    
    - **高频参数**（梯度大且频繁更新的参数）降低学习率，避免震荡；
    - **低频参数**（梯度小或稀疏的参数）增大学习率，加速收敛。 通过累积历史梯度平方和，Adagrad 自动适应不同参数的特征，特别适合稀疏数据（如自然语言处理任务）
    
    AdaGrad算法会在单个坐标层面动态降低学习率
    
  - 算法步骤：

    - 初始化：

      - 全局学习率$\eta$
      - 梯度平方累计变量$r=0$
      - 小常数$\epsilon$

    - 迭代更新

      - 计算当前梯度
  
      - 累积梯度平方和：$r_t=r_{t-1}+g_t\odot g_t$
  
        > $\odot$哈达玛积：两个同维矩阵的元素对应相乘
  
      - 调整学习率；$\eta_t = \frac{\eta}{\sqrt{r_t+\epsilon}}$
  
      - 更新参数
  
  - 优缺点
  
    | **优点**                                                                  | **缺点**                                                                             |
    | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
    | ✅ **自适应学习率**：无需手动调节，适合高维稀疏数据（如 NLP 中的词向量）。 | ⚠️ **学习率衰减过快**：累积梯度平方和持续增大，导致后期学习率趋近于零，模型停止更新。 |
    | ✅ **梯度方向优化**：对低频参数更敏感，加速稀疏特征的学习。                | ⚠️ **内存消耗高**：需为每个参数存储梯度平方累积量，参数量大时内存占用显著。           |
    | ✅ **数值稳定性**：引入 ϵ 避免分母为零，适合非凸优化问题。                 | ⚠️ **依赖初始学习率**：初始 η 设置不当可能导致早期收敛慢或震荡。                      |



- RMSprop（Root Mean Square Propagation）

   **Adagrad 算法的改进版本**，旨在解决 Adagrad 因累积全部历史梯度导致学习率过早衰减的问题，Adagrad:$s(i,t+1)=s(i,t)+(\partial_if(x))^2$由于缺乏规范化，没有约束力，$s_t$持续增长，几乎上是在算法收敛时呈线性递增。

  - 核心思想：$s(i,t+1)=\gamma s(i,t)+(1-\gamma)(\partial_if(x))^2$
    - **指数加权平均**：通过衰减系数（如 0.9）动态调整历史梯度的影响，仅保留近期梯度信息，避免长期累积导致学习率过小。
    - **自适应学习率**：每个参数的学习率根据其梯度幅度的均方根（RMS）动态调整，梯度大的参数降低学习率，梯度小的参数增大学习率

- Adadelta是AdaGrad的另一种变体

  - 核心思想：减少了学习率适应坐标的数量。 此外，广义上Adadelta被称为没有学习率，因为它使用变化量本身作为未来变化的校准

  - 算法步骤：

    1. 泄漏梯度平方更新
  
    维护历史梯度平方的指数加权平均（ρ为衰减因子）：
    $$s_t = \rho s_{t-1} + (1 - \rho) g_t'^2 \tag{11.9.1}$$
  
    2. 参数更新规则
  
    使用调整后的梯度进行参数更新：
    $$x_t = x_{t-1} - g_t' \tag{11.9.2}$$
  
    3. 梯度调整计算
  
    通过历史参数更新量缩放原始梯度（ϵ为数值稳定项）：
    $$g_t' = \frac{\sqrt{\Delta x_{t-1} + \epsilon}}{\sqrt{s_t + \epsilon}} \odot g_t \tag{11.9.3}$$

    4. 参数更新量更新

    维护参数更新量的指数加权平均：
    $$\Delta x_t = \rho \Delta x_{t-1} + (1 - \rho) g_t'^2 \tag{11.9.4}$$
  
- Adam（Adaptive Moment Estimation）

  一种自适应学习率优化算法，**融合了动量法（Momentum）和RMSProp的优点**。将很多优化技术汇总到了这一个方法里面，但是也有一些问题yogi改进

  - 核心思想：

    - **一阶矩估计（动量）**：跟踪梯度的指数加权平均，保持参数更新方向的稳定性。
    - **二阶矩估计（自适应学习率）**：计算梯度平方的指数加权平均，根据梯度幅值调整学习率。
    - **偏差校正**：修正初始阶段因零初始化导致的矩估计偏差，确保更新量的准确性。

  - 算法步骤：

    - l初始化：

      - 初始参数：$ \theta_0 $
      - 学习率：$ \alpha = 0.001 $（默认）
      - 一阶矩衰减率：$ \beta_1 = 0.9 $
      - 二阶矩衰减率：$ \beta_2 = 0.999 $
      - 数值稳定常数：$ \epsilon = 10^{-8} $
      - 初始一阶矩 $ m_0 = 0 $，二阶矩 $ v_0 = 0 $

    - 迭代过程（时间步 $ t=1,2,\dots $）：

      1. **计算当前梯度**：

      $$
         g_t = \nabla_\theta L(\theta_{t-1})
       
      $$
      
    2. **更新一阶矩**（动量）：
    
    $$
         m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
       
    $$
    
      3. **更新二阶矩**（自适应学习率）：

    $$
       v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
       
    $$
    
      4. **偏差校正**（消除初始零偏置）：
    
    $$
         \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
     
    $$
    
      5. **更新参数**：
    
    $$
    \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t
    $$

  - 优缺点

    | **优势**           | **说明**                                                             |
    | ------------------ | -------------------------------------------------------------------- |
    | ✅ **自适应学习率** | 每个参数独立调整学习率，梯度大的方向步长减小，梯度小的方向步长增大。 |
    | ✅ **高效收敛**     | 结合动量与自适应机制，在非凸优化问题中收敛速度通常快于SGD、RMSProp。 |
    | ✅ **处理稀疏梯度** | 对低频参数（如NLP中的词向量）自动增大更新步长，提升训练效率。        |
    | ✅ **抗噪声能力强** | 通过指数加权平均平滑梯度噪声，适应高噪声数据场景。                   |

    | **局限性**             | **说明**                                                         |
    | ---------------------- | ---------------------------------------------------------------- |
    | ⚠️ **超参数敏感**       | β1,β2,α 需调优，不当设置可能导致收敛不稳定。                     |
    | ⚠️ **内存占用较高**     | 需存储一阶和二阶矩变量，参数量较大时内存消耗显著。               |
    | ⚠️ **局部最优风险**     | 在部分非凸问题中可能陷入鞍点，需结合预热（Warmup）或学习率衰减。 |
    | ⚠️ **长期训练性能下降** | 因自适应学习率随训练衰减，后期可能不如带动量的SGD                |



### 控制论视角下的神经网络

Nature Communication上的PIDAO论文：

gradient-based optimizations can be interpreted as continuous-time dynamical systems  将优化过程建模为**连续时间动力系统**，并引入**PID控制器**设计新的优化器，通过反馈控制机制改进优化动态

### hw

作业：机器学习中的优化算法

- 学习神经网络的基本原理，主要从无约束优化的角度，了解反向传播的求解方法，主要参考资料：[An Introduction to Optimization: With Applications to Machine Learning - Edwin K. P. Chong, Wu-Sheng Lu, Stanislaw H. Zak - Google Books](https://books.google.co.jp/books?hl=en&lr=&id=uEDUEAAAQBAJ&oi=fnd&pg=PR15&dq=info:WSaWZthIywYJ:scholar.google.com&ots=qAtUVnxFtM&sig=Sl77RBaLcYWOmv7FMLzQZg-qcGg&redir_esc=y#v=onepage&q&f=false)——这本书的ch13 Unconstrained Optimization and Neural Networks  
- 学习机器学习中重要的梯度类算法
  - SGD、Nestonov等
  - Adagrad优化算法
  - RMSprop和Adam优化算法
- 针对以上的学习内容：撰写一篇小论文，阐述机器学习中的优化问题与加速算法
- 附加研究兴趣题目：深度学习优化中的梯度流、信息几何、动力系统等方法，参考Nature Communication上的PIDAO论文




# References

- https://zhuanlan.zhihu.com/p/629131647
- [2023-2024春夏许超老师最优化与最优控制课程分享 - CC98论坛](https://www.cc98.org/topic/5923253)

- [Tutorial — Ceres Solver (ceres-solver.org)](http://ceres-solver.org/tutorial.html)