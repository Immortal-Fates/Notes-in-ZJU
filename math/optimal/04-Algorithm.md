# Optimal Algorithm

优化算法



## Overview

最优化算法的解决思路遵循一个自然的等级制度：

1. **基础层：二次规划（Quadratic Optimization）**，可以通过求解一组线性方程组（KKT系统）直接得到精确解,。
2. **中间层：牛顿法（Newton's Method）**，通过将无约束或等式约束问题转化为一系列二次规划问题来求解。
3. **最高层：内点法（Interior-point Methods）**，通过求解一系列无约束或等式约束问题来处理带有不等式约束的复杂问题

## Unconstrained minimization

### Unconstrained minimization problems

目标形式
$$
\min_{x\in\mathbb{R}^n} f(x)
$$
常见假设是 $f$ 可微或二次可微。

- 最优性条件
  
  1. 一阶必要条件：若 $x^*$ 为局部极小点且 $f$ 可微，则
     $$
     \nabla f(x^*)=0
     $$
  
  2. 二阶必要条件：若 $f$ 二次可微，则
     $$
     \nabla^2 f(x^*)\succeq 0
     $$
  
  3. 二阶充分条件：若
     $$
     \nabla f(x^*)=0,\quad \nabla^2 f(x^*)\succ 0
     $$
     则 $x^*$ 为严格局部极小点
  
  4. 若 $f$ 为凸函数，满足 $\nabla f(x^*)=0$ 即为全局最优
  
- 迭代算法 
  大多数问题无法通过解析方法求解，因此需要采用迭代算法生成极小化序列
  $$
  x^{(0)}, x^{(1)}, \dots
  $$
  使得
  $$
  f\bigl(x^{(k)}\bigr) \to p^*
  $$

- 初始点与水平集 
  算法需要给定初始点 $x^{(0)}$，并要求其初始水平集
  $$
  S = \{ x \in \mathrm{dom}\,f \mid f(x) \le f(x^{(0)}) \}
  $$
  是闭集。

- 强凸性  
  通常假设 $f$ 是强凸的，即存在 $m>0$ 使得
  $$
  \nabla^2 f(x) \succeq m I
  $$
  该条件保证最优解唯一，并给出次优性界
  $$
  f(x) - p^* \le \frac{1}{2m} \lVert \nabla f(x) \rVert_2^2
  $$

### Descent methods

下降方法遵循如下更新规则
$$
x^{(k+1)} = x^{(k)} + t^{(k)} \Delta x^{(k)}
$$
其中 $t^{(k)} > 0$ 为步长，$\Delta x^{(k)}$ 为搜索方向。

- **下降方向**  
  搜索方向需满足
  $$
  \nabla f(x)^{\mathsf T} \Delta x < 0
  $$
  即与负梯度方向成锐角。

- **线搜索**
  - **精确线搜索**  
    沿给定方向寻找使目标函数最小的步长 $t$。
    
  - **回溯线搜索**  
    一种非精确但高效的方法，选取参数 $\alpha \in (0,0.5)$、$\beta \in (0,1)$，不断缩小步长直到满足下降条件。
    
    下面介绍一下线搜素的两个步长准则，来保证每一步既下降又“不过分保守”
    
    - Armijo 条件
    
      - 形式：取 $0<c_1<1$，寻找满足
        $$
        f(x^{(k)}+t p^{(k)})\le f(x^{(k)})+c_1 t \nabla f(x^{(k)})^\top p^{(k)}
        $$
        的 $t$，通常从 $t=1$ 开始按 $t\leftarrow \beta t$ 缩小
    
      - 作用：保证函数值至少达到一阶泰勒下降的某个比例，防止步长过大导致不下降
    
    - Wolfe 条件
    
      - 形式：Armijo 条件 + 曲率条件
    
        取 $0<c_1<c_2<1$，在 Armijo 的基础上再要求
        $$
        \nabla f(x^{(k)}+t p^{(k)})^\top p^{(k)}
        \ge
        c_2\,\nabla f(x^{(k)})^\top p^{(k)}
        $$
    
      - 作用：防止步长过小，确保沿方向走到“接近一维最优”而不是停在很近的起点

### Gradient descent method

迭代方向
$$
p^{(k)} = - \nabla f(x^{(k)})
$$

基本性质

1. 若 $\nabla f$ 为 $L$-Lipschitz，取固定步长 $t\in(0,2/L)$ 可保证下降
2. 若 $f$ 为 $m$-强凸且 $L$-光滑，取 $t=1/L$ 时有线性收敛
   $$
   f(x^{(k)})-f^*\le \left(1-\frac{m}{L}\right)^k\big(f(x^{(0)})-f^*\big)
   $$

常见问题与改进

1. 条件数大时出现之字形迭代，收敛慢
2. 可通过变量尺度变换、预条件、拟牛顿法或 LM 修正改善

### Steepest descent method

基本思想是在给定范数约束下寻找下降最快的方向。

1. 规范化方向
   $$
   p_{nsd}=\arg\min_{\|v\|\le 1}\ \nabla f(x)^\top v
   $$
   等价解为
   $$
   p_{nsd}=-\frac{\nabla f(x)}{\|\nabla f(x)\|_*}
   $$
   其中 $\|\cdot\|_*$ 为对偶范数
2. 非规范化方向
   $$
   p_{sd}=-\nabla f(x)
   $$

常见范数

1. 在 $l_2$ 范数下，最速下降方向与梯度方向一致
2. 在 $P$-二次范数 $\|v\|_P^2=v^\top P v$ 下
   $$
   p_{sd}=-P^{-1}\nabla f(x)
   $$
   对应变量尺度变换或预条件

### Newton’s method

核心思想是用二阶模型近似目标函数。

二阶模型
$$
m_k(p)=f(x^{(k)})+\nabla f(x^{(k)})^\top p+\frac12 p^\top \nabla^2 f(x^{(k)}) p
$$

牛顿方向
$$
p_N^{(k)}=-\left[\nabla^2 f(x^{(k)})\right]^{-1}\nabla f(x^{(k)})
$$

性质

1. 对严格凸二次函数具有一次终止性

2. 在最优点附近若 Hessian 正定，局部二次收敛

3. 计算代价高，需要求解线性方程组

4. 若 Hessian 不正定，方向可能不是下降方向

- **牛顿减量**  
  定义为
  $$
  \lambda(x) = \bigl( \nabla f(x)^{\mathsf T} \nabla^2 f(x)^{-1} \nabla f(x) \bigr)^{1/2}
  $$
  是衡量次优程度的重要量，常用作停止准则。

- **仿射不变性**  
  牛顿法对线性坐标变换不敏感，这是其相对于梯度法的重要优势。

- **收敛阶段**  
  算法通常经历阻尼牛顿阶段，呈线性收敛；当接近最优解时进入纯牛顿阶段，表现为二次收敛。

全局化与修正

1. 阻尼牛顿  
   用线搜索控制步长
   $$
   x_{k+1}=x_k+\alpha_k p_N^{(k)},\quad \alpha_k\in(0,1]
   $$
   常用 Armijo 或 Wolfe 条件保证下降
2. Levenberg–Marquardt 修正  
   通过加正则保证正定
   $$
   p_{LM}^{(k)}=-\left[\nabla^2 f(x^{(k)})+\mu_k I\right]^{-1}\nabla f(x^{(k)})
   $$
   $\mu_k\to 0$ 时趋近牛顿法，$\mu_k\to\infty$ 时趋近梯度下降法
3. 拟牛顿法  
   用近似 Hessian 或其逆替代真实 Hessian
   $$
   B_{k+1}s_k=y_k,\quad s_k=x_{k+1}-x_k,\ y_k=g_{k+1}-g_k
   $$
   常见方法包括 DFP、BFGS、L-BFGS。若满足 $s_k^\top y_k>0$，BFGS 近似可保持正定，并常配合 Wolfe 条件使用

### Self-concordance

这是 Nesterov 和 Nemirovski 提出的新分析框架，旨在摆脱对未知常数的依赖

自共轭性是对三阶导数的控制条件，用来分析牛顿法的全局收敛性。

定义形式
$$
|D^3 f(x)[h,h,h]|\le 2\,(h^\top \nabla^2 f(x) h)^{3/2}
$$

要点

1. 对数障碍函数是典型的自共轭函数
2. 在自共轭假设下，可用阻尼牛顿获得全局收敛，并在进入邻域后使用全步长
3. 该框架是内点法复杂度分析的基础
## Equality constrained minimization

- **消除法 (Elimination)**：通过对约束条件进行参数化，将问题简化为等价的无约束优化问题。
  设约束为 $Ax=b$ 且 $A\in\mathbb{R}^{m\times n}$ 满行秩。取可行基点 $x_0$ 满足 $Ax_0=b$，再取 $N$ 为 $A$ 的零空间基，令
  $$
  x = x_0 + Nz
  $$
  则原问题化为无约束最小化
  $$
  \min_z\ f(x_0+Nz)
  $$
  其梯度与 Hessian 分别为
  $$
  \nabla_z f = N^\top \nabla f(x_0+Nz),\quad \nabla^2_{zz} f = N^\top \nabla^2 f(x_0+Nz) N
  $$
  优点是规模降维、约束自动满足；缺点是需要构造零空间基，数值稳定性依赖于 $N$ 的质量。
- **等式约束牛顿法**：在保持迭代点**可行**（满足 $Ax=b$）的前提下，通过求解 KKT 系统来确定牛顿步
  设当前点可行，二阶近似子问题为
  $$
  \min_p\ \nabla f(x)^\top p + \frac12 p^\top H p
  \quad \text{s.t. } A(x+p)=b
  $$
  即 $Ap=0$，KKT 条件给出线性方程组
  $$
  \begin{bmatrix}
  H & A^\top\\
  A & 0
  \end{bmatrix}
  \begin{bmatrix}
  p\\
  \lambda
  \end{bmatrix}
  =
  \begin{bmatrix}
  -\nabla f(x)\\
  0
  \end{bmatrix}
  $$
  其中 $H=\nabla^2 f(x)$。若 $H$ 在约束切空间上正定，则该步是下降方向，可配合可行线搜索保持可行性。
- **不可行起点牛顿法 (Infeasible Start Newton Method)**：允许初始点不满足 $Ax=b$，在迭代过程中同时追求函数值下降和残差（Residual）减小，直至达到可行和最优
  设残差 $r_p = Ax-b$，KKT 残差 $r_d = \nabla f(x)+A^\top \lambda$。牛顿方向通过线性化
  $$
  \begin{bmatrix}
  H & A^\top\\
  A & 0
  \end{bmatrix}
  \begin{bmatrix}
  \Delta x\\
  \Delta \lambda
  \end{bmatrix}
  =
  -
  \begin{bmatrix}
  r_d\\
  r_p
  \end{bmatrix}
  $$
  典型更新为 $x^+ = x + \alpha \Delta x,\ \lambda^+ = \lambda + \alpha \Delta \lambda$，其中 $\alpha$ 通过线搜索确保目标下降且残差缩小。该方法不要求初始可行点，更适合结合内点法的外层求解。

## Interior-point methods

用于求解带有不等式约束的凸优化问题。

- **对数障碍函数 (Logarithmic Barrier)**：引入 $\phi(x) = -\sum \log(-f_i(x))$ 来近似表示不等式约束。
  适用于标准形式 $f_i(x)\le 0$。当 $x$ 接近边界时 $\phi(x)\to +\infty$，从而阻止越界。障碍参数 $t$ 控制近似精度，$t$ 越大，解越接近原问题。

- **中心路径 (Central Path)**：对于参数 *t*>0，定义 $x^∗(t)$ 为最小化 $tf_0+\phi$ 的点。当 $t→\infty$ 时，$x(t)$ 收敛于原问题的最优解 $x$。
  一般写作
  $$
  x^*(t)=\arg\min_x\ tf_0(x)+\phi(x)
  $$
  并满足一阶条件
  $$
  t\nabla f_0(x)+\sum_i \frac{1}{-f_i(x)}\nabla f_i(x)=0
  $$
  中心路径提供从“可行内点”到最优解的连续轨迹，外层通过增大 $t$ 跟踪该路径。

- **障碍方法 (Barrier Method)**：
  1. **外迭代**：增加参数 $t$ 的值（通常乘以倍数 $μ$）。
  2. **内迭代（中心化步骤）**：使用牛顿法求解当前的无约束优化问题,。
  内层牛顿方向通常由 Hessian 与梯度构成，若用回溯线搜索，需保证 $f_i(x)<0$。常见停止准则为
  $$
  \frac{m}{t}\le \varepsilon
  $$
  其中 $m$ 为不等式个数，$\varepsilon$ 为精度要求。

- **可行性与阶段 I (Phase I)**：如果找不到初始的可行点，需要先通过一个专门的优化问题来寻找可行起点或证明问题不可行,。
  典型构造为
  $$
  \min_{x,s}\ s \quad \text{s.t. } f_i(x)\le s,\ i=1,\dots,m
  $$
  若最优值 $s^*\le 0$ 则得到可行点，否则原问题不可行。Phase I 可与障碍方法共享同一套求解器。

- **原始-对偶内点法 (Primal-Dual Methods)**：比障碍方法更高效的变体，**不再区分内、外迭代**，它同时更新原始变量和对偶变量，通常具有超线性的收敛速度
  以不等式约束 $f_i(x)\le 0$ 的 KKT 为例，引入拉格朗日乘子 $\lambda_i\ge 0$，互补条件为 $\lambda_i f_i(x)=0$。对偶残差与互补残差可写为
  $$
  r_d = \nabla f_0(x)+\sum_i \lambda_i \nabla f_i(x),\quad
  r_c = -\lambda\circ f(x)
  $$
  通过牛顿法求解原始-对偶系统，并用中心化参数推动 $\lambda_i f_i(x)\approx \mu$。该方法往往迭代次数少，对高精度更有优势。

## 收敛性分析	

### 预备知识

#### 收敛性与收敛速度

- 复杂度视角
  - oracle complexity：达到精度 ε 所需的 oracle 调用次数
  - arithmetical complexity：达到精度 ε 的总运算量
  - ε-次优：f(x^{(k)})-f(x^*)\le \varepsilon

- 线性收敛
  $$
  f(x^{(k)})-f(x^*)\le c^k\big(f(x^{(0)})-f(x^*)\big),\quad 0<c<1
  $$
  迭代复杂度
  $$
  O(\log(1/\varepsilon))
  $$

- 二次收敛
  $$
  \|x^{(k+1)}-x^*\|\le C\|x^{(k)}-x^*\|^2
  $$
  迭代复杂度
  $$
  O(\log\log(1/\varepsilon))
  $$

- 次线性收敛
  $$
  f(x^{(k)})-f(x^*)\le \frac{K}{k^\alpha}
  $$
  迭代复杂度
  $$
  O\big((1/\varepsilon)^{1/\alpha}\big)
  $$

- oracle 模型是对算法访问信息的抽象，通常只允许查询 f(x) 和 \nabla f(x)

#### 强凸性

定义：若存在常数 m>0，使得对任意 x\in S
$$
\nabla^2 f(x)\succeq m I
$$
则 f 在 S 上强凸。

基本性质

1. 强凸性推出严格凸性
2. 强凸性依赖于定义区域 S
3. 严格凸不必然强凸

二次下界

若 f 强凸，则对任意 x,y\in S
$$
f(y)\ge f(x)+\nabla f(x)^\top (y-x)+\frac{m}{2}\|y-x\|^2
$$

强凸与光滑的二次上下界

若满足
$$
m I\preceq \nabla^2 f(x)\preceq M I,\quad \forall x\in S
$$
则
$$
f(y)\ge f(x)+\nabla f(x)^\top (y-x)+\frac{m}{2}\|y-x\|^2
$$
$$
f(y)\le f(x)+\nabla f(x)^\top (y-x)+\frac{M}{2}\|y-x\|^2
$$

m 与 M 的来源

1. m=\inf_{x\in S} \lambda_{min}(\nabla^2 f(x))，M=\sup_{x\in S} \lambda_{max}(\nabla^2 f(x))
2. 二次函数 f(x)=\frac12 x^T Q x+b^T x 时，\nabla^2 f=Q，直接取特征值上下界

#### 梯度 Lipschitz 连续

定义：若存在 L>0，使得对任意 x,y\in S
$$
\|\nabla f(y)-\nabla f(x)\|\le L\|y-x\|
$$

等价充分条件

若 f 二次可微且
$$
\|\nabla^2 f(x)\|\le L,\quad \forall x\in S
$$
则 \nabla f Lipschitz 连续。

二次上界

若 \nabla f Lipschitz 连续，则
$$
f(y)\le f(x)+\nabla f(x)^\top (y-x)+\frac{L}{2}\|y-x\|^2
$$

最优值间隙下界

若 \nabla f 为 L-Lipschitz 且全局极小点 x^* 存在，则
$$
f(x)-f(x^*)\ge \frac{1}{2L}\|\nabla f(x)\|^2
$$

#### 寻优间隙的上下界

在强凸且梯度 Lipschitz 连续条件下

1. 解误差与梯度关系
   $$
   \frac{1}{M}\|\nabla f(x)\|\le \|x-x^*\|\le \frac{1}{m}\|\nabla f(x)\|
   $$
2. 目标值间隙界
   $$
   \frac{1}{2M}\|\nabla f(x)\|^2\le f(x)-f(x^*)\le \frac{1}{2m}\|\nabla f(x)\|^2
   $$

### 下降法

强凸条件下

1. 精确线搜索与回溯线搜索都具有线性收敛
2. 收敛速度与条件数有关
   $$
   \kappa=\frac{M}{m}
   $$

精确线搜索下的线性收敛因子

设 m I\preceq \nabla^2 f(x)\preceq M I，则
$$
f(x^+)-f(x^*)\le c\big(f(x)-f(x^*)\big),\quad c=1-\frac{m}{M}
$$
因此迭代复杂度为 O(\kappa\log(1/\varepsilon))。

回溯线搜索下界

Armijo 条件
$$
f(x+td)\le f(x)+\alpha t\nabla f(x)^\top d,\quad \alpha\in(0,0.5]
$$
回溯规则 t\leftarrow \beta t，\beta\in(0,1)。若 d=-\nabla f(x)，可得终止步长下界
$$
t_f\ge \min\left(1,\frac{\beta}{M}\right)
$$
并有线性收敛
$$
f(x^+)-f(x^*)\le \big(1-2m\alpha t_f\big)\big(f(x)-f(x^*)\big)
$$

弱条件下的结论

1. 非凸 + Lipschitz 连续：收敛到驻点，复杂度 O(1/\varepsilon)
2. 凸 + Lipschitz 连续：次优值收敛，复杂度 O(1/\varepsilon)
3. 加速方法可提升到 O(1/\sqrt{\varepsilon})

### 牛顿法

#### 经典分析法

纯牛顿法全局收敛性差，需配合阻尼或线搜索。

阻尼牛顿法特征

1. 初期阻尼阶段通常线性收敛
2. 接近最优点后步长趋近 1，恢复二次收敛

经典牛顿法的局部二次收敛

若 f 强凸且 Hessian Lipschitz 连续
$$
\|\nabla^2 f(y)-\nabla^2 f(x)\|_2\le L_H\|y-x\|_2
$$
则存在二次收敛区域。当
$$
\frac{L_H}{2m}\|x^{(0)}-x^*\|_2<1
$$
时，迭代满足二次收敛。

牛顿减量

定义
$$
\lambda(x)=\sqrt{\nabla f(x)^T\nabla^2 f(x)^{-1}\nabla f(x)}
$$
性质

1. \lambda(x)^2=-\nabla f(x)^T\Delta x_{nt}
2. m\|\Delta x_{nt}\|_2^2\le \lambda(x)^2\le M\|\Delta x_{nt}\|_2^2
3. 可用于停止准则与判断是否进入二次收敛区

#### 自和谐函数法

自和谐函数框架下，可用阻尼牛顿获得全局收敛，并在邻域内用全步长保证快速收敛。
## Review

1. 梯度下降法的迭代方向是什么？为什么会出现“之”字形？

  - 方向：
    $$
    p^{(k)}=-\nabla f(x^{(k)})
    $$
  - 原因：等高线狭长时，梯度方向在两侧来回摆动，导致“之”字形收敛。

2. 最速下降法在 $$P$$-二次范数下的最速下降方向是什么？

  - 设 $$\|x\|_P^2=x^T P x$$，则最速下降方向为
    $$
    p=-P^{-1}\nabla f(x)
    $$

3. 最优步长（精确线搜索）的解析表达式是什么？

  - 在方向 $$p^{(k)}$$ 上：
    $$
    t_k^*=-\frac{\nabla f(x^{(k)})^T p^{(k)}}{p^{(k)T}\nabla^2 f(x^{(k)})p^{(k)}}
    $$

4. Armijo 条件是什么？回溯直线搜索如何做？

  - Armijo：
    $$
    f(x^{(k)}+t_k p^{(k)})\le f(x^{(k)})+\alpha t_k \nabla f(x^{(k)})^T p^{(k)}
    $$
  - 回溯：从初始步长开始，若不满足则按 $$t\leftarrow \beta t$$ 逐步缩小。

5. 牛顿法的迭代方向与优缺点分别是什么？

  - 方向：
    $$
    p_N^{(k)}=-\left[\nabla^2 f(x^{(k)})\right]^{-1}\nabla f(x^{(k)})
    $$
  - 优点：局部二次收敛；严格凸二次函数一步到位。
  - 缺点：计算 Hessian 代价高，且 Hessian 非正定时不一定下降。

6. Levenberg–Marquardt 修正的作用与方向是什么？

  - 作用：使 Hessian 变正定，保证下降方向。
  - 方向：
    $$
    p_{LM}^{(k)}=-\left[\nabla^2 f(x^{(k)})+\mu_k I\right]^{-1}\nabla f(x^{(k)})
    $$

7. 梯度下降法的收敛速率（强凸 + 光滑）表达式是什么？

  - 若 $$f$$ 为 $$m$$-强凸且梯度 $$L$$-Lipschitz，取固定步长
    $$
    t\in\left(0,\frac{2}{L}\right)
    $$
    则有线性收敛：
    $$
    f(x^{(k)})-f^*\le \left(1-\frac{m}{L}\right)^k \left(f(x^{(0)})-f^*\right)
    $$

8. 对 $$f(x)=x^T\mathrm{diag}(1,4)\,x$$ 的梯度下降收敛速率怎么写？写出P-二次范数最速下降法的最速下降方向

  - Hessian：
    $$
    \nabla^2 f = 2\cdot \mathrm{diag}(1,4)
    $$
    因此 $$m=2,\ L=8,\ \kappa=L/m=4$$。
    
  - 取固定步长 $$t=1/L=1/8$$ 时：
    $$
    f(x^{(k)})-f^*\le \left(1-\frac{m}{L}\right)^k\!\left(f(x^{(0)})-f^*\right)
    =\left(\frac{3}{4}\right)^k\!\left(f(x^{(0)})-f^*\right)
    $$
  - 若用最优固定步长或精确线搜索，线性收敛因子与条件数有关，典型为
    $$
    \left(\frac{\kappa-1}{\kappa+1}\right)^2=\left(\frac{3}{5}\right)^2
    $$
    给定正定矩阵 $P\succ0$，定义
    $$
    \|d\|_P=\sqrt{d^\top P d}
    $$
    最速下降方向的定义是解下面这个方向选择问题
    $$
    \min_d\ \nabla f(x)^\top d
    \quad \text{s.t. }\|d\|_P\le 1
    $$
    设 $g=\nabla f(x)$。用拉格朗日法可得最优解满足
    $$
    g+2\lambda P d=0
    \Rightarrow d=-\frac{1}{2\lambda}P^{-1}g
    $$
    再用约束 $\|d\|_P=1$ 求出尺度，得到单位化最速方向
    $$
    d^\*=
    -\frac{P^{-1}g}{\sqrt{g^\top P^{-1}g}}
    $$
    因此最速下降方向写成你题目里的 $\Delta x_{sd}$ 是
    $$
    \Delta x_{sd}=
    -\frac{P^{-1}\nabla f(x)}{\sqrt{\nabla f(x)^\top P^{-1}\nabla f(x)}}
    $$

9. 原始-对偶内点法对迭代点的可行性要求是什么？为何仍叫“内点”？

  - 迭代点通常保持在不等式约束严格可行域内（如 $$g_j(x)<0$$）。
  - 不可行内点法允许暂时违反等式或不等式，但保持松弛变量/对偶变量严格正，从而始终在锥的“内部”，因此仍称内点法。

10. Zoutendijk 可行方向法如何判断是否存在下降方向？

  - 解线性规划得最优值 $$\eta$$：
    - 若 $$\eta<0$$，存在可行下降方向；
    - 若 $$\eta=0$$，算法终止。

11. 阻尼牛顿法的思想与“牛顿减量”是什么？

  - 思想：牛顿方向 + 回溯线搜索，保证全局收敛与局部二次收敛。
  - 牛顿减量：
    $$
    \lambda(x)=\sqrt{\nabla f(x)^T \nabla^2 f(x)^{-1}\nabla f(x)}
    $$
    用于衡量距最优点的“牛顿距离”，并作为停止准则。

12. 列举两种可收敛到非凸问题 KKT 点的算法，并给出局部极小性判别方法。

  - 算法示例（任选其二即可）：
    - SQP（序列二次规划）
    - 增广拉格朗日法
    - 罚函数法 / 内点法 / 信赖域-罚函数法
  - 局部极小性判别（典型二阶充分条件）：
    - 满足 KKT 且在临界锥 $$C(x^*,\lambda^*,\nu^*)$$ 上
      $$
      p^T \nabla_{xx}^2 L(x^*,\lambda^*,\nu^*)\,p>0,\quad \forall p\in C\setminus\{0\}
      $$

1 序列二次规划 SQP

每一步在当前点 $x_k$ 处对拉格朗日函数做二阶近似，对约束做一阶线性化，解一个二次规划子问题得到方向，再配合线搜索或信赖域更新。
 在常见光滑性与步长策略条件下，迭代点的极限点满足原问题的 KKT 条件。

2 增广拉格朗日法 ALM

把约束通过乘子和二次罚项并入目标，迭代地

- 近似最小化增广拉格朗日函数
- 更新乘子与罚参数
   在适当条件下，收敛点满足原问题 KKT 条件。对一般非凸约束问题也常用。

