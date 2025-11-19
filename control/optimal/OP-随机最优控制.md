# Main Takeaway

随机最优控制Stochastic Optimal Control

- [ ] 掌握观测器，滤波器
- [ ] 理解随机系统的数学建模
- [ ] 熟练掌握卡尔曼滤波器原理与实现
- [ ] 掌握LQG控制器设计
- [ ] 掌握最优估计理论（MAP, MMSE）

<!--more-->

# 随机系统基础理论

本节主要内容：

- 观测器的基本概念
- 随机变量复习
- 滤波

## 为什么需要随机最优控制？

在实际工程系统中，我们面临以下挑战：

- 状态不完全可观测：无法直接测量所有系统状态
- 测量噪声：传感器存在不可避免的噪声干扰
- 过程噪声：系统动力学存在不确定性和外部扰动
- 模型误差：数学模型与实际系统存在偏差

测量模型
$$
y = g(x) + v
$$
其中：

- $y$ 为测量输出（观测值）
- $x$ 为真实系统状态
- $g(\cdot)$ 为测量函数
- $v$ 为测量噪声

随机最优控制的核心目标：
$$
\min_{\pi_\theta} E[J(x,u)]
$$
确定一个策略 $\pi_\theta$，使得期望代价函数最小。

## 观测器理论

常常state $x$是不知道或者不准确的，于是需要观测器得到较为准确的观测变量

观测器中常有的一种状态估计就是LQE(stand for Linear-Quadratic Estimator)，类似于LQR，它配置估计器极点，使其估计误差的平方和最小。Luenverger observer and Kalman filter就是其中的栗子

### Luenberger oberver

Luenberger观测器（Luenberger Observer）通过结合系统模型和实际测量数据，实现对无法直接测量的系统内部状态的实时估计

下面给出离散系统的Luenberger Observer
$$
\hat{x}_{k+1} = A\hat{x}_{k} + Bu_k + L(y_k - \hat{y}_k) \\
\hat{y}_k = C\hat{x}_k + D u_k
$$
Luenberger Observer with separate predict/update
$$
predict~ step:  \hat{x}_{k+1}^{-} = A\hat{x}_{k}^{-} + Bu_k \\
update~ step: \hat{x}_{k+1}^+ = \hat{x}_{k+1}^-+A^{-1}L(y_k - \hat{y}_k) \\
\quad \hat{y}_k = C\hat{x}_k^{-}
$$
观测器实质上多了一项$L$，使用估计输出和测量输出的偏差来进行状态估计。

下面进行观测器稳定性分析

令$e_k = x_k - \hat{x}_k$，将上述估计公式代入状态空间方程可得
$$
\hat{x}_{k+1} = A\hat{x}_{k} + Bu_k + LCe_k
$$

$$
e_{k+1} = (A-LC)e_k
$$

为了收敛，$A-LC$的特征值必须在单位圆内，其衡量了估计器收敛到真实值有多快，而更慢的估计器会放大测量噪声$v_k$
$$
\hat{x}_{k+1} = A\hat{x}_{k} + Bu_k + L((y_k+v_k) - \hat{y}_k)
$$

### 分离性原理（Separation Principle）

指针对随机控制系统设计控制器时，可将其分解为**状态估计**和**确定性反馈控制**两个独立部分分别设计的理论框架

我们已经得到观测器动态$$e_{k+1} = (A - LC)e_k$$

将误差方程重新排列为 $\hat{x}_k = x_k - e_k$，控制器可以重写为：

$$
u_k = -K(x_k - e_k)
$$
$$x_{k+1} = Ax_k + Bu_k$$
$$x_{k+1} = Ax_k - BK(x_k - e_k)$$
$$x_{k+1} = (A - BK)x_k + BKe_k$$

$$
\begin{bmatrix} x_{k+1} \\ e_{k+1} \end{bmatrix} = \begin{bmatrix} A - BK & BK \\ 0 & A - LC \end{bmatrix} \begin{bmatrix} x_k \\ e_k \end{bmatrix}
$$
由于闭环系统矩阵是三角形的，特征值是 $A - BK$ 和 $A - LC$ 的特征值。因此，反馈控制器和观测器的稳定性是独立的

# 随机变量与概率论基础

概率密度函数（PDF）是随机变量的函数，其在样本空间（可能测量值的范围）中的给定样本（测量值）处的值是该样本发生的概率$p(x)$
$$
E[f(x)] = \int_{-\infty}^{\infty} f(x) p(x)dx
$$
令$x$的均值为$\bar x$，则有
$$
E[x - \bar x] = 0
$$

$$
var(x) = \sigma^2 = E[(x-\bar{x})^2]
$$

联合概率密度函数(Joint probability density functions)$p(x,y)$
$$
E[f(x,y)] = \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} f(x,y) p(x,y)dxdy
$$
Covariance$cov(x,y)$ and Correlation$\rho(x,y)$
$$
cov(x,y) = \Sigma_{xy} = E[(x-\bar{x})(y-\bar{y})]
$$

$$
\rho(x,y) = \frac{\Sigma_{xy}}{\sqrt{\Sigma_{xx}\Sigma_{yy}}},|\rho(x,y)| \le 1
$$

我们假设$x$是一个random vectors，于是Covariance matrix
$$
\Sigma = cov(x,x)
$$

# LQG控制理论

线性二次高斯（LQG）控制是随机最优控制中的一个特殊但重要的情况，它结合了LQR控制和高斯噪声处理。

## 系统模型

状态方程：
$$
\begin{aligned}
x_{n+1} &= Ax_n + Bu_n + w_n, \quad w_n \sim \mathcal{N}(0,W) \\
y_n &= Cx_n + v_n, \quad v_n \sim \mathcal{N}(0,V)
\end{aligned}
$$
其中：

- $w_n$ 为过程噪声（系统不确定性）
- $v_n$ 为测量噪声（传感器噪声）
- $W, V$ 分别为过程噪声和测量噪声的协方差矩阵

代价函数
$$
J = E\left[x_N^TQ_Nx_N + \sum_{n=1}^{N-1} (x_n^TQx_n + u_n^TRu_n)\right]
$$

## 动态规划求解

值函数递推：

从终端条件开始：
$$
V_N(x) = E[x_N^TQ_Nx_N] = E[x_N^TP_Nx_N]
$$
对于 $k = N-1$：
$$
\begin{aligned}
V_{N-1}(x) &= \min_u E[x_{N-1}^TQx_{N-1} + u_{N-1}^TRu_{N-1} \\
&\quad + (Ax_{N-1} + Bu_{N-1} + w_{N-1})^TP_N(Ax_{N-1} + Bu_{N-1} + w_{N-1})]
\end{aligned}
$$
由于 $w_{N-1}$ 与 $(Ax_{N-1} + Bu_{N-1})$ 不相关：
$$
E[(Ax_{N-1} + Bu_{N-1})^TP_Nw_{N-1}] = 0
$$
简化后的优化问题：
$$
\min_u E[x_{N-1}^TQx_{N-1} + u_{N-1}^TRu_{N-1} + (Ax_{N-1} + Bu_{N-1})^TP_N(Ax_{N-1} + Bu_{N-1})]
$$
这与标准LQR问题完全相同！

于是得到最优控制律：
$$
u_{n} = -K_n E[x_n|y_{1:n}] = -K_n \hat{x}_{n|n}
$$
其中：

- $K_n$ 为LQR增益矩阵
- $\hat{x}_{n|n}$ 为基于所有可用测量的状态估计

LQG = LQR + 卡尔曼滤波：

1. 使用卡尔曼滤波器估计状态：$\hat{x}_{n|n}$
2. 将估计状态代入LQR控制律：$u_n = -K_n\hat{x}_{n|n}$

# 最优估计理论

## Objective of Optimization

- **最大后验估计MAP (Maximum a Posteriori Estimation)**:

    $$
    \hat{x} = \arg\max_x p(x|y)
    $$

- **最小均方误差估计MMSE (Minimum Mean-Squared Error)**:

    $$
    \begin{align*}
    \hat{x} &= \arg\min_{\hat{x}} E[(x-\hat{x})^T(x-\hat{x})] \\
    &= \arg\min_{\hat{x}} E[tr((x-\hat{x})^T(x-\hat{x}))] \\
    &= \arg\min_{\hat{x}} E[(x-\hat{x})(x-\hat{x})^T] \\
    &= \arg\min_{\hat{x}} tr(\Sigma)
    \end{align*}
    $$

## Kalman Filter

卡尔曼滤波器是递推的线性MMSE估计器：

$$
\hat{x}_{n|k} = \mathbb{E}[x_n | y_{1:k}]
$$

- **Prediction Step**:

    $$
    \hat{x}_{n+1|n} = A\hat{x}_{n|n} + Bu_n
    $$

    $$
    \Sigma_{n+1|n} = A\Sigma_{n|n}A^T + W
    $$

- **Measurement Update**:

    The error signal is fed into the estimator to update (the innovation):

    $$
    z_{n+1} = y_{n+1} - C\hat{x}_{n+1|n}
    $$

    $$
    S_{n+1} = C\sigma_{n+1|n}C^T + V
    $$

- **State Update with Kalman Gain** (can run at different frequencies):

    $$
    \hat{x}_{n+1|n+1} = \hat{x}_{n+1|n} + L_{n+1}z_{n+1}
    $$

- **Covariant Update with Joseph Form**:

    $$
    \Sigma_{n+1|n+1} = (I-L_{n+1}C)\Sigma_{n+1|n} (I-L_{n+1}C)^T + L_{k+1}VL_{k+1}
    $$

- **Kalman Gain**:

    $$
    L_{n+1} = \Sigma_{n+1|n}C^TS^{-1}_{n+1}
    $$

- ⭐ **Kalman Filter Algorithm Summary**:

    **Initialize**:

    $$
    \hat{x}_{0|0}, \Sigma_{0|0}, W, v
    $$

    **Predict**:

    $$
    \hat{x}_{n+1|n} = A\hat{x}_{n|n} + Bu_n
    $$
    $$
    \Sigma_{n+1|n} = A\Sigma_{n|n}A^T + W
    $$

    **Calculate Innovation + Covariance**:

    $$
    z_{n+1} = y_{n+1} - C\hat{x}_{n+1|n}
    $$
    $$
    S_{n+1} = C\sigma_{n+1|n}C^T + V
    $$

    **Calculate Kalman Gain**:

    $$
    L_{n+1} = \Sigma_{n+1|n}C^TS^{-1}_{n+1}
    $$

    **Update**:

    $$
    \hat{x}_{n+1|n+1} = \hat{x}_{n+1|n} + L_{n+1}z_{n+1}
    $$
    $$
    \Sigma_{n+1|n+1} = (I-L_{n+1}C)\Sigma_{n+1|n} (I-L_{n+1}C)^T + L_{k+1}VL_{k+1}
    $$

# 鲁棒控制

TODO
