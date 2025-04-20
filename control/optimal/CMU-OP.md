# Main Takeaway

配套CMU-16-745 Optimal Control and Reinforcement Learning食用



<!--more-->



# CMU-16-745

## Lec 1 系统状态方程、平衡点与稳定性

Optimal Control and RL are the same thing

- 连续系统状态方程（Continuous Time Dynamics）
- 仿射系统状态方程（Control-Affine System）
- 机械臂系统状态方程（Manipulator Dynamics）
- 线性系统
- 平衡点（Equilibria）
- 平衡点的稳定性（Stability）

[Lecture 1 系统状态方程、平衡点与稳定性 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/629135263)



## Lec 2 离散状态方程、数值积分与稳定性

- 状态方程离散
  - 离散状态方程
  - 稳定性分析
  - 案例分析
    - 前向欧拉积分
    - 龙格库塔法RK4
    - 后向欧拉积分
- 控制量的离散

[Lecture 2 离散状态方程、数值积分与稳定性 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/629135862)

non-causal非因果关系

- 要小心离散情况下的ODE，特别是对于临界的情况，最好做sanity check（检查energy）



## Lec 3 求根法与无约束的最优化问题

- 符号约定（Notation）
- 方程求根（Root Finding）
  - 牛顿法（Newton's method）
  - 不动点迭代法Fixed point Iteration）
- 最小化问题（Minimization）
  - 充分条件与必要条件
  - 正则化（regularization）
  - 线搜索（line search）

[Lecture 3 求根法与无约束的最优化问题 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/629137277)

在牛顿法中，若Hessian矩阵正定，Cholesky分解可快速计算 ( $H^{-1}$ )，避免直接求逆。并且也可以通过分解判断是否正定，如果不正定使用阻尼牛顿法

## Lec 4 约束最优化问题

- 等式约束
  - 一阶必要条件与牛顿法
  
  - 高斯牛顿法
  
    高斯牛顿法在实际中往往比较常用，因为每次迭代比较快，而且具有**超线性**的收敛性
  
- 不等式约束
  - 一阶必要条件（KKT条件）
  - [Active-Set法](https://zhida.zhihu.com/search?content_id=227940219&content_type=Article&match_order=1&q=Active-Set法&zhida_source=entity)
  - [障碍函数法](https://zhida.zhihu.com/search?content_id=227940219&content_type=Article&match_order=1&q=障碍函数法&zhida_source=entity)/[内点法](https://zhida.zhihu.com/search?content_id=227940219&content_type=Article&match_order=1&q=内点法&zhida_source=entity)
  - [罚函数法](https://zhida.zhihu.com/search?content_id=227940219&content_type=Article&match_order=1&q=罚函数法&zhida_source=entity)
  - 增广拉格朗日法
  
- 二次规划QP

[Lecture 4 约束最优化问题 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/629139142)

heuristic启发式

机器人QP问题：将机器人实际需求（如关节运动、力分配、轨迹跟踪）转化为**带约束的二次优化问题**，通过求解该问题获得满足物理限制的最优解

二次规划（QP）问题的标准形式

**目标函数**：
$$
\min_x \quad \frac{1}{2} x^T H x + c^T x
$$

**约束条件**：
1. **等式约束**：
$$
A x = b
$$

2. **不等式约束**：
$$
lb \leq x \leq ub
$$

---

变量说明

- $  x  $：优化变量（如关节力矩、足底力）。
- $  H  $：正定矩阵，保证问题为凸优化，有唯一解。
- $  A  $：等式约束矩阵。
- $  b  $：等式约束向量。
- $  lb  $、$  ub  $：变量的下界和上界。

**QP问题的求解方法**

1. **求解器类型**：
   - **Active-set方法**（如qpOASES）：适合中小规模问题，实时性强，常用于嵌入式系统。
   - **内点法**（如IPOPT、OSQP）：适合大规模问题，稳定性高。
   - **凸优化库**（CVXPY、CasADi）：提供建模接口，简化问题构建。
2. **机器人中的实际使用**：
   - **MIT Cheetah**：使用qpOASES实时求解足底力分配。
   - **工业机械臂**：通过OSQP生成无碰撞轨迹。
   - **无人机集群**：利用CVXPY建模多机协同避障。



## Lec 5 带约束线搜索与正则化

- [对偶性](https://zhida.zhihu.com/search?content_id=227940583&content_type=Article&match_order=1&q=对偶性&zhida_source=entity)与[正则化](https://zhida.zhihu.com/search?content_id=227940583&content_type=Article&match_order=1&q=正则化&zhida_source=entity)（Regularization and Duality）
- 指标函数与线搜索（Merit Function）
- 带约束最小化问题

[Lecture 5 带约束线搜索与正则化 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/629140808)



## Lec 6 确定性最优控制

- 控制简史
- 确定性最优控制（Deterministic Optimal Control）
  - 连续时间
  - 离散时间
- [极小值原理](https://zhida.zhihu.com/search?content_id=227941610&content_type=Article&match_order=1&q=极小值原理&zhida_source=entity)（Pontryagin's Minimum Principle）

[Lecture 6 确定性最优控制（Deterministic Optimal Control） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/629145483)

前面Lecture 1-5我们一直在打基础，学习了如何离散化模型，如何做一些优化，从这一节开始，我们正式进入最优控制的学习

最后，教授总结，当前研究的困难点有

- 如何找到通用的处理接触的理论
- 如何将model-base控制和model-free的强化学习结合
- 如何在强化学习中加入先验知识让RL更加数据高效
- 如何保证不确定非线性系统的控制安全裕度
- 在一个非协作的环境中，如何处理其他adversaial 课题



确定性最优控制是指系统的模型是确定的，在当前状态x给一个特定的u，根据系统方程我就一定可以知道下一步状态会怎么转移。

- 通过解这个最优化解出来的轨迹是一个开环轨迹**（因此，要么就解的特别快，总是用前面的开环轨迹执行，如MPC问题，要么就离线解，在线用一个很好的反馈控制器来跟踪）**。与此相反，在**随机最优控制问题**中，由于问题的定义是充满噪声的，因此解随机最优控制问题必须要得到一个闭环的轨迹。
- (1)式指定的优化问题在大部分时候，是没有解析解的，但是少部分很特殊的情况有，如LQR问题。



极小值原理（Pontryagin's Minimum Principle）

KKT条件[Karush-Kuhn-Tucker (KKT)条件 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/38163970)



## Let 7 LQR（Linear Quadratic Regulator）的三种解法

- 间接法（indirect method）、打靶法（shooting method）
- 二次规划[QP解法](https://zhida.zhihu.com/search?content_id=227941801&content_type=Article&match_order=1&q=QP解法&zhida_source=entity)
- [Riccati迭代](https://zhida.zhihu.com/search?content_id=227941801&content_type=Article&match_order=1&q=Riccati迭代&zhida_source=entity)(Riccati Recursion)

[Leture 7 LQR（Linear Quadratic Regulator）的三种解法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/629146365)

[LQR解析解推导：从LQR到迭代Riccati方程 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/636305927)

Riccati方程的核心应用在于**最优控制与状态估计**，具体包括：

1. **线性二次调节器（LQR）**： 设计反馈控制器，最小化二次型性能指标 J=∫(x⊤Qx+u⊤Ru)dt，通过求解CARE得到最优反馈矩阵 K=R−1B⊤P，使闭环系统稳定。
2. **卡尔曼滤波（LQG）**： 在含噪声的系统中估计状态，通过DARE求解最优滤波器增益，最小化估计误差协方差。
3. **经济与金融优化**： 处理时间不一致控制问题，如均衡Riccati方程用于动态资源分配。

[(22 封私信 / 80 条消息) 请详细介绍下黎卡提方程在控制理论中的重要性和作用？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/20081582)

标准的代数Riccati方程分为如下两种：

- 连续时间代数Riccati方程（CARE）：
- 离散时间代数Riccati方程（DARE）



**哈密顿矩阵法**?



# References

- https://zhuanlan.zhihu.com/p/629131647
- [2023-2024春夏许超老师最优化与最优控制课程分享 - CC98论坛](https://www.cc98.org/topic/5923253)

- [Tutorial — Ceres Solver (ceres-solver.org)](http://ceres-solver.org/tutorial.html)