# Main Takeaway

强化学习 Reinforcement Learning

- [ ] 理解强化学习基本概念
- [ ] 熟练掌握Q学习算法
- [ ] 理解策略梯度方法
- [ ] 掌握Actor-Critic算法

<!--more-->


# 强化学习简介 🚀

强化学习（RL）是机器学习的一种类型，智能体通过在环境中采取行动来学习决策，以最大化奖励。在最优控制的背景下，RL提供了各种算法和方法来找到最佳控制策略。

强化学习的本质：一种自适应的最优控制

- **理论锚点**：直接继承动态规划的贝尔曼方程框架
- 三大创新方向：
  - **无模型化（Model-Free）**
    避开系统建模，通过试错直接学习（如 Q-Learning）
  - **高维函数逼近**
    用深度神经网络拟合价值函数/策略（DQN, PPO 等）
  - **策略梯度方法**
    直接优化随机策略 $π_θ(a∣s) $（如 REINFORCE 算法）



# RL方法

以下是一些流行的RL方法：

## Model-based RL 🧠

在基于模型的RL中，智能体试图学习环境的模型，用 \( f \) 表示。

$$
\min_\theta ||x_{n+1}-f_\theta(x,u)||_2^2 
$$

🔴 **缺点：** 
- 智能体可能学习到环境中不必要的细节。
- 即使在学习之后，仍然需要求解模型。

## Q-learning 🎮

Q-learning是一种基于价值的方法，智能体学习Q函数，这与动态规划密切相关。

$$
V_{n-1}(x) = \underbrace{\min_u l(x,u) + V_n(f(x,u))}_{Q(x,u)}
$$

🔴 **缺点：** 
- 无法泛化到新任务。
- 具有**高偏差**。
- Q值的过高估计。解决方案包括使用多个Q值或调整更新率。

## Policy Gradient 📈

在Policy Gradient中，智能体直接优化由 \( \theta \) 参数化的策略。

$$
\min_\theta J = \sum_{n=1}^{N-1} l(x_n, u_\theta(x_n)) + l_N(x_n) \quad s.t. \quad x_{n+1} = f(x_n, u_\theta(x_n))
$$

更新方法为：

$$
\min_\theta E_{P(\tau;\theta)}[J(\tau)] = \min \int_\text{all trajectory} J(\tau)p(\tau;\theta)d\tau
$$

其梯度为：

$$
\nabla_\theta E_{p(\tau;\theta)}[J(\tau)] = E_{p(\tau;\theta)}[J(\tau)\nabla_\theta \log(p(\tau;\theta))]
$$

这个梯度可以使用蒙特卡洛搜索来估计。

🔴 **缺点：** 
- 样本效率低。
- 学习不稳定。
- 高方差。解决方案包括信任域方法、协方差调度、梯度裁剪和使用优势函数。

## Actor-Critic 🎭

在演员-评论家方法中，策略梯度中的价值估计被神经网络替代。或者，可以使用特定函数来解决连续动作情况下的Q网络。

🟢 **优点：** 
- 结合了基于价值和基于策略方法的优势。
- 可以处理连续动作空间。

## Conclusion 🌟

有了好的模拟器和准确的成本函数，许多RL问题都可以在很大程度上得到解决。方法的选择取决于具体问题、可用数据和计算资源。