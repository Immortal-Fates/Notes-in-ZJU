# Intro

介绍数理统计





## EM算法

- Takeaway: Expectation Maximization(EM) 期望最大化算法，用两步来迭代更新模型

- Target Problem

  设观测数据是
  $$
  X=\{x_1,x_2,\dots,x_n\}
  $$
  隐藏变量是
  $$
  Z=\{z_1,z_2,\dots,z_n\}
  $$
  模型参数是$\theta$

  我们想让观测数据的似然最大：
  $$
  \log p(X \mid \theta)
  $$
  但因为隐藏变量 $Z$ 不知道，所以真实的观测似然需要把所有可能的 $Z$ 都加起来：
  $$
  p(X \mid \theta)=\sum_Z p(X,Z \mid \theta)
  $$
  于是目标函数变成：
  $$
  \log p(X \mid \theta)
  =
  \log \sum_Z p(X,Z \mid \theta)
  $$
  这个式子难处理的地方是：$\log$外面套着求和。这会让直接求最大值变得困难。

- Core Mechanism

  EM 的核心想法就是：**既然隐藏变量不知道，那就先根据当前参数估计隐藏变量，再用估计出来的隐藏变量更新参数。**

  EM 每轮迭代分成两步。

  1. Expectation step
  
     固定当前参数 $\theta^{old}$，估计隐藏变量 $Z$ 的后验分布。
     $$
     p(Z \mid X,\theta^{old})
     $$

  2. Maximization step

     固定 E 步得到的隐藏变量分布，更新模型参数 $\theta$。

     EM 会构造一个辅助函数：
     $$
     Q(\theta,\theta^{old})
     =
     \mathbb{E}_{Z \mid X,\theta^{old}}
     \left[
     \log p(X,Z \mid \theta)
     \right]
     =
     \sum_Z
     p(Z \mid X,\theta^{old})
     \log p(X,Z \mid \theta)
     $$
     表示在当前参数下，先估计隐藏变量的概率分布，然后计算完整数据对数似然的期望。

     M 步就是最大化这个函数：
     $$
     \theta^{new}
     =
     \arg\max_{\theta} Q(\theta,\theta^{old})
     $$
  
  
  所以 EM 的流程可以理解为：
  $$
  \theta^{old}
  \longrightarrow
  p(Z \mid X,\theta^{old})
  \longrightarrow
  \theta^{new}
  $$
  用旧参数估计隐藏变量，再用隐藏变量更新新参数。
  
  > [!NOTE]
  >
  > 为什么 EM 不是直接优化原目标？
  >
  > 原目标是：
  > $$
  > \log p(X \mid \theta)
  > =
  > \log \sum_Z p(X,Z \mid \theta)
  > $$
  > 这个式子不好直接最大化。
  >
  > EM 的做法是构造一个更容易优化的下界。每次迭代时，E 步让这个下界贴近当前参数处的真实目标，M 步提高这个下界。
  >
  > 所以 EM 不是直接硬求：
  > $$
  > \arg\max_{\theta} \log p(X \mid \theta)
  > $$
  > 而是不断优化：
  > $$
  > Q(\theta,\theta^{old})
  > $$
  > 这样可以保证每次迭代后观测数据似然不会下降：
  > $$
  > \log p(X \mid \theta^{new})
  > \ge
  > \log p(X \mid \theta^{old})
  > $$
  > 但它只能保证似然不下降，不能保证一定找到全局最优

- Application：GMM高斯混合模型，kmeans
