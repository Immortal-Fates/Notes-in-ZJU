# Main Takeaway

在振宇指导下的入门RL第一步，训练一个倒立摆

<!--more-->

# Quick Start

环境配置，使用conda环境装好（这里我是用的是python 3.10，基本什么都支持）

1. 安装Gymnasium

   ```
   pip install Gymnasium
   ```

2. 安装pytorch，这里需要直接去`torch.org`官网上去找符合你CUDA Version的下载版本

   使用`nvidia-smi`查看显卡各种信息

   > 这种强依赖一般在前面安装

3. 安装Stable-Baseline3

   ```
   pip install stable-baselines3
   ```

4. 安装`tensorflow`

   > 不适用也可以跳过

5. 其他安装，例如pygame等

# Envs

下面对使用RL的环境进行一下介绍

- **Gymnasium** 是 OpenAI Gym 的官方继承者，由 Farama Foundation 维护的**强化学习环境**标准化库。
  - **标准化环境接口**: 提供统一的环境API，让不同算法可以无缝切换环境
  - **丰富的预置环境**: 包含经典控制、Atari游戏、机器人控制等多种任务
  - **环境注册系统**: 方便创建和管理自定义环境
- **Stable Baselines3** 是一个高质量的强化学习算法实现库，基于PyTorch构建

# Train

用随便一个LLM写好代码就可以训练，建议先用Gymnasium内置的几个环境训练

使用tensorboard可视化训练过程：

```
tensorboard --logdir=`your trained model path`
```

下面是对训练指标详解

# TensorBoard PPO训练指标详解

在使用Stable Baselines3训练PPO模型时，TensorBoard会显示多个重要的训练指标。以下是对每个指标的详细解释：

## Rollout 指标 (数据收集阶段)

- `rollout/ep_len_mean`

  - **含义**: 平均每个episode的长度（步数）

  - **上升趋势**: 表示智能体存活时间越来越长，性能改善

  - **波动大**: 说明策略还不稳定

- `rollout/ep_rew_mean`

  - **含义**: 平均每个episode的累积奖励

  - **持续上升**: 训练进展良好

  - **停滞或下降**: 可能遇到训练瓶颈

## Time 指标 (性能监控)

- `time/fps`

  - **含义**: 每秒处理的帧数（frames per second）

  - **重要性**: 训练效率指标

  - **典型值**: 1000-5000 FPS属于正常范围

  - **影响因素**: 网络复杂度、并行环境数量、硬件性能

- `time/iterations`
  - **含义**: 完成的训练迭代次数

- `time/total_timesteps`

  - **含义**: 总的环境交互步数

  - **重要性**: 衡量智能体的经验积累量

## Train 指标 (核心学习指标)

- `train/approx_kl`

  - **含义**: 新旧策略之间的KL散度（近似值）

  - **数值解读**:

    - **0.01-0.03**: 理想范围，表示策略更新幅度适中

    - **> 0.05**: 策略变化太大，可能不稳定

    - **< 0.005**: 策略变化太小，学习缓慢

  - **调整建议**: 如果持续过高，降低学习率或减小clip_range

- `train/clip_fraction`

  - **含义**: 被裁剪的策略比率梯度的比例

  - **数值解读**:

    - **0.1-0.3**: 正常范围

    - **> 0.5**: 裁剪过多，学习率可能太高

    - **< 0.05**: 很少裁剪，可能可以提高学习率

  - **PPO核心**: 反映PPO裁剪机制的工作情况

- `train/clip_range`

  - **含义**: PPO的裁剪范围参数

  - **典型值**: 0.2（通常固定）

  - **作用**: 限制策略更新幅度，保证训练稳定性

- `train/entropy_loss`

  - **含义**: 策略熵的损失（负值）

  - **数值解读**:

    - **绝对值大**: 策略探索性强（随机性高）

    - **绝对值小**: 策略变得确定性（收敛）

    - **逐渐减小**: 正常的学习过程

  - **调整**: 通过ent_coef参数控制探索-利用平衡

- `train/explained_variance`

  - **含义**: 价值函数预测准确性

  - **数值解读**:

    - **0.8-0.95**: 价值函数学习良好

    - **0.5-0.8**: 中等水平

    - **< 0.3**: 价值函数学习困难

  - **负值**: 预测效果很差，需要调试

- `train/learning_rate`

  - **含义**: 当前学习率

  - **用途**: 如果使用学习率调度，可以观察其变化

- `train/loss`

  - **含义**: 总的策略损失

  - **趋势**: 应该总体呈下降趋势

  - **波动**: 适度波动正常，剧烈波动表示不稳定

- `train/policy_gradient_loss`

  - **含义**: 策略梯度损失

  - **重要性**: PPO的核心损失函数

  - **趋势**: 训练过程中应该逐渐减小

- `train/std`

  - **含义**: 连续动作空间中动作分布的标准差

  - **仅适用于**: 连续控制任务（如Pendulum）

  - **数值解读**:

    - **初期较大**: 探索性强

    - **逐渐减小**: 策略变得更确定

    - **过小**: 可能收敛过早，缺乏探索

- `train/value_loss`

  - **含义**: 价值函数的损失

  - **重要性**: 反映价值网络学习质量

  - **期望**: 逐渐下降，与explained_variance相关

## 指标监控建议

健康的训练过程应该显示：

```python
# 良好的训练指标范围
rollout/ep_rew_mean: 持续上升
train/approx_kl: 0.01-0.03
train/clip_fraction: 0.1-0.3
train/explained_variance: > 0.7
train/entropy_loss: 逐渐减小（绝对值）
train/policy_gradient_loss: 逐渐减小
train/value_loss: 逐渐减小
```

异常情况警报：

```python
# 需要注意的异常指标
train/approx_kl > 0.05: 学习率太高
train/explained_variance < 0.3: 价值函数学习困难
rollout/ep_rew_mean 长期停滞: 算法收敛或遇到瓶颈
train/clip_fraction > 0.5: 策略更新过于激进
```

调试建议：

```python
# 根据指标调整超参数
if train/approx_kl > 0.05:
    learning_rate *= 0.5  # 降低学习率

if train/explained_variance < 0.5:
    net_arch=dict(vf=[256, 256])  # 增加价值网络容量

if train/entropy_loss绝对值过小:
    ent_coef *= 2  # 增加探索
```

通过监控这些指标，你可以实时了解训练进展，及时发现问题并调整超参数，确保训练的稳定性和效果。

# References

- [Stable-Baselines3 Docs - Reliable Reinforcement Learning Implementations — Stable Baselines3 2.6.1a1 documentation](https://stable-baselines3.readthedocs.io/en/master/index.html)
- [Basic Usage - Gymnasium Documentation (farama.org)](https://gymnasium.farama.org/introduction/basic_usage/)
- [PyTorch](https://pytorch.org/)
