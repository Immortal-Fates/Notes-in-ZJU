---
title: 12-Practical-Implementation
date: 2026-03-02
tags:
course: AI
status: draft
---
# Practical Implementation

[TOC]

## Environment Interface

- **Gymnasium-style Environment**.

  - Takeaway: RL 实现的第一步是确保 environment API 正确。多数训练异常不是算法公式错，而是 observation、reward、done handling 或 action scaling 有问题。

  - Core Mechanism:

    ```python
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    ```

    `terminated` 表示任务自然结束，`truncated` 表示时间限制等外部截断；value bootstrap 时二者处理方式可能不同。

  - Pipeline:

    1. 检查 observation shape、dtype、范围。
    2. 检查 action space 与 policy output 是否一致。
    3. 检查 reward scale 和 termination。
    4. 用 random policy 跑通 rollouts。
    5. 用已知 baseline 验证环境可学习。

---

## Training Loop

- **RL Training Loop**.

  - Takeaway: 大多数 RL 算法都遵循“collect data → estimate target → update model → evaluate”的循环。差异在于数据是否 on-policy、target 是 return/TD/Q/advantage、是否使用 replay buffer。

  - Pipeline:

    1. Initialize policy/value networks。
    2. Collect transitions or rollouts。
    3. Store data in rollout buffer or replay buffer。
    4. Compute returns、TD targets、advantages 或 Q targets。
    5. Update actor、critic、Q network 或 dynamics model。
    6. Evaluate policy without exploration noise。
    7. Log return、loss、entropy、KL、value error、success rate。

  - Cons:

    - 训练曲线高噪声，不能只看单次 run。
    - Evaluation 必须与 training exploration 区分。

---

## Debug Checklist

- **Common RL Bugs**.

  - Takeaway: RL debug 应优先检查数据接口和估计目标，再检查网络结构。很多问题会表现为 return 不涨，但根因可能是 done mask、log-prob shape、advantage normalization 或 action scaling。

  - Core Mechanism:

    - **Done handling**: terminal state 不应继续 bootstrap；time-limit truncation 可按任务选择是否 bootstrap。
    - **Reward scale**: reward 太大或太小都会影响 critic 和 entropy temperature。
    - **Action scaling**: policy 输出范围必须映射到 environment action bounds。
    - **Old policy freeze**: PPO 必须保存 old log-prob，不能用更新后的 policy 重算。
    - **Advantage normalization**: 通常对 batch advantage 做 normalization，提高稳定性。
    - **Replay buffer**: off-policy 算法要确认 transition 顺序、done mask、next observation 正确。

  - Pipeline:

    1. Random policy sanity check。
    2. Overfit tiny environment or tiny dataset。
    3. Compare against Stable-Baselines3 baseline。
    4. Run multiple seeds。
    5. Log intermediate targets: returns、advantages、TD error、Q values。

## References

- [Gymnasium Basic Usage](https://gymnasium.farama.org/introduction/basic_usage/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/)
- [PyTorch](https://pytorch.org/)
