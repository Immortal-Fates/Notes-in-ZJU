---
title: 01-Control-to-RL
date: 2026-03-02
tags:
course: AI
status: draft
---
# From Optimal Control to Reinforcement Learning

[TOC]

<!--more-->

## Optimal Control

- **Optimal Control**.

  - Takeaway: 最优控制在已知 dynamics 约束下寻找最优 control law；RL 可以看作最优控制在 unknown dynamics、高维 observation 和 data-driven setting 下的学习化版本。

  - Core Mechanism:

    - **Objective**:

      $$
      J=\phi(x(t_f),t_f)+\int_{t_0}^{t_f}L(x(t),u(t),t)\,dt
      $$

      $x(t)$ 是状态，$u(t)$ 是控制，$L$ 是 running cost，$\phi$ 是 terminal cost。

    - **Dynamics constraint**:

      $$
      \dot{x}(t)=f(x(t),u(t),t)
      $$

      控制问题的优化变量不是任意轨迹，而是必须满足系统动力学的轨迹。

    - **Dynamic programming view**:

      $$
      V^*(s)=\min_a\mathbb{E}\left[c(s,a)+\gamma V^*(s')\right]
      $$

      把 cost 换成 reward 的负数，就得到 RL 中 Bellman optimality equation 的基本形态。

  - Cons:

    - 需要较准确的 dynamics model。
    - 高维状态离散化会导致 curse of dimensionality。
    - 对复杂 perception input 难以直接设计解析 controller。

---

## RL as Adaptive Control

- **RL and Control Mapping**.

  - Takeaway: RL 继承 Bellman principle，但把“已知模型求解控制律”改成“从 interaction data 中估计 value、policy 或 model”。它更适合模型未知、状态复杂、reward 可定义但 optimal behavior 难手写的任务。

  - Core Mechanism:

    $$
    r_t=-c_t,\qquad
    \max_\pi\mathbb{E}\left[\sum_t\gamma^t r_t\right]
    \equiv
    \min_\pi\mathbb{E}\left[\sum_t\gamma^t c_t\right]
    $$

    控制和 RL 的目标本质一致：在 dynamics 约束下优化长期累计指标。

    | Optimal Control | Reinforcement Learning | Typical Method |
    | --- | --- | --- |
    | Dynamic Programming | Value Iteration / Q-learning | Bellman backup |
    | LQR / iLQR | DDPG / TD3 | continuous control |
    | MPC | Model-Based RL | receding-horizon planning |
    | Inverse Optimal Control | Inverse RL / GAIL | infer reward |

  - Pipeline:

    1. 把任务写成 MDP。
    2. 判断 dynamics 是否已知、可学习或完全不可用。
    3. 已知模型优先考虑 planning / MPC。
    4. 可在线交互则考虑 model-free 或 model-based RL。
    5. 只能用历史数据则考虑 offline RL / imitation learning。

## References

- [从最优控制到强化学习](https://www.bilibili.com/video/BV1MDWPzMEnw?vd_source=93bb338120537438ee9180881deab9c1)
