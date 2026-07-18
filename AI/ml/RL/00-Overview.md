---
title: 00-Overview
date: 2026-03-02
tags:
course: AI
status: draft
---
# Reinforcement Learning Overview

[TOC]

<!--more-->

## Learning Route

- **RL Learning Route**.

  - Takeaway: 强化学习可以按“问题建模 → value 递推 → 算法分类 → model-free / model-based → 实践实现”来学习。这样组织能把 Bellman equation、policy gradient、planning 和 modern RL algorithms 放在同一张路线图里。

  - Core Mechanism:

    1. **Problem Setting**: MDP, state/observation, action space, reward, return, policy。
    2. **Value Foundation**: $V^\pi$、$Q^\pi$、$A^\pi$、Bellman expectation / optimality equations。
    3. **Algorithm Taxonomy**: 按是否使用 environment model 分成 model-free 与 model-based。
    4. **Model-Free RL**: MC、TD、SARSA、Q-learning、DQN、policy gradient、PPO、SAC、offline RL、RLHF。
    5. **Model-Based RL**: known model planning、MPC、Dyna、learned dynamics、model rollout。
    6. **Practical Implementation**: environment API、training loop、debug checklist。

---

## Core Problem

- **Sequential Decision Making**.

  - Takeaway: RL 的核心问题是 agent 如何通过 trial and error 学到一个最大化 long-term return 的 policy。它不同于监督学习，因为 action 会改变未来 state distribution，训练数据由 policy 自己诱导出来。

  - Core Mechanism:

    $$
    J(\pi)=\mathbb{E}_{\tau\sim\pi}\left[\sum_{t=0}^{T}\gamma^t r_t\right]
    $$

    $J(\pi)$ 是 policy 的 expected return，$\tau$ 是由 policy 和 environment 共同生成的 trajectory。

  - Pipeline:

    1. 设计 environment: state/observation、action、reward、termination。
    2. 选择算法族: value-based、policy-based、actor-critic、model-based。
    3. 收集 rollouts 或使用 offline dataset。
    4. 估计 value、Q、advantage 或 model。
    5. 更新 policy 或 value function。
    6. 用 average return、sample efficiency、seed stability、constraint violation 评估。

  - Cons:

    - Reward 设计错误会导致 agent 学到非预期策略。
    - exploration 成本高，真实系统中还涉及安全约束。
    - 单 seed 曲线不可靠，RL 对随机性和超参数敏感。

## References

- [OpenAI Spinning Up: Introduction to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [OpenAI Spinning Up: Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)
- [Sutton and Barto: Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- lilianweng`s blog[A (Long) Peek into Reinforcement Learning | Lil'Log](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) (so good)
- a math insight pdf: [Deep Reinforcement Learning Through Policy Optimization](http://joschu.net/docs/thesis.pdf)
- 研究项目的建议：[Tim Rocktäschel](https://rockt.ai/2018/08/29/msc-advice)
- google`s advice to this field: [ML engineering for AI safety & robustness: a Google Brain engineer's guide to entering the field | 80,000 Hours](https://80000hours.org/articles/ml-engineering-career-transition-guide/)
