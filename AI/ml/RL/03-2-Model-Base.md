---
title: 03-2-Model-Base
date: 2026-03-02
tags:
course: AI
status: draft
---
# Model-Base Reinforcement Learning

[TOC]

## Model-Based RL

- **Model-Based RL**.

  - Takeaway: Model-based RL 学习或使用 environment model，通过 planning、imagined rollout 或 model-augmented updates 提高 sample efficiency。

  - Core Mechanism:

    model-based 方法用模型把一部分真实采样替换为规划或模拟

  - Pros:

    - Sample efficiency 
    - 当部分 physics 或 simulator 已知时，可以直接利用 domain knowledge。
  
  - Cons:
  
    - 与真实环境有差距
    - learned model 对 out-of-distribution action 不可靠。

---

## Pure Planning

- **Planning with Known Model**.

  - Takeaway: 如果 dynamics 已知，可以不显式学习 policy，而是在每个 state 上直接规划 action。Dynamic programming、trajectory optimization 和 tree search 都属于这一类思想。

  - Core Mechanism:

    - **Value iteration**:

      $$
      V_{k+1}(s)=
      \max_a\sum_{s'}P(s'|s,a)
      \left[R(s,a,s')+\gamma V_k(s')\right]
      $$

    - **Trajectory optimization**:

      $$
      a_{0:H}^*=
      \arg\max_{a_{0:H}}
      \sum_{t=0}^{H}\gamma^t r(s_t,a_t)
      $$

      在 continuous control 中常用 shooting、CEM、iLQR 等方法搜索 action sequence。

  - Pros:

    - 不需要先训练 policy，给定模型即可规划 action。
    - 可以直接处理 changing goals 或 explicit constraints。
    - 在 low-dimensional known dynamics 中通常解释性强。

  - Cons:

    - 需要可靠模型。
    - 在线规划开销可能很高。
    - 高维 observation 或复杂 dynamics 下规划难度迅速上升。

---

## MPC

- **Model Predictive Control**.

  - Takeaway: MPC 每一步都解一个 finite-horizon planning problem，只执行第一个 action，然后重新观测并重新规划。它通过 receding horizon 降低长期模型误差影响，是控制和 model-based RL 的重要连接点。

  - Core Mechanism:

    $$
    a_{t:t+H}^*=
    \arg\max_{a_{t:t+H}}
    \sum_{k=0}^{H}\gamma^k\hat{r}_{t+k}
    $$

    执行 $a_t^*$ 后丢弃剩余计划，下一步用新 state 再规划。

  - Pipeline:

    1. Observe current state $s_t$。
    2. 用 model 评估多个 candidate action sequences。
    3. 选择 predicted return 最高的 sequence。
    4. 执行第一个 action。
    5. 进入下一 state 后重新规划。

  - Pros:

    - 可自然加入 action constraint 和 safety constraint。
    - 每步重规划能缓解模型误差。

  - Cons:

    - horizon 太短会短视，太长会放大 model error。
    - real-time planning 对计算资源要求高。

---

## Dyna

- **Dyna-style Model-Based Learning**.

  - Takeaway: Dyna 同时使用真实经验和模型生成的 simulated experience 更新 value 或 policy。它把 model learning、planning 和 model-free update 放进同一个循环。

  - Pipeline:

    1. 与真实环境交互，收集 transition。
    2. 更新 dynamics / reward model。
    3. 用真实 transition 更新 value 或 policy。
    4. 从模型中采样 imagined transitions。
    5. 用 imagined transitions 继续更新 value 或 policy。

  - Pros:

    - 能复用少量真实经验产生更多训练信号。
    - 与 Q-learning、actor-critic 等 model-free learner 可以组合。

  - Cons:

    - 错误模型生成的数据会系统性污染 policy。
    - 通常需要限制 rollout horizon 或使用 uncertainty filtering。

---

## Modern Model-Based RL

- **Learned Models for Policy Improvement**.

  - Takeaway: 现代 model-based RL 常把 learned model 当作 sample-efficiency accelerator，而不是完全替代真实环境。常见策略是短 rollout、ensemble uncertainty、model predictive control 或把 simulated data 混入 replay buffer。

  - Core Mechanism:

    - **Short model rollout**: 只 rollout 少量步数，降低 compounding error。
    - **Model ensemble**: 多个模型估计 uncertainty，避免过度相信单个 learned dynamics。
    - **Hybrid training**: real data 保持 anchor，model data 增加 update frequency。
    - **Planning + value**: planning 只看短 horizon，远期价值由 learned value function 补足。

  - Pros:

    - Short rollout 能在提高 sample efficiency 的同时限制 compounding error。
    - Ensemble uncertainty 可以帮助识别模型不确定区域。
    - 与 off-policy replay buffer 结合时能显著增加训练信号。
    - Planning + value function 能平衡短期模型预测和长期价值估计。

  - Cons:

    - 模型训练、uncertainty 估计和 policy learning 耦合复杂。
    - 高维视觉输入下 learned dynamics 更难准确预测。
    - 如果 uncertainty 估计不准，policy 仍可能 exploit model error。

## References

- [OpenAI Spinning Up: Model-Based RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html#model-based-rl)
- [Sutton and Barto: Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
