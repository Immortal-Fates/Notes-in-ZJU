# RL Algorithms

![rl_algorithms_9_15](./assets/03-RL-Algorithms.assets/rl_algorithms_9_15.svg)

- Model Free vs. Model Base: **whether the agent has access to (or learns) a model of the environment**.


- 学习什么
  - policies, either stochastic or deterministic,
  - action-value functions (Q-functions),
  - value functions,
  - and/or environment models.

## Model-Free RL

- Pros: easier to implement and tune

There are two main approaches to representing and training agents with model-free RL: **Policy Optimization** and **Q-Learning**

### Policy Optimization

Methods in this family represent a policy explicitly as ![\pi_{\theta}(a|s)](https://spinningup.openai.com/en/latest/_images/math/400068784a9d13ffe96c61f29b4ab26ad5557376.svg). They optimize the parameters ![\theta](https://spinningup.openai.com/en/latest/_images/math/ce5edddd490112350f4bd555d9390e0e845f754a.svg) either directly by gradient ascent on the performance objective ![J(\pi_{\theta})](https://spinningup.openai.com/en/latest/_images/math/96b876944de9cf0f980fe261562e8e07029245bf.svg), or indirectly, by maximizing local approximations of ![J(\pi_{\theta})](https://spinningup.openai.com/en/latest/_images/math/96b876944de9cf0f980fe261562e8e07029245bf.svg). This optimization is almost always performed **on-policy**, which means that each update only uses data collected while acting according to the most recent version of the policy. Policy optimization also usually involves learning an approximator ![V_{\phi}(s)](https://spinningup.openai.com/en/latest/_images/math/693bb706835fbd5903ad9758837acecd07ef13b1.svg) for the on-policy value function ![V^{\pi}(s)](https://spinningup.openai.com/en/latest/_images/math/a81303323c25fc13cd0652ca46d7596276e5cb7e.svg), which gets used in figuring out how to update the policy.

- [A2C / A3C](https://arxiv.org/abs/1602.01783), which performs gradient ascent to directly maximize performance,
- [PPO](https://arxiv.org/abs/1707.06347), whose updates indirectly maximize performance, by instead maximizing a *surrogate objective* function which gives a conservative estimate for how much ![J(\pi_{\theta})](https://spinningup.openai.com/en/latest/_images/math/96b876944de9cf0f980fe261562e8e07029245bf.svg) will change as a result of the update.
- Pros: stable and reliable in the sense that *you directly optimize for the thing you want*
- Cons:

#### PPO

> check the link: https://github.com/DLR-RM/stable-baselines3/blob/master/docs/modules/ppo.rst

- Takeaway: The [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) algorithm combines ideas from A2C (having multiple workers) and TRPO (it uses a trust region to improve the actor).

  The main idea is that after an update, the new policy should be not too far from the old policy. For that, ppo uses clipping to avoid too large update.



### Q-Learning

Methods in this family learn an approximator ![Q_{\theta}(s,a)](https://spinningup.openai.com/en/latest/_images/math/de947d14fdcfaa155ef3301fc39efcf9e6c9449c.svg) for the optimal action-value function, ![Q^*(s,a)](https://spinningup.openai.com/en/latest/_images/math/cbed396f671d6fb54f6df5c044b82ab3f052d63e.svg). Typically they use an objective function based on the [Bellman equation](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#bellman-equations). This optimization is almost always performed **off-policy**, which means that each update can use data collected at any point during training, regardless of how the agent was choosing to explore the environment when the data was obtained. The corresponding policy is obtained via the connection between ![Q^*](https://spinningup.openai.com/en/latest/_images/math/c2e969d09ae88d847429eac9a8494cc89cabe4bd.svg) and ![\pi^*](https://spinningup.openai.com/en/latest/_images/math/1fbf259ac070c92161e32b93c0f64705a8f18f0a.svg): the actions taken by the Q-learning agent are given by

![a(s) = \arg \max_a Q_{\theta}(s,a).](https://spinningup.openai.com/en/latest/_images/math/d353412962e458573b92aac78df3fbe0a10d998d.svg)

Examples of Q-learning methods include

- [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), a classic which substantially launched the field of deep RL,
- [C51](https://arxiv.org/abs/1707.06887), a variant that learns a distribution over return whose expectation is ![Q^*](https://spinningup.openai.com/en/latest/_images/math/c2e969d09ae88d847429eac9a8494cc89cabe4bd.svg).

- Pros: gain the advantage of being substantially more sample efficient when they do work, because they can reuse data more effectively than policy optimization techniques.
- Cons: *indirectly* optimize for agent performance. There are many failure modes for this kind of learning, so it tends to be less stable

### Interpolating Between Policy Optimization and Q-Learning

Serendipitously, policy optimization and Q-learning are not incompatible (and under some circumstances, it turns out, [equivalent](https://arxiv.org/abs/1704.06440)), and there exist a range of algorithms that live in between the two extremes. Algorithms that live on this spectrum are able to carefully trade-off between the strengths and weaknesses of either side. Examples include

- [DDPG](https://arxiv.org/abs/1509.02971), an algorithm which concurrently learns a deterministic policy and a Q-function by using each to improve the other,
- and [SAC](https://arxiv.org/abs/1801.01290), a variant which uses stochastic policies, entropy regularization, and a few other tricks to stabilize learning and score higher than DDPG on standard benchmarks.

## Model-Based RL

- Pros: When this works, it can result in a substantial improvement in sample efficiency over methods that don’t have a model.
- Cons: a ground-truth model of the environment is usually not available to the agent

**Background: Pure Planning.** The most basic approach *never* explicitly represents the policy, and instead, uses pure planning techniques like [model-predictive control](https://en.wikipedia.org/wiki/Model_predictive_control) (MPC) to select actions. In MPC, each time the agent observes the environment, it computes a plan which is optimal with respect to the model, where the plan describes all actions to take over some fixed window of time after the present. (Future rewards beyond the horizon may be considered by the planning algorithm through the use of a learned value function.) The agent then executes the first action of the plan, and immediately discards the rest of it. It computes a new plan each time it prepares to interact with the environment, to avoid using an action from a plan with a shorter-than-desired planning horizon.

## References

- [kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)