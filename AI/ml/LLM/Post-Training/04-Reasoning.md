# Reasoning



## Takeaway

提升模型的复杂推理能力

1. **Self-Refine for Reasoning**

2. **Reinforcement Learning for Reasoning**

## Self-Refine

- Takeaway

  让模型生成答案后，再检查和修改自己的输出



## RL for Reasoning

把 reasoning 建模成 MDP，reward function 是核心。它指出，reasoning reward 不像游戏分数那样天然明确，需要 carefully engineered reward。常见 reward 包括最终答案正确性、step-wise correctness、自一致性、偏好模型评分等。



### GRPO

PPO变种，introduced by DeepSeek-R1-Zero