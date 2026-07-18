# Alignment

RLHF、RLAIF、DPO

## Takeaway

让模型输出更符合人类偏好



- **PPO** 是传统 RLHF 常用的 policy optimization 方法。
 **RLHF** 用人类反馈训练 reward model，再优化 policy。
  **DPO** 直接用偏好对优化模型，绕过显式 reward model。
  **GRPO** 用同一个问题的一组回答做相对比较，减少 critic 依赖。

## RLHF



- Core Mechnism

  RLHF 通过偏好标签或奖励信号训练 reward model，再用它调整 policy

  > [!Note]
  >
  > Q: 
  >
  > 人类反馈是什么形式？
  >
  > reward model 如何训练？
  >
  > policy learning 如何避免模型跑偏？

```
收集人类偏好
    ↓
训练 reward model
    ↓
用 reward model 指导 policy learning
    ↓
得到更符合人类偏好的模型
```



## RLAIF

RLAIF 是 Reinforcement Learning from AI Feedback。
它用 AI 反馈替代部分人类反馈

## DPO

直接使用 chosen / rejected pair 做优化，而不显式训练 reward model

```
同一个 prompt 下
提高 chosen response 的相对概率
降低 rejected response 的相对概率
```

- Pros
  - to RLHF: 训练更简单
  - to SFT: 可以表达哪个回答更好



DPO、Iterative DPO、CPO、SimPO、TDPO 等一系列 alignment 方法