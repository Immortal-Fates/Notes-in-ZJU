# PostTraining Intro

后训练介绍

## Takeaway

调整模型行为

## Motivation

1. 模型很强，但是不听话且不会干活
2. 不安全
3. 不符合特定任务
4. 不会长思考

## Core Mechanism

| 模块            | 数据形式            | 训练目标                   | 主要解决什么                           |
| --------------- | ------------------- | -------------------------- | -------------------------------------- |
| SFT             | 指令和标准回答      | 模仿高质量答案             | 会听指令，会按格式回答                 |
| Reward Model    | chosen / rejected   | 学习偏好分数               | 把人类偏好转成奖励                     |
| RLHF / PPO      | prompt + reward     | 最大化 reward              | 提升 helpfulness、safety、偏好一致性   |
| DPO             | chosen / rejected   | 直接偏好优化               | 简化 RLHF，降低训练复杂度              |
| RLAIF           | AI 反馈             | 用 AI 偏好替代部分人工偏好 | 降低标注成本                           |
| RLVR            | 可验证任务          | 用规则奖励训练推理         | 强化数学、代码、逻辑能力               |
| GRPO            | 同题多回答 + reward | group 内相对优化           | 训练 reasoning model，减少 critic 成本 |
| Tool-use SFT/RL | 工具调用轨迹        | 学会使用工具               | Agent、RAG、函数调用                   |
| Safety tuning   | 安全数据、拒答数据  | 控制危险输出               | 安全、合规、稳健                       |

> [!TIP]
>
> post-training值一整个训练阶段，fine-tuning也是post-training的一部分

## Map

1. 早期的任务适配
2. 发展到指令跟随和人类偏好对齐
3. 再发展到 reasoning RL、效率优化、多模态和领域适配



## References

- https://aclanthology.org/2025.acl-long.140/?utm_source=chatgpt.com
- __A Survey on Post-training of Large Language Models.__ *Guiyao Tie et al.* __arXiv, 2025__ [(Arxiv)](https://arxiv.org/abs/2503.06072) 