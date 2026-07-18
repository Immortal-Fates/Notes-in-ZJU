# Fine Tuning

SFT、instruction tuning、prompt/prefix tuning、RFT

## Takeaway

用有标签数据或任务数据调整参数，最基础的一类方法

分为三类

## SFT

Supervised Fine-Tuning，SFT

- Takeaway: 让模型学会instruction following

- Motivation: 不会常见的任务格式（问答，翻译等），只会补全续写

- Core Mechanism

  给定一个输入 prompt 和一个标准答案 response，让模型学习在这个 prompt 后生成这个 response

  Loss: CE
  $$
  \mathcal{L}
  =
  -\frac{1}{|\mathcal{A}|}
  \sum_{t \in \mathcal{A}}
  \log p_\theta(y_t \mid y_{<t})
  $$
  

- Pipeline

  1. Data

     SFT 数据一般是 instruction-response 格式。SFT 对数据噪声很敏感，高质量小数据集往往比低质量大数据集更有效

     有不同的format格式，常见多轮对话形式：

     ```
     [
       {"role": "user", "content": "什么是 LoRA？"},
       {"role": "assistant", "content": "LoRA 是一种参数高效微调方法……"},
       {"role": "user", "content": "它为什么省显存？"},
       {"role": "assistant", "content": "因为它冻结原模型参数，只训练低秩矩阵……"}
     ]
     ```

     不同的模型要转为不同的token序列：使用chat template

     ```
     tokenizer.apply_chat_template(messages, tokenize=False)
     ```

- Cons: 

  - 纯监督数据的质量和覆盖范围有限，而且难以捕捉更细微的人类偏好
  - catastrophic forgetting通用能力下降


两种训练方法

1. Full Fine-Tuning: 更新模型全部参数
2. Parameter-Efficient Fine-Tuning: 只训练一小部分参数eg:LoRA

### Parameter-Efficient Fine-Tuning

PEFT

#### LoRA

- Takeaway: 冻结原模型参数，只在部分线性层旁边并联可训练的低秩矩阵

- trainable rank decomposition matrices

- 

- Core Mechanism

  冻结原始权重 $W_0$，在部分线性层旁边并联一个 low-rank update：

  $$
  W = W_0 + \frac{\alpha}{r} \Delta W
  $$
  其中：

  $$
  \Delta W = BA \\
  B \in \mathbb{R}^{d_{\text{out}} \times r}, \quad A \in \mathbb{R}^{r \times d_{\text{in}}}, \quad r \ll \min(d_{\text{in}}, d_{\text{out}})
  $$
  所以原本 full fine-tuning 要训练：$$d_{\text{out}} \times d_{\text{in}}$$

  LoRA 只训练：$$r(d_{\text{out}} + d_{\text{in}})$$

  训练完成后，可以把 LoRA 分支合并进原权重：

  $$
  W_{\text{merged}} = W_0 + \frac{\alpha}{r}BA
  $$

- Pipeline

  ```
  target_modules = [
      "q_proj", "k_proj", "v_proj", "o_proj",
      "gate_proj", "up_proj", "down_proj"
  ]
  ```

  

#### QLoRA

- https://arxiv.org/abs/2305.14314?utm_source=chatgpt.com

- Takeaway: quantized base model + LoRA adapters



#### AdaLoRA

- Takeaway

- Motivation: LoRA 的一个问题是：所有层都给相同 rank $r$。但不同层的重要性不同

- Core Mechanism

  根据重要性adaptive rank allocation

#### IA^3

Infused Adapter by Inhibiting and Amplifying Inner Activations

- Takeaway: 学习activation scaling vectors

- Core Mechanism

  形式：

  $$
  h' = l \odot h
  $$
  其中 $l$ 是可训练 scaling vector，$\odot$ 是 element-wise product。

  它通常作用在 attention / FFN 的中间激活上



| 方法             | 训练参数 | 表达能力 | 显存 | 推理延迟              | 工程推荐            |
| ---------------- | -------- | -------- | ---- | --------------------- | ------------------- |
| Full Fine-Tuning | 最高     | 最强     | 最高 | 无额外延迟            | 算力充足时          |
| LoRA             | 低       | 强       | 低   | 可 merge，无额外延迟  | 默认首选            |
| QLoRA            | 低       | 强       | 很低 | 量化推理影响速度/精度 | 单卡/小显存         |
| AdaLoRA          | 低       | 强       | 低   | 可 merge              | 低 budget 研究      |
| IA³              | 极低     | 中       | 极低 | 很低                  | few-shot / baseline |
| Prefix Tuning    | 低       | 中       | 低   | KV cache 增加         | NLG 任务            |
| Prompt Tuning    | 极低     | 中低     | 极低 | 占 prompt length      | 大模型简单任务      |
| Adapter          | 中       | 中强     | 中   | 有额外模块延迟        | 多任务模块化        |

### Full Fine-Tuning



## AFT

Adaptive Fine-Tuning

- Takeaway: 更贴近真实应用的一类适配策略

   包括 instruction tuning、prefix-tuning、prompt-tuning 等。

- Core Mechanism

   常组合使用多种tricks
   $$
   \min_\theta
   \underbrace{\mathcal{L}_{\text{target}}}_{\text{适配目标任务}}
   +
   \lambda
   \underbrace{\mathcal{L}_{\text{preserve}}}_{\text{保留通用能力}}
   +
   \beta
   \underbrace{D_{\text{KL}}(\pi_\theta \Vert \pi_{\text{ref}})}_{\text{防止模型漂移}}
   $$

## RFT

Reinforcement Fine-Tuning，RFT

- Takeaway: 用 reward signal 直接优化模型行为
- Cons:
  - 训练不稳定
  - reward hacking

见后面的md

## Relation

| 维度                 | SFT                        | AFT                       | RFT                            |
| -------------------- | -------------------------- | ------------------------- | ------------------------------ |
| 核心目标             | 学会 instruction following | 适配 domain/task/context  | 最大化 reward                  |
| 数据                 | instruction-response       | domain/task/feedback data | prompts + reward signal        |
| 监督信号             | reference answer           | target distribution       | reward / preference / verifier |
| 优化方式             | CE loss                    | CE + KL + replay + PEFT   | policy gradient / PPO / GRPO   |
| 是否需要模型采样     | 不需要                     | 通常不需要                | 需要                           |
| 是否能超越 reference | 较难                       | 取决于数据                | 有可能                         |
| 主要风险             | 模仿错误答案               | forgetting / overfit      | reward hacking / instability   |





## References

