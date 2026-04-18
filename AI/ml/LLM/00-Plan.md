# Intro

开始大模型学习之旅



## 学习计划

| 日期 | 今日目标                                           | 学习资料 论文 / 官方资料                                     | 学习资料 代码                                                | 实践任务                                                     | 自检 / 复盘                                                  |
| ---- | -------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Day1 | 建立大模型全景图，知道 LLM 的整体链路是什么        | Hugging Face LLM Course 第 1 章 ([Hugging Face](https://huggingface.co/learn/llm-course/en/chapter1/1?utm_source=chatgpt.com))Stanford CS336 课程主页 ([Stanford CS336](https://cs336.stanford.edu/?utm_source=chatgpt.com))Attention Is All You Need 论文 ([arXiv](https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com))Transformers Quickstart 官方文档 ([Hugging Face](https://huggingface.co/docs/transformers/en/quicktour?utm_source=chatgpt.com)) | [2025科协暑期基础技能培训——大模型基础](https://www.bilibili.com/video/BV1gs8hzaEoL/?share_source=copy_web&vd_source=93bb338120537438ee9180881deab9c1)<br />huggingface/transformers 仓库 ([GitHub](https://github.com/huggingface/transformers?utm_source=chatgpt.com)) | 画一张大模型总图，包含数据、tokenizer、embedding、attention、FFN、训练目标、推理、微调、评估。跑通一个最小文本生成脚本。写一页笔记，总结大模型训练与使用流程。 | 什么是 next token prediction。Transformer 为什么比 RNN 更适合并行。tokenizer、model、decoder 各自做什么。预训练与微调有什么区别。不看资料能否口述一遍文本生成流程。 |
| Day2 | 搞懂 Transformer 核心结构，重点吃透 attention      | Attention Is All You Need 论文 ([arXiv](https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com))PyTorch Transformer 官方文档 ([PyTorch 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html?utm_source=chatgpt.com)) | karpathy 做了 minGPT 这个教学型仓库，适合逐行读 GPT 主干代码 ([GitHub](https://github.com/karpathy/minGPT?utm_source=chatgpt.com)) | 手写单头 self attention 前向过程。在 minGPT 中找到 token embedding、position embedding、attention block、MLP、lm head。画一张更细的 Transformer 数据流图。 | Q、K、V 分别是什么。attention score 为什么除以根号下维度。多头注意力解决了什么问题。FFN 起什么作用。causal mask 为什么对 GPT 必需。 |
| Day3 | 搞懂 tokenizer 和训练数据，理解文本如何进入模型    | SentencePiece 论文 ([arXiv](https://arxiv.org/abs/1808.06226?utm_source=chatgpt.com))Hugging Face LLM Course 中从零训练 causal language model 的章节 ([Hugging Face](https://huggingface.co/learn/llm-course/en/chapter7/6?utm_source=chatgpt.com)) | google 做了 SentencePiece 官方实现 ([GitHub](https://github.com/google/sentencepiece?utm_source=chatgpt.com))OpenAI 做了 tiktoken 官方仓库 ([GitHub](https://github.com/openai/tiktoken?utm_source=chatgpt.com)) | 用自己的小语料训练一个 SentencePiece tokenizer。试 3 组不同词表大小。用 tiktoken 切同一段文本，比较 token 数量。记录词表大小变化对序列长度和切分效果的影响。 | 为什么不用按整词切分。BPE 和 unigram 的直觉差别是什么。为什么不同 tokenizer 会导致 token 数不同。词表过大和过小各有什么代价。训练语料进入模型前至少经过哪些步骤。 |
| Day4 | 从零训练一个极小语言模型，把结构、数据、训练串起来 | Language Models are Few-Shot Learners 论文 ([arXiv](https://arxiv.org/abs/2005.14165?utm_source=chatgpt.com))Training Compute-Optimal Large Language Models 论文 ([arXiv](https://arxiv.org/abs/2203.15556?utm_source=chatgpt.com))Hugging Face Causal Language Modeling 官方文档 ([Hugging Face](https://huggingface.co/docs/transformers/en/tasks/language_modeling?utm_source=chatgpt.com)) | karpathy 做了 nanoGPT，适合跑最小训练实验 ([GitHub](https://github.com/karpathy/nanogpt?utm_source=chatgpt.com)) | 用 nanoGPT 在小语料上训练 toy model。记录词表大小、层数、hidden size、context length、batch size、learning rate。观察 train loss 与 val loss。定期采样生成文本。写一页训练复盘。 | train loss 下降说明什么。val loss 为什么更重要。context length 变大会带来什么影响。模型参数量与训练 token 为什么要一起考虑。如果生成效果差，优先从数据、模型、训练哪一侧排查。 |
| Day5 | 理解推理、解码与加速，知道模型回答时发生了什么     | Hugging Face Generation Strategies 文档 ([Hugging Face](https://huggingface.co/docs/transformers/en/generation_strategies?utm_source=chatgpt.com))Hugging Face KV Cache 文档 ([Hugging Face](https://huggingface.co/docs/transformers/cache_explanation?utm_source=chatgpt.com))vLLM 官方文档 ([vLLM](https://docs.vllm.ai/en/latest/?utm_source=chatgpt.com))FlashAttention 论文 ([arXiv](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)) | vllm-project/vllm 仓库 ([GitHub](https://github.com/vllm-project/vllm?utm_source=chatgpt.com))ggml-org/llama.cpp 仓库 ([GitHub](https://github.com/ggml-org/llama.cpp?utm_source=chatgpt.com)) | 用同一个模型比较 greedy 和 sampling 输出差异。调 temperature 和 top p，观察稳定性与多样性变化。本地跑通一条推理链路，机器弱就优先试 llama.cpp。记录输入长度变化对推理速度和显存的影响。 | 为什么推理阶段特别依赖缓存。KV cache 复用了什么。greedy 和 sampling 适合什么任务。推理框架和模型本体有什么区别。FlashAttention 解决的本质瓶颈是什么。 |
| Day6 | 学习微调，重点掌握 LoRA 和 QLoRA                   | LoRA 论文 ([arXiv](https://arxiv.org/abs/2106.09685?utm_source=chatgpt.com))QLoRA 论文 ([arXiv](https://arxiv.org/abs/2305.14314?utm_source=chatgpt.com))Hugging Face PEFT Quicktour ([Hugging Face](https://huggingface.co/docs/peft/quicktour?utm_source=chatgpt.com))Hugging Face PEFT Quantization 文档 ([Hugging Face](https://huggingface.co/docs/peft/developer_guides/quantization?utm_source=chatgpt.com))TRL 的 SFT Trainer 文档 ([Hugging Face](https://huggingface.co/docs/trl/sft_trainer?utm_source=chatgpt.com)) | huggingface/peft 仓库 ([GitHub](https://github.com/huggingface/peft?utm_source=chatgpt.com))huggingface/trl 仓库 ([GitHub](https://github.com/huggingface/trl?utm_source=chatgpt.com)) | 找一个小型 instruction 数据集。选一个小模型做一次 LoRA 微调。保存 adapter 权重。对比微调前后回答差异。条件允许就试量化加载加 PEFT。 | 为什么全量微调成本高。LoRA 为什么能减少训练参数。adapter 权重和基础模型权重是什么关系。QLoRA 比 LoRA 多解决了什么问题。监督微调本质上改变了模型什么能力。 |
| Day7 | 理解对齐和评估，形成完整闭环                       | InstructGPT 论文 ([arXiv](https://arxiv.org/abs/2203.02155?utm_source=chatgpt.com))TRL 官方文档 ([Hugging Face](https://huggingface.co/docs/trl/en/index?utm_source=chatgpt.com))Stanford CS336 课程主页，回看整体框架 ([Stanford CS336](https://cs336.stanford.edu/?utm_source=chatgpt.com)) | EleutherAI 做了 lm-evaluation-harness 统一评测框架 ([GitHub](https://github.com/EleutherAI/lm-evaluation-harness?utm_source=chatgpt.com)) | 给本周训练或微调过的模型设计一套 20 条以上的小评测集。从是否答非所问、是否重复、是否事实捏造、是否风格稳定四个维度评测。条件允许就接入 lm-evaluation-harness。写 2 到 3 页总结，题目可设为我如何理解大模型从数据到对齐的全流程。 | 监督微调和对齐是不是一回事。为什么不能只看 loss。为什么会答题不等于好用。人类偏好数据在对齐中扮演什么角色。现在能否完整讲出数据到 tokenizer 到预训练到推理到微调到对齐到评估这条链路。 |

## 知识库

【要成为大模型算法工程师，至少应该掌握哪些内容？来自一线算法工程师的建议】 https://www.bilibili.com/video/BV1ix6UBcEp2/?share_source=copy_web&vd_source=93bb338120537438ee9180881deab9c1

成为大模型算法工程师，最少要什么：

1. 数学 done

   大学三件套走天下了

2. 深度学习 done

   梯度下降，loss function, dropout/BN, resnet, adam, LR, CNN(RNN已经被淘汰了)

3. 大模型基础 done

   - transformer: qkv动手啃【强烈推荐新人入门LLM的方法——逐行调试一个小模型】 https://www.bilibili.com/video/BV1BNy9BREno/?share_source=copy_web&vd_source=93bb338120537438ee9180881deab9c1

     两条线

     - bert: embedding
     - gpt: 

   - 生态：**huggingface**, transformers库

   - SFT, LoRA, QLoRA: DeepSpeed, 混合精度， 显存与规模的估算（如何设置batch size，要不要梯度累计，自己手动推导）

     - 对齐：RL(DPO,PPO,GRPO)

   - 量化：kvcache, 怎么部署

   - 测评：各种任务的指标

   - 顶会论文

4. 计算机基础 done

5. 数据工程：算法工程师，90%时间都是在跟数据打交道



《大白话概率论讲义：从数学直觉到AI视野》

《一个大模型算法工程师的诞生》



什么是attention???

cross-attention: 建立两个语言之间的桥梁

多头注意力：给attention加上卷积

transformer:全局建模能力很强。小镇做题家模型，需要大量数据

scaling law

SFT:监督式微调（不新，所以我们要用强化学习）

强化学习：chatgpt先训练了一个打分器

RLHF诞生了

图文对齐：CLIP

当预训练把不管什么智慧敲进KV矩阵后，另外一个神经网络就有信息把它提取出来（生成Q，然后cross-attention）

未来：SFT+RHLF

空间感知智能





