# LLM Basic Concepts and Full Pipeline

- 本章目标：画一张大模型总图，包含数据、tokenizer、embedding、attention、FFN、训练目标、推理、微调、评估。跑通一个最小文本生成脚本。写一页笔记，总结大模型训练与使用流程。

## Basic Concepts

- natural language processing (NLP) is still challenging.
- LLM
  - characters
    - **Scale**: They contain millions, billions, or even hundreds of billions of parameters
    - **General capabilities**: They can perform multiple tasks without task-specific training
    - **In-context learning**: They can learn from examples provided in the prompt
    - **Emergent abilities**: As these models grow in size, they demonstrate capabilities that weren’t explicitly programmed or anticipated
  - limitations
    - **Hallucinations**: They can generate incorrect information confidently
    - **Lack of true understanding**: They lack true understanding of the world and operate purely on statistical patterns
    - **Bias**: They may reproduce biases present in their training data or inputs.
    - **Context windows**: They have limited context windows (though this is improving)
    - **Computational resources**: They require significant computational resources

## Full Pipeline

```mermaid
flowchart TD

    A[数据] --> B[Tokenizer]
    B --> C[Embedding]
    C --> D[Transformer主干]
    D --> D1[Attention]
    D --> D2[FFN]
    D1 --> E[训练目标]
    D2 --> E
    E --> F[预训练模型]
    F --> G[微调与对齐]
    G --> H[推理生成]
    H --> I[评估]
    I --> J[部署与应用]
```

下面我们来分别看每一步部分的大概实现

### 数据

```mermaid
flowchart TD

    A[原始语料] --> B[采集]
    B --> C[清洗]
    C --> D[去重]
    D --> E[质量过滤]
    E --> F[训练集验证集测试集划分]
    F --> G[可用于训练的文本数据]
```

### Tokenizer

```mermaid
flowchart TD

    A[文本数据] --> B[学习词表]
    B --> C[确定切分规则]
    C --> D[文本切成token]
    D --> E[token转成id]
    E --> F[模型可读取的离散序列]
```

### 输入表达

```mermaid
flowchart TD

    A[token id序列] --> B[Token Embedding]
    A --> C[位置编码]
    B --> D[向量表示]
    C --> D
    D --> E[送入Transformer]
```

### Transformer 内部模块

```mermaid
flowchart TD

    A[输入向量] --> B[多头自注意力]
    B --> C[残差连接与归一化]
    C --> D[前馈网络 FFN]
    D --> E[残差连接与归一化]
    E --> F[输出向量]
```

他们分别的作用

```mermaid
flowchart LR

    A[Attention] --> A1[看上下文]
    A --> A2[建立token之间关系]
    A --> A3[信息聚合]

    B[FFN] --> B1[逐位置非线性变换]
    B --> B2[增强特征表达]
    B --> B3[提升表示能力]
```

### training

```mermaid
flowchart TD

    A[输入token序列] --> B[模型前向计算]
    B --> C[预测下一个token]
    C --> D[计算loss]
    D --> E[反向传播]
    E --> F[参数更新]
    F --> G[模型能力提升]
```

### 微调与对齐模块

```mermaid
flowchart TD

    A[预训练模型] --> B[监督微调]
    B --> C[学会按指令回答]
    C --> D[偏好对齐]
    D --> E[输出更符合人类偏好]
    C --> F[领域微调]
    F --> G[适应垂直任务]
    C --> H[参数高效微调]
    H --> I[低成本适配]
```

### 推理模块

```mermaid
flowchart TD

    A[用户输入] --> B[Tokenizer编码]
    B --> C[Embedding]
    C --> D[Transformer前向计算]
    D --> E[预测下一个token]
    E --> F[解码策略选择]
    F --> G[生成一个token]
    G --> H[拼接回输入]
    H --> D
    H --> I[直到结束标记]
    I --> J[输出文本]
```

解码策略

```mermaid
flowchart LR

    A[下一个token概率] --> B[Greedy]
    A --> C[Top k]
    A --> D[Top p]
    A --> E[Temperature]
    B --> F[最终输出风格]
    C --> F
    D --> F
    E --> F
```

### 评估与应用模块

```mermaid
flowchart TD

    A[模型输出] --> B[能力评估]
    B --> C[通用基准]
    B --> D[任务基准]
    B --> E[人工评测]
    B --> F[安全性评测]
    B --> G[事实性评测]
    B --> H[效率评测]
    H --> I[部署与应用]
    G --> I
```

## References

- Hugging Face LLM Course 第 1 章 ([Hugging Face](https://huggingface.co/learn/llm-course/en/chapter1/1?utm_source=chatgpt.com))
- Stanford CS336 课程主页 ([Stanford CS336](https://cs336.stanford.edu/?utm_source=chatgpt.com))
- Attention Is All You Need 论文 ([arXiv](https://arxiv.org/abs/1706.03762?utm_source=chatgpt.com))
- Transformers Quickstart 官方文档 ([Hugging Face](https://huggingface.co/docs/transformers/en/quicktour?utm_source=chatgpt.com))