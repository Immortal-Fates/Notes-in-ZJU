# Milestone

介绍一些LLM上的重要论文



## Zoo

- __Scaling Vision Transformers.__ *Xiaohua Zhai et al.* __arXiv, 2021__ [(Arxiv)](https://arxiv.org/abs/2106.04560) 

  - Takeaway

    *Scaling Vision Transformers* shows that Vision Transformers have strong and predictable scaling behavior: as model size, data size, and compute increase together, performance keeps improving. It also presents a refined ViT training and architecture recipe that enables a 2B-parameter model to reach **90.45% top-1 on ImageNet**.

  - Motivation

    Previous work had already shown that scale is a key driver of success for Transformers in language, but it was still unclear **how Vision Transformers scale** with respect to model size, dataset size, and compute. 

    The paper is motivated by the need to understand these scaling laws for vision, so that future ViT systems can be designed more systematically rather than by trial and error.

    Another motivation is practical: standard ViT training becomes increasingly expensive and memory-hungry at large scale. The authors therefore not only study scaling behavior, but also refine the architecture and training setup to reduce memory use and improve accuracy.

  - Core Mechanism

    The core idea is to study ViT through the lens of **scaling laws**. Instead of proposing a completely new backbone, the paper systematically scales three things:

    1. **Model size**
    2. **Training data size**
    3. **Training compute**

    The paper models the relationship between error and scale using a power-law style formulation. A simplified form is:

    $$
    \mathrm{Error}(N) \approx aN^{-\alpha} + b
    $$

    where \(N\) can represent a scaling variable such as model size, dataset size, or compute budget, \(a\) and \(b\) are constants, and \(\alpha\) is the scaling exponent. The key message is that ViT error decreases in a predictable way as scale increases. This power-law view is the central analytical tool of the paper.

    In addition, the authors refine the standard ViT recipe to make large-scale training feasible and more effective. The paper summary explicitly states that they refine both **architecture and training**, reducing memory consumption and increasing accuracy while scaling up to a **2B-parameter ViT**.

  - Pipeline

    1. Start from the Vision Transformer backbone and prepare variants at different scales. :contentReference[oaicite:6]{index=6}
    2. Scale the **model size** up and down across a broad range. :contentReference[oaicite:7]{index=7}
    3. Scale the **training dataset size** up and down to study its interaction with model size. :contentReference[oaicite:8]{index=8}
    4. Measure performance as a function of **training compute** and fit scaling trends. :contentReference[oaicite:9]{index=9}
    5. Refine the ViT architecture and training setup to improve memory efficiency and optimization at large scale. :contentReference[oaicite:10]{index=10}
    6. Train an extremely large ViT, then evaluate it on ImageNet and few-shot transfer tasks. The paper reports **90.45% top-1 on ImageNet** and **84.86% top-1 with only 10 examples per class** in few-shot transfer. :contentReference[oaicite:11]{index=11}

  - Math Formula

    A concise way to express the paper’s scaling-law viewpoint is:

    $$
    \mathcal{L}(x) = A x^{-\alpha} + B
    $$

    where:

    - \(\mathcal{L}(x)\) is the loss or error
    - \(x\) is a scale variable such as model size, dataset size, or compute
    - \(A\) and \(B\) are constants
    - \(\alpha\) is the scaling exponent

    This formula captures the main empirical observation of the paper: performance improves with scale in a regular, power-law-like manner. :contentReference[oaicite:12]{index=12}

    Since the paper studies Vision Transformers, the underlying self-attention block still follows the standard Transformer form:

    $$
    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
    $$

    and the overall scaling study examines how networks built from these blocks behave as their size and training resources grow. :contentReference[oaicite:13]{index=13}

  - Pros

    - It provides one of the clearest early demonstrations that **Vision Transformers obey useful scaling laws**, which gives researchers a principled way to think about future ViT design. :contentReference[oaicite:14]{index=14}
    - It shows that very large ViTs can achieve **state-of-the-art accuracy**, including **90.45% ImageNet top-1**. :contentReference[oaicite:15]{index=15}
    - It improves not only full-data classification, but also **few-shot transfer**, indicating strong representation quality at scale. :contentReference[oaicite:16]{index=16}
    - It does not just analyze scaling; it also refines training and architecture to reduce memory consumption and improve practical large-scale training. :contentReference[oaicite:17]{index=17}

  - Cons

    - The strongest results rely on **extreme scale**, including very large models and large compute budgets, so the recipe is not easily accessible to ordinary researchers or smaller labs. This is a reasonable inference from the paper’s reported 2B-parameter setting and large-scale study scope. :contentReference[oaicite:18]{index=18}
    - The paper is more about **scaling behavior and training recipe** than about introducing a fundamentally new architecture, so its novelty is less architectural than some other ViT papers. This is an interpretation based on the paper summary. :contentReference[oaicite:19]{index=19}
    - Because it emphasizes scaling, some conclusions are less directly useful when compute, data, or memory are limited. This is an inference from the paper’s large-scale experimental setup. :contentReference[oaicite:20]{index=20}

## Relation

```mermaid
graph TD
  A[Seq2Seq 2014] -->|soft alignment| B[Attention NMT 2014]
  B -->|remove recurrence, full attention| C[Transformer 2017]

  %% Encoder-oriented branch
  C -->|bidirectional pretraining| D[BERT 2018]
  D -->|unified text-to-text transfer| E[T5 2019]

  %% Decoder-only branch
  C -->|decoder-only LM| F[GPT 2018]
  F -->|scale parameters and data| G[GPT-2 2019]
  G -->|in-context learning at scale| H[GPT-3 2020]
  H -->|instruction tuning plus RLHF| I[InstructGPT 2022]
  I -->|chat alignment| J[ChatGPT 2022]

  %% Scaling and open-weight branch
  H -->|scaling law and dense training| K[PaLM 2022]
  H -->|efficient open-weight recipe| L[LLaMA 2023]
  L -->|instruction-tuned open models| M[LLaMA 2 2023]
  M -->|stronger data and post-training| N[LLaMA 3 2024]

  %% Mixture-of-experts / multimodal trend
  K -->|sparse MoE scaling| O[Mixtral 2024]
  J -->|multimodal extension| P[GPT-4V / GPT-4o]
```

- One useful way to read this trajectory is: **Attention -> Transformer -> foundation-model scaling -> alignment -> open-weight / multimodal expansion**.
