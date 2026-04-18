---
title: 00-OD-Trends
date: 2026-03-02
tags:
course: AI
status: draft
---
# Trends
[TOC]

介绍目标检测的未来大趋势，现在还啥都不懂，等我再多看点

1. kaiming引领的unsupervised learning，妥妥撸起袖子干一个检测友好的unsupervised pretrain model especially for object detection
2. FAIR最近火爆的DETR，其实去掉NMS这个事情今年也一直在弄，搞的思路一直不太对，也没搞出啥名堂，还是DETR花500个epoch引领了一下这个潮流，指了个门道，当然方向有了，具体走成啥样，还是八仙过海，各显神通啦

## 类别

### one stage vs two stage

check [here](../01-Object-Detection-Basics.md##One-Stage Object Detectors)

### anchor free vs anchor base

- anchor free: predict boxes directly without predefined anchors.

  只是指不需要人为预先设定anchor，但实际上还是有anchor的

- anchor base: 人为预先设定

### dense prediction vs sparse prediction

- dense detection head: 在每个grid都进行预测

  - Pros

    - recall 高

    - 能覆盖整个图像

    - 小目标检测能力强

  - Cons

    - 产生大量冗余候选框，需要 **NMS**

      > [!NOTE]
      >
      > NMS 的问题：
      >
      > 1. 不是 end-to-end
      > 2. 破坏梯度传播
      > 3. 推理增加 latency
      > 4. 对 crowded scenes 不稳定
      >
      > 所以也有很多论文在做NMS trick engineering

    - label assignment是核心难点、

    - 优化目标常常不一致

      > [!NOTE]
      >
      > Dense detector 的 loss 通常是：
      > $$
      > L = L_{cls} + L_{reg}
      > $$
      > 例如 RetinaNet：
      > $$
      > L = L_{focal} + L_{box}
      > $$
      > 但问题是：
      >
      > ```
      > training objective ≠ evaluation metric
      > ```
      >
      > 评估指标是：AP
      >
      > 而 AP 依赖
      >
      > ```
      > ranking + NMS
      > ```
      >
      > 这导致训练优化方向和最终 evaluation 不完全一致。


- Sparse Detection Head(不需要nms) : 这些模型通过 **query / proposal** 直接预测目标

  - Pros

    - 不需要 NMS
    - 预测结果更干净
    - end-to-end 训练

  - Cons

    - recall 可能较低
    - 训练难度较高
    - 小目标 detection 有时困难

  | 模型            | 类型   |
  | --------------- | ------ |
  | DETR            | sparse |
  | Deformable DETR | sparse |
  | DINO            | sparse |
  | Sparse R-CNN    | sparse |

#### Q&A

那么为什么使用transfomer可以来解决dense prediction的问题，实现sparse呢？

> [!NOTE]
>
> Sparse detection 最大的困难：目标之间的竞争
>
> 目标检测不是独立预测问题，而是 **集合预测问题**。
>
> 假设图像中有 $M$ 个目标：
> $$
> \{y_1, y_2, ..., y_M\}
> $$
> 模型预测 $K$ 个候选：
> $$
> \{\hat y_1, \hat y_2, ..., \hat y_K\}
> $$
> 要求：
> $$
> K \ge M
> $$
> 问题是：哪些 prediction 对应哪些目标
>
> 如果没有机制协调预测，就会出现：多个 prediction 预测同一个目标，这就是 **duplicate boxes** 问题。
>
> Dense detector 用NMS解决。
>
> Sparse detector需要一种 **预测之间互相协调的机制**
>
> 现在就回到原始的问题，为什么CNN不行而transfomer可以
>
> - CNN不行是因为每个位置的预测是互相独立的，
>   $$
>   P(y_i∣x)
>   $$
>   但目标检测需要：
>   $$
>   P(y_i | x, y_{-i})
>   $$
>   也就是：一个预测必须知道其他预测在干什么
>
> - transfomer的self-attention解决了这个问题
>
>   ```
>   Q = query
>   K = key
>   V = value
>   ```
>
>   在 detection 中：
>
>   ```
>   query = object queries
>   ```
>
>   因此每个 query 都可以看到所有其他 query。这样就可以避免duplicate boxes，下面举个例子
>
>   ```
>   query 1 预测 dog
>   query 2 看到 query 1
>   query 2 会避免再预测 dog
>   ```
>
>   这就是 implicit NMS
>
> - Cross attention找到目标
>
>   Decoder 还有 **cross-attention**：
>   $$
>   \text{Attention}(Q_{obj}, K_{img}, V_{img})
>   $$
>   其中：
>
>   ```
>   Q = object query
>   K,V = image feature
>   ```
>
>   含义是：
>
>   ```
>   query 在 feature map 中寻找目标
>   ```
>
>   于是 detection 过程变成：
>
>   ```
>   query → search object
>   ```
>
>   而不是：
>
>   ```
>   pixel → predict object
>   ```
>
>   这正是 **dense → sparse** 的转变。
