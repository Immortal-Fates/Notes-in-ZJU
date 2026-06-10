---
title: 01-Basic-Model-Zoo
date: 2026-03-02
tags:
course: AI
status: draft
---
# Basic Model Zoo
[TOC]

## CNN Zoo

- **Deep learning**. LeCun Yann et.al. **Nature**, **2015-5-27**, ([link](https://doi.org/10.1038/nature14539)).

  - 监督学习就是“定义可微损失 + 反向传播算梯度 +（小批量）SGD 优化 + 测试集检验泛化”

  - 高维非凸里，真正的绊脚石多是鞍点而非“坏极小值”，因此带噪声的小批量 SGD、本质上的随机性与动量等会有帮助穿过鞍点

    > 当然现在优化器有更多，例如AdamW,LAMB等

  - ReLU 在多层网络中通常学得更快，（直观理解）ReLU 是分段线性、计算简单；其“零–正”两段让很多激活在任一前向传播中为 0，形成**稀疏激活**

  - CNN:local connections, shared weights, pooling and the use of many layers

  - Distributed representations:特征由向量表示（每个维度代表一个微特征），彼此并不排斥，能因子化复杂的输入——输出关系

  - 到端的“CNN+RNN+强化学习”的主动感知/视觉决策

- **Gradient-based learning applied to document recognition**. Lecun Y. et.al. **Proc. IEEE**, **1998**, ([link](https://doi.org/10.1109/5.726791)). -- LeNet-5

  只在MNIST上训练（5w images and 10 classes）

- **ImageNet classification with deep convolutional neural networks**. Krizhevsky Alex et.al. **Commun. ACM**, **2017-5-24**, ([link](https://doi.org/10.1145/3065386)). -- AlexNet

  - Takeaway: scale up模型更大，数据更多。还有很多工程上的tricks
    ![alt text](assets/01-Basic-Model-Zoo.assets/image.png)
  - Core Mechanism
    - data scaling: ImageNet 1.28 million images and 1000 classes
    - model scaling: 60 million参数
    - reducing overfitting
      - 使用数据增广来获得更多训练样本，通过平移、灰度变换等增广方式来扩充数据，使网络适应更多情况
      - 对网络中间层加入Dropout， 即随机使部分神经元不工作
    - group conv
    - 采用ReLU函数来替代Sigmoid函数，降低了计算量的同时，还避免了极端输入导致的梯度消失
    - 使用动量参数和学习率降低策略来加速收敛， 每当学习陷入瓶颈时学习率就会降低（手动）
    - GPU training

- **Very Deep Convolutional Networks for Large-Scale Image Recognition**. Karen Simonyan et.al. **arxiv**, **2014**, ([Arxiv](https://arxiv.org/pdf/1409.4842)) ([link](http://arxiv.org/abs/1409.1556v6)). VGGNet

  - 不同于以往的大卷积核，此网络中卷积核尺寸均为3× 3，相对于更大的卷积核而言减少了参数， 使得网络的层数能够得到加深，这样也能更好地保留图像的特征
  - deeper is better

VGG是分阶段训练，先训练浅层的再添加更多的层数，然后微调。但这与我们end-to-end是违背的。既然知道了deeper is better，我们就需要深层的网络训练，但这是很困难的：

1. 反向传播会累计problem
2. 梯度爆炸和消失

我们需要好的初始化来解决这些问题。比如Xavier来使得VGG可以end-to-end地训练

- **Going deeper with convolutions**. Szegedy Christian et.al. **No journal**, **2015-6**, ([link](https://doi.org/10.1109/cvpr.2015.7298594)) -- GoogLeNet (Inception v1)

  - Core Mechanism

    ![image-20260516213206621](./assets/01-Basic-Model-Zoo.assets/image-20260516213206621.png)

    多分支结构，主要是1*1分支的使用

    打破了initialization assumptions，因此人们来时研究normalization归一化


### ResNet

- **Deep ResidualZoo Learning for Image Recognition**. He Kaiming et.al. **No journal**, **2016-6** ([link](https://doi.org/10.1109/cvpr.2016.90)).

  - Motivation

    在ResNet之前，无法做得更深，20layers就会退化，很难训练。这和常理不符，下面来一个思想训练

    ![image-20260516214614823](./assets/01-Basic-Model-Zoo.assets/image-20260516214614823.png)

    我们训练一个浅层，然后在其中添加一些extra layers使其变成深层，让extra layers为恒等映射，就可以得到深层网络至少和浅层网络有同样的训练精度。我们认为深层网络的表达能力更强

  - ResNet：从求导来看即使梯度小，因为是加法所以还是能进行训练的

    ![image-20251029133629385](assets/01-Basic-Model-Zoo.assets/image-20251029133629385.png)

    ![image-20251029134613108](assets/01-Basic-Model-Zoo.assets/image-20251029134613108.png)

  - bottleneck的设计：右边先降维，再升维，这样就能做得更深

    做得深，就可以使用更多的通道数（可看作特征），用更复杂的特征向量来表示

    ![image-20251029135705447](assets/01-Basic-Model-Zoo.assets/image-20251029135705447.png)

  > [!NOTE]
  >
  > 为什么ResNet改变了世界？
  > 因为有了残差连接之后，我们只需要关注每个block的设计就可以了，然后堆叠起来（用残差连接）


### ResNeXt

- grouped conv

### ShuffleNet

- 分组卷积 + 通道混洗

- **Densely Connected Convolutional Networks**. Huang Gao et.al. **No journal**, **2017-7** ([link](https://doi.org/10.1109/cvpr.2017.243)). -- DenseNet

- **Rethinking the Inception Architecture for Computer Vision**. Szegedy Christian et.al. **No journal**, **2016-6** ([link](https://doi.org/10.1109/cvpr.2016.308)). -- Inception-v3

- __Deformable Convolutional Networks.__ *Jifeng Dai et al.* __arXiv, 2017__ [(Arxiv)](https://arxiv.org/abs/1703.06211) 

  - Takeaway: 

    Deformable Convolutional Networks enhance standard convolution by learning spatial offsets for sampling locations, allowing the network to adapt its receptive field to object geometry. This significantly improves performance in tasks with geometric variations such as object detection and segmentation.

  - Prior: 双线性插值

    ![image-20260303105239600](assets/01-Basic-Model-Zoo.assets/image-20260303105239600.png)

    1. 四个点先对x方向进行插值，得到两个点P1,P2
    2. 再对P1,P2，对y方向进行插值

  - Core Mechanism: deformable convolution + deformable RoI pooling

    - deformable convolution

      DCN augments each sampling location with a learnable offset $\Delta p_n$:
      $$
      y(p_0) = \sum_{p_n \in \mathcal{R}} w(p_n)\, x(p_0 + p_n + \Delta p_n)
      $$
      Since $p=p_0 + p_n + \Delta p_n$ is generally fractional, bilinear interpolation is used:
      $$
      x(p) = \sum_{q} G(q, p)\, x(q)
      $$
      Where $G(q, p)$ is the bilinear interpolation kernel.

      $G$ 为二维，可拆为两个一维核之积
      $$
      G(q,p) = g(q_x,p_x)\, g(q_y,p_y)
      $$
      其中：
      $$
      g(a,b) = \max(0, 1 - |a-b|)
      $$
      This keeps the operation differentiable and trainable end-to-end. 用周围四个点双线性插值来代替这个可能不为整数的 $p$

      ![image-20260303103244326](assets/01-Basic-Model-Zoo.assets/image-20260303103244326.png)

      - 这里offset field的2N是因为有N个点，每个点有x,y两个坐标，所以一共2N个偏移的值
      - 根据我们输入的值决定的：也是一种self attention

    - deformable RoI pooling

      ![x7](assets/01-Basic-Model-Zoo.assets/x7.png)


- Deformable convolutional networks v2


## Attention Zoo

- __Neural Machine Translation by Jointly Learning to Align and Translate.__ *Dzmitry Bahdanau et al.* __CoRR, 2014__ [(Arxiv)](https://arxiv.org/abs/1409.0473) [(S2)](https://www.semanticscholar.org/paper/fa72afa9b2cbc8f0d7b05d52548906610ffbb9c5) (Citations __28426__)

  - Takeaway: This paper introduces **attention-based neural machine translation (NMT)**.

      ![encoder-decoder-attention](assets/01-Basic-Model-Zoo.assets/encoder-decoder-attention.png)

- **Attention Is All You Need**. Ashish Vaswani et.al. **arxiv**, **2017**, ([link](https://arxiv.org/abs/1706.03762v7))([details](../../05-Attention.md)).

  check LLM part

- __Transformers in Vision: A Survey.__ *Salman Hameed Khan et al.* __ACM Computing Surveys (CSUR), 2021__ [(Link)](https://doi.org/10.1145/3505244) [(S2)](https://www.semanticscholar.org/paper/3a906b77fa218adc171fecb28bb81c24c14dcc7b) (Citations __3005__)

- **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. Jacob Devlin et.al. **arxiv**, **2018**, ([link](http://arxiv.org/abs/1810.04805v2)).

## Relation

```mermaid
graph TD
  A[LeNet] -->|deeper/bigger| B[AlexNet]

  %% Left branch
  B -->|1x1 conv| C[NiN]
  C -->|"inception: split-transform-merge"| D[GoogLeNet]
  D -->|batch normalization| E[BN-Inception]
  E -->|"updated inception, label smooth"| F[InceptionV3]
  F -->|residual connection| G[Inception-ResNet]
  F -->|"inception to depthwise separable conv"| H[Xception]

  %% Right branch
  B -->|3x3 conv| I[VGG]
  I -->|"1x1 conv bottleneck"| J[SqueezeNet]
  J -->|"+ to concat"| K[DenseNet]
  I -->|residual| L[ResNet]
  L -->|grouped conv| M[ResNeXt]
  M -->|"shuffle channels among groups"| N[ShuffleNet]
  L -->|squeeze-excite| O[SENet]

```

## References

- [Great Intro to DL](https://www.bilibili.com/video/BV1Vw4m1d7pf/?share_source=copy_web&vd_source=93bb338120537438ee9180881deab9c1)
- [DERT blog](https://ai.meta.com/blog/end-to-end-object-detection-with-transformers/)
- [Great blog about attention family](https://lilianweng.github.io/posts/2018-06-24-attention/)

- [Vit blog](https://research.google/blog/transformers-for-image-recognition-at-scale/?m=1)
