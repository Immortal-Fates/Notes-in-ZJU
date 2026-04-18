---
title: 02-OD-Model-Zoo
date: 2026-03-02
tags:
course: AI
status: draft
---
# Object Detection Model Zoo

Focus on object detection models

[TOC]

## Model Zoo

### MobileNet Zoo

- **MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications**. Andrew G. Howard et.al. **arxiv**, **2017**, ([link](https://arxiv.org/abs/1704.04861v1)).

  - Takeaway: MobileNet v1 replaces standard convs with **depthwise + pointwise** convs (DSC) to cut FLOPs/params for mobile.

  - Motivation: Classical CNNs (VGG, ResNet) use full 3×3 convolutions with cost:

    - Computation: `H × W × Cin × Cout × K²`
    - Parameters: `Cin × Cout × K²`

    This is too heavy for phones and embedded devices.

  - Core Mechanism: Depthwise Separable Convolutions(DSC): Replace a standard `K×K` conv with:

    - **Depthwise conv**: `K×K` per input channel, no channel mixing
    - **Pointwise conv**: `1×1` conv to mix channels

    <img src="assets/02-OD-Model-Zoo.assets/image-20251119220902654.png" alt="image-20251119220902654" style="zoom:50%;" />

    This factorizes computation:

    - Original MACs: `H × W × Cin × Cout × K²`
    - DSC MACs: `H × W × (Cin × K² + Cin × Cout)`

    For typical settings (e.g., `K=3`, `Cin ~ Cout`), this greatly reduces compute. Additionally, MobileNet v1 introduces:

    - width multiplier $\alpha$: thinner models, controls the number of channels in each layer

    - resolution multiplier $\rho$: reduce the computational cost, controls the input image resolution
      $$
      D_K \times D_K \times \alpha M \times \rho D_F \times \rho D_F+\alpha M \times \alpha N \times  \rho D_F \times \rho D_F
      $$

    Together, they form a simple knob set for accuracy–efficiency trade-offs.

    > [!TIP]
    >
    > Less regularization and data augmentation techniques because **small models have less trouble with overfitting**.

  - Pros

    - Massive reduction in FLOPs and parameters vs full convs.

  - Cons

    - Pure DSC networks can be **memory-bound** (low compute/memory ratio).
    - Representational power is weaker than advanced backbones (e.g., ResNet, MobileNet v2/v3).
    - No explicit mechanism to handle **information loss** in low-dimensional bottlenecks.

- **MobileNetV2: Inverted Residuals and Linear Bottlenecks**. Mark Sandler et.al. **arxiv**, **2018**, ([link](https://arxiv.org/abs/1801.04381v4)).

  - Takeaway: MobileNet v2 introduces the **Inverted Residual + Linear Bottleneck** block, improving over v1.

  - Motivation: MobileNet v1’s DSC is efficient but suffers from information loss in narrow intermediate layers. And there`s no residual connections in most layers.

  - Core Mechanism:

    a novel layer module: the inverted residual with linear bottleneck

    1. **Expansion**:
       The input is first passed through a **1×1 convolution** that **expands** the number of channels, increasing the model’s representational capacity.
    2. **Depthwise Separable Convolution**:
       After the expansion, a **depthwise separable convolution** (which is more computationally efficient than a regular convolution) is applied. This operation is performed on each channel separately, rather than combining all channels together, reducing the number of parameters and computation.
    3. **Projection (Linear Bottleneck)**:
       The **output** of the depthwise separable convolution is then passed through a **1×1 convolution** that **projects** the feature map back to a smaller number of channels.

    ![image-20251119220845255](assets/02-OD-Model-Zoo.assets/image-20251119220845255.png)

    > [!NOTE]
    >
    > ReLU6: $y = min(max(x,0),6)$, cut the value in 6

    Key tricks:

    - linear bottlenecks

      No non-linearity (e.g., ReLU) after the final projection to avoid losing information in low-dimensional space (maintain representational power).

    - Inverted residuals

      - Normal bottleneck use $1\times 1$ layers to reduce and then increase(restore) dimensions, which helps in reducing computational cost. However, this comes with a trade-off, as you need to balance the reduction in dimensions with the need to preserve sufficient information.
      - Inverted residuals use $1\times 1$ layers to increase and then reduce dimensions

  - Pros

    - Remains highly efficient and widely adopted in detection/segmentation backbones.

  - Cons

    - Still heavily reliant on depthwise conv (memory-bound).
    - This design is not memory-efficient for both inference and training.

- __Searching for MobileNetV3.__ *Andrew G. Howard et al.* __2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019__ [(Arxiv)](https://arxiv.org/abs/1905.02244) [(S2)](https://www.semanticscholar.org/paper/5e19eba1e6644f7c83f607383d256deea71f87ae) ([code_link](https://github.com/d-li14/mobilenetv3.pytorch))(Citations __8097__)

  - Takeaway: MobileNet v3 combines（有点缝合怪的感觉）:

    - **Inverted residual blocks from v2**,
    - **Squeeze-and-Excitation (SE)** for channel attention,
    - **NAS-based layer configuration** and customized **nonlinearities** (h-swish, h-sigmoid), to push mobile efficiency further while keeping FLOPs low.

  - Motivation: Google’s work on NAS and SE (SENet, EfficientNet) motivates an automated and more refined design.

  - Core Mechanism:

    MobilenetV3 block: MobileNetV2 + Squeeze-and-Excite

    ![image-20251201002400620](assets/02-OD-Model-Zoo.assets/image-20251201002400620.png)

    Key tricks

    - Introduces **h-swish** (hard-swish) and **h-sigmoid** as cheap approximations of swish/sigmoid, better suited for mobile hardware.
      $$
      \text{hsigmoid}(x) = \frac{ReLU6(x+3)}{6},\quad \text{h-swish}(x) = x\cdot \frac{ReLU6(x+3)}{6}
      $$
      ![image-20251201004538749](assets/02-OD-Model-Zoo.assets/image-20251201004538749.png)

    - Overall architecture (widths, kernel sizes, presence of SE, activation type) is discovered via **NAS** under latency constraints.

  - Pipeline:

    ![image-20251201003814413](assets/02-OD-Model-Zoo.assets/image-20251201003814413.png)

  - Pros

    - Architecture tailored for target latency on specific hardware.
    - **Excellent backbone** for mobile classification and detection.

  - Cons

    - Harder to reason about or modify compared to simple v1/v2 patterns.

### R-CNN Zoo

check [here](./02-4-RCNN-Zoo.md)

### FCN

用于处理语义分割的问题：语义分割是这针对像素而言的，要求每个像素都要确定其具体的所属类别

- FCN(Fully Convolutional Networks)

  - Motivation

    图像分类网络往往输出是全连接层的输出,也就是输出一维的张量,然后Softmax归一化为每一个类别的概率,但是这样一维的特征向量就显式的丢失了空间信息。而语义分割需要空间信息，因此我们最后输出需要是一张二维的特征图，然后对每个像素做softmax分类

  - Core Mechanism

    其实将图像分类网络的最后的全连接层换成卷积层就可以输出二维特征图了,这样整个网络都是由卷积层组成,这样的网络叫做FCN(全卷积网络)

    - Architecture

      ![image-20260401150008317](./assets/02-OD-Model-Zoo.assets/image-20260401150008317.png)

      > 这里21是因为在数据集PascalVOC进行的实验,共20类（21=20+1）

    - skip connection融合浅层特征

### FPN Zoo

- __Feature Pyramid Networks for Object Detection.__ *Tsung-Yi Lin et al.* __arXiv, 2016__ [(Arxiv)](https://arxiv.org/abs/1612.03144) 

  - Takeaway

    Feature Pyramid Networks (FPN) is a multi-scale feature fusion architecture for object detection. Its key idea is to combine the **strong semantics of deep layers** with the **high resolution of shallow layers** through a **top-down pathway with lateral connections**, producing feature maps at multiple scales that are all semantically strong. 

  - Motivation

    Object detection must handle objects of very different sizes, but standard ConvNet backbones naturally create a hierarchy where:

    - shallow layers have **high resolution** but **weak semantics**
    - deep layers have **strong semantics** but **low resolution**

    Traditional image pyramids can address scale variation, but they are expensive in computation and memory. Before FPN, many detectors either relied mainly on the final deep feature map, which hurts small-object detection, or built pyramids in less efficient ways. FPN was proposed to exploit the **inherent pyramidal hierarchy** inside ConvNets and build a strong feature pyramid with only marginal extra cost.

  - Core Mechanism

    ![image-20260316181708113](assets/02-OD-Model-Zoo.assets/image-20260316181708113.png)

    FPN has three core parts:

    1. backbone to extract features
       \[
       C_2,\; C_3,\; C_4,\; C_5
       \]
       
    2. Top-down pathway
       
       Starting from the deepest feature map, FPN upsamples higher-level semantic features and sends them downward.

    3. Lateral connections
       
       At each scale, the upsampled high-level feature is merged with the corresponding backbone feature using a lateral \(1\times1\) convolution.

    The fused pyramid features are denoted:

    \[
    P_2,\; P_3,\; P_4,\; P_5
    \]

    A common way to write the fusion is:

    \[
    P_l = \mathrm{Conv}_{3\times3}\!\left(\mathrm{Conv}_{1\times1}(C_l) + \mathrm{Up}(P_{l+1})\right)
    \]

    > [!note]
    >
    > several ways to fuse the feature
    >
    > 1. use 1×1 Convolution to  align channels and add element-wisely
    >    - cheap and keeps channel dimension fixed
    > 2. just channel concatenation

    where:

    - \(C_l\) is the backbone feature at level \(l\)
    - \(P_{l+1}\) is the higher-level pyramid feature
    - \(\mathrm{Up}(\cdot)\) is usually 2× upsampling
    - \(\mathrm{Conv}_{1\times1}\) aligns channel dimensions
    - \(\mathrm{Conv}_{3\times3}\) reduces aliasing after fusion

    The highest pyramid level is usually initialized as:

    \[
    P_5 = \mathrm{Conv}_{1\times1}(C_5)
    \]

    So the whole design can be understood as:

    \[
    \text{high-level semantics} + \text{low-level spatial detail} \rightarrow \text{multi-scale strong features}
    \]

    This is the central reason FPN works especially well for detection across scales, including small objects. 

  - Pros

    - Strong, efficient and simple

  - Cons

    - Fusion is relatively simple: FPN mainly uses element-wise addition, which may not be the most expressive cross-scale fusion strategy.
    - Information flow is mostly top-down: lower-level detailed information is enhanced by high-level semantics, but reverse aggregation is limited.
      - Sol: later methods such as PANet were designed partly to address this.

    - Scale assignment is heuristic: assigning objects to feature levels is not fully adaptive in the original design.

- __Path Aggregation Network for Instance Segmentation.__ *Shu Liu et al.* __arXiv, 2018__ [(Arxiv)](https://arxiv.org/abs/1803.01534) [(Code)](https://github.com/ShuLiu1993/PANet)

  - Takeaway
  
    Path Aggregation Network (PANet) is an extension of FPN-based instance segmentation that improves information flow across feature levels. Its main idea is to strengthen **bottom-up localization propagation**, **multi-level RoI feature aggregation**, and **mask prediction fusion**, so proposal-based segmentation and detection can use both strong semantics and precise spatial details more effectively. :contentReference[oaicite:0]{index=0}
  
  - Motivation
  
    In FPN-based frameworks such as Mask R-CNN, high-level features contain strong semantics, but accurate localization cues often originate from lower layers. The original PANet paper argues that the path from shallow localization features to topmost features is too long, which can weaken precise spatial information for downstream proposal classification, box regression, and mask prediction. PANet was proposed to shorten this path and let proposal subnetworks access useful information from **all** pyramid levels more directly.
  
  - Core Mechanism
  
    ![x1](assets/02-OD-Model-Zoo.assets/x1-1773709312211-1.png)
  
    Figure: Illustration of our framework. (a) FPN backbone. (b) Bottom-up path augmentation. (c) Adaptive feature pooling. (d) Box branch. (e) Fully-connected fusion. Note that we omit channel dimension of feature maps in (a) and (b) for brevity.
  
    > [!TIP]
    >
    > the dashed green line is a shortcut
  
    1. Bottom-up Path Augmentation
       
       On top of the standard FPN top-down pyramid, PANet adds an extra **bottom-up path** so lower-level localization signals can travel upward more efficiently. This strengthens the whole feature hierarchy with spatially precise information.
  
       $$
       P_l = \text{Conv}_{3\times3}\left(\text{Conv}_{1\times1}(C_l) + \text{Up}(P_{l+1})\right) \\
       N_{l+1} = \text{Conv}_{3\times3}\left(P_{l+1} + \text{Conv}_{3\times3}^{stride=2}(N_l)\right)
       $$
       
    2. Adaptive Feature Pooling
       
       In standard FPN-based RoI assignment, each proposal is usually mapped to only one pyramid level. PANet instead pools RoI features from **all feature levels** and fuses them, so each proposal can use multi-scale information directly.
  
       A concise mathematical view is:
       
       \[
       \mathbf{r} = \sum_{l} \alpha_l \, \mathrm{RoIAlign}(N_l, \mathrm{RoI}),
       \]
       
       where:
       
       - \(N_l\) is the feature map at level \(l\)
       - \(\mathrm{RoIAlign}(N_l, \mathrm{RoI})\) extracts proposal features from level \(l\)
       - \(\alpha_l\) is a fusion weight
  
       > [!TIP]
       >
       > The paper’s core point is not the exact symbol choice, but the idea that proposal features **should aggregate information from all pyramid levels rather than only one**.
       
    3. Fully-Connected Fusion for Mask Prediction
       
       PANet adds a complementary branch for mask prediction that captures a different view of each proposal, then fuses it with the normal FCN-style mask branch. This improves mask quality by combining local dense prediction with more global proposal-level information. 
  
       A simple fusion expression is:
       
       \[
       \mathbf{m}_{\text{final}} = \mathbf{m}_{\text{fcn}} + \mathbf{m}_{\text{fc}},
       \]
       
       where:
       
       - \(\mathbf{m}_{\text{fcn}}\) is the mask from the convolutional mask branch
       - \(\mathbf{m}_{\text{fc}}\) is the mask from the fully-connected branch
  
       Again, this is a compact mathematical summary of the fusion idea described in the paper.

### Solo Zoo

- __SOLO: Segmenting Objects by Locations.__ *Xinlong Wang et al.* __European Conference on Computer Vision, 2019__ [(Arxiv)](https://arxiv.org/abs/1912.04488) [(S2)](https://www.semanticscholar.org/paper/4c4f040d0c4ed6d434534fc278e886db31c0d8b4) (Citations __807__)

  - Takeaway

    A new, embarrassingly simple approach to instance segmentation in images by introducing the notion of "instance categories", which assigns categories to each pixel within an instance according to the instance's location and size thus nicely converting instance mask segmentation into a classification-solvable problem.根据实例的位置和大小为实例中的每个像素分配类别

  - Motivation

    之前的实例分割方法都是阶段性的,要不先分出来实例再分割,要不就是先分割再分出来实例,能不能实现一个直接端到端的方法?直接输出不同的实例呢?——位置和形状，那么怎么表示呢

    - 位置：FCOS里面的网格,不同的网格的感受野内的特征是不一样的，因此不同位置上的实例就是不同的实例，转化为了位置分类
    - 形状：形状变化很大，这里用大小来表示

  - Core Mechanism

    - Architecture

      ![image-20260401165317980](./assets/02-OD-Model-Zoo.assets/image-20260401165317980.png)

      将实例分割重新表述为两个同时的、类别预测和实例掩码生成问题。具体而
      言,将输入图像划分为统一的网格,即 S × S 。如果对象的中心落入网格单
      元,则该网格单元负责该实例对象的语义类别预测以及mask预测

    - Head

      ![image-20260401170323016](./assets/02-OD-Model-Zoo.assets/image-20260401170323016.png)

      > [!NOTE]
      >
      > 上面分支是$S\times S$，每个网格对应instance mash branch的一个channel。即分类通道上的一个网格是和掩码分支输出的一个通道有一对一的关系。
      >
      > 那么掩码分支的通道和分类分支的网格有对应的关系,所以,不同位置的特征也要去该位置对应的通道上输出mask。但是卷积具有平移不变性,一个特征无论在哪个位置,其都是不变的。那我应该如何根据特征来决定将其输出到对应的mask通道上呢?所以,在mask分支中,会引入位置
      > 编码
      >
      > ![image-20260401171928787](./assets/02-OD-Model-Zoo.assets/image-20260401171928787.png)

    - Matrix NMS

      Matrix NMS 本质上是 Soft-NMS 的并行实现。它引入了一个“衰减因子”的概念。也就是说对于任何一个框,他最终**被保留的概率取决于他自己原本的分数和他被比他分数高的框的抑制程度**。该方法结合了Soft-NMS和Fast-NMS的方法

      - 传统NMS:只要一个框的IOU和得分最高的框的IOU大于阈值就被抑制，串行计算，很慢
      - FastNMS:假设所有的框是同时存在的,直接通过IOU判断哪些框应该被抑制,实现了并行计算NMS,速度很快
      - SoftNMS:不会像传统NMS那样直接删除掉框,而是衰减其分数,这样能够一定程度上的缓解密集检测的问题
      - Matrix NMS:结合将FastNMS和Soft-NMS，既并行计算，又不会直接删框，A,B两个框,A的得分高,B在计算被A抑制的程度的时候,也要考虑到A本身是否已经被更高得分的框抑制了,如果A已经被抑制了,那么对B的一直程度就应该减少一点,因为A本来就是一个被抑制的框

- __SOLOv2: Dynamic, Faster and Stronger.__ *Xinlong Wang et al.* __ArXiv, 2020__ [(Arxiv)](https://arxiv.org/abs/2003.10152) [(S2)](https://www.semanticscholar.org/paper/fab853583c8465c01e2b8244debaa2bcd6be18d6) (Citations __99__)

  > Arxiv和S2上面的论文名字还不一样的，奇怪

  - Takeaway

  - Motivation

    SOLOv1中提出的mask head的通道数是网格的数量，很多都是冗余的，因此想要对其做一些改进

  - Core Mechanism

    - Architecture

      ![image-20260401174249122](./assets/02-OD-Model-Zoo.assets/image-20260401174249122.png)

    - decoupled head

      ![image-20260401174712122](./assets/02-OD-Model-Zoo.assets/image-20260401174712122.png)

      每个通道不再预测mask,而是预测X和Y方向两个向量,类似的,每个网格都有一个X,Y向量。第 i个网格的Mask直接取出来第i个X,Y向量相乘即可得到

    - 动态卷积

      > 对照结构图查看

      将mask head分为两个branch。上面这个branch输出的是卷积核的参数,通道数就是卷积核参数的个数,也就是说每个网格都会对应一个卷积核。下面这个分支输出的就是一张特征图。当需要第 i个网格的mask的时候,就将第 i个网格逐通道取出来卷积核的参数,然后取下面分支输出的特征图上进行卷积操作,再经过sigmoid就得到了最终的mask


### GhostNet Zoo

- **GhostNet: More Features from Cheap Operations**. Kai Han et.al. **arxiv**, **2019**, ([link](https://arxiv.org/abs/1911.11907v2))([code link](https://github.com/huawei-noah/Efficient-AI-Backbones)) (Citations __5494__).

  - Takeaway: GhostNet dramatically reduces the cost of convolution by observing that many feature maps in standard CNNs are *redundant* and can be generated by cheap linear transformations instead of expensive convolutions.

  - Motivation: Standard CNNs generate feature maps like:$Y = Conv(X)$. But analysis shows:

    - Many feature maps are **highly correlated**
    - Much of the computation is **producing redundant information**
    - Depthwise conv reduces compute but becomes **memory-bound**
    - Mobile models (MobileNetV1/V2/V3) still require substantial 1×1 conv operations

  - Core Mechanism: Ghost Module

    ![image-20251128232247387](assets/02-OD-Model-Zoo.assets/image-20251128232247387.png)

    GhostModule proposes that output feature maps consist of:

    - **Intrinsic features:** small set of essential feature maps (computed by real convolution)

    - **Ghost features:** redundant maps derived from intrinsic ones (via cheap ops)

      > [!NOTE]
      >
      > Here cheap operation is actually group convolution, when group number == input channel number, which is equivalent to depthwise separable convolution. Of course, we can apply other ops like affine transformation, wavelet transformation, shift etc.

    GhostBottleneck

    ![image-20251129130620952](assets/02-OD-Model-Zoo.assets/image-20251129130620952.png)

    ```
    Input
      → GhostModule (expand)
        → DepthwiseConv (if stride=2)
          → Squeeze-and-Excitation (optional)
            → GhostModule (project)
    + Shortcut (identity or depthwise+pointwise)
    ```

    > [!TIP]
    >
    > We found that the construction process of GhostNet is to use Ghost bottleneck to replace the bottleneck in MobileNetV3.

  - Pipeline

    ```mermaid
    flowchart LR

    A[Intrinsic Feature Generation<br/>Apply standard 1x1 or 3x3 conv<br/>Produce a small set of intrinsic feature maps]
        --> B[Ghost Feature Generation<br/>Apply cheap linear ops -- DW conv etc.<br/>Generate additional ghost feature maps]

    B --> C[Concatenation<br/>Merge intrinsic + ghost features<br/>Match full conv output dimensions]

    C --> D[Optional Squeeze-and-Excitation<br/>Channel attention to refine features]

    D --> E[Stack Ghost Bottlenecks<br/>Form GhostNet blocks and full network]

    ```

  - Pros

    - Massive reduction of FLOPs
    - Generalizable: Ghost Module can replace conv in ResNet, MobileNet, etc.

  - Cons

    - Depthwise convolution is memory-bound: Latency improvements may vary by hardware.
    - Cheap operations (depthwise conv, linear transforms) are inherently local. Missing global or long-range feature interactions.

- __GhostNetV2: Enhance Cheap Operation with Long-Range Attention.__ *Yehui Tang et al.* __ArXiv, 2022__ [(Arxiv)](https://arxiv.org/abs/2211.12905) [(S2)](https://www.semanticscholar.org/paper/3e420beb7f5d1bc370470b31908dd766ba35eedd) (Citations __582__)

  - Takeaway: GhostNetV2 = GhostNet + long-range spatial modeling (DFC) with almost zero extra cost.

  - Motivation

    - Cheap operations (depthwise conv, linear transforms) are inherently local.
    - Transformers provide global attention, but are too costly for mobile.

  - Core Mechanism

    GhostNetV2 introduces the **Decoupled Fully Connected (DFC)** mechanism — a computationally cheap yet globally aware operator.

    DFC is a spatial long-range attention operator that approximates a **fully connected** layer over the spatial dimension **but decouples it into two 1D projections**, making it extremely efficient.

    One way to implement an attention map using an FC layer is
    $$
    {a}_{hw}=\sum_{h^\prime,w^\prime}{F_{hw,h^\prime,w^\prime}\odot
    {z}_{h^\prime,w^\prime}} \tag{3}
    $$
    $ \odot $ represents the element-wise multiplication，$ F^{HW\times H\times W} $is the learnable weight. Still $O(H^2W^2)$

    CNN features are 2D, and this 2D shape naturally provides a perspective to reduce the computational load of the FC layer. The author decomposes Equation 3 into 2 FC layers and aggregates features along the horizontal and vertical directions respectively.
    $$
    {a}_{hw}^\prime=\sum_{h^\prime=1}^{H}{F_{h,h^\prime w}^H\odot {z}_{h^\prime w}},h=1,2,\cdot\cdot\cdot,H,w=1,2,\cdot\cdot\cdot,W \tag{4}
    $$

    $$
    {a}_{hw}=\sum_{w^\prime=1}^{W}{F_{w,h w^\prime}^W\odot {a}_{hw^\prime}^\prime},h=1,2,\cdot\cdot\cdot,H,w=1,2,\cdot\cdot\cdot,W \tag{5}
    $$

    The computational complexity of the attention module can be reduced to $O(H^2W+HW^2)$.

    ![image-20251202164623610](assets/02-OD-Model-Zoo.assets/image-20251202164623610.png)

    GhostNetV2 bottleneck

    ![image-20251202164808202](assets/02-OD-Model-Zoo.assets/image-20251202164808202.png)

  - Pros:

    - Adds long-range spatial reasoning
    - **Versatile**: works for classification, detection, segmentation, and mobile vision tasks

  - Cons:

    - More complex
    - Less mathematically expressive than Transformers

- __GhostNets on Heterogeneous Devices via Cheap Operations.__ *Kai Han et al.* __International Journal of Computer Vision, 2022__ [(Arxiv)](https://arxiv.org/abs/2201.03297) [(S2)](https://www.semanticscholar.org/paper/c3a302ed0a8687f8b7bc50e4a1dff0f96b4fbf52) (Citations __165__)

  - Takeaway: This paper generalizes GhostNet to **heterogeneous hardware (CPU and GPU)** by:

    - Designing a **CPU-efficient Ghost module (C-Ghost)** that operates at the feature-map level with cheap operations, and
    - Designing a **GPU-efficient Ghost stage (G-Ghost)** that exploits **stage-wise redundancy** while avoiding GPU-inefficient ops like heavy depthwise conv.

  - Motivation:

    -

  - Core Mechanism: G-Ghost

    ![image-20251207095649092](assets/02-OD-Model-Zoo.assets/image-20251207095649092.png)

    why mix: There might be a lack of deep information that needs to be extracted in multiple layers later on, so add the rich expressive power of the middle layer, and then mix

    how mix: Global average pooling

    ![add based fusion](assets/02-OD-Model-Zoo.assets/image-20251207095707549.png)

  - Pros

    - plug-and-play
    - GPU-friendly

  - Cons

    - Less “unified” than a single-architecture solution

- __RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization.__ *Chengpeng Chen et al.* __ArXiv, 2022__ [(Arxiv)](https://arxiv.org/abs/2211.06088) [(S2)](https://www.semanticscholar.org/paper/d8d754d93d4a4fcc62838429fd36f795cb8f5d98) (Citations __144__)

  - Takeaway: RepGhost replaces the original Ghost module’s feature concatenation with a re-parameterizable, add-based design that implicitly reuses features in the weight space instead of the feature space.
  - Motivation:
    - [CPU vs GPU] We call the original Ghost as C-Ghost because cheap operations such as Depthwise are more friendly to mobile devices such as pipelined CPUs and ARM, but are not so "cheap" for GPUs with strong parallel computing capabilities. Because the computational density of Depthwise operations is relatively low. So we want to dive into a module more GPU-friendly.
    - [Concat vs Add] `concat` vs `add` on ARM:
      - Same params and FLOPs,
      - But `concat` is about **2× slower** than `add` due to memory access overhead.

### NanoDet Zoo

- NanoDet. ([Intro](https://zhuanlan.zhihu.com/p/306530300))

- __NanoDet-Plus: Super fast and high accuracy lightweight anchor-free object detection model.__ *RangiLyu.* __GitHub software repository, 2021__ [(Intro)](https://zhuanlan.zhihu.com/p/449912627) [(Code)](https://github.com/RangiLyu/nanodet). 

  - Takeaway: 

    lightweight, anchor-free, one-stage object detector. Tiny and fast with better feature fusion (Ghost-PAN) and better label assignment during training (AGM + DSLA). 良心涨点

  - Prior:

    - FCOS-style anchor-free detection: `backbone+FPN+head`

    - GFL
  
      - **QFL (Quality Focal Loss):** Solve "Does classification score reflect positioning quality?
  
      - **DFL (Distribution Focal Loss):** Solve "Can regression express positioning uncertainty?"
  
        For bounding box regression, each side offset is modeled as a **discrete distribution**.
  
    - GhostBlock
  
      ![image-20251128232247387](assets/02-OD-Model-Zoo.assets/image-20251128232247387.png)
  
      GhostModule proposes that output feature maps consist of:
  
      - **Intrinsic features:** small set of essential feature maps (computed by real convolution)
  
      - **Ghost features:** redundant maps derived from intrinsic ones (via cheap ops)
  
  - Motivation:
  
    作者把标签分配说成目标检测训练里最核心的问题之一。原版 NanoDet 用的是 ATSS，这类方法虽然会动态选样本，但本质上还是比较依赖中心点、anchor 这类先验信息，属于偏静态的匹配。与此同时，DETR、OTA、YOLOX 这一类方法开始流行基于 matching cost 的动态匹配，这些方法在大模型上效果很好。
  
    问题在于，**大模型能用，不代表小模型也能直接用**。作者明确指出，把这种依赖预测结果的动态匹配直接搬到轻量检测模型上，会遇到大模型没有的困难. 为什么呢? 这里我们需要来看看动态匹配到底是什么
  
    > [!NOTE]
    >
    > 基于Matching Cost的动态匹配：简单来说，就是直接使用模型检测头的输出，与每一个Ground Truth计算一个**匹配的代价**，这个代价一般由分类loss和回归loss组成。Feature Map上所有的点（N个）的预测值与所有的Ground Truth（M个）计算得到的**NxM的矩阵**，就是所谓的**Cost Matrix**，基于这个Cost Matrix进行二分图匹配也好还是传输优化也好再或者直接取TopK也好，就是一种动态匹配策略。这种策略与之前的基于Anchor算IOU的匹配最大的不同就是，它**不再只依赖先验的静态的信息**，而是使用当前的预测结果去动态寻找最优的匹配，只要模型预测的越准确，匹配算法求得的结果也会更优秀。
  
    既然标签匹配需要依赖预测输出，但预测输出又是依赖标签匹配去训练的，但我的模型一开始是**随机初始化**的，啥也没有呀？那这不就成了一个**鸡生蛋，蛋生鸡的问题**了吗？不过好在神经网络天生具有抗噪能力，即使一开始随机初始化的时候给模型随机分配一些点去训练，只要这些点在对应的GT框内，模型也能够逐渐的去拟合那些最容易学到的特征。因此对于除了DETR这种稀疏预测以外，稠密的目标检测的动态标签匹配都会加上一些**位置约束**，比如OTA和SimOTA都使用了一个5x5的中心区域去**限制匹配的自由程度**。
  
    但是这样会有一个问题,轻量模型的检测头太轻了。NanoDet 的 head 很小，只用了很少的深度可分离卷积去同时做分类和回归；和大模型里那种更重、更强的检测头相比，表达能力差很多。所以你让这样一个从随机初始化开始、表达能力又有限的小 head，在训练初期就去产出可靠预测，再拿这些预测反过来指导标签匹配，这件事本身就很难
  
  - Core Mechanism
  
    ```
    ShuffleNetV2 Backbone → GhostPAN → NanoDetPlusHead
                          └── GhostPAN_copy → aux_head (training only)
    ```
  
    ![nanodet-plus-arch](assets/02-OD-Model-Zoo.assets/nanodet-plus-arch.png)
  
    > [!WARNING]
    >
    > 这个图片画的有问题,因为没有直接从backbone输入到assign guidance module的部分
  
    - GFL-style box representation: combine Quality Focal Loss, Distribution Focal Loss, and GIoU Loss
      $$
      \mathcal{L}
      =
      \mathcal{L}_{\text{QFL}}
      + \lambda_{\text{bbox}}\,\mathcal{L}_{\text{GIoU}}
      + \lambda_{\text{DFL}}\,\mathcal{L}_{\text{DFL}}.
      $$
  
    - Ghost-PAN(a light feature pyramid) for lightweight multi-scale fusion
  
      Ghost-PAN: add ghost blocks to PAN module
      $$
      \{C_3,C_4,C_5\}
      \;\xrightarrow{\text{1×1 reduce conv}}\;
      \{\tilde C_3,\tilde C_4,\tilde C_5\} \\
      \text{Top-down: }\;
      P_{l}^{td}=\text{GhostBlock}\big(\operatorname{Concat}(\operatorname{Up}(P_{l+1}^{td}),\tilde C_l)\big) \\
      \text{Bottom-up: }\;
      P_{l+1}^{out}=\text{GhostBlock}\big(\operatorname{Concat}(\operatorname{Down}(P_l^{out}),P_{l+1}^{td})\big)
      $$
  
      Pipeline
  
      1. reduce all input feature C channels to a fixed value
      2. Top-down: upsample P(bilinear), cancat with C, and send into the ghostblock to get the fused feature
      3. Bottom-up: downsample N(conv with stride=2), cancat with P, and send into the ghostblock to get the fused feature
      4. extra layer: 为了增强对更大目标的检测。this combines:
         - a stride-2 transform of the original deepest reduced backbone feature
         - a stride-2 transform of the current deepest PAN output
  
      ```python
          def forward(self, inputs):
              """
              Args:
                  inputs (tuple[Tensor]): input features.
              Returns:
                  tuple[Tensor]: multi level features.
              """
              assert len(inputs) == len(self.in_channels)
              inputs = [
                  reduce(input_x) for input_x, reduce in zip(inputs, self.reduce_layers)
              ]
              # top-down path
              inner_outs = [inputs[-1]]
              for idx in range(len(self.in_channels) - 1, 0, -1):
                  feat_heigh = inner_outs[0]
                  feat_low = inputs[idx - 1]
      
                  inner_outs[0] = feat_heigh
      
                  upsample_feat = self.upsample(feat_heigh)
      
                  inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                      torch.cat([upsample_feat, feat_low], 1)
                  )
                  inner_outs.insert(0, inner_out)
      
              # bottom-up path
              outs = [inner_outs[0]]
              for idx in range(len(self.in_channels) - 1):
                  feat_low = outs[-1]
                  feat_height = inner_outs[idx + 1]
                  downsample_feat = self.downsamples[idx](feat_low)
                  out = self.bottom_up_blocks[idx](
                      torch.cat([downsample_feat, feat_height], 1)
                  )
                  outs.append(out)
      
              # extra layers
              for extra_in_layer, extra_out_layer in zip(
                  self.extra_lvl_in_conv, self.extra_lvl_out_conv
              ):
                  outs.append(extra_in_layer(inputs[-1]) + extra_out_layer(outs[-1]))
      
              return tuple(outs)
      ```
  
    - head: output num_cls+4*(reg_max+1) where the reg_max is the distribution for each side
  
      > [!NOTE]
      >
      > - 对于小网络, 独立的head会好点
      > - 对于大网络, share的head会收敛快点
  
      ```python
      self.gfl_cls = nn.ModuleList(
          [
              nn.Conv2d(
                  self.feat_channels,
                  self.num_classes + 4 * (self.reg_max + 1),
                  1,
                  padding=0,
              )
              for _ in self.strides
          ]
      )
      for feat, cls_convs, gfl_cls in zip(
          feats,
          self.cls_convs,
          self.gfl_cls,
      ):
          for conv in cls_convs:
              # 两层
              feat = conv(feat)
          output = gfl_cls(feat)
          outputs.append(output.flatten(start_dim=2))
      outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
      ```
  
    - label assignment: AGM + DSLA
  
      > [!IMPORTANT]
      >
      > Idea: 用更强大branch的来指导head做匹配(4 convs)
  
      - AGM(Assign Guidance Module): guide the head to do label assignment.
  
        > [!TIP]
        >
        > Training-only auxiliary branch
        >
        > aux head 需要 detach，Reason：它在 NanoDet-Plus 里主要是做 assignment guidance，不希望这条辅助分配路径持续反向干扰 backbone 和主 FPN 的特征学习
  
        Pipeline
  
        1. deepcopy fpn as aux_fpn and concat the fpn_feat and aux_fpn_feat as dual_fpn_feat to send into the aux_head 通道信息更丰富
        2. AGM用4个3x3的conv+1个conv对每个slot进行预测，得到预测类概率和检测框送进DSLA
  
           > [!NOTE]
           >
           > 是在不同featuremap的每个slot上预测cls and reg，也就是每个grid cell的顶点
  
      - DSLA(Dynamic Soft Label Assigner): 利用AGM的结果计算cost_matrix，然后对main head进行动态分配
        
        - cost matrix 计算`cost_matrix = cls_cost + iou_cost * self.iou_factor`
        - dynamic_k_matching: 用联合代价挑正样本，并让每个 GT 的正样本数量由当前 IoU 质量自适应决定
        
        > [!WARNING]
        >
        > 训练后期还使用aux head这样好吗???
        
        Pipeline:
        
        1. prior center 过滤候选slot: 先默认全是背景0
        
           > [!TIP]
           >
           > ignore 不是默认就有的，它只在配置了 gt_bboxes_ignore 且 ignore_iof_thr > 0 时启用，见 nanodet/model/head/assigner/dsl_assigner.py:123
           >
           > 然后会计算预测框和 ignore 区域的 IOF(Intersection over Foreground)，如果超过阈值：
           > ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
           > assigned_gt_inds[ignore_idxs] = -1
           >
           > 这样就会被当作ignore不参与训练(即有潜力被当作正样本的样本会被忽略,避免影响)
        
        2. 对每个slot的anchor计算cost matrix
        
        3. 取 k = max(每个 GT 的 top-k IoU（topk=13）下取整数, 1)
        
           ```python
                   # calculate dynamic k for each gt
                   dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
                   for gt_idx in range(num_gt):
                       _, pos_idx = torch.topk(
                           cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                       )
                       matching_matrix[:, gt_idx][pos_idx] = 1.0
           ```
        
        4. 选cost最小的k个prior
        
        5. 后处理：解决一个 prior 匹配多个 GT 的冲突：只保留cost最小的gt
        
      - Loss的计算，nanodet-plus中有两套loss，因为aux head也有自己的一套同构loss
  
        1. 先用 `aux_preds` 做 assignment(DSLA)
        2. 用这个 assignment 结果算主 head 的 loss
        3. 再用同一个 assignment 结果，给 `aux head` 也算一份同构的 loss
        4. 最后：$\mathcal L_{\text{total}}=\mathcal L_{\text{main}}+\mathcal L_{\text{aux}}$
  
      | Method               | COCO mAP 0.5:0.95 |
      | -------------------- | ----------------- |
      | NanoDet              | 20.6              |
      | NanoDet + DSLA       | 21.9              |
      | NanoDet + DSLA + AGM | 22.7              |
  
    - 后处理筛选求解
  
      对于这么多slot,我们后处理是通过很多步骤来进行处理的
  
      1. 先解码
      2. 一轮筛选: 只保留 score(分类分支输出经过 sigmoid 之后得到的每类分数) 大于 score_thr=0.05
      3. 二轮筛选: 做NMS,只保留其中最好的几个,IoU 阈值这里是 0.6
      4. 限制: 最后最多保留 100 个检测框
  
  - Experiment
  
    - Config: AdamW+CosineAnnealingLR+EMA
  
    | Model                   | Resolution | mAPval 0.5:0.95 | CPU Latency (i7-8700) | ARM Latency (4xA76) | FLOPS     | Params    | Model Size                         |
    | ----------------------- | ---------- | --------------- | --------------------- | ------------------- | --------- | --------- | ---------------------------------- |
    | NanoDet-m               | 320*320    | 20.6            | **4.98ms**            | **10.23ms**         | **0.72G** | **0.95M** | **1.8MB(FP16)** \| **980KB(INT8)** |
    | **NanoDet-Plus-m**      | 320*320    | **27.0**        | **5.25ms**            | **11.97ms**         | **0.9G**  | **1.17M** | **2.3MB(FP16)** \| **1.2MB(INT8)** |
    | **NanoDet-Plus-m**      | 416*416    | **30.4**        | **8.32ms**            | **19.77ms**         | **1.52G** | **1.17M** | **2.3MB(FP16)** \| **1.2MB(INT8)** |
    | **NanoDet-Plus-m-1.5x** | 320*320    | **29.9**        | **7.21ms**            | **15.90ms**         | **1.75G** | **2.44M** | **4.7MB(FP16)** \| **2.3MB(INT8)** |
    | **NanoDet-Plus-m-1.5x** | 416*416    | **34.1**        | **11.50ms**           | **25.49ms**         | **2.97G** | **2.44M** | **4.7MB(FP16)** \| **2.3MB(INT8)** |
    | YOLOv3-Tiny             | 416*416    | 16.6            | -                     | 37.6ms              | 5.62G     | 8.86M     | 33.7MB                             |
    | YOLOv4-Tiny             | 416*416    | 21.7            | -                     | 32.81ms             | 6.96G     | 6.06M     | 23.0MB                             |
    | YOLOX-Nano              | 416*416    | 25.8            | -                     | 23.08ms             | 1.08G     | 0.91M     | 1.8MB(FP16)                        |
    | YOLOv5-n                | 640*640    | 28.4            | -                     | 44.39ms             | 4.5G      | 1.9M      | 3.8MB(FP16)                        |
    | FBNetV5                 | 320*640    | 30.4            | -                     | -                   | 1.8G      | -         | -                                  |
    | MobileDet               | 320*320    | 25.6            | -                     | -                   | 0.9G      | -         | -                                  |
  
  - Cons
  
    - Small-object performance is still challenging
  
    - 正负样本分配差距很严重,很多背景作为负样本,不然打开ignore这样很多纯背景也没有被当作副样本
  
      > ?被loss处理过可能没啥问题???
  
    - 训练后期还使用aux head这样好吗???

### Yolo Zoo

check [here](02-2-YOLO-Zoo.md)


### FCOS Zoo

- __FCOS: Fully Convolutional One-Stage Object Detection.__ *Zhi Tian et al.* __2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019__ [(Arxiv)](https://arxiv.org/abs/1904.01355) [(S2)](https://www.semanticscholar.org/paper/e2751a898867ce6687e08a5cc7bdb562e999b841) [(Code)](https://github.com/tianzhi0549/FCOS/) (Citations __5666__)

  - Takeaway: FCOS is an **anchor-free, proposal-free, one-stage detector** that predicts objects **per pixel** using a fully convolutional head.

    > 第一个这样做得比较好的，直接回归点到四边的距离

  - Motivation

    我们想要实现anchor-free, proposal-free, one-stage detector，那么就需要判断哪里有物体（换句话说就是每个像素是否有物体），物体是哪一类，边界框在哪。这就和FCN语义分割比较像

  - Core Mechanism
  
    - Architure:
  
      ![image-20251223212240953](assets/02-OD-Model-Zoo.assets/image-20251223212240953.png)
  
      就是比语义分割多了一个画框的问题，那么在FCN的基础上再参考anchor-base的方法，多用一个head来预测边界框即可。这也是为什么输出是二维特征图的原因
  
      ```
      backbone+FPN+head
      ```
  
    - backbone
      
    - FPN: fuse the features
      
    - head：共享权重的检测头
  
      > [!NOTE]
      >
      > 检测头有共享权重的，也有不共享的
      
      - 归一化方法：使用Group Normalization
      
      Per location on a feature map, FCOS predicts three things： **classification, box regression, centerness**
      
      - Box regression uses distances to four sides of the target box
        $$
        \mathbf{t} = (l, t, r, b)
        $$
        <img src="assets/02-OD-Model-Zoo.assets/image-20251223212106670.png" alt="image-20251223212106670" style="zoom: 67%;" />
      
        For a feature-map location mapped to image coordinates $(x, y)$ and a ground-truth box with corners $(x_0, y_0)$ and $(x_1, y_1)$
        $$
        l = x - x_0,\quad t = y - y_0,\quad r = x_1 - x,\quad b = y_1 - y
        $$
      
      - Centerness down-weights locations near box edges: a **localization quality indicator**
        $$
        \text{centerness} =
        \sqrt{
        \frac{\min(l, r)}{\max(l, r)}
        \cdot
        \frac{\min(t, b)}{\max(t, b)}
        }
        $$
      
        > [!TIP]
        >
        > We employ sqrt here to slow down the decay of the centerness
      
        ![image-20260401152824005](./assets/02-OD-Model-Zoo.assets/image-20260401152824005.png)
      
        > [!NOTE]
        >
        > 现在我们来讨论一下为什么需要一个centerness？中心度
        >
        > 因为进行正负样本选择时发现分布在框边缘的那些点得到的框效果是不好的，这里有几个解释
        >
        > 1. 其感受野是有限的，更多地依赖于非此物体的信息
        > 2. 更容易回归，中心点预测得到的四个边界值的差距不会很大，比较平滑
        >
        > 因为设置了一个center region(大小由超参数设置)，这样会加剧正负样本不平衡的问题，因此为了缓解这个问题，cls branch使用的是focal loss
      
      Final score at inference multiplies classification confidence and centerness
      $$
      s = \sigma(p_{\text{cls}})\cdot \sigma(p_{\text{ctr}})
      $$
      Training objective combines classification, regression, and centerness losses
      $$
      L = L_{\text{cls}} + \lambda L_{\text{reg}} + \gamma L_{\text{ctr}}
      $$
      A common regression choice in FCOS is IoU or GIoU loss
      $$
      L_{\text{reg}} = 1 - \mathrm{IoU}(B, B^{gt})
      $$
      
      $$
      L_{\text{reg}} = 1 - \mathrm{GIoU}(B, B^{gt})
      $$
  
    Cons
  
    - FCOS的centerness分支在轻量级的模型上很难收敛
      - Sol: GFL 完美去掉了Centerness分支

这里来总结一下anchor-base and anchor-free

- Anchor-base和Anchor-free实质上就是一个东西,就拿acnhor-based方法来说,如果把每个点上的anchor数量设置为1,所生成的anchor尺寸都设置为0,这部就变为了anchor-free类似的方法

  那么区别其实在于正负样本的选择和回归方法

- 正负样本选择：Anchor-base根据IOU选择正负样本,Anchor-free根据位置选择正负样本(其实思想差不多)，在这个框架下，本质就是去解决如何处理正负样本不均的问题

- anchor-free检出率更高，recall更高，也会有更多的误检，因此常通过re-weight来检测出结果（fcos里面的centerness就是如此）

### EfficientNet Zoo

### DETR Zoo

[check here](02-3-DETR-Zoo.md)

### Shuffle-Net Zoo

- **ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices**. Xiangyu Zhang et.al. **arxiv**, **2017**, ([link](https://arxiv.org/abs/1707.01083v2)).

  - Takeaway: ShuffleNet designs extremely efficient CNNs using:

    - **Group Convolutions** to reduce computation,
    - **Channel Shuffle** to preserve information flow across groups, making it suitable for very low FLOP budgets.

  - Core Mechanism: Group convolutions and shuffling

    ![image-20251119220945103](assets/02-OD-Model-Zoo.assets/image-20251119220945103.png)

    ![image-20251126165157180](assets/02-OD-Model-Zoo.assets/image-20251126165157180.png)

  - Cons

    - Channel shuffle pattern can be non-trivial for some inference engines.
    - Accuracy still limited compared to newer designs like MobileNet v3, GhostNet, MobileOne at similar budgets.

- __ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design.__ *Ningning Ma et al.* __ArXiv, 2018__ [(Arxiv)](https://arxiv.org/abs/1807.11164) [(S2)](https://www.semanticscholar.org/paper/c02b909a514af6b9255315e2d50112845ca5ed0e) (Citations __6114__)

### SAM

- __Artificial General Intelligence for Medical Imaging Analysis.__ *Xiang Li et al.* __IEEE Reviews in Biomedical Engineering, 2023__ [(Arxiv)](https://arxiv.org/abs/2306.05480) [(S2)](https://www.semanticscholar.org/paper/d818f40ea693a335e02f32dab520351d271c58bf) (Citations __57__)

### Small Object

[check here](02-1-Small-Object.md)

### SSD Zoo

- **SSD: Single Shot MultiBox Detector**. Liu Wei et.al. **No journal**, **2016**, ([link](https://doi.org/10.1007/978-3-319-46448-0_2)) [(Code)](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection?tab=readme-ov-file).

  - Takeaway: SSD is a **single-stage, anchor-based** detector that predicts on **multi-scale feature maps** in one pass.

  - Motivation: YOLO showed that single-stage detection is fast but initially had localization and small-object issues. There was a need for a detector that:

    - is single-shot (no proposal stage),
    - uses multi-scale features for better small-object detection,
    - keeps good speed–accuracy trade-of

  - Core Mechanism: Architecture

    ```
    backbone+FPN+head
    ```

    - Use a backbone network (e.g., VGG, MobileNet) and attach **extra conv layers** to produce a **feature pyramid**.
    - On each selected feature map:
      - Define a set of **default boxes (anchors)** with different aspect ratios and scales.
      - Use small conv filters to predict:
        - class scores for each default box,
        - bounding box offsets for each default box.
    - Combine predictions from all feature maps and apply NMS.

    ![image-20251122205203672](assets/02-OD-Model-Zoo.assets/image-20251122205203672.png)

    > [!NOTE]
    >
    > This is why it's called **single-shot**: All predictions happen in **one forward pass**, **on multiple scales**.

  - Pros:

    - Eliminate proposal generation and resampling entirely.
    - Multi-scale feature maps improve detection across object sizes.

  - Cons:

    - Performance on **very small objects** is weaker than some later methods (e.g., FPN-based detectors), since SSD relies on relatively shallow high-resolution maps with limited semantics.
    - The hand-designed scales/aspect ratios of default boxes require tuning for new datasets

### MobileOne

- **MobileOne: An Improved One millisecond Mobile Backbone**. Pavan Kumar Anasosalu Vasu et.al. **arxiv**, **2022**, ([link](https://arxiv.org/abs/2206.04040v2)).

  - Takeaway: MobileOne trains **multi-branch** blocks, then **fuses to a single conv** for fast mobile inference.

  - Motivation: Structural reparameterization (e.g., RepVGG) shows we can **train multi-branch, infer single-branch** by fusing conv+BN branches into one conv.

    > [!IMPORTANT]
    >
    > The relationship between these two indicators(**floating-point operations (FLOPs) and parameter count**) and the specific latency of the model is not so clear. For the **specific latency**, we should also consider **memory access cost(MAC) and degree of parallelism.**

  - Core Mechanism: Architectural Blocks(MobileOne block)

    <img src="assets/02-OD-Model-Zoo.assets/image-20251122105130631.png" alt="MobileOne block" style="zoom:80%;" />

    Use structural re-parameterization to decouple the *training* architecture from the *inference* architecture

    - training time: each of those convs (depthwise and pointwise) is expanded into a multi-branch structure (over-parameterized)

    - inference time: all these branches are algebraically fused into a single conv per stage, so the runtime block is very simple

      > [!TIP]
      >
      > Straight cylinder shape： this structure is chosen to minimize latency and memory access cost on mobile hardware.

    - the DSC module is integrated by "scale branch", "skip branch" and "conv branches"
  
      - `rbr_scale`: center-only 1×1 path (after padding) that improves channel-wise scaling flexibility
      - `rbr_skip`: identity + BN path providing residual-like behavior and extra affine freedom
      - `rbr_conv`: main expressive conv paths (3×3 or 1×1)

    ![Model Scaling](assets/02-OD-Model-Zoo.assets/image-20251122174746980.png)

  - Pipeline

    - 先来看看结构

      ```python
      # 可以看到只有前2个小模型的"num_conv_branches": 4,其他都是2,说明branch不是越大越好的
      PARAMS = {
          "s0small": {"width_multipliers": (0.5, 0.75, 0.5, 0.5), "num_conv_branches": 4},
          "s0": {"width_multipliers": (0.75, 1.0, 1.0, 2.0), "num_conv_branches": 4},
          "s0_b2": {"width_multipliers": (0.75, 1.0, 1.0, 2.0), "num_conv_branches": 2},
          "s1": {"width_multipliers": (1.5, 1.5, 2.0, 2.5)},
          "s2": {"width_multipliers": (1.5, 2.0, 2.5, 4.0)},
          "s3": {"width_multipliers": (2.0, 2.5, 3.0, 4.0)},
          "s4": {"width_multipliers": (3.0, 3.5, 3.5, 4.0), "use_se": True},
      }
      
      def mobileone(
          num_classes: int = 1000,
          inference_mode: bool = False,
          variant: str = "s0",
          stagetype=None,
      ) -> nn.Module:
          """Get MobileOne model.
      
          :param num_classes: Number of classes in the dataset.
          :param inference_mode: If True, instantiates model in inference mode.
          :param variant: Which type of model to generate.
          :param stagetype:
          :return: MobileOne model."""
          return MobileOne(
              num_classes=num_classes,
              inference_mode=inference_mode,
              stagetype=stagetype,
              **(PARAMS[variant]),
          )
      ```
      
      然后我们来看看MobileOne的具体构造
      
      ```
      # Build stages
              self.stage0 = MobileOneBlock(
                  in_channels=3,
                  out_channels=self.in_planes,
                  kernel_size=3,
                  stride=2,
                  padding=1,
                  inference_mode=self.inference_mode,
              )
              self.cur_layer_idx = 1
              self.stage1 = self._make_stage(
                  int(64 * width_multipliers[0]), num_blocks_per_stage[0], num_se_blocks=0
              )
              self.stage2 = self._make_stage(
                  int(128 * width_multipliers[1]), num_blocks_per_stage[1], num_se_blocks=0
              )
              if self.stagetype == "stage3s1":
                  self.stage3 = self._make_stride1_stage(
                      int(256 * width_multipliers[2]),
                      num_blocks_per_stage[2],
                      num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0,
                  )
              else:
                  self.stage3 = self._make_stage(
                      int(256 * width_multipliers[2]),
                      num_blocks_per_stage[2],
                      num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0,
                  )
                  self.stage4 = self._make_stage(
                      int(512 * width_multipliers[3]),
                      num_blocks_per_stage[3],
                      num_se_blocks=num_blocks_per_stage[3] if use_se else 0,
                      stride=1 if self.stagetype == "stage4s1" else 2,
                  )
                  self.gap = nn.AdaptiveAvgPool2d(output_size=1)
                  self.linear = nn.Linear(int(512 * width_multipliers[3]), num_classes)
      ```
      
      构造了四个stage,然后一个池化层+线性层,直接输出类别
      
      ```
      def _make_stage(
          self, planes: int, num_blocks: int, num_se_blocks: int, stride: int = 2
      ) -> nn.Sequential:
      
              # Get strides for all layers
              strides = [stride] + [1] * (num_blocks - 1)
              blocks = []
              for ix, stride in enumerate(strides):
                  use_se = False
                  if num_se_blocks > num_blocks:
                      raise ValueError("Number of SE blocks cannot exceed number of layers.")
                  if ix >= (num_blocks - num_se_blocks):
                      use_se = True
      
                  # Depthwise conv
                  blocks.append(
                      MobileOneBlock(
                          in_channels=self.in_planes,
                          out_channels=self.in_planes,
                          kernel_size=3,
                          stride=stride,
                          padding=1,
                          groups=self.in_planes,
                          inference_mode=self.inference_mode,
                          use_se=use_se,
                          num_conv_branches=self.num_conv_branches,
                      )
                  )
                  # Pointwise conv
                  blocks.append(
                      MobileOneBlock(
                          in_channels=self.in_planes,
                          out_channels=planes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          groups=1,
                          inference_mode=self.inference_mode,
                          use_se=use_se,
                          num_conv_branches=self.num_conv_branches,
                      )
                  )
                  self.in_planes = planes
                  self.cur_layer_idx += 1
              return nn.Sequential(*blocks)
      ```
      
    - reparameterize
  
      ```python
      def reparameterize(self):
          """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
          https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
          architecture used at training time to obtain a plain CNN-like structure
          for inference.
          """
          if self.inference_mode:
              return
          kernel, bias = self._get_kernel_bias()
          self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                        out_channels=self.rbr_conv[0].conv.out_channels,
                                        kernel_size=self.rbr_conv[0].conv.kernel_size,
                                        stride=self.rbr_conv[0].conv.stride,
                                        padding=self.rbr_conv[0].conv.padding,
                                        dilation=self.rbr_conv[0].conv.dilation,
                                        groups=self.rbr_conv[0].conv.groups,
                                        bias=True)
          self.reparam_conv.weight.data = kernel
          self.reparam_conv.bias.data = bias
      
          # Delete un-used branches
          for para in self.parameters():
              para.detach_()
          self.__delattr__('rbr_conv')
          self.__delattr__('rbr_scale')
          if hasattr(self, 'rbr_skip'):
              self.__delattr__('rbr_skip')
      
          self.inference_mode = True
      ```
  
      当然主要是`._get_kernel_bias`的计算
  
      ```python
      def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
          """ Method to obtain re-parameterized kernel and bias.
          Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83
      
          :return: Tuple of (kernel, bias) after fusing branches.
          """
          # get weights and bias of scale branch
          kernel_scale = 0
          bias_scale = 0
          if self.rbr_scale is not None:
              kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
              # Pad scale branch kernel to match conv branch kernel size.
              pad = self.kernel_size // 2
              kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                     [pad, pad, pad, pad])
      
          # get weights and bias of skip branch
          kernel_identity = 0
          bias_identity = 0
          if self.rbr_skip is not None:
              kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)
      
          # get weights and bias of conv branches
          kernel_conv = 0
          bias_conv = 0
          for ix in range(self.num_conv_branches):
              _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
              kernel_conv += _kernel
              bias_conv += _bias
      
          kernel_final = kernel_conv + kernel_scale + kernel_identity
          bias_final = bias_conv + bias_scale + bias_identity
          return kernel_final, bias_final
      
      def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
          """ Method to fuse batchnorm layer with preceeding conv layer.
          Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95
      
          :param branch:
          :return: Tuple of (kernel, bias) after fusing batchnorm.
          """
          if isinstance(branch, nn.Sequential):
              kernel = branch.conv.weight
              running_mean = branch.bn.running_mean
              running_var = branch.bn.running_var
              gamma = branch.bn.weight
              beta = branch.bn.bias
              eps = branch.bn.eps
          else:
              assert isinstance(branch, nn.BatchNorm2d)
              if not hasattr(self, 'id_tensor'):
                  input_dim = self.in_channels // self.groups
                  kernel_value = torch.zeros((self.in_channels,
                                              input_dim,
                                              self.kernel_size,
                                              self.kernel_size),
                                             dtype=branch.weight.dtype,
                                             device=branch.weight.device)
                  for i in range(self.in_channels):
                      kernel_value[i, i % input_dim,
                                   self.kernel_size // 2,
                                   self.kernel_size // 2] = 1
                  self.id_tensor = kernel_value
              kernel = self.id_tensor
              running_mean = branch.running_mean
              running_var = branch.running_var
              gamma = branch.weight
              beta = branch.bias
              eps = branch.eps
          std = (running_var + eps).sqrt()
          t = (gamma / std).reshape(-1, 1, 1, 1)
          return kernel * t, beta - running_mean * gamma / std
      
      def _conv_bn(self,
                   kernel_size: int,
                   padding: int) -> nn.Sequential:
          """ Helper method to construct conv-batchnorm layers.
      
          :param kernel_size: Size of the convolution kernel.
          :param padding: Zero-padding size.
          :return: Conv-BN module.
          """
          mod_list = nn.Sequential()
          mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                                out_channels=self.out_channels,
                                                kernel_size=kernel_size,
                                                stride=self.stride,
                                                padding=padding,
                                                groups=self.groups,
                                                bias=False))
          mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
          return mod_list
      ```
  
      
  
  - Pros
  
    - Extremely fast at inference due to: Single-path structure/Fewer ops, better cache behavior.
  
  - Cons
  
    - Once fused, the model loses its multi-branch flexibility (harder to fine-tune structurally).



### Vit Zoo

- __An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.__ *Alexey Dosovitskiy et al.* __ArXiv, 2020__ [(Arxiv)](https://arxiv.org/abs/2010.11929) [(S2)](https://www.semanticscholar.org/paper/268d347e8a55b5eb82fb5e7d2f800e33c75ab18a) (Citations __52634__)

  - Takeaway: ViT tokenizes image **patches** and runs a **Transformer encoder** with global self-attention.

  - Motivation: Try transformer in vision.

  - Core Machanism:

    ViT replaces convolution with **patch-level tokens** processed by a **Transformer encoder**.

    - the input of transformer encoder includes: $N$ patches vecters of $D$ dimension concats the `[CLS]` token which is one vector of D dimension. And add the positional embedding by element-wise addition.

      ```
      encoder input = patch embeddings + class token + positional embeddings.
      ```

    ![image-20251203210356823](assets/02-OD-Model-Zoo.assets/image-20251203210356823.png)

  - Pros

    - Global Receptive Field from the Start
    - Foundation for Many Vision Transformers

  - Cons

    - Requires Large-Scale Data
    - Quadratic Cost of Self-Attention

- __FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization.__ *Pavan Kumar Anasosalu Vasu et al.* __arXiv, 2023__ [(Arxiv)](https://arxiv.org/abs/2303.14189) 

  > MobileOne 原班人马打造，可以看做是 MobileOne 的方法在 Transformer 上的一个改进型的应用

  - Takeaway: a hybrid vision transformer architecture that obtains the state-of-the-art latency-accuracy trade-off. 引入了一种新的 token mixer，叫做 RepMixer，它使用结构重新参数化技术，通过删除网络中的 Shortcut 来降低内存访问成本。效果也很好

  - Core Mechanism

    - Architecture: FastViT 是一个 hybrid vision transformer，也就是混合式视觉 Transformer。作者把网络分成了四个 stage。前 3 个 stage 主要用 RepMixer 来做 token mixing，第 4 个 stage 才使用 self attention

      ![image-20260324182524700](assets/02-OD-Model-Zoo.assets/image-20260324182524700.png)

      - 每个stage分辨率减半，通道数加倍

    - a new token mixer: RepMixer。它的目标不是像标准注意力那样做全局交互，而是更像一个高效的局部信息搅拌器，用深度卷积去混合空间信息。下面介绍一下主要特点

      > [!TIP]
      >
      > skip connection由于增加了内存访问成本 (memory access cost)，这些跳过连接在延迟方面占了很大的开销。所以这里想到了使用**结构重参数化**来删除 skip-connection
    
      1. use structural reparameterization to remove skip connection
    
      2. 为主要的层添加一些过参数化的额外的分支，以在训练时提升模型的精度，在推理时全部消除
    
      3. 使用了大核卷积在前几个阶段替换掉 self-attention
    
         主要是在FFN和Patch Embedding中加入




### Others

- __Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition.__ *Kaiming He et al.* __IEEE Transactions on Pattern Analysis and Machine Intelligence, 2014__ [(Arxiv)](https://arxiv.org/abs/1406.4729) [(S2)](https://www.semanticscholar.org/paper/cbb19236820a96038d000dc629225d36e0b6294a) (Citations __12112__)

  - Takeaway: SPP uses **multi-level pooling** to produce fixed-length features from **any input size**.

  - Motivation: Before SPP, standard CNN (e.g., AlexNet, VGG-style networks) required **fixed-size inputs**

  - Core Mechanism

    Apply pooling over **spatial bins of different granularities**, producing a **fixed-dimensional output** independent of the input feature map size.

  <img src="assets/02-OD-Model-Zoo.assets/image-20251213221137178.png" alt="image-20251213221137178" style="zoom:80%;" />

- __VarifocalNet: An IoU-aware Dense Object Detector.__ *Haoyang Zhang et al.* __2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020__ [(Arxiv)](__VarifocalNet: An IoU-aware Dense Object Detector.__ *Haoyang Zhang et al.* __2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020__ [(Arxiv)](https://arxiv.org/abs/2008.13367) [(S2)](https://www.semanticscholar.org/paper/14c3510e4f4b370d5cd0420037406024533f4b6f) (Citations __851__)) [(S2)](https://www.semanticscholar.org/paper/14c3510e4f4b370d5cd0420037406024533f4b6f) (Citations __1305__)

- __A Dual Weighting Label Assignment Scheme for Object Detection.__ *Shuai Li et al.* __2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022__ [(Arxiv)](https://arxiv.org/abs/2203.09730) [(S2)](https://www.semanticscholar.org/paper/8e68ea6bf41335d341cf629fa03b91463531bf98) (Citations __100__)

- __Once for All: Train One Network and Specialize it for Efficient Deployment.__ *Han Cai et al.* __ArXiv, 2019__ [(Arxiv)](https://arxiv.org/abs/1908.09791) [(S2)](https://www.semanticscholar.org/paper/7823292e5c4b05c47af91ab6ddf671a0da709e82) (Citations __1414__)

- __PBADet: A One-Stage Anchor-Free Approach for Part-Body Association.__ *Zhongpai Gao et al.* __ArXiv, 2024__ [(Arxiv)](https://arxiv.org/abs/2402.07814) [(S2)](https://www.semanticscholar.org/paper/1ba604d5766f632f5430b8a5d8f9d656645ac8a6) (Citations __1__)

  - Takeaway: PBADet predicts part boxes plus a **part → body-center offset** for one-stage association.

    ![x2](assets/02-OD-Model-Zoo.assets/x2.png)

  - Motivation: Earlier pipelines often used **two-stage** designs (detect bodies and parts separately, then associate) or **body → part offsets** (e.g., predicting multiple offsets from each body to many parts), which can become **channel-heavy** as part types grow and can be brittle under occlusion/invisibility. PBADet instead flips the direction to **part → body** with a single universal offset.

  - Core Mechanism: For each part candidate (dense point/feature location), PBADet predicts:

    1. the **part bounding box** + classification, and
    2. a **2D vector** that points from the part to its **owning body center**.
        This keeps the association head **constant-sized** regardless of the number of part categories.

  - Pros:

    - Simple & fast: one-stage, anchor-free; lightweight association head
    - Scalable: offset head does **not** grow with part category count

  - Cons:

    - **Still relies on post-processing matching** (not fully end-to-end assignment)
    - **Sensitive to body detection quality**: missed/shifted body boxes can break association. Crowded/overlapping people may confuse the assignments.

- __TOOD: Task-aligned One-stage Object Detection.__ *Chengjian Feng et al.* __2021 IEEE/CVF International Conference on Computer Vision (ICCV), 2021__ [(Arxiv)](https://arxiv.org/abs/2108.07755) [(S2)](https://www.semanticscholar.org/paper/7438524bf00d7c5a22cb8799797f57c3a794b220) (Citations __1029__)


## Loss Function

check the [OD-Loss-Zoo](./03-OD-Loss-Zoo)

## Module Design

chech the [Module Design](../../../Efficient-AI/02-Module-Design.md)

## Key Points Detection

chech the [kpts](05-Kpts-OD.md)



## References

- [Depthwise Convolution explanation]( https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)
- [MobileNetv2 explanation]( https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html)
- [MobileNetV2 explained video](https://www.youtube.com/watch?v=DkNIBBBvcPs)
- [MobileNetV1_intro](https://research.google/blog/mobilenets-open-source-models-for-efficient-on-device-vision/?_gl=1)
- [Selective-search](https://learnopencv.com/selective-search-for-object-detection-cpp-python/)
