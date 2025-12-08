# Lightweight Models

Focus on lightweight models and their optimizations for embedded systems.

[TOC]

## Model Zoo

### MobileNet Zoo

- **MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications**. Andrew G. Howard et.al. **arxiv**, **2017**, ([link](https://arxiv.org/abs/1704.04861v1)).

  - Takeaway: MobileNet v1 introduces [Depthwise Separable Convolutions(DSC)](06-Module-Design.md#Depthwise Separable Convolutions(DSC)) as a drop-in replacement for standard convs, using **depthwise conv + pointwise conv** to drastically reduce FLOPs and parameters while keeping acceptable accuracy for mobile/embedded deployment.

  - Prior: Classical CNNs (VGG, ResNet) use full 3×3 convolutions with cost:

    - Computation: `H × W × Cin × Cout × K²`
    - Parameters: `Cin × Cout × K²`
    
    This is too heavy for phones and embedded devices. 
    
  - Core Mechanism: Depthwise Separable Convolutions(DSC): Replace a standard `K×K` conv with:

    - **Depthwise conv**: `K×K` per input channel, no channel mixing  
    - **Pointwise conv**: `1×1` conv to mix channels

    <img src="./assets/09-Lightweight-Models.assets/image-20251119220902654.png" alt="image-20251119220902654" style="zoom:50%;" />

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

  - Prior: MobileNet v1’s DSC is efficient but suffers from information loss in narrow intermediate layers. And there`s no residual connections in most layers.

  - Core Mechanism:

    a novel layer module: the inverted residual with linear bottleneck

    1. **Expansion**:
       The input is first passed through a **1×1 convolution** that **expands** the number of channels, increasing the model’s representational capacity.
    2. **Depthwise Separable Convolution**:
       After the expansion, a **depthwise separable convolution** (which is more computationally efficient than a regular convolution) is applied. This operation is performed on each channel separately, rather than combining all channels together, reducing the number of parameters and computation.
    3. **Projection (Linear Bottleneck)**:
       The **output** of the depthwise separable convolution is then passed through a **1×1 convolution** that **projects** the feature map back to a smaller number of channels.

    ![image-20251119220845255](./assets/09-Lightweight-Models.assets/image-20251119220845255.png)

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

  - Prior: Google’s work on NAS and SE (SENet, EfficientNet) motivates an automated and more refined design.

  - Core Mechanism:

    MobilenetV3 block: MobileNetV2 + Squeeze-and-Excite

    ![image-20251201002400620](./assets/09-Lightweight-Models.assets/image-20251201002400620.png)

    Key tricks

    - Introduces **h-swish** (hard-swish) and **h-sigmoid** as cheap approximations of swish/sigmoid, better suited for mobile hardware.
      $$
      \text{hsigmoid}(x) = \frac{ReLU6(x+3)}{6},\quad \text{h-swish}(x) = x\cdot \frac{ReLU6(x+3)}{6}
      $$
      ![image-20251201004538749](./assets/09-Lightweight-Models.assets/image-20251201004538749.png)

    - Overall architecture (widths, kernel sizes, presence of SE, activation type) is discovered via **NAS** under latency constraints.

  - Pipeline:

    ![image-20251201003814413](./assets/09-Lightweight-Models.assets/image-20251201003814413.png)

  - Pros

    - Architecture tailored for target latency on specific hardware.
    - **Excellent backbone** for mobile classification and detection.

  - Cons

    - Harder to reason about or modify compared to simple v1/v2 patterns.


### R-CNN Zoo

- **Rich feature hierarchies for accurate object detection and semantic segmentation**. Ross Girshick et.al. **arxiv**, **2013**, ([link](https://arxiv.org/abs/1311.2524v5)).

  > R-CNN: Regions with CNN features

  - CNN on region proposals (Selective Search): run the CNN **on each region proposals**.

    ![image-20251121203041130](./assets/09-Lightweight-Models.assets/image-20251121203041130.png)

    -  Module design: Region proposals(Selective Search) + Feature extraction(4096-dimensional vector using pre-trained CNN) +  classspecific linear SVMs
    - drawback: multi-stage / non end-to-end, slow, require large disk space

  - To solve the labeled datais scarce: use unsupervised pre-training, followed by supervised fine-tuning/supervised pre-training on a large auxiliary dataset (ILSVRC), followed by domainspecific fine-tuning on a small dataset(is also effective)

  - Cons

    - very slow: need to do ~2k independent forward passes for each image.

- **Fast R-CNN**. Ross Girshick et.al. **arxiv**, **2015**, ([link](https://arxiv.org/abs/1504.08083v2)).

  - Run the CNN **once per image** to get a feature map, then use **ROI pooling** to reuse convolutional features for all proposals. Train classification and bbox regression jointly with a single softmax + regression head.

    ![image-20251121204749852](./assets/09-Lightweight-Models.assets/image-20251121204749852.png)

    > how to project: using the network’s total stride sss to map box coordinates from image space to feature-map space: divide coordinates by sss, then crop that sub-region from the conv feature map.

    - Joint loss: classification cross-entropy + smooth L1 bbox regression loss.
    - The RoI pooling layer uses max pooling to convert the features inside any valid region of interest into a small feature map with a fixed spatial extent of H × W (e.g., 7 × 7).

  - Cons:
    - Proposals are the test-time computational bottleneck in state-of-the-art detection systems.

- **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**. Shaoqing Ren et.al. **arxiv**, **2015**, ([link](https://arxiv.org/abs/1506.01497v3)). 

  - Takeaway: Faster R-CNN is a **two-stage object detector** that integrates **Region Proposal Network (RPN)** with the detection head, replacing external proposal methods and enabling end-to-end training with better speed and accuracy than R-CNN / Fast R-CNN.

    Faster R-CNN’s key idea:  **Learn region proposals with a CNN (RPN) that shares features with the detector.**

  - Prior

    - **R-CNN**: Selective Search proposals + per-region CNN; very slow.
    - **Fast R-CNN**: Single CNN per image + ROI pooling; faster but still depends on external region proposal (e.g., Selective Search), which is CPU-bound and slow.
    - The bottleneck: generating region proposals outside the CNN.

  - Core Mechanism: Attach an **RPN head** on top which slides small conv over the feature map.For each spatial position, predicts:

    - objectness scores for multiple anchors,
    - bounding box regressions for anchors.

    Then generate proposals from RPN and apply **ROI pooling / ROI Align** on shared features.

    <img src="./assets/09-Lightweight-Models.assets/image-20251121222439105.png" alt="image-20251121222439105" style="zoom:50%;" />

    Faster R-CNN is a single, unified network for object detection. The RPN module serves as the ‘attention’ of this unified network.

  - Pros:

    - End-to-end CNN-based detection with learned proposals.
    - Flexible: works with various backbones (ResNet, MobileNet, etc.).

  - Cons: 

    - relatively heavy / two-stage: 1.RPN to generate proposals. 2.ROI head to classify and refine them.
    - Anchor-based design: many hyperparameters, inefficiency

- **Mask R-CNN**. Kaiming He et.al. **arxiv**, **2017**, ([link](https://arxiv.org/abs/1703.06870v3)).

> [!TIP]
>
> The R-CNN universe is not used any more because all of they require large calculation

---


### GhostNet Zoo

- **GhostNet: More Features from Cheap Operations**. Kai Han et.al. **arxiv**, **2019**, ([link](https://arxiv.org/abs/1911.11907v2))([code link](https://github.com/huawei-noah/Efficient-AI-Backbones)) (Citations __5494__).

  - Takeaway: GhostNet dramatically reduces the cost of convolution by observing that many feature maps in standard CNNs are *redundant* and can be generated by cheap linear transformations instead of expensive convolutions.

  - Motivation: Standard CNNs generate feature maps like:$Y = Conv(X)$. But analysis shows:

    - Many feature maps are **highly correlated**
    - Much of the computation is **producing redundant information**
    - Depthwise conv reduces compute but becomes **memory-bound**
    - Mobile models (MobileNetV1/V2/V3) still require substantial 1×1 conv operations

  - Core Mechanism: Ghost Module

    ![image-20251128232247387](./assets/09-Lightweight-Models.assets/image-20251128232247387.png)

    GhostModule proposes that output feature maps consist of:

    - **Intrinsic features:** small set of essential feature maps (computed by real convolution)

    - **Ghost features:** redundant maps derived from intrinsic ones (via cheap ops)

      > [!NOTE]
      >
      > Here cheap operation is actually group convolution, group number = input channel number, which is equivalent to depthwise separable convolution. Of course,we can apply other ops like affine transformation, wavelet transformation, shift etc.

    GhostBottleneck

    ![image-20251129130620952](./assets/09-Lightweight-Models.assets/image-20251129130620952.png)

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

  - Prior

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

    ![image-20251202164623610](./assets/09-Lightweight-Models.assets/image-20251202164623610.png)

    GhostNetV2 bottleneck

    ![image-20251202164808202](./assets/09-Lightweight-Models.assets/image-20251202164808202.png)

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

  - Prior: 

    - 

  - Core Mechanism: G-Ghost

    

    ![image-20251207095649092](./assets/09-Lightweight-Models.assets/image-20251207095649092.png)

    why mix: There might be a lack of deep information that needs to be extracted in multiple layers later on, so add the rich expressive power of the middle layer, and then mix

    how mix: Global average pooling

    ![add based fusion](./assets/09-Lightweight-Models.assets/image-20251207095707549.png)

  - Pros

    - plug-and-play
    - GPU-friendly

  - Cons

    - Less “unified” than a single-architecture solution

- __RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization.__ *Chengpeng Chen et al.* __ArXiv, 2022__ [(Arxiv)](https://arxiv.org/abs/2211.06088) [(S2)](https://www.semanticscholar.org/paper/d8d754d93d4a4fcc62838429fd36f795cb8f5d98) (Citations __144__)

  - Takeaway: RepGhost replaces the original Ghost module’s feature concatenation with a re-parameterizable, add-based design that implicitly reuses features in the weight space instead of the feature space.
  - Prior: 
    - [CPU vs GPU] We call the original Ghost as C-Ghost because cheap operations such as Depthwise are more friendly to mobile devices such as pipelined CPUs and ARM, but are not so "cheap" for GPUs with strong parallel computing capabilities. Because the computational density of Depthwise operations is relatively low. So we want to dive into a module more GPU-friendly.
    - [Concat vs Add] `concat` vs `add` on ARM:
      - Same params and FLOPs,
      - But `concat` is about **2× slower** than `add` due to memory access overhead.


### Yolo Zoo

- __You Only Look Once: Unified, Real-Time Object Detection.__ *Joseph Redmon et al.* __2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015__ [(Arxiv)](https://arxiv.org/abs/1506.02640) [(S2)](https://www.semanticscholar.org/paper/f8e79ac0ea341056ef20f2616628b3e964764cfd) (Citations __41521__)

  - Takeaway: YOLO reframes object detection as a single, unified regression problem from image pixels to bounding boxes and class probabilities, using **one convolutional network, one forward pass**.

  - Prior: Before YOLO, dominant detectors were region-based like R-CNN zoo which is multi-stage and slow at inference.

  - Core Mechanism:

    Divide the image into $S\times S$ grids and for each grid cell predict:

    - B bounding box(with position and confidence score $Pr(object)\times IOU_{\text{truth}}^{\text{pred}}$)
    - C class probabilities

    ![image-20251203135417881](./assets/09-Lightweight-Models.assets/image-20251203135417881.png)

  - Pipeline

    ![image-20251203135258693](./assets/09-Lightweight-Models.assets/image-20251203135258693.png)

  - Pros

    - real-time, simple architecture
    - foundational impact: established the one-stage detection paradigm

  - Cons

    - coarse localization
    - struggles with small objects

### Others

- **ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices**. Xiangyu Zhang et.al. **arxiv**, **2017**, ([link](https://arxiv.org/abs/1707.01083v2)).

  - Takeaway: ShuffleNet designs extremely efficient CNNs using:

    - **Group Convolutions** to reduce computation,
    - **Channel Shuffle** to preserve information flow across groups, making it suitable for very low FLOP budgets.

  - Core Mechanism: Group convolutions and shuffling

    ![image-20251119220945103](./assets/09-Lightweight-Models.assets/image-20251119220945103.png)

    ![image-20251126165157180](./assets/09-Lightweight-Models.assets/image-20251126165157180.png)

  - Cons

    - Channel shuffle pattern can be non-trivial for some inference engines.
    - Accuracy still limited compared to newer designs like MobileNet v3, GhostNet, MobileOne at similar budgets.

- **SSD: Single Shot MultiBox Detector**. Liu Wei et.al. **No journal**, **2016**, ([link](https://doi.org/10.1007/978-3-319-46448-0_2)).

  - Takeaway: SSD is a **single-stage, anchor-based detector** that predicts bounding boxes and class scores **directly from multiple feature maps** in one forward pass, avoiding region proposals and enabling real-time detection with reasonable accuracy.

  - Prior: YOLO showed that single-stage detection is fast but initially had localization and small-object issues. There was a need for a detector that:

    - is single-shot (no proposal stage),

  - uses multi-scale features for better small-object detection,

    - keeps good speed–accuracy trade-of

  - Core Mechanism: Architecture

    - Use a backbone network (e.g., VGG, MobileNet) and attach **extra conv layers** to produce a **feature pyramid**.
    - On each selected feature map:
      - Define a set of **default boxes (anchors)** with different aspect ratios and scales.
      - Use small conv filters to predict:
        - class scores for each default box,
        - bounding box offsets for each default box.
    - Combine predictions from all feature maps and apply NMS.

    ![image-20251122205203672](./assets/09-Lightweight-Models.assets/image-20251122205203672.png)

    > [!NOTE]
    >
    > This is why it's called **single-shot**: All predictions happen in **one forward pass**, **on multiple scales**.

  - Pros:

    - Eliminate proposal generation and resampling entirely.
    - Multi-scale feature maps improve detection across object sizes.

  - Cons:

    - Performance on **very small objects** is weaker than some later methods (e.g., FPN-based detectors), since SSD relies on relatively shallow high-resolution maps with limited semantics.
    - The hand-designed scales/aspect ratios of default boxes require tuning for new datasets

- **MobileOne: An Improved One millisecond Mobile Backbone**. Pavan Kumar Anasosalu Vasu et.al. **arxiv**, **2022**, ([link](https://arxiv.org/abs/2206.04040v2)).

  - Takeaway: MobileOne uses **structural reparameterization**: train with a **multi-branch over-parameterized block** (for accuracy and optimization), then **fuse all branches into a single 3×3 conv** at inference, achieving extremely fast, deployment-friendly mobile backbones.

  - Prior: Structural reparameterization (e.g., RepVGG) shows we can **train multi-branch, infer single-branch** by fusing conv+BN branches into one conv.

    > [!IMPORTANT]
    >
    > The relationship between these two indicators(**floating-point operations (FLOPs) and parameter count**) and the specific latency of the model is not so clear. For the **specific latency**, we should also consider **memory access cost(MAC) and degree of parallelism.**

  - Core Mechanism: Architectural Blocks(MobileOne block)

    <img src="./assets/09-Lightweight-Models.assets/image-20251122105130631.png" alt="MobileOne block" style="zoom:80%;" />

    Use structural re-parameterization to decouple the *training* architecture from the *inference* architecture

    - training time: each of those convs (depthwise and pointwise) is expanded into a multi-branch structure (over-parameterized)

    - inference time: all these branches are algebraically fused into a single conv per stage, so the runtime block is very simple

      > Straight cylinder shape： this structure is chosen to minimize latency and memory access cost on mobile hardware.

    - the DSC module is integrated by "scale branch", "skip branch" and "conv branches"

      - `rbr_scale`: center-only 1×1 path (after padding) that improves channel-wise scaling flexibility
      - `rbr_skip`: identity + BN path providing residual-like behavior and extra affine freedom
      - `rbr_conv`: main expressive conv paths (3×3 or 1×1)

    ![Model Scaling](./assets/09-Lightweight-Models.assets/image-20251122174746980.png)

  - Pros

    - Extremely fast at inference due to: Single-path structure/Fewer ops, better cache behavior.

  - Cons

    - Once fused, the model loses its multi-branch flexibility (harder to fine-tune structurally).

## Module Design

chech the [Module Design](../Model-Compression/06-Module-Design.md)

## References

- [Depthwise Convolution explanation]( https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)
- [MobileNetv2 explanation]( https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html)
- [MobileNetV2 explained video](https://www.youtube.com/watch?v=DkNIBBBvcPs)
- [MobileNetV1_intro](https://research.google/blog/mobilenets-open-source-models-for-efficient-on-device-vision/?_gl=1)
- [Selective-search](https://learnopencv.com/selective-search-for-object-detection-cpp-python/)
