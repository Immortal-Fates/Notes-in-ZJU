---
title: 04-Training-Strategies
date: 2026-03-02
tags:
course: AI
status: draft
---
# Training Strategies
[TOC]

## Data Augmentation

- color augmentation, random affine, random flip, and mosaic

- **Bag of Freebies for Training Object Detection Neural Networks**. Zhi Zhang et.al. **arxiv**, **2019**, [(Arxiv)](https://arxiv.org/abs/1902.04103) [(S2)](https://www.semanticscholar.org/paper/Bag-of-Freebies-for-Training-Object-Detection-Zhang-He/0ba8182fa99559257e99a56d790bf2c705c42537).
  - Takeaway: Systematically explores training tweaks that improve Faster R-CNN and YOLOv3 without architecture changes.
  
  - Motivation: Classical object detectors often overlook training details; many simple improvements can be combined for cumulative gains.
  
  - Core Mechanism:
    - **Visually Coherent Image Mixup**: Specialized mixup for object detection preserving spatial alignment
    - **Label Smoothing**: Regularization for classification head, smooths one-hot targets
    - **Cosine Learning Rate Schedule**: Outperforms traditional step LR schedules, smoother decay
    - **Synchronized Batch Normalization**: Better multi-GPU training performance
    - Random geometry transformations with careful application
  
  - Pros:
    - Up to 5% absolute precision improvement
    - No inference cost increase (pure training tricks)
    - Generalizable across different detectors
  
  - Cons:
    - Requires careful hyperparameter tuning
    - Some techniques may not transfer to all datasets
  
- **Simple Copy-Paste is a Strong Data Augmentation Method**. Golnaz Ghiasi et.al. **CVPR**, **2021**, [(Arxiv)](https://arxiv.org/abs/2012.07177) [(S2)](https://www.semanticscholar.org/paper/Simple-Copy-Paste-is-a-Strong-Data-Augmentation-for-Ghiasi-Cui/914a593b7f2e980470075a9955f1407641669a8f).

  - Takeaway: Copy-Paste augmentation pastes objects from one image onto another, achieving SOTA on multiple datasets.

  - Motivation: Object detection needs diverse training data; complex augmentations can be hard to design.

  - Core Mechanism:
    - **Copy-Paste**: Randomly paste objects from source image onto target image
    - Combines with self-training for additive gains
    - Preserves instance masks, enabling simultaneous detection and segmentation

  - Pros:
    - Simple to implement
    - State-of-the-art on COCO, LVIS, PASCAL
    - Effective for instance segmentation

  - Cons:
    - May create unrealistic scenes
    - Limited to datasets with instance masks

## Class Imbalance

- **Focal Loss for Dense Object Detection**. Tsung-Yi Lin et.al. **ICCV**, **2017**, [(Arxiv)](https://arxiv.org/abs/1708.02002) [(S2)](https://www.semanticscholar.org/paper/Focal-Loss-for-Dense-Object-Detection-Lin-Goyal/1a857da1a8ce47b2aa185b91b5cb215ddef24de7).

  - Takeaway: Focal Loss down-weights easy examples to address extreme foreground-background class imbalance in dense detectors.

  - Motivation: Dense detectors evaluate huge candidate locations; most are easy background, dominating standard cross-entropy loss.

  - Core Mechanism:
    - Binary cross-entropy with modulating factor
    - Let $y \in \{0,1\}$ be label, $p \in [0,1]$ be model prediction:
    - $p_t = \begin{cases} p & \text{if } y=1 \\ 1-p & \text{if } y=0 \end{cases}$
    - Cross Entropy: $\mathrm{CE}(p_t) = -\log(p_t)$
    - Focal Loss: $\mathrm{FL}(p_t) = -(1-p_t)^{\gamma}\log(p_t)$
    - $\gamma \geq 0$ is focusing parameter (typical: $\gamma=2$)
    - Easy examples ($p_t \approx 1$): $(1-p_t)^\gamma \approx 0$, loss heavily reduced
    - Hard examples ($p_t \ll 1$): modulating factor stays large, focused learning

  - Pros:
    - Eliminates need for hard negative mining
    - Foundation for RetinaNet and subsequent methods
    - Addresses two imbalance types: positive/negative and easy/hard

  - Cons:
    - Requires tuning $\gamma$ for different tasks
    - May still struggle with extreme long-tail distributions

- **Equalized Focal Loss**. Bing Li et.al. **CVPR**, **2022**, [(Arxiv)](https://arxiv.org/abs/2201.02593) [(S2)](https://www.semanticscholar.org/paper/Equalized-Focal-Loss-for-Dense-Long-Tailed-Object-Li-Yao/d1d75ac25fd457166360c346cf89005e2531a5fc).

  - Takeaway: EFL addresses long-tailed distribution with category-relevant modulating factors, dynamically adjusting loss based on imbalance degrees.

  - Motivation: Focal Loss treats all categories equally, but real-world datasets have severe class frequency imbalance (e.g., LVIS).

  - Core Mechanism:
    - **Category-relevant modulating factor**: Different modulating parameters for different categories
    - Reweights loss contribution based on class frequency
    - Balances gradients between frequent and rare classes
    - Adaptive to dataset's tail distribution

  - Pros:
    - Strong results on LVIS v1 benchmark
    - Better handles long-tailed distributions than standard FL
    - Dynamically adjusts per-category learning

  - Cons:
    - Additional hyperparameters per category
    - Requires class frequency statistics

## Hard Example Mining

## Label Assignment Strategy

### Intro

- Takeaway: label assignment就是在训练时，哪些预测框是正样本（positive）？哪些是负样本（negative）？

  > [!TIP]
  >
  > 仅在dense perdiction中存在，sparse prediction已经解决了这个问题

- Motivation: 如果 assignment 不合理：
  - 正样本太少 → 学不到
  - 正样本太多 → 噪声大
  - 小目标没有 anchor → 小目标 recall 崩
  - 遮挡目标被分错 → AP50 还行，AP75 很差

- 查看label assignment: 需要可视化看anchor如何与gt进行匹配的

- 一些insight: assignment 本质是 heuristic，没有一个真正理论最优解。

  因此 dense detection 论文大量时间都在研究：

  ```
  better label assignment
  ```

  而不是模型本身

下面介绍一些主流的方法：

1. 规则驱动：早几年的研究，如ATSS

2. 预测驱动：用模型当前输出结果来参与正样本选择，动态样本分配
   - Taxonomy
   
     - 基于Matching Cost的动态分配
   
       直接使用模型检测头的输出，与每一个Ground Truth计算一个匹配的代价，这个代价一般由分类loss和回归loss组成。Feature Map上所有的点（N个）的预测值与所有的Ground Truth（M个）计算得到的**NxM的矩阵**，就是所谓的**Cost Matrix**
   
       那么基于这个Cost Matrix进行二分图匹配/传输优化/取topk 都是一种动态匹配
   
       > [!TIP]
       >
       > 虽然看着是一个鸡生蛋，蛋生鸡的问题，然是NN还是能够训练学习
   
   - Pros: 更强
   
   - Cons:
     - **训练初期不稳定**，因为预测还不准
       - Sol: 使用 warmup / 混合规则 + 预测

### Taxonomy

#### 规则驱动

- Fixed IoU Threshold

- Center-Prior Assignment

    - Positive only if anchor center is inside GT center region (radius/ratio), then apply IoU rule.
    - Medium effort, reduces noisy positives near borders.

- __Bridging the Gap Between Anchor-Based and Anchor-Free Detection via Adaptive Training Sample Selection.__ *Shifeng Zhang et al.* __CVPR, 2020__ [(Arxiv)](https://arxiv.org/abs/1912.02424) [(S2)](https://www.semanticscholar.org/paper/db160e36aec4b43cc0651039eb1fc1e63527b090) (Citations __1965__) -- ATSS ([My PDF](https://drive.google.com/file/d/1pjk--V3r9oXrp0tIsSChaH9jjGlGgGNr/view?usp=drivesdk))

  - Takeaway: （规则驱动）ATSS argues that the key gap between anchor-based and anchor-free dense detectors is not the anchor itself, but **how positives/negatives are assigned**. 它用 per-GT 的自适应 IoU 阈值选正样本，在不增加推理开销的前提下显著提升 RetinaNet / FCOS。

    Anchor design 不是关键，sample selection 才是。

    > CNN 成立；放到后来的 Transformer / NMS-free / query-based detector，这个结论就不再能原样照搬。

  - Motivation:

    - 传统 anchor-based detector 用固定 IoU threshold（如 0.5/0.4）分配正负样本，hyperparameter 很敏感。
    - FCOS 这类 anchor-free detector 用 spatial + scale constraint 选样本，看起来更强，但作者追问的是：提升到底来自 “anchor-free”，还是来自 **assignment strategy**？
    - 论文先做 controlled comparison，发现只要把正负样本定义统一，anchor box regression 和 point regression 的性能差距几乎消失。

  - Core Mechanism:

    ![image-20260421142018860](./assets/04-Training-Strategies.assets/image-20260421142018860.png)

    - **Step 1: per-level candidate mining**. 对每个 GT $g$，在每个 FPN level 里选出中心点距离 GT center 最近的 $k$ 个 anchors，组成 candidate set
      $$
      \mathcal{C}_g = \bigcup_{i=1}^{\mathcal{L}} \mathcal{S}_i,
      \quad |\mathcal{S}_i| = k,
      \quad |\mathcal{C}_g| = k\mathcal{L}.
      $$
      这样先保证每个尺度层都有机会参与，不再靠人工指定某个 level 负责某类目标。

    - **Step 2: adaptive IoU threshold**. 计算这些 candidates 与 GT 的 IoU 分布
      $$
      \mathcal{D}_g = IoU(\mathcal{C}_g, g),
      \qquad
      t_g = m_g + v_g,
      $$
      其中 $m_g = \mathrm{Mean}(\mathcal{D}_g)$，$v_g = \mathrm{Std}(\mathcal{D}_g)$。

      - $m_g$ 高：说明这个 GT 和预设 anchor 很匹配，threshold 应该更高。
      - $v_g$ 高：说明只有少数 pyramid levels 特别适合它，ATSS 会更倾向只从这些 level 里挑 positives。

    - **Step 3: final positive selection**. 若 candidate 满足 $IoU(c,g) \ge t_g$ 且 its center lies inside the GT box，则标为 positive；若一个 anchor 同时匹配多个 GT，就分给 IoU 最大的那个 GT。

      > [!NOTE]
      >
      > 本质上，ATSS 还是先用几何先验（center distance + IoU）筛 candidates，再做自适应 threshold，因此它比固定规则灵活，但还不是后面那种 fully prediction-driven matching。

  - Pipeline:

    1. 输入 image，经过 FPN 得到 multi-level dense anchors / points。
    2. 对每个 GT，在每个 level 选 top-$k$ closest candidates。中心点之间的欧式距离来计算
    3. 统计 candidate IoU 的 mean/std，得到该 GT 自己的 threshold $t_g$。
    4. 选出满足 threshold 且 center in GT 的 positives，用于 classification / regression 训练。
    5. 剩余样本视为 negatives；推理阶段不改 detector head，因此没有额外 inference overhead。

  - Pros:
    - 把 fixed IoU threshold / fixed scale range 变成 per-object adaptive assignment，鲁棒性更强。
    - 统一解释了 anchor-based 和 anchor-free 的差别：关键在 assignment，而不是 box vs point。
    - 几乎不引入额外超参，核心只剩一个较稳健的 $k$（默认 9）。
    - 在 paper 的 COCO minival 验证里，RetinaNet (#A=1) 用 ATSS 可从 $37.0$ AP 提升到 $39.3$ AP；FCOS full version 也有明显提升。

  - Cons:
    - 虽然叫 adaptive，但仍建立在 center prior、IoU 和 FPN level 这些 hand-crafted inductive bias 上。
    - 仍属于 rule/statistics-driven assignment，不会利用当前分类分数或回归质量做真正的 prediction-driven matching。
    - “anchor 本身不重要” 这个结论主要适用于当时的 CNN dense detectors；放到 DETR-style / NMS-free 范式需要重新审视。

近年来正样本的选择（label assignment）**由模型当前预测结果决定**，而不是固定规则决定。

#### 预测驱动

- **TOOD: Task-aligned One-stage Object Detection**. Chengjian Feng et.al. **ICCV**, **2021**, [(Arxiv)](https://arxiv.org/abs/2108.07755) [(S2)](https://www.semanticscholar.org/paper/7438524bf00d7c5a22cb8799797f57c3a794b220) [(Code)](https://github.com/fcjian/TOOD). -- TAL

  - Takeaway: （预测驱动）TOOD aligns classification and localization objectives with task-aligned learning, improving ranking consistency and final AP.

  - Motivation: One-stage detectors often optimize classification and box regression separately, causing score-IoU mismatch during NMS ranking.

    两个任务的最优样本不一致，这就是Misalignment

  - Core Mechanism: 为了让两个任务共享同一套对齐标准，有如下三点

    ![image-20260302102627093](assets/04-Training-Strategies.assets/image-20260302102627093.png)

    1. **T-Head (Task-aligned Head)**: Shared interactive features are learned for classification and localization branches.

    2. **Task Alignment Learning (TAL)**: Uses an alignment metric to jointly evaluate class confidence and localization quality for positive sample selection.
       $$
       score = (cls)^{\alpha} \cdot (IoU)^{\beta}
       $$

    3. **Task-aligned Loss**: Optimizes both tasks toward consistent high-quality predictions so classification scores better reflect box quality.

    ![T-head](assets/04-Training-Strategies.assets/T-head.png)

    - training: 前 4 个 epoch 用 ATSS 稳定收敛;之后用 TAL：样本分配、分类监督、回归权重都围绕同一个对齐指标。

      > [!tip]
      >
      > loss function也会跟着改变，先用focal_loss_with_prob再用task_aligned_focal_loss

  - Pros:
    - Mitigates classification-localization misalignment effectively.
    - Strong gains on COCO with one-stage detectors.
    - Plug-and-play assignment idea, later adopted by many detectors.

  - Cons:
    - Adds assignment/loss design complexity compared with fixed IoU rules.
    - Sensitive to alignment metric and hyperparameter settings.

- PAA（Probabilistic Anchor Assignment）

  PAA 假设正负样本的联合损耗分布遵循高斯分布。因此，它使用 GMM 拟合正负样本分布，然后以正样本分布中心作为*正**负*分界

- DERT

  Hungarian matching

  - Pros
    - Globally optimal assignment（一对一）严格限制一个 GT 只匹配一个预测

- __OTA: Optimal Transport Assignment for Object Detection.__ *Zheng Ge et al.* __arXiv, 2021__ [(Arxiv)](https://arxiv.org/abs/2103.14259) [(Code)](https://github.com/Megvii-BaseDetection/OTA)

  - Takeaway: OTA formulates label assignment as **optimal transport** problem, finding globally optimal assignment for all anchors simultaneously.
  - Motivation: Most assignment strategies assign anchors per-GT locally greedily, ignoring global optimal configuration. 多个 GT 争抢同一个候选框
  - Core Mechanism:
    - **Optimal Transport (OT) formulation**: Global perspective for all anchors
    - Transportation cost: Weighted sum of classification and regression losses
    - Solves OT problem to find optimal one-to-one matching TaskAlignedAssigner. 直接计算难度较大，Sinkhorn-Knopp 近似求解
  - Pros:
    - Globally optimal assignment（多对多）更适合 dense detector
    - Theoretically principled approach
    - Outperforms greedy assignment strategies
  - Cons:
    - Higher computational cost (solving OT problem)
    - Complex implementation 人脸 / 手势检测常用。后来常用simOTA，效果接近更简单

- **YOLOX: Exceeding YOLO Series in 2021**. Zheng Ge et.al. **arXiv**, **2021**, [(Arxiv)](https://arxiv.org/abs/2107.08430) [(Code)](https://github.com/Megvii-BaseDetection/YOLOX). -- SimOTA

  - Takeaway: SimOTA 是 YOLOX 里对 OTA 的工程化简化版。它保留了 **loss-aware cost + center prior + dynamic number of positives** 这几个关键思想，但不再真的去解 Optimal Transport，而是直接用 dynamic top-$k$ 做近似匹配，因此训练更快、实现更简单。

  - Motivation:

    - OTA 很强，但要用 Sinkhorn-Knopp 求 OT，YOLOX 作者实测会带来约 25% 的额外训练时间。
    - 对实时检测器来说，label assignment 不能太重，否则训练成本过高，不利于大规模工程使用。
    - 所以 SimOTA 的核心目标不是再追求“最优传输”的理论完备性，而是：保留 OTA 的有效成分，丢掉最贵的求解器。

  - Core Mechanism:

    - **Pair-wise matching cost**. 对每个 GT $g_i$ 和 prediction $p_j$，先计算匹配代价
      $$
      c_{ij} = L_{ij}^{cls} + \lambda L_{ij}^{reg},
      $$
      其中 $L_{ij}^{cls}$ 是分类损失，$L_{ij}^{reg}$ 是回归损失，$\lambda$ 是平衡系数。

      这个 cost 本质上在回答：哪个 prediction 对这个 GT 来说“又分得对，又框得准”。

    - **Center prior**. SimOTA 不是在全图所有 predictions 上选 positives，而是先限制在一个 fixed center region 内再做匹配。

      这样做的原因是：靠近 GT center 的 grids 更可能是高质量正样本，也能减少训练初期不稳定的低质量匹配。

    - **Dynamic top-$k$ matching**. 对每个 GT，不是固定分配 1 个或固定 $k$ 个正样本，而是在候选中心区域内选 cost 最小的 top-$k$ predictions 作为 positives。

      这里的 $k$ 不是常数，而是 dynamic 的。YOLOX 这篇 paper 把具体估计细节引用到 OTA：对每个 GT，先找出与它 IoU 最高的 top-$q$ predictions（OTA 里默认 $q=20$），再把这些 IoU 相加，得到该 GT 需要的正样本数量估计：
      $$
      k_i \approx \sum_{j \in \operatorname{Top}q(IoU)} IoU(p_j, g_i).
      $$
      实际实现里通常会再把它转成整数（至少为 1），作为这个 GT 的 dynamic $k$。

      直觉上，如果一个 GT 周围本来就有更多 predictions 能回归得很好，那么高 IoU prediction 的总和就更大，这个 GT 就应该分到更多 positives；反之，如果它本身难回归、可用候选少，那它的 $k$ 也会更小。

      > [!NOTE]
      >
      > 所以 SimOTA 可以看成：用 cost-based ranking + dynamic top-$k$，去近似 OTA 里的全局最优分配；它保留了“动态正样本数”这个很关键的 insight，但放弃了真正的 OT solver。

  - Pipeline:

    1. 对每个 GT，先根据 center prior 缩小候选预测范围。
    2. 计算候选 predictions 与该 GT 的 pair-wise cost。
    3. 用 dynamic $k$ 估计这个 GT 该分配多少个 positives。
    4. 选出该 GT 下 cost 最小的 top-$k$ predictions 作为 positives。
    5. 对应 grids 标为正样本，其余为负样本；不需要 Sinkhorn-Knopp 或 OT 求解。

  - Pros:
    - 相比 OTA，训练更省时，且不再引入 Sinkhorn 求解器相关的额外超参。
    - 保留了 loss-aware matching、center prior、dynamic positive count 这些真正有效的部分。
    - 很适合 one-stage real-time detector，因此后来在 YOLO 系里非常常见。
    - 在 YOLOX-DarkNet53 的 roadmap 里，加入 SimOTA 后 AP 从 $45.0$ 提升到 $47.3$。

  - Cons:
    - 它是对 OTA 的近似，不再显式保证 global optimal assignment。
    - center prior 仍然是一种 hand-crafted bias，候选区域外的预测基本没有机会成为正样本。
    - dynamic $k$ 估计仍然依赖当前预测框的 IoU 质量，因此训练初期的 matching 质量会受模型状态影响。

- __Category-Aware Dynamic Label Assignment with High-Quality Oriented Proposal.__ *Mingkui Feng et al.* __ArXiv, 2024__ [(Arxiv)](https://arxiv.org/abs/2407.03205) [(S2)](https://www.semanticscholar.org/paper/ccb0d094cf39cb17cf29214cb930f0dce9ca3211) (Citations __4__)

- __Integrating Diverse Assignment Strategies into DETRs.__ *Yiwei Zhang et al.* __arXiv, 2026__ [(Arxiv)](https://arxiv.org/abs/2601.09247)

- __Point2RBox-v3: Self-Bootstrapping from Point Annotations via Integrated Pseudo-Label Refinement and Utilization.__ *Teng Zhang et al.* __arXiv, 2025__ [(Arxiv)](https://arxiv.org/abs/2509.26281)

- __Improving Object Detection by Label Assignment Distillation.__ *Chuong H. Nguyen et al.* __arXiv, 2021__ [(Arxiv)](https://arxiv.org/abs/2108.10520)  蒸馏引导label assignment

- https://openaccess.thecvf.com/content/CVPR2025/html/Liu_FSHNet_Fully_Sparse_Hybrid_Network_for_3D_Object_Detection_CVPR_2025_paper.html

### Practice

现在我在做一个手头脸的检测人物，label assignment会从以下方面直接影响我的检测效果

- 小目标
- 密集场景
- 类间重叠：face inside head，存在层级关系，assignment可能混淆，分类不稳定
- 遮挡：漏检

## Initialization

### 权重初始化

- Xavier initialization [(paper)](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) [(Good Intro)](https://zhuanlan.zhihu.com/p/653754525)

  - Takeaway: 

    Xavier initialization sets the initial weights so that the variance of activations stays roughly stable across layers. 

  - Motivation: 

    Glorot and Bengio proposed it to reduce vanishing and exploding signals in deep networks.

  - Core Mechanism

    Xavier initialization chooses weights with zero mean and a variance that depends on both the number of input connections and the number of output connections.

    每个输出$o_i$都可以表示为
    $$
    o_i = \text{activation}(\sum_{j=1}^{n_{in}} w_{ij} x_j + b_i) \\
    $$
    那么$\sum_{j=1}^{n_{in}} w_{ij} x_j$的方差将为$n_{in}\times Var(w)\times Var(x)$，为了让每一层的输出方差接近其输入的方差，设置权重$w$的初始方差为
    $$
    \mathrm{Var}(W) \approx \frac{2}{fan_{in} + fan_{out}}
    $$
    然后权重会从，均值0和这样的方差的正态分布或者从以下均匀分布中抽取

    The Xavier normal form is
    $$
    W_{ij} \sim \mathcal{N}\left(0,\; \frac{2}{fan_{in} + fan_{out}}\right)
    $$
    The Xavier uniform form is
    $$
    W_{ij} \sim U\left[-a,\; a\right]
    $$
    where
    $$
    a = \sqrt{\frac{6}{fan_{in} + fan_{out}}}
    $$

  - Pros

    - 一定程度避免梯度消失和爆炸
    - 加速收敛

  - Cons

    - It is not the best choice for **ReLU** family activations, where He initialization is usually better.

    - It may still perform poorly in very deep or highly specialized architectures if used without adaptation.

- Guassian initialization

  - Core Mechanism

    In its most basic form, Gaussian initialization sets each weight as
    $$
    W_{ij} \sim \mathcal{N}(\mu,\sigma^2)
    $$
    In practice, people often use zero mean:
    $$
    W_{ij} \sim \mathcal{N}(0,\sigma^2)
    $$

  - Pros
    - a general random initialization method
  - Cons
    - Its quality depends heavily on the choice of $\sigma$.

- He initialization

## Scheduling & Optimization

- **Libra R-CNN: Towards Balanced Learning**. Jiangmiao Pang et.al. **CVPR**, **2019**, [(Arxiv)](https://arxiv.org/abs/1904.02701) [(S2)](https://www.semanticscholar.org/paper/Libra-R-CNN%3A-Towards-Balanced-Learning-for-Object-Pang-Chen/32a69681c103807704f71b838454c7924ceec5ce).

  - Takeaway: Libra R-CNN addresses three imbalance levels (sample, feature, objective) with IoU-balanced sampling, Balanced Feature Pyramid, and Balanced L1 Loss.

  - Motivation: Object detection suffers from multiple imbalance types: foreground-background, feature pyramid levels, and regression loss contributions.

  - Core Mechanism:
    - **IoU-balanced sampling**: Reduces sample-level imbalance by resampling based on IoU
    - **Balanced Feature Pyramid**: Reduces feature-level imbalance with integrated features
    - **Balanced L1 Loss**: Reduces objective-level imbalance by controlling regression loss gradients
    - Addresses imbalance at three levels systematically

  - Pros:
    - +2.5 AP over FPN Faster R-CNN
    - +2.0 AP over RetinaNet
    - Comprehensive solution to multiple imbalance types

  - Cons:
    - Adds complexity to training pipeline
    - More hyperparameters to tune

- **DN-DETR: Accelerate DETR Training by Introducing Query DeNoising**. Feng Li et.al. **CVPR**, **2022**, [(Arxiv)](https://arxiv.org/abs/2203.01305) [(S2)](https://www.semanticscholar.org/paper/DN-DETR%3A-Accelerate-DETR-Training-by-Introducing-Li-Zhang/78d02f2909a582c624eca2d0f67c91ee91974180).

  - Takeaway: DN-DETR accelerates DETR convergence by feeding noised ground-truth boxes into decoder and reconstructing original boxes.

  - Motivation: DETR converges slowly (requires 500+ epochs) due to unstable bipartite matching in early training.

  - Core Mechanism:
    - **Denoising Training**:
      1. Add noise to ground-truth boxes (random offset, scaling)
      2. Feed noised boxes as decoder queries
      3. Reconstruct original unnoised boxes
    - Stabilizes bipartite matching by providing better supervision early
    - Reduces matching difficulty in early training stages
    - Auxiliary denoising loss

  - Pros:
    - Significantly accelerates convergence
    - Maintains DETR's simplicity (no NMS)
    - Reduces training epochs from 500 to ~150

  - Cons:
    - Adds auxiliary loss and computational cost during training
    - Requires tuning noise parameters

## Others

- EMA (Exponential Moving Average)
  - **What it is:** maintain a smoothed copy of model parameters updated every step.
  - **Update rule:** $\theta^{EMA}_t = \beta \theta^{EMA}_{t-1} + (1 - \beta) \theta_t$, where $\beta \in [0.9, 0.9999]$.
  - **How to use:** train with normal weights $\theta_t$, but **evaluate and/or export** using $\theta^{EMA}_t$.
  - **Why it helps:** reduces noise from SGD updates, improves stability, and often yields better validation/inference performance.
- earlystop
- label smoothing
- multi-stage training

掩蔽文本训练和跨实例对比学习

- zero-shot setting是什么

- 伪掩码注释，该数据集作为掩码头的主要训练数据

- 零样本检测表现
- 冻结骨干,新增关键头进行训练(不同关键点头负责不同内容分别进行训练)
