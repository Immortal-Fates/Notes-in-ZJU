# Training Strategies

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

## Label Assignment Strategy

- **ATSS: Adaptive Training Sample Selection**. Shifeng Zhang et.al. **CVPR**, **2020**, [(Arxiv)](https://arxiv.org/abs/1912.02424) [(S2)](https://www.semanticscholar.org/paper/Bridging-the-Gap-Between-Anchor-Based-and-Detection-Zhang-Chi/448529da2bf004cf79084401ad3cbd6b511e4969).
  - Takeaway: ATSS automatically selects positive samples based on statistical characteristics, bridging anchor-based and anchor-free detectors.
  
  - Motivation: Label assignment in detection relies on fixed IoU thresholds; anchor-based vs anchor-free detectors have different assignment strategies.
  
  - Core Mechanism:
    - **Per-level selection**: Select $k$ anchors closest to GT center per pyramid level
    - **Dynamic IoU threshold**: $t_g = m_g + v_g$ (mean + std of IoU values)
    - Statistical characteristics determine assignment, not hand-tuned thresholds
    - Unified approach works for both anchor-based and anchor-free
  
  - Pros:
    - No hyperparameters for IoU threshold
    - Bridges gap between detection paradigms
    - Better performance than fixed-threshold methods
  
  - Cons:
    - Still level-based assignment
    - May not handle extreme scale variations optimally
  
- **OTA: Optimal Transport Assignment**. Zhishuai Ge et.al. **CVPR**, **2021**, [(Arxiv)](https://arxiv.org/abs/2103.14259) [(S2)](https://www.semanticscholar.org/paper/OTA%3A-Optimal-Transport-Assignment-for-Object-Ge-Liu/3bc6e930f0114202f668d35cd733c29f6dca7ebb).
  - Takeaway: OTA formulates label assignment as optimal transport problem, finding globally optimal assignment for all anchors simultaneously.
  
  - Motivation: Most assignment strategies assign anchors per-GT greedily, ignoring global optimal configuration.
  
  - Core Mechanism:
    - **Optimal Transport (OT) formulation**: Global perspective for all anchors
    - Transportation cost: Weighted sum of classification and regression losses
    - Solves OT problem to find optimal one-to-one matching
    - Global optimal vs per-GT local optimal
    - Addresses issue where one anchor might be better matched to a different GT
  
  - Pros:
    - Globally optimal assignment
    - Theoretically principled approach
    - Outperforms greedy assignment strategies
  
  - Cons:
    - Higher computational cost (solving OT problem)
    - Complex implementation
  
- __Category-Aware Dynamic Label Assignment with High-Quality Oriented Proposal.__ *Mingkui Feng et al.* __ArXiv, 2024__ [(Arxiv)](https://arxiv.org/abs/2407.03205) [(S2)](https://www.semanticscholar.org/paper/ccb0d094cf39cb17cf29214cb930f0dce9ca3211) (Citations __4__)

- __Integrating Diverse Assignment Strategies into DETRs.__ *Yiwei Zhang et al.* __arXiv, 2026__ [(Arxiv)](https://arxiv.org/abs/2601.09247) 
- __Point2RBox-v3: Self-Bootstrapping from Point Annotations via Integrated Pseudo-Label Refinement and Utilization.__ *Teng Zhang et al.* __arXiv, 2025__ [(Arxiv)](https://arxiv.org/abs/2509.26281) 
- https://openaccess.thecvf.com/content/CVPR2025/html/Liu_FSHNet_Fully_Sparse_Hybrid_Network_for_3D_Object_Detection_CVPR_2025_paper.html







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

掩蔽文本训练和跨实例对比学习

- zero-shot setting是什么

- 伪掩码注释，该数据集作为掩码头的主要训练数据

- 零样本检测表现
- 冻结骨干,新增关键头进行训练(不同关键点头负责不同内容分别进行训练)
