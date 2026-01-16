# Tips and Tricks

introduction to the engineer tips and tricks.



## Training Skills

- EMA (Exponential Moving Average)
  - **What it is:** maintain a smoothed copy of model parameters updated every step.
  - **Update rule:** $\theta^{EMA}_t = \beta \theta^{EMA}_{t-1} + (1 - \beta) \theta_t$, where $\beta \in [0.9, 0.9999]$.
  - **How to use:** train with normal weights $\theta_t$, but **evaluate and/or export** using $\theta^{EMA}_t$.
  - **Why it helps:** reduces noise from SGD updates, improves stability, and often yields better validation/inference performance.
- earlystop
- label smoothing
- multi-stage training



## Open-Src Toolkit

- __A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit.__ *Rafael Padilla et al.* __Electronics, 2021__ [(Link)](https://doi.org/10.3390/electronics10030279) [(S2)](https://www.semanticscholar.org/paper/7d72fb27184c1e5793e382681dbbc853fe055093) [(code_link_old)](https://github.com/rafaelpadilla/Object-Detection-Metrics)[(code_link_new)](https://github.com/rafaelpadilla/review_object_detection_metrics)(Citations __546__)

## Learning Rate Schedulers

a) Warmup + Cosine

- **Linear warmup** at the start (1k–5k iterations) → stabilizes early training.
- **Cosine annealing LR decay** — dominant choice in recent models (YOLOv7/v8, DETR variants).

b) OneCycle Policy

From Leslie Smith; used effectively in many PyTorch pipelines:

- LR rises then falls (supervised with momentum adjustments).
- Works well with mixed precision and large batch regimes.

c) StepLR (legacy, still used)

- Classic “drop LR by factor at preset epochs.”
- Less common now in cutting‑edge pipelines (replaced mostly by cosine/one‑cycle), but still used in some frameworks and ablation baselines.

**Default modern choice:**
 **Warmup → Cosine decay** (with optional OneCycle) is standard across COCO/benchmark training.

## Data Augmentations

- https://paddlepedia.readthedocs.io/en/latest/tutorials/computer_vision/image_augmentation/index.html
- **AutoAugment** is an automatic data augmentation technique designed to improve the generalization and robustness of deep learning models by augmenting training data in a way that is optimized for the specific task. It was introduced in the paper "AutoAugment: Learning Augmentation Strategies from Data" by Ekin D. Cubuk, Barret Zoph, Dandelion Mané, Vijay Vasudevan, and Quoc V. Le (2019).

介绍有哪些数据增强的方法

- **Albumentations**：如果安装，含随机裁剪、模糊、中值模糊、灰度、CLAHE、亮度对比度、Gamma、JPEG 压缩等，可用于检测和分类（[Albumentations](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)、[classify_albumentations](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)）。
- **归一化/反归一化**：[normalize](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)、[denormalize](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)。
- **颜色变换**：[augment_hsv](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) (HSV 随机扰动)、[hist_equalize](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) (直方图均衡/CLAHE)。
- **目标复制**：[replicate](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)（复制最小目标的一半）。
- **信箱缩放**：[letterbox](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)（按比例缩放并填充）。
- **随机透视/仿射**：[random_perspective](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)（旋转、平移、缩放、剪切、透视）。
- **Copy-Paste**：[copy_paste](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)（水平翻转后粘贴分割目标）。
- **Cutout**：[cutout](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)（随机遮挡并过滤被遮挡标签）。
- **MixUp**：[mixup](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)（图像线性混合并合并标签）。
- **候选框筛选**：[box_candidates](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)（过滤失真框）。
- **分类基本变换**：[classify_transforms](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)（CenterCrop+ToTensor+Normalize）。
- **常用预处理类**：[LetterBox](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)、[CenterCrop](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)、[ToTensor](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)（BGR→RGB，HWC→CHW，归一化，支持 FP16）。



- **数据增强**：Mosaic与MixUp（4图拼接+线性混合）、HSV抖动、翻转/缩放/平移/仿射、随机透视与剪裁、Albumentations 可选增强；utils/augmentations.py 与 datasets.py 中实现。
- **标签与样本策略**：标签平滑（smooth_BCE）、多标签处理、cls/obj 正负样本平衡、Focal/QFocal 可选（utils/loss.py）。
- **损失设计**：CIoU/DIoU/GIoU 盒回归，分离的 obj/cls BCE，类别自平衡，自动目标平衡（autobalance）调整各层 obj 权重；分割版额外原型掩码损失（utils/segment/loss.py）。
- **锚框与分辨率自适应**：自动锚框聚类与检查（utils/autoanchor.py），动态图像缩放/letterbox 保持纵横比，步幅对齐填充（datasets.py）。
- **批大小与显存优化**：自动批大小估计（utils/autobatch.py），半精度/混合精度训练 (--amp)，梯度累积，梯度裁剪，EMA 滑动平均权重（train.py）。
- **训练调度**：Cosine/Step lr 调度，warmup；多种超参预设 (data/hyps/*.yaml)，早停；多卡 DDP、SyncBN、分布式评估。
- **正负样本分配**：基于网格的候选匹配（anchor-matching），IoU/比例筛选，TTA 测试时多尺度+翻转融合（val.py）。
- **推理侧优化**：NMS 结合置信度与类权重，跨类/类内选择；导出多后端（ONNX/TensorRT/CoreML/OpenVINO等）；可用 --half 推理、--augment 多尺度 TTA；utils/general.py/export.py。
- **模型结构**：CSPDarknet + PAN/FPN 颈部，Focus/Conv/BN/Silu 组合；无池化，全卷积；轻量化版本 (s/m/l/x)、分割/姿态/检测多头同框架。
- **日志与可视化**：丰富的 logger（TensorBoard/ClearML/W&B/Comet/CSV），训练曲线、混淆矩阵、PR 曲线、样本可视化（utils/loggers/*）。
- **其他工程细节**：多种数据缓存（RAM/磁盘）、重复数据检测；自动下载权重与数据校验 (utils/downloads.py)，健壮的 resume/断点续训，随机种子控制，cudnn.benchmark 自适应。

## Modern Recipe

For a COCO‑like object detection training:

**Model:** Anchor‑free or YOLO‑family variant
 **Loss:**

- Classification: Focal/Varifocal
- Localization: CIoU/SIoU
- Objectness: BCE / IoU‑aware score
   **Augmentation:**
- Mosaic + MixUp + HSV
- RandAugment
- Multi‑scale training
   **Optimizer:** AdamW or SGD with momentum
   **LR Schedule:** Warmup → Cosine decay
   **Precision:** AMP (FP16)
   **Regularization:** Label smoothing + weight decay schedule
   **Inference:** NMS/Soft‑NMS; TTA optional

## References

