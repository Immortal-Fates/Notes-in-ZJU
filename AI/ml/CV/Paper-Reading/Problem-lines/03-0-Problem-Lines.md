# Problem Lines Overview

[TOC]

本部分按照 **research problem / 问题线** 组织目标检测论文，而不是按照模型家族组织。

完整论文笔记仍然保留在 `02-*` model-zoo 文件，或者保留在 archive 类型的详细笔记中。`03-*` 文件只负责梳理问题线，主要包含 **Paper Matrix** 和紧凑的 `## Relation` 图。

## Problem Lines

| 文件 | 核心问题 | 主要来源 |
| --- | --- | --- |
| [03-1-Loss](03-1-Loss.md) | detection loss 如何统一分类、定位、质量估计和排序？ | `03-1-Loss.md`, `02-OD-Model-Zoo.md`, `02-2-YOLO-Zoo.md` |
| [03-2-NMS-and-End-to-End](03-2-NMS-and-End-to-End.md) | 如何减少或去掉 NMS 这类 heuristic post-processing？ | `02-3-DETR-Zoo.md`, `02-2-YOLO-Zoo.md` |
| [03-3-Label-Assignment](03-3-Label-Assignment.md) | 训练时 prediction / anchor / query 应该如何分配给 GT？ | `03-3-Label-Assignment.md`, `02-3-DETR-Zoo.md`, `02-2-YOLO-Zoo.md` |
| [03-4-Small-Object](03-4-Small-Object.md) | 小目标为什么容易失败？哪些机制能提升 recall 和 localization？ | `03-4-Small-Object.md`, `02-3-DETR-Zoo.md`, `02-OD-Model-Zoo.md` |
| [03-5-Multi-Scale-Feature-Fusion](03-5-Multi-Scale-Feature-Fusion.md) | detector 如何同时利用高分辨率细节和高层语义？ | `02-OD-Model-Zoo.md`, `02-2-YOLO-Zoo.md`, `02-3-DETR-Zoo.md` |
| [03-6-Class-Imbalance](03-6-Class-Imbalance.md) | 如何处理前景/背景、easy/hard example、long-tail imbalance？ | `03-6-Class-Imbalance.md`, `03-1-Loss.md` |
| [03-7-Efficient-Deployment](03-7-Efficient-Deployment.md) | 如何提升真实部署中的 latency、memory 和推理简洁性？ | `02-OD-Model-Zoo.md`, `02-2-YOLO-Zoo.md`, `02-5-Vit-Zoo.md` |

## Rule

`03-*` 笔记只回答：**一篇论文为什么属于这条问题线**。如果开始需要完整公式、代码、pipeline 或详细结构说明，就说明这部分应该保留在对应的 model-zoo note 中，然后在 matrix 里链接过去。
