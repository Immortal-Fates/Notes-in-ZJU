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

## 发展脉络

- 从滑窗和部件模型，到 proposal-based CNN，再到 dense one-stage detector 到sparse detector(也就是query-based set prediction)
- 从 anchor 到 anchor-free，再到 query-based set prediction
- 从固定类别监督学习，到自监督预训练和视觉语言开放词表检测

check the excalidraw
