# Operation Design

[TOC]

## Structural Reparameterization

(RepVGG, ACNet, DBB)
 Train with complex multi-branch blocks → deploy as single-path conv.

## Dynamic Inference / Conditional Computation

- Early exiting (e.g., BranchyNet)
- Dynamic depth (skip layers)
- Dynamic width (Adaptive channel selection)
- Token pruning for Transformers
- Mixture-of-Experts routing

## Activation Compression

- Activation quantization
- Activation sparsification
- Checkpointing for memory reductions
- Reversible networks (RevNets)

## Progressive Compression / Multi-Stage Methods

- Prune → retrain → quantize → distill
- Compound compression pipelines

## Weight Sharing

 Weight sharing by scalar quantization (top) and centroids fine-tuning (bottom).

![image-20251123150437396](./assets/07-Operation-Design.assets/image-20251123150437396.png)



- HashNet-style weight hashing
- Shared-weight architectures (Cell-based NAS blocks)

## Efficient Attention Mechanisms

(Used especially for Vision Transformers)

- Linformer
- Performer (kernelized attention)
- Nyströmformer
- Sparse or block attention
- Low-rank attention

