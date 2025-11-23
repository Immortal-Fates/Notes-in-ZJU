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

- HashNet-style weight hashing
- Shared-weight architectures (Cell-based NAS blocks)

## Efficient Attention Mechanisms

(Used especially for Vision Transformers)

- Linformer
- Performer (kernelized attention)
- Nyströmformer
- Sparse or block attention
- Low-rank attention

