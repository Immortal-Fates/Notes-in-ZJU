# Model Compression

Model compression techniques aim to make deep learning models smaller, faster, and more efficient, without severely sacrificing accuracy. The objective is to reduce computational and storage costs while maintaining acceptable performance.

---

## Applications

| Application            | Objective                                 | Typical Techniques         |
| ---------------------- | ----------------------------------------- | -------------------------- |
| Mobile and Embedded AI | Reduce memory and latency                 | Quantization, pruning      |
| Edge Computing         | Efficient deployment on low-power devices | Pruning, distillation      |
| Large Language Models  | Minimize inference cost                   | Distillation, quantization |
| Autonomous Systems     | Real-time inference                       | Structured pruning         |

## Summary

| Technique               | Main Idea                          | Advantages                     | Challenges             |
| ----------------------- | ---------------------------------- | ------------------------------ | ---------------------- |
| Pruning                 | Remove redundant weights           | Reduces size, faster inference | Needs fine-tuning      |
| Quantization            | Lower numerical precision          | Memory and energy efficiency   | Accuracy degradation   |
| Knowledge Distillation  | Transfer from large to small model | High efficiency                | Requires teacher model |
| Binary/Ternary Networks | Extreme quantization               | Maximum compression            | Accuracy drop          |

## References

- **Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding**. Song Han et.al. **arxiv**, **2015**, ([link](https://arxiv.org/abs/1510.00149v5)).
- [MIT course](https://hanlab.mit.edu/courses/2024-fall-65940)