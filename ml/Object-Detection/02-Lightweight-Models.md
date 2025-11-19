# Lightweight Nets



# Depthwise Separable Convolutions(DSC)

> - **Evaluation Of Trainable Parameters**
> - **Evaluation Of Computational Cost**

- why: a technique to reduce model size and training/inference cost without a significant loss in validation accuracys

- what: The main difference between a regular convolution and a DSC is that a DSC is composed of 2 convolutions as described below:

  1. A **depthwise grouped convolution**, w[here](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) the number of input channels m is equal to the number of output channels such that each output channel is affected only by a single input channel. In PyTorch, this is called a "grouped" convolution.
  2. A **pointwise convolution** (filter size=1), which operates like a regular convolution such that each of the n filters operates on all m input channels to produce a single output value.

- how: The “grouped” convolutions have m filters, each of which has *dₖ x dₖ* learnable parameters which produces m output channels. This results in a total of *m x dₖ x dₖ* learnable parameters. The pointwise convolution has n filters of size *m x 1 x 1* which adds up to *n x m x 1 x 1* learnable parameters.

  ```
  # a tiny example
  class DepthwiseSeparableConv(nn.Sequential):
      def __init__(self, chin, chout, dk):
          super().__init__(
              # Depthwise convolution
              nn.Conv2d(chin, chin, kernel_size=dk, stride=1, padding=dk-2, bias=False, groups=chin),
              # Pointwise convolution
              nn.Conv2d(chin, chout, kernel_size=1, bias=False),
          )
  ```
  





# Lightweight Nets

- {1704.04861}
- {1801.04381}
- 







bottleneck ?? is what

why: residual connections, shortcuts enable faster training and better accuracy

# References

- [Depthwise Convolution explanation]( https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)
- [MobileNetv2 explanation]( https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html)
- [MobileNetV2 explained video](https://www.youtube.com/watch?v=DkNIBBBvcPs)
- [MobileNetV1](https://research.google/blog/mobilenets-open-source-models-for-efficient-on-device-vision/?_gl=1)