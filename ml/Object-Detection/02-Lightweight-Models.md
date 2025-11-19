# Lightweight Models

Focus on lightweight models and their optimizations for embedded systems.

- ResNet (residual learning)
- MobileNet v1 (depthwise conv)
- MobileNet v2 (Inverted Residual + Linear Bottleneck)
- ShuffleNet (channel shuffle)
- GhostNet (Ghost Module – Cheap Operations)

# Depthwise Separable Convolutions(DSC)

> - **Evaluation Of Trainable Parameters**
> - **Evaluation Of Computational Cost**

- why: The main reason for using DSC is **efficiency**. It significantly reduces the computational cost and the number of parameters compared to standard convolutions. And it also maintain performance.(a bit worse)

- what: Depthwise Separable Convolutions (DSC) are a variant of the standard convolution operation used in Convolutional Neural Networks (CNNs). Unlike regular convolutions, DSC splits the convolution operation into two parts:

  1. A **depthwise grouped convolution**, w[here](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) the number of input channels m is equal to the number of output channels such that each output channel is affected only by a single input channel. In PyTorch, this is called a "grouped" convolution.
  2. A **pointwise convolution** (filter size=1), which operates like a regular convolution such that each of the n filters operates on all m input channels to produce a single output value.

- how: For an input feature map of size \( $H \times W \times C $\) (Height x Width x Channels):

  - **Standard Convolution** would use $ K_h \times K_w \times C_{in} \times C_{out} $ parameters, where $ K_h $  and $ K_w $ are the height and width of the kernel, $ C_{in} $ is the number of input channels, and $ C_{out} $ is the number of output channels.

  - **Depthwise Separable Convolution** would use:
    - $$ K_h \times K_w \times C_{in} $$ parameters for the depthwise convolution
    - $$ 1 \times 1 \times C_{in} \times C_{out} $$ parameters for the pointwise convolution.

  This reduces the number of parameters and the computational complexity.

  ```
  # a tiny example
  class DepthwiseSeparableConv(nn.Module):
      def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
          super(DepthwiseSeparableConv, self).__init__()

          # Depthwise Convolution
          self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, groups=in_channels)

          # Pointwise Convolution
          self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

      def forward(self, x):
          # Apply depthwise convolution
          x = self.depthwise(x)
          # Apply pointwise convolution
          x = self.pointwise(x)
          return x

  # Example of using the Depthwise Separable Convolution layer
  input_tensor = torch.randn(1, 3, 64, 64)  # Example input with batch size 1, 3 channels, 64x64 image
  model = DepthwiseSeparableConv(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
  output_tensor = model(input_tensor)

  print(f'Output shape: {output_tensor.shape}')
  ```

# Lightweight Nets

- **MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications**. Andrew G. Howard et.al. **arxiv**, **2017**, ([link](https://arxiv.org/abs/1704.04861v1)).

  - Depthwise Separable Convolutions(DSC)

  - two simple global hyperparameters that efficiently trade off between latency and accuracy

    - width multiplier $\alpha$: thinner models, controls the number of channels in each layer

    - resolution multiplier $\rho$: reduce the computational cost, controls the input image resolution
      $$
      D_K \times D_K \times \alpha M \times \rho D_F \times \rho D_F+\alpha M \times \alpha N \times  \rho D_F \times \rho D_F
      $$

  - less regularization and data augmentation techniques because small models have less trouble with overfitting

- **MobileNetV2: Inverted Residuals and Linear Bottlenecks**. Mark Sandler et.al. **arxiv**, **2018**, ([link](https://arxiv.org/abs/1801.04381v4)).

  - a novel layer module: the inverted residual with linear bottleneck

    This module takes as an input a low-dimensional compressed representation which is first expanded to high dimension and filtered with a lightweight depthwise convolution. Features are subsequently projected back to a low-dimensional representation with a linear convolution

  - remove non-linearities in the narrow layers in order to maintain representational power

- **ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices**. Xiangyu Zhang et.al. **arxiv**, **2017**, ([link](https://arxiv.org/abs/1707.01083v2)).

- **GhostNet: More Features from Cheap Operations**. Kai Han et.al. **arxiv**, **2019**, ([link](https://arxiv.org/abs/1911.11907v2)).

bottleneck ?? is what

why: residual connections, shortcuts enable faster training and better accuracy

# References

- [Depthwise Convolution explanation]( https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)
- [MobileNetv2 explanation]( https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html)
- [MobileNetV2 explained video](https://www.youtube.com/watch?v=DkNIBBBvcPs)
- [MobileNetV1](https://research.google/blog/mobilenets-open-source-models-for-efficient-on-device-vision/?_gl=1)
