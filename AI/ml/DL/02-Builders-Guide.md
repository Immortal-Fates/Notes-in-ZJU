# Main Takeaway

usage manual

<!--more-->

## Workflow

show workflow when training on local or remote server.

### Useful Cmds

| Task                  | Command                                 | Explanation                           |
| --------------------- | --------------------------------------- | ------------------------------------- |
| GPU monitoring        | `watch -n 1 nvidia-smi`                 | Real-time GPU usage.                  |
| CPU/memory monitoring | `htop/top`                              | Interactive system usage.             |
| Tail logs             | `tail -f log.txt`                       | Monitor logs live.                    |
| Run TensorBoard       | `tensorboard --logdir runs --port 6006` | Launch TensorBoard in its own window. |

- [visualize nn architecture](https://netron.app/)

### Remote Server

- first connect

  Generate a Key Pair(local machine)

  ```
  ssh-keygen -t rsa(whatever, you may already have)
  ```

  Copy Public Key to Remote Server

  - if work:

    ```
    ssh-copy-id username@remote_ip
    ```

    ```
    ssh-copy-id -i .ssh/id_ed25519 foobar@remote
    ```

  - else

    ```
    cat .ssh/id_rsa.pub | ssh foobar@remote 'cat >> ~/.ssh/authorized_keys'
    ```

  After this, you can log in without password.

- copy files

  - ssh+tee
  - scp
  - **rsync**
  - fillzilla

- normal

| Task                     | Command                                                      | Explanation                               |
| ------------------------ | ------------------------------------------------------------ | ----------------------------------------- |
| ssh connect              | `ssh user@ip`                                                | remote connect                            |
| Sync code                | `git` or `rsync -avz --delete ~/ws/remote-project/ root@connect.nmb2.seetacloud.com:/root/Test-Example/` | Sync the code                             |
| Start training           | `tmux new -s train` → `python train.py`                      | Begin training safely inside tmux.        |
| Detach                   | `Ctrl-b d`                                                   | Training continues even if SSH drops.     |
| Resume                   | `tmux a -t train`                                            | Reconnect anytime.                        |
| Open tb forward the port | On the server: `tensorboard --logdir experiments --port 6006` | Add a TB writer and then forward the port |
| Local open TB            | `ssh -L 6006:localhost:6006 user@server`<br />[http://localhost:6006](http://localhost:6006/) | Check TB on local machine                 |
| Open log window          | `Ctrl-b c` → `tail -f log.txt`                               | Monitor logs separately.                  |
| Open GPU monitor         | Split pane → `watch -n 1 nvidia-smi`                         | Real-time GPU usage.                      |

### Hyperparameter tuning

use **Optuna** for hyperparameter tuning

## Layers and Modules

- 使用pytorch提供的块

  ```
  import torch
  from torch import nn
  from torch.nn import functional as F

  net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

  X = torch.rand(2, 20)
  net(X)
  ```

- Custom Layers

  自定义块的基本功能

  1. 将输入数据作为其前向传播函数的参数。
  1. 通过前向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。
  1. 计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的。
  1. 存储和访问前向传播计算所需的参数。
  1. 根据需要初始化模型参数。

  需要写自己的构造函数和前向传播

  ```
  class MLP(nn.Module):
      # 用模型参数声明层。这里，我们声明两个全连接的层
      def __init__(self):
          # 调用MLP的父类Module的构造函数来执行必要的初始化。
          # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
          super().__init__()
          self.hidden = nn.Linear(20, 256)  # 隐藏层
          self.out = nn.Linear(256, 10)  # 输出层
  
      # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
      def forward(self, X):
          # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
          return self.out(F.relu(self.hidden(X)))
  ```

## Parameter Management

- Parameter Access

  参数是复合对象，包括值，梯度和额外信息

- Tied Parameter参数绑定：有时我们希望在多个层间共享参数，我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数

  ```
  # 我们需要给共享层一个名称，以便可以引用它的参数
  shared = nn.Linear(8, 8)
  net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                      shared, nn.ReLU(),
                      shared, nn.ReLU(),
                      nn.Linear(8, 1))
  ```

  在反向传播时，**它的梯度会自动累加来自多个使用位置的梯度**

## Deferred-Init

深度学习框架无法判断网络的输入维度是什么。这里的诀窍是框架的**延后初始化**（defers initialization），即直到数据第一次通过模型传递时，框架才会动态地推断出每个层的大小。

```
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
```

## File I/O

- Takeaway: how to load and store both individual weight vectors and entire models


深度学习框架提供了内置函数来保存和加载整个网络。需要注意的一个重要细节是，这将保存模型的参数而不是保存整个模型。例如，如果我们有一个3层多层感知机，我们需要单独指定架构。因为模型本身可以包含任意代码，所以模型本身难以序列化。因此，为了恢复模型，我们需要用代码生成架构，然后从磁盘加载参数。

- 保存：

  ```
  torch.save(net.state_dict(), 'mlp.params')
  ```

- 加载：需要先实例化一个模型

  ```
  clone = MLP()
  clone.load_state_dict(torch.load('mlp.params'))
  clone.eval()
  ```

## GPUs

在PyTorch中，每个数组都有一个设备（device），我们通常将其称为环境（context）。默认情况下，所有变量和相关的计算都分配给CPU。有时环境可能是GPU。当我们跨多个服务器部署作业时，事情会变得更加棘手。通过智能地将数组分配给环境，我们可以最大限度地减少在设备之间传输数据的时间。例如，当在带有GPU的服务器上训练神经网络时，我们通常希望模型的参数在GPU上。

- 查看gpu

  ```
  torch.device('cuda')
  ```

  ```
  def try_gpu(i=0):  #@save
      """如果存在，则返回gpu(i)，否则返回cpu()"""
      if torch.cuda.device_count() >= i + 1:
          return torch.device(f'cuda:{i}')
      return torch.device('cpu')

  ```

- 我们可以[**查询张量所在的设备。**]，默认是cpu

  ```
  x = torch.tensor([1, 2, 3])
  x.device
  ```

- 存储在gpu上

  ```
  X = torch.ones(2, 3, device=try_gpu())
  ```

  在不同GPU上进行操作需要使用cuda（到时候再学）

## Loss Function

Just check the [Loss function document](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.L1Loss.html) which includes the math formula and how to use it in pytorch.

- BCE loss
- check the [loss function](./CV/Paper-Reading/03-OD-Loss-Zoo.md)
- Focal loss
  - Quality Focal Loss (QFL) / Varifocal Loss (VFL) / Generalized Focal Loss (GFL, v1, v2)


## Activation Function

Just check the [Activation function document](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.ELU.html)

| Name             | Formula                                                      | Pros                                                         | Cons                                                         |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Sigmoid**      | $f(x)=\frac{1}{1+e^{-x}}$                                    | Smooth; outputs in $(0,1)$; probabilistic interpretation     | Severe vanishing gradients; not zero-centered; slow convergence |
| **Tanh**         | $f(x)=\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$                | Zero-centered; stronger gradients than sigmoid; good for hidden layers | Still vanishes for large \|x\|; expensive to compute         |
| **ReLU**         | $f(x)=\max(0,x)$                                             | Very fast; simple; no vanishing gradient for $x>0$; sparse activations | “Dying ReLU” problem; unbounded output can cause instability |
| **Leaky ReLU**   | $f(x)=\max(\alpha x, x)$                                     | Fixes dying ReLU; small negative slope keeps gradients alive | Slightly more computation; $\alpha$ is a hyperparameter      |
| **PReLU**        | $f(x)=\max(a x, x)$ (learnable $a$)                          | Learns negative slope; flexible                              | Extra parameters; risk of overfitting                        |
| **ELU**          | $f(x)=\begin{cases}x & x\ge0 \\ \alpha(e^x-1) & x<0\end{cases}$ | Negative values push mean toward zero; smooth                | Exponential is expensive; saturates for $x \ll 0$            |
| **SELU**         | $f(x)=\lambda\begin{cases}x & x\ge0 \\ \alpha(e^x-1) & x<0\end{cases}$ | Self-normalizing networks (stable variance)                  | Must follow specific architecture rules; unstable in CNNs    |
| **Softplus**     | $f(x)=\ln(1+e^x)$                                            | Smooth ReLU approximation; avoids sharp corners              | Slow; saturates for large negative x                         |
| **Swish / SiLU** | $f(x)=x \cdot \sigma(x)$                                     | Smooth and non-monotonic; improves accuracy; used in EfficientNet/YOLOv5 | More expensive (sigmoid); harder to tune analytically        |
| **Mish**         | $f(x)=x\cdot\tanh(\ln(1+e^x))$                               | Very smooth; strong empirical performance (YOLOv4)           | Expensive; more complex derivative                           |
| **GELU**         | $f(x)=x\Phi(x)$ (Gaussian CDF)                               | Default for Transformers; excellent for deep MLPs            | Expensive; less interpretable                                |
| **Hard-Swish**   | $f(x)=x\cdot \frac{\text{ReLU6}(x+3)}{6}$                    | Efficient Swish approximation; mobile-friendly               | Less smooth; heuristic design                                |
| **ReLU6**        | $f(x)=\min(\max(0,x),6)$                                     | Supports quantization; mobile-efficient                      | Caps activation → may limit representation                   |

## Learning Scheduler

check the pytorch [scheduler](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LRScheduler.html#lrscheduler)

- LambdaLR

ema

## Model Indicator

Introduce indicators for measuring the complexity of a model.

### Summary

| Indicator                       | Type               | What it measures                                             | How to calculate (short)                                     | Effect when reduced (if accuracy maintained)                 |
| ------------------------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| FLOP                            | Unit               | A single floating-point arithmetic operation (e.g., `+`, `-`, `*`, `/`). | Atomic unit; algorithms are counted as a sum of many FLOPs.  | Fewer FLOPs in an algorithm → less compute, potentially lower latency/energy. |
| FLOPs (total ops per inference) | Model              | Total floating-point operations for one forward pass (computational complexity). | Sum over layers; e.g., Conv MACs: `H_out * W_out * C_out * (k_h * k_w * C_in)`; FLOPs ≈ `2 * MACs` (mul + add). | Lower theoretical compute cost, usually faster and more energy-efficient. |
| FLOPS (FLOP/s, throughput)      | Hardware           | Floating-point operations per second (hardware compute capability). | Given by hardware spec, e.g., `10 TFLOPS = 10 * 10^12 FLOP/s`. | Not a model property; higher FLOPS → lower latency for same model FLOPs. |
| MACs                            | Model              | Multiply–accumulate operations per inference.                | Conv MACs: `H_out * W_out * C_out * (k_h * k_w * C_in)` (1 MAC = 1 or 2 FLOPs depending on convention). | Same trend as FLOPs; good proxy for compute.                 |
| Parameter Count (#params)       | Model              | Number of learnable weights (capacity / structural complexity). | Sum of params over all layers; Conv: `k_h * k_w * C_in * C_out (+ bias)`; Linear: `in_dim * out_dim (+ bias)`. | Smaller weight memory, easier deployment, but less representational capacity. |
| Memory Access                   | Unit               | The total amount of data that must be read from or written to memory during an operation (feature maps + weights). | Approximate as: Input size + Output size + Weight size  (in number of elements or bytes). | Lower bandwidth demand → faster inference on memory-bound hardware; reduced energy; better real-time performance. |
| Model Size on Disk              | Model              | File size of stored weights (storage / download cost).       | Approx. `#params * bits_per_param / 8` (bytes) `+` metadata. | Smaller files, faster download, fits into limited flash/ROM. |
| Peak Runtime Memory             | Model+<br>Hardware | Max RAM/VRAM usage (weights + activations + buffers) during inference. | ≈ `model_memory + max_live_activations_memory`; usually measured empirically on target device. | Enables deployment on memory-limited devices; allows larger batch sizes. |
| Latency                         | Runtime            | Wall-clock time for one inference (or per batch).            | Measure average `end_time - start_time` over many runs on target hardware. | Directly impacts responsiveness / fps in real-time systems.  |
| Throughput (samples / second)   | Runtime            | Number of inputs processed per second.                       | `throughput = batch_size / batch_latency`.                   | Higher throughput = more streams/users on same hardware.     |
| Energy per inference / Power    | Runtime            | Energy or average power draw for running the model.          | Measured via power APIs/meters; ≈ `#ops * energy_per_op + memory_energy`. | Critical for battery / thermal limits; lower values extend device lifetime. |
| Compression Ratio               | Summary            | How much smaller a compressed model is vs baseline.          | `compression_ratio = size_original / size_compressed` (or using `#params`). | Higher ratio = more aggressive compression; summarizes pruning + quantization. |
| Sparsity / Density              | Model              | Fraction of zero (or non-zero) weights; pruning level.       | `sparsity = #zero_params / #total_params`; `density = 1 - sparsity`. | Can reduce effective FLOPs and size if hardware/libraries exploit sparsity. |
| Bitwidth / Numerical Precision  | Model              | Bits used per weight/activation (FP32, FP16, INT8, INT4, etc.). | Defined by quantization scheme; effective size per param = `bitwidth` bits. | Reduces model size and memory bandwidth; often speeds up inference on LP units. |
| Network Depth                   | Architecture       | Number of layers in the network.                             | Count of sequential learnable layers (e.g., 50 in ResNet-50). | Fewer layers → fewer params/FLOPs but lower representational power. |
| Network Width                   | Architecture       | Number of channels/units per layer.                          | Channel/hidden size per layer; e.g., width multiplier `α` scales channels: `C' = α * C`. | Narrower network → fewer params/FLOPs; too narrow hurts accuracy. |

### Calculation

- model size

  - conv layer
    $$
    \text{param}_{w} = K\times K \times channel_{in} \times channel_{out}
    $$

    $$
    \text{param}_{b} = K\times K \times channel_{out}
    $$

    $$
    so \quad \text{param}  =  \text{param}_{w}+\text{param}_{b}
    $$

  - FC layer
    $$
    \text{param}_{w} = N_{in}\times N_{out}
    $$

    $$
    \text{param}_{b} = N_{out}
    $$

    $$
    so \quad \text{param}  =  \text{param}_{w}+\text{param}_{b}
    $$

- activation memory
  $$
  \text{memory}_{act} = H\times W\times Channel_{out} \times \text{bytes per element}
  $$


- FLOPs

  - conv layer(consider bais)

    只需在 parameters 的基础上再乘以 feature map 的大小即可，即对于某个卷积层，它的 FLOPs 数量为：

    $$
    \text{FLOPs} = 2 K_{in} \times K_{out} \times C_{in} \times C_{out} \times H_{out} \times W_{out} \tag{1} \\
    addition:(K_{in} \times K_{out} \times C_{in} - 1) \times C_{out} \times H_{out} \times W_{out} \\
    multiplication:K_{in} \times K_{out} \times C_{in} \times C_{out} \times H_{out} \times W_{out} \\
    bais:1 \times C_{out} \times H_{out} \times W_{out}
    $$

  - FC layer
    $$
    FLOPs = 2(N_{in}-1) N_{out} \\
    addition: (N_{in}-1)N_out \\
    multiplication: N_{in}N_{out}
    $$

- MACs
  $$
  1MACs \approx 2FLOPs
  $$

- Memory Access

  - conv layer
    $$
    \text{Memory Access} = H_{in}W_{in} C_{in} + H_{out}W_{out} C_{out} + C_{out}C_{in} K^2
    $$

- Computation / Memory Access Ratio

  - High ratio → computation dominates (GPU-friendly)
  - Low ratio → memory access dominates (mobile/edge bottleneck)


### Model Compression

When comes to model compression. We usually care about the three parameters:

- Model Size
- Runtime Memory
- Number of computing operations: two ways to calculate: 
  - FLOPS
  - MACs

### Open-Src Toolkit

- __A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit.__ *Rafael Padilla et al.* __Electronics, 2021__ [(Link)](https://doi.org/10.3390/electronics10030279) [(S2)](https://www.semanticscholar.org/paper/7d72fb27184c1e5793e382681dbbc853fe055093) [(code_link_old)](https://github.com/rafaelpadilla/Object-Detection-Metrics)[(code_link_new)](https://github.com/rafaelpadilla/review_object_detection_metrics)(Citations __546__)


## Tips and Tricks

check this [file](./11-Tips-and-Tricks.md)

## References

- [6. Builders’ Guide — Dive into Deep Learning 1.0.3 documentation](https://d2l.ai/chapter_builders-guide/index.html)
- [python-numpy-tutorial](https://cs231n.github.io/python-numpy-tutorial/)
