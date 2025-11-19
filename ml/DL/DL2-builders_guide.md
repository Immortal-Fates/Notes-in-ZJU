# Main Takeaway

builders guide

<!--more-->



# Layers and Modules

- 使用pytorch提供的块

  ```
  import torch
  from torch import nn
  from torch.nn import functional as F
  
  net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
  
  X = torch.rand(2, 20)
  net(X)
  ```

- 自定义块

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

# Parameter Management

- Parameter Access

  参数是复合对象，包括值，梯度和额外信息

- Tied Parameter参数绑定：有时我们希望在多个层间共享参数：我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数

  ```
  # 我们需要给共享层一个名称，以便可以引用它的参数
  shared = nn.Linear(8, 8)
  net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                      shared, nn.ReLU(),
                      shared, nn.ReLU(),
                      nn.Linear(8, 1))
  ```

  在反向传播时，**它的梯度会自动累加来自多个使用位置的梯度**

  

# Deferred-Init

深度学习框架无法判断网络的输入维度是什么。这里的诀窍是框架的**延后初始化**（defers initialization），即直到数据第一次通过模型传递时，框架才会动态地推断出每个层的大小。

```
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
```



# Custom Layers

```
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

# File I/O

how to load and store both individual weight vectors and entire models

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



# GPUs

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



# References

- [6. Builders’ Guide — Dive into Deep Learning 1.0.3 documentation](https://d2l.ai/chapter_builders-guide/index.html)