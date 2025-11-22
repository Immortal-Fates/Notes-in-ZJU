# Main Takeaway

回归模型

<!--more-->

机器学习模型中的关键要素是训练数据、损失函数、优化算法，还有模型本身

## Linear-Regression

### Mathematica

$$
\hat{y} = w_1  x_1 + ... + w_d  x_d + b.
$$

将所有特征放到向量$\mathbf{x} \in \mathbb{R}^d$中，并将所有权重放到向量$\mathbf{w} \in \mathbb{R}^d$中，

我们可以用点积形式来简洁地表达模型：
$$
\hat{y} = \mathbf{w}^\top \mathbf{x} + b.
$$
我们通过对噪声分布的假设来解读平方损失目标函数。

正态分布和线性回归之间的关系很密切。正态分布（normal distribution），也称为**高斯分布**（Gaussian distribution），

简单的说，若随机变量$x$具有均值$\mu$和方差$\sigma^2$（标准差$\sigma$），其正态分布概率密度函数如下：
$$
p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right).
$$
就像我们所看到的，改变均值会产生沿$x$轴的偏移，增加方差将会分散分布、降低其峰值。

均方误差损失函数（简称均方损失）可以用于线性回归的一个原因是：我们假设了观测中包含噪声，其中噪声服从正态分布。噪声正态分布如下式:
$$
y = \mathbf{w}^\top \mathbf{x} + b + \epsilon,
$$
其中，$\epsilon \sim \mathcal{N}(0, \sigma^2)$。

因此，我们现在可以写出通过给定的$\mathbf{x}$观测到特定$y$的**似然**（likelihood）：
$$
P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right).
$$
现在，根据极大似然估计法，参数$\mathbf{w}$和$b$的最优值是使整个数据集的**似然**最大的值：
$$
P(\mathbf y \mid \mathbf X) = \prod_{i=1}^{n} p(y^{(i)}|\mathbf{x}^{(i)}).
$$
根据极大似然估计法选择的估计量称为**极大似然估计量**。虽然使许多指数函数的乘积最大化看起来很困难，但是我们可以在不改变目标的前提下，通过最大化似然对数来简化。由于历史原因，优化通常是说最小化而不是最大化。

我们可以改为**最小化负对数似然**$-\log P(\mathbf y \mid \mathbf X)$。由此可以得到的数学公式是：
$$
-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2.
$$
现在我们只需要假设$\sigma$是某个固定常数就可以忽略第一项，

- 因为第一项不依赖于$\mathbf{w}$和$b$。
- 现在第二项除了常数$\frac{1}{\sigma^2}$外，其余部分和前面介绍的均方误差是一样的。

得到的均方误差（MSE）损失函数
$$
\min_{\mathbf{w},b} \sum(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b)^2
$$
幸运的是，上面式子的解并不依赖于$\sigma$。

因此，在高斯噪声的假设下，最小化均方误差等价于对线性模型的极大似然估计。

### FC layer

全连接层是“完全”连接的，可能有很多可学习的参数。

具体来说，对于任何具有$d$个输入和$q$个输出的全连接层，参数开销为$\mathcal{O}(dq)$，这个数字在实践中可能高得令人望而却步。幸运的是，将$d$个输入转换为$q$个输出的成本可以减少到$\mathcal{O}(\frac{dq}{n})$，其中超参数$n$可以由我们灵活指定，以在实际应用中平衡参数节约和模型有效性

## Softmax-Regression

> classification

softmax函数能够将未规范化的预测变换为非负数并且总和为1，同时让模型保持可导的性质。为了完成这一目标，我们首先对每个未规范化的预测求幂，这样可以确保输出非负。为了确保最终输出的概率值总和为1，我们再让每个求幂后的结果除以它们的总和。如下式：
$$
\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{其中}\quad \hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}
$$
这里，对于所有的$j$总有$0 \leq \hat{y}_j \leq 1$。因此，$\hat{\mathbf{y}}$可以视为一个正确的概率分布。

softmax运算不会改变未规范化的预测$\mathbf{o}$之间的大小次序，只会确定分配给每个类别的概率。因此，在预测过程中，我们仍然可以用下式来选择最有可能的类别。
$$
\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j.
$$
尽管softmax是一个非线性函数，但softmax回归的输出仍然由输入特征的仿射变换决定。

因此，softmax回归是一个**线性模型**（linear model）

这里我们需要交叉熵损失函数
$$
 l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j
$$
利用softmax的定义，我们得到：
$$
\begin{aligned}

l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\

&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j\\

&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.

\end{aligned}
$$
考虑相对于任何未规范化的预测$o_j$的导数，我们得到：
$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.
$$
换句话说，导数是我们softmax模型分配的概率与实际发生的情况（由独热标签向量表示）之间的差异。

从这个意义上讲，这与我们在回归中看到的非常相似，其中梯度是观测值$y$和估计值$\hat{y}$之间的差异。这不是巧合，在任何指数族分布模型中（参见[link](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/distributions.html)），对数似然的梯度正是由此得出的。这使梯度计算在实践中变得容易很多。

### 实现

我们计算了模型的输出，然后将此输出送入交叉熵损失。

从数学上讲，这是一件完全合理的事情。然而，从计算角度来看，指数可能会造成数值稳定性问题。回想一下，softmax函数$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$，其中$\hat y_j$是预测的概率分布。$o_j$是未规范化的预测$\mathbf{o}$的第$j$个元素。如果$o_k$中的一些数值非常大，那么$\exp(o_k)$可能大于数据类型容许的最大数字，即**上溢**（overflow）。这将使分母或分子变为`inf`（无穷大），最后得到的是0、`inf`或`nan`（不是数字）的$\hat y_j$。在这些情况下，我们无法得到一个明确定义的交叉熵值。

解决这个问题的一个技巧是：在继续softmax计算之前，先从所有$o_k$中减去$\max(o_k)$。这里可以看到每个$o_k$按常数进行的移动不会改变softmax的返回值：
$$
\begin{aligned}

\hat y_j & =  \frac{\exp(o_j - \max(o_k))\exp(\max(o_k))}{\sum_k \exp(o_k - \max(o_k))\exp(\max(o_k))} \\

& = \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}.

\end{aligned}
$$
在减法和规范化步骤之后，可能有些$o_j - \max(o_k)$具有较大的负值。由于精度受限，$\exp(o_j - \max(o_k))$将有接近零的值，即**下溢**（underflow）。这些值可能会四舍五入为零，使$\hat y_j$为零，并且使得$\log(\hat y_j)$的值为`-inf`。反向传播几步后，我们可能会发现自己面对一屏幕可怕的`nan`结果。尽管我们要计算指数函数，但我们最终在计算交叉熵损失时会取它们的对数。

通过将softmax和交叉熵结合在一起，可以避免反向传播过程中可能会困扰我们的数值稳定性问题。如下面的等式所示，我们避免计算$\exp(o_j - \max(o_k))$，而可以直接使用$o_j - \max(o_k)$，因为$\log(\exp(\cdot))$被抵消了。
$$
\begin{aligned}

\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}\right) \\

& = \log{(\exp(o_j - \max(o_k)))}-\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)} \\

& = o_j - \max(o_k) -\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)}.

\end{aligned}
$$
我们也希望保留传统的softmax函数，以备我们需要评估通过模型输出的概率。但是，我们没有将softmax概率传递到损失函数中，而是[**在交叉熵损失函数中传递未规范化的预测，并同时计算softmax及其对数**]，这是一种类似["LogSumExp技巧"](https://en.wikipedia.org/wiki/LogSumExp)的聪明方式。

> `LogSumExp`（**对数–指数求和技巧**）是机器学习和深度学习中非常常见的一个数学技巧，用来 **避免数值溢出、提高稳定性**。
>
> - 核心思想
>
>   在计算如下表达式时：
>
>   $$
>   \log \left( \sum_i e^{z_i} \right)
>   $$
>   直接计算可能导致数值溢出或下溢，因为：
>
>   - 若 \(z_i\) 很大，\(e^{z_i}\) 会非常大 → **上溢 (overflow)**
>   - 若 \(z_i\) 很小，\(e^{z_i}\) 会非常接近 0 → **下溢 (underflow)**
>
>   ---
>
>   💡 解决方法：提取最大值 \($m = \max_i z_i$\)
>   $$
>   \log \left( \sum_i e^{z_i} \right)
>   = \log \left( e^{m} \sum_i e^{z_i - m} \right)
>   = m + \log \left( \sum_i e^{z_i - m} \right)
>   $$
>   这样：
>
>   - 所有的 \(z_i - m \le 0\)，因此 \(e^{z_i - m} \in (0, 1]\)
>   - 避免了指数运算的数值爆炸或消失
>   - 数学上完全等价，数值上更加稳定

### 小批量样本矢量化

为了提高计算效率并且充分利用GPU，我们通常会对小批量样本的数据执行矢量计算。

假设我们读取了一个批量的样本$\mathbf{X}$，其中特征维度（输入数量）为$d$，批量大小为$n$。此外，假设我们在输出中有$q$个类别。那么小批量样本的特征为$\mathbf{X} \in \mathbb{R}^{n \times d}$，权重为$\mathbf{W} \in \mathbb{R}^{d \times q}$，偏置为$\mathbf{b} \in \mathbb{R}^{1\times q}$。

softmax回归的矢量计算表达式为：
$$
\begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned}
$$
相对于一次处理一个样本，小批量样本的矢量化加快了$\mathbf{X}和\mathbf{W}$的矩阵-向量乘法。

由于$\mathbf{X}$中的每一行代表一个数据样本，那么softmax运算可以**按行**（rowwise）执行：

对于$\mathbf{O}$的每一行，我们先对所有项进行幂运算，然后通过求和对它们进行标准化。

### Loss Function

- 均方误差（MSE）损失函数
  $$
  \min_{\mathbf{w},b} \sum(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b)^2
  $$

- 交叉熵损失函数
  $$
   l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j
  $$

## 实现

1. 引入库

   ```python
   import numpy as np
   import torch
   from torch.utils import data
   from d2l import torch as d2l
   ```

2. 读数据

   ```python
   def load_data_fashion_mnist(batch_size, resize=None):  #@save
       """下载Fashion-MNIST数据集，然后将其加载到内存中"""
       trans = [transforms.ToTensor()]
       if resize:
           trans.insert(0, transforms.Resize(resize))
       trans = transforms.Compose(trans)
       mnist_train = torchvision.datasets.FashionMNIST(
           root="../data", train=True, transform=trans, download=True)
       mnist_test = torchvision.datasets.FashionMNIST(
           root="../data", train=False, transform=trans, download=True)
       return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                               num_workers=get_dataloader_workers()),
               data.DataLoader(mnist_test, batch_size, shuffle=False,
                               num_workers=get_dataloader_workers()))

       train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
   ```

3. 定义模型

   ```
   # nn是神经网络的缩写
   from torch import nn

   net = nn.Sequential(nn.Linear(2, 1))
   ```

   初始话模型参数（可能不需要）

   ```
   net[0].weight.data.normal_(0, 0.01)
   net[0].bias.data.fill_(0)
   ```

4. 定义损失函数

   ```
   loss = nn.MSELoss()
   ```

5. 定义优化算法

   ```
   trainer = torch.optim.SGD(net.parameters(), lr=0.03)
   ```

6. 训练

   ```
   num_epochs = 3
   for epoch in range(num_epochs):
       for X, y in data_iter:
           l = loss(net(X) ,y)
           trainer.zero_grad()
           l.backward()
           trainer.step()
       l = loss(net(features), labels)
       print(f'epoch {epoch + 1}, loss {l:f}')
   ```
