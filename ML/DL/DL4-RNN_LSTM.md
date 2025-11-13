# LSTM

> LSTM（Long Short-Term Memory，长短期记忆网络）是一种特殊的循环神经网络（RNN）
>
> 高级一点的RNN

LSTM从被设计之初就被用于解决一般递归神经网络中普遍存在的**长期依赖问题**，使用LSTM可以有效的传递和表达长时间序列中的信息并且不会导致长时间前的有用信息被忽略（遗忘）。与此同时，LSTM还可以解决RNN中的梯度消失/爆炸问题。

- 直觉解释：“长短期记忆”——只有一部分的信息需要长期的记忆，而有的信息可以不记下来。



## 重要结构

RNN什么信息它都存下来，因为它没有挑选的能力，而LSTM不一样，它会选择性的存储信息，因为它能力强，它有**门控装置**，它可以尽情的选择。

长短期记忆网络引入了*记忆元*（memory cell），或简称为*单元*（cell）。 有些文献认为记忆元是隐状态的一种特殊类型， 它们与隐状态具有相同的形状，其设计目的是用于记录附加的信息。 为了控制记忆元，我们需要许多门。 

其隐藏层输出包括“隐状态”和“记忆元”，只有隐状态会传递到输出层，而记忆元完全属于内部信息。

### 输入门、忘记门和输出门

就如在门控循环单元中一样， 当前时间步的输入和前一个时间步的隐状态 作为数据送入长短期记忆网络的门中。 它们由三个具有sigmoid激活函数的全连接层处理， 以计算输入门、遗忘门和输出门的值。 因此，这三个门的值都在$(0,1)$的范围内。

假设有 $  h  $ 个隐藏单元，批量大小为 $  n  $，输入数为 $  d  $。因此，输入为 $  \mathbf{X}_t \in \mathbb{R}^{n \times d}  $，前一时间步的隐状态为 $  \mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}  $。相应地，时间步 $  t  $ 的门被定义如下：输入门是 $  \mathbf{I}_t \in \mathbb{R}^{n \times h}  $，遗忘门是 $  \mathbf{F}_t \in \mathbb{R}^{n \times h}  $，输出门是 $  \mathbf{O}_t \in \mathbb{R}^{n \times h}  $。

- 输入门Input Gate：$I_t = \sigma(X_t W_{xi} + H_{t-1} W_{hi} + b_i)$

  决定何时将数据读入单元

- 遗忘门Forget Gate：$F_t = \sigma(X_t W_{xf} + H_{t-1} W_{hf} + b_f)$

  重置单元的内容

- 输出门Output Gate：$O_t = \sigma(X_t W_{xo} + H_{t-1} W_{ho} + b_o)$

  从单元中输出条目

其中 $\mathbf{W}_{xi}, \mathbf{W}_{xf}, \mathbf{W}_{xo} \in \mathbb{R}^{d \times h}$ 和 $\mathbf{W}_{hi}, \mathbf{W}_{hf}, \mathbf{W}_{ho} \in \mathbb{R}^{h \times h}$ 是权重参数，$\mathbf{b}_i, \mathbf{b}_f, \mathbf{b}_o \in \mathbb{R}^{1 \times h}$ 是偏置参数。

### 候选记忆元

由于还没有指定各种门的操作，所以先介绍候选记忆元（candidate memory cell） $  \tilde{\mathbf{C}}_t \in \mathbb{R}^{n \times h}  $。它的计算与上面描述的三个门的计算类似，但是使用 $  \tanh  $ 函数作为激活函数，函数的值范围为 $ (-1, 1) $。下面导出在时间步 $  t  $ 处的方程：

- 候选单元状态：$\tilde{C}_t = \tanh(X_t W_{xc} + H_{t-1} W_{hc} + b_c)$

其中 $\mathbf{W}_{xc} \in \mathbb{R}^{d \times h}  $ 和 $  \mathbf{W}_{hc} \in \mathbb{R}^{h \times h}$ 是权重参数，$  \mathbf{b}_c \in \mathbb{R}^{1 \times h}$ 是偏置参数。

### 记忆元

在门控循环单元中，有一种机制来控制输入和遗忘（或跳过）。类似地，在长短期记忆网络中，也有两个门用于这样的目的：输入门 $  \mathbf{I}_t  $ 控制采用多少来自 $  \tilde{\mathbf{C}}_t  $ 的新数据，而遗忘门 $  \mathbf{F}_t  $ 控制保留多少过去的记忆元 $  \mathbf{C}_{t-1} \in \mathbb{R}^{n \times h}  $ 的内容。使用按元素乘法，得出：

- 单元状态更新：$$C_t = F_t \odot C_{t-1} + I_t \odot \tilde{C}_t$$

如果遗忘门始终为 1 且输入门始终为 0，则过去的记忆元 $  \mathbf{C}_{t-1}  $ 将随时间被保存并传递到当前时间步。引入这种设计是为了缓解梯度消失问题，并更好地捕获序列中的长距离依赖关系。

### 隐状态

最后，我们需要定义如何计算隐状态 $  \mathbf{H}_t \in \mathbb{R}^{n \times h}  $，这就是输出门发挥作用的地方。在长短期记忆网络中，它仅仅是记忆元的 $  \tanh  $ 的门控版本。这就确保了 $  \mathbf{H}_t  $ 的值始终在区间 $ (-1, 1) $ 内：

- 隐藏状态更新：$$H_t = O_t \odot \tanh(C_t)$$

只要输出门接近 1，我们就能够有效地将所有记忆信息传递给预测部分，而对于输出门接近 0，我们只保留记忆元内的所有信息，而不需要更新隐状态。

![lstm-3](./assets/LSTM.assets/lstm-3.svg)

## 总结

长短期记忆网络是典型的具有重要状态控制的隐变量自回归模型。 多年来已经提出了其许多变体，例如，多层、残差连接、不同类型的正则化。 然而，由于序列的长距离依赖性，训练长短期记忆网络 和其他序列模型（例如门控循环单元）的成本是相当高的。 在后面的内容中，我们将讲述更高级的替代模型，如Transformer。



# 经验

- teacher foring

  一种引导和加速模型学习过程的方法，在序列的每一步都为其提供正确的输入，而不是让它根据之前的输出来生成下一步。

  - 优点：使用Teacher Forcing技术后模型的收敛速度更快。在训练初期，模型的预测结果非常糟糕。如果我们不使用Teacher Forcing，模型的隐藏状态就会被一连串错误的预测所更新，错误就会累积，模型就很难从中学习。
  - 缺点：在推理过程中，由于通常没有Ground Truth可用，RNN 模型需要将自己之前的预测反馈给自己，以进行下一次预测。因此，训练和推理之间数据分布存在差异，这可能会导致模型性能不佳和不稳定。这在学术界被称为Exposure Bias

- 测试阶段因为用预测值继续进行预测而造成累计误差：

  - 预测后用真实值代替history再继续预测
  - 修改预测len，使得长度和想要预测的最长值一样

- 自相关性，看着预测就是真实值滞后偏移产生的

  消除自相关性的办法就是进行差分运算，也就是我们可以将当前时刻与前一时刻的差值作为我们的回归目标

- 模型均值坍缩



# Papers

```
set PYTHONUTF8=1
autoliter -i ./LSTM.md -o ../../../papers/models/LSTM/
```

- **Long Short-Term Memory**. Hochreiter Sepp et.al. **Neural Computation**, **1997-11-1**, ([pdf](..\..\..\papers\models\LSTM\Long_Short-Term_Memory.pdf))([link](https://doi.org/10.1162/neco.1997.9.8.1735)).
- **LSTM: A Search Space Odyssey**. Greff Klaus et.al. **IEEE Trans. Neural Netw. Learning Syst.**, **2017-10**, ([pdf](..\..\..\papers\models\LSTM\LSTM:_A_Search_Space_Odyssey.pdf))([link](https://doi.org/10.1109/tnnls.2016.2582924)).
- **Bidirectional LSTM Networks for Improved Phoneme Classification and Recognition**. Graves Alex et.al. **No journal**, **2005**, ([pdf](..\..\..\papers\models\LSTM\Bidirectional_LSTM_Networks_for_Improved_Phoneme_Classification_and_Recognition.pdf))([link](https://doi.org/10.1007/11550907_126)).



# References

- [(37 封私信 / 80 条消息) LSTM - 长短期记忆递归神经网络](https://zhuanlan.zhihu.com/p/123857569)
- 超详细：[ML Lecture 21-1: Recurrent Neural Network (Part I)](https://www.youtube.com/watch?v=xCGidAeyS4M)
- [9.2. 长短期记忆网络（LSTM） — 动手学深度学习 2.0.0 documentation](https://zh-v2.d2l.ai/chapter_recurrent-modern/lstm.html)
- [(39 封私信 / 80 条消息) LSTM如何来避免梯度弥散和梯度爆炸？ - 知乎](https://www.zhihu.com/question/34878706)
- [长期短期记忆 (LSTM)代码解读](https://nn.labml.ai/zh/lstm/index.html)
