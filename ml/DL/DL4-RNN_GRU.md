# GRU

门控循环单元（GRU）

GRU是LSTM稍微简化的变体，通常能提供同等的效果，且速度明显更快。它和RNN之间的关键区别在于：GRU支持隐状态的门控。这意味着有专门的机制来确定何时更新隐状态，以及应该何如重置隐状态

## 重要结构

两个门的值都在$(0,1)$的范围内，我们使用sigmoid函数-将输入值转换到区间$(0,1)$。

- 更新门update gate：$$\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xz} + \mathbf{H}_{t-1} \mathbf{W}_{hz} + \mathbf{b}_z),$$

  控制新状态中有多少是旧状态的副本

- 重置门reset gate：$$\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xr} + \mathbf{H}_{t-1} \mathbf{W}_{hr} + \mathbf{b}_r),$$

  控制“可能还想记住”的过去状态的数量

- 候选隐藏状态candidate hidden state：$$\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xh} + (\mathbf{R}_t \odot \mathbf{H}_{t-1}) \mathbf{W}_{hh} + \mathbf{b}_h),$$

  符号$\odot$是Hadamard积（按元素乘积）运算符。 在这里，我们使用tanh非线性激活函数来确保候选隐状态中的值保持在区间$(-1.1)$中。

- 隐状态：$$\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1} + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.$$

  每当更新门 $  \mathbf{Z}_t  $ 接近 1 时，模型就倾向只保留旧状态。此时，来自 $  \mathbf{X}_t  $ 的信息基本上被忽略，从而有效地跳过了依赖链条中的时间步 $  t  $。相反，当 $  \mathbf{Z}_t  $ 接近 0 时，新的隐状态 $  \mathbf{H}_t  $ 就会接近候选隐状态 $  \tilde{\mathbf{H}}_t  $。这些设计可以帮助我们处理循环神经网络中的梯度消失问题，并更好地捕获时间步距离很长的序列的依赖关系。例如，如果整个子序列的所有时间步的更新门都接近于 1，则无论序列的长度如何，在序列起始时间步的旧隐状态都将很容易保留并传递到序列结束。



![gru-3](./assets/GRU.assets/gru-3.svg)

门控循环单元具有以下两个显著特征：

- 重置门有助于捕获序列中的短期依赖关系；
- 更新门有助于捕获序列中的长期依赖关系。