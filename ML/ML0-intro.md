# Main Takeaway

重学Machine Learning

<!--more-->

![relation](./assets/ML0-intro.assets/a388b49cb06e128515de182fc3e51800.png)

# preliminaries

- 深度学习是关于优化的学习

- autograd:

  深度学习框架通过自动计算导数，即**自动微分**（automatic differentiation）来加快求导。实际中，根据设计好的模型，系统会构建一个**计算图**（computational graph），来跟踪计算是哪些数据通过哪些操作组合起来产生输出。自动微分使系统能够随后反向传播梯度。这里，**反向传播**（backpropagate）意味着跟踪整个计算图，填充关于每个参数的偏导数。

