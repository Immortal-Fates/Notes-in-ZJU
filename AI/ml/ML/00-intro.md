# Main Takeaway

重学Machine Learning

<!--more-->

![relation](./assets/ML0-intro.assets/a388b49cb06e128515de182fc3e51800.png)

## Prior

- 深度学习是关于优化的学习

- autograd:

  深度学习框架通过自动计算导数，即**自动微分**（automatic differentiation）来加快求导。实际中，根据设计好的模型，系统会构建一个**计算图**（computational graph），来跟踪计算是哪些数据通过哪些操作组合起来产生输出。自动微分使系统能够随后反向传播梯度。这里，**反向传播**（backpropagate）意味着跟踪整个计算图，填充关于每个参数的偏导数。

## Softmax and Its Variants

## Softmax

- Takeaway: Softmax is used when a model must **choose one class** among many. It produces smooth, continuous probabilities but **does not perform sampling**.

  > [!NOTE]
  >
  > Here we just need the expectation not sampling.

- What: Softmax transforms a vector of real-valued logits into a **probability distribution**:
  $$
  p_i = \frac{\exp(z_i)}{\sum_{j=1}^K \exp(z_j)}
  $$

  Properties:

  - \(p_i > 0\)
  - \(\sum_i p_i = 1\)
  - Differentiable and suitable for backpropagation

- Cons

  - Softmax cannot generate **discrete one-hot samples**. This leads to the need for **Gumbel-Softmax**.

---

## Gumbel-Softmax

- Takeaway: Gumbel-Softmax provides a **differentiable approximation** to sampling from a **categorical distribution**.

- What: Ordinary one-hot sampling uses a non-differentiable argmax:
  $$
  y = \text{one\_hot}(\arg\max_i z_i)
  $$

  > [!NOTE]
  >
  > Use differentiable softmax to replace the argmax function here.

  Gumbel-Softmax enables **backpropagation through sampling**.

  1. Step 1 — Add Gumbel Noise

     Sample Gumbel noise:
     $$
     g_i = -\log(-\log U_i), \quad U_i \sim \text{Uniform}(0,1)
     $$

     Perturb logits:
     $$
     z_i = \log(p_j) \\
     \tilde{z}_i = z_i + g_i = \begin{cases}
     1,i = \arg \max_j(\log(p_j)+g_i)\\
     0,\text{otherwise}
     \end{cases}
     $$

  2. Step 2 — Temperature-controlled Softmax

     Apply Softmax with temperature \(\tau\):

     $$
     y_i = \frac{\exp(\tilde{z}_i / \tau)}{\sum_{j=1}^K \exp(\tilde{z}_j / \tau)}
     $$

     - High \(\tau\): soft distribution  
     - Low \(\tau\): nearly one-hot  
     - \(\tau \to 0\): exactly argmax behavior

     Thus: **Gumbel-Softmax ≈ differentiable categorical sampling.**


- How: Straight-Through Gumbel-Softmax. Used when a hard sample is required:

  - Forward: take argmax (one-hot)  $x_{sample} = x\times y$
  - Backward: use soft gradients from Gumbel-Softmax  

  This trick keeps sampling discrete but training differentiable.

