# Transformer

- {{1706.03762v7}}



![ModalNet-21](./assets/transformer.assets/ModalNet-21.png)

```mathematica
Encoder:
[Input] → [Embedding + Positional Encoding] → [Multi-Head Self-Attention] → [Feed-Forward] → [Output Representation]

Decoder:
[Shifted Output] → [Embedding + Positional Encoding] → [Masked Multi-Head Attention]
                 → [Encoder-Decoder Attention]
                 → [Feed-Forward] → [Predicted Output]

```

输入

- Attention：KQV
  $$
  \text{Attention}(Q, K, V) = {\text{softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)} {V}
  $$

  - **Query (Q, 查询矩阵):** 希望关注或查询的信息  
  - **Key (K, 键矩阵):** 标识信息，供其他位置判断相关程度  
  - **Value (V, 值矩阵):** 具体信息内容，被关注后实际使用的信息  

- Multi-Head：同时并行地做多次自注意力计算

  相当于为每个单词计算了一个偏移量

- Add & Norm：残差连接与层归一化

  将原始向量加上偏移量再进行归一化

- Feed Forward：先升维再降维是典型的特征提取与去噪手段，经典图像算法经常用到

输出

- Masked：盖住未来的信息，shifted right不断右移，做题看答案
- Linear：将向量转换



# References

- 【20分钟读懂AI神级论文《Attention Is All You Need》】https://www.bilibili.com/video/BV1dyW9zsEk1?vd_source=93bb338120537438ee9180881deab9c1