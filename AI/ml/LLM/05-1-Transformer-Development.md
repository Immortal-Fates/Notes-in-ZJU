# Transformer 与大模型基础
## 目录

1. [动机：为什么需要 Transformer？](#第一章动机为什么需要-transformer)
2. [Self-Attention 机制精讲](#第二章self-attention-机制精讲)
3. [完整 Transformer 架构](#第三章完整-transformer-架构)
4. [从 Transformer 到现代大模型](#第四章从-transformer-到现代大模型)
5. [工程优化与架构进阶](#第五章工程优化与架构进阶)（KV Cache · MQA/GQA · Flash Attention · LoRA · **MoE · Gated Attention · Attention Residuals**）
6. [知识点串联与总结](#第六章知识点串联与总结)

---

## 第一章　动机：为什么需要 Transformer？

### 1.1　RNN / LSTM 的本质缺陷

**核心问题：信息瓶颈 + 梯度消失**

- RNN 将所有历史信息压缩进固定维度的隐状态 `h_t`，序列越长，信息损耗越大
- 梯度需要跨时间步反传，连乘衰减问题：
  ```
  ∂L/∂h₁ = ∂L/∂hₙ × ∏ᵢ (∂hᵢ/∂hᵢ₋₁)
  ```
- 即便 LSTM 引入门控机制缓解了梯度消失，但**串行计算无法并行**，训练速度极慢
- GPU 核心数以千计，但 RNN 每步必须等上一步完成，利用率极低

> 💡 **比喻**：RNN 像一个每次只能看当前这页、且必须把前面所有记忆浓缩成一张纸条的读者。

---

### 1.2　Seq2Seq + Bahdanau Attention 的过渡

- 2014 年 Bahdanau 在 Seq2Seq 框架中引入 Attention，允许 Decoder 每一步动态查询 Encoder 所有隐状态
- 核心改进：摆脱固定长度向量瓶颈，引入了 Query-Key 匹配思想
- **局限**：底层仍然依赖 RNN，并行问题没有解决，只是注意力机制的雏形

---

### 1.3　Attention is All You Need（2017）

**核心贡献**：完全抛弃 RNN，用纯注意力机制构建序列模型，实现完全并行化

- 所有 token 同时互相计算注意力，充分利用 GPU 并行算力
- 任意两个位置之间的路径长度为 `O(1)`，解决长程依赖问题
- 配合残差连接和层归一化，深层网络训练稳定
- 从此开启大模型时代的序幕

---

## 第二章　Self-Attention 机制精讲

### 2.1　三个核心矩阵：Q、K、V

**线性投影**：输入序列 `X ∈ ℝⁿˣᵈ` 通过三个独立权重矩阵投影

```
Q = X · Wq     K = X · Wk     V = X · Wv
```

- `Wq`、`Wk`、`Wv` 均为**可学习参数**，不是固定变换——这是与传统检索的本质区别
- **Query（查询）**：当前 token 想要查询什么信息
- **Key（键）**：每个 token 对外声明自己包含什么信息
- **Value（值）**：每个 token 实际携带的信息内容
- 数据库类比：Query = 搜索词，Key = 文档索引，Value = 文档正文

> 💡 Q 和 K 做内积衡量「相关度」，相关度高的 Value 权重就大，最终加权求和得到输出。

---

### 2.2　Scaled Dot-Product Attention

完整计算公式：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

#### 为什么要除以 √dₖ？

**技术原因**：防止 softmax 进入梯度消失区域

- 假设 Q、K 每个元素独立均值为 0、方差为 1，则 `QKᵀ` 的每个元素**方差为 dₖ**
- 当 `dₖ` 很大时（如 64、128），点积绝对值很大，softmax 分布趋于 one-hot
- one-hot 区域梯度接近于零，反传信号消失，训练崩溃
- 除以 `√dₖ` 将方差归一化为 1，softmax 输入保持合适的数值范围

#### 计算复杂度分析

| 维度 | 复杂度 | 说明 |
|------|--------|------|
| 时间 | `O(n²·d)` | 注意力矩阵为 n×n，每个元素需要 d 次乘加 |
| 空间 | `O(n²)` | 需要存储完整注意力矩阵 |

> 实际含义：序列长度 4096 tokens → 注意力矩阵 4096×4096 = **16M 个元素**。这是 Transformer 处理超长序列的根本瓶颈，后续所有优化都围绕此展开。

---

### 2.3　Multi-Head Attention（多头注意力）

将注意力机制并行运行 h 次，每次使用不同的投影子空间：

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h) \cdot W_O$$

$$\text{head}_i = \text{Attention}(Q \cdot W_{qi},\ K \cdot W_{ki},\ V \cdot W_{vi})$$

- 每个 head 的维度为 `dₖ = d_model / h`，因此**总计算量与单头相同**
- 不同 head 可以并行学习不同语义维度：句法依赖、指代关系、语义相似性
- h 个 head 的输出拼接后，经 `Wo` 线性投影回 `d_model` 维
- 实验（BertViz 可视化）表明：不同 head 确实关注到不同的语言现象

> 💡 多头不是多次独立运行，而是维度切分后并行——工程实现上是一次大矩阵乘法。

---

### 2.4　Causal Mask（因果掩码）

**核心问题**：生成式模型训练时为何需要 Mask？

- **自回归生成**：推理时逐 token 生成，位置 `i` 只能看到 `j ≤ i` 的 token
- **训练时并行**：把整个序列一次性喂入，但必须模拟推理时的因果约束
- 实现方式：将注意力矩阵**上三角**设为 `-∞`，经 softmax 后权重变为 0
- **训练-推理的关键不对称性**：训练时全并行，推理时串行自回归

```
# Causal mask 示意（n=4）
mask = [[ 0,  -∞, -∞, -∞],
        [ 0,   0, -∞, -∞],
        [ 0,   0,  0, -∞],
        [ 0,   0,  0,  0]]
```

---

## 第三章　完整 Transformer 架构

### 3.1　整体结构

```
Input Tokens
    ↓
Token Embedding + Positional Encoding
    ↓
┌─────────────────────────────────────┐
│         Encoder Block × N           │  ← 双向注意力（BERT 系列）
│  Multi-Head Attention               │
│  Add & LayerNorm                    │
│  Feed-Forward Network               │
│  Add & LayerNorm                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│         Decoder Block × N           │  ← 因果注意力（GPT 系列）
│  Masked Multi-Head Attention        │
│  Add & LayerNorm                    │
│  Cross-Attention（Encoder-Decoder） │
│  Add & LayerNorm                    │
│  Feed-Forward Network               │
│  Add & LayerNorm                    │
└─────────────────────────────────────┘
    ↓
Linear + Softmax → Output Token Prob
```

---

### 3.2　Positional Encoding（位置编码）

**为什么需要？** Attention 是置换等变的（permutation-equivariant），打乱 token 顺序注意力结果不变。

原始 Sinusoidal 编码公式：

$$PE_{(pos,\ 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos,\ 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

- 不同频率的正弦波叠加，每个位置有唯一的编码向量
- 可以通过线性变换表达相对位置，理论上支持长度外推
- 与词向量**直接相加**（不拼接），不增加维度

**位置编码进化路线：**

```
Sinusoidal PE（原版）
    │ 外推性差
    ↓
Learned PE（BERT / GPT-2）
    │ 超出训练长度就失效
    ↓
RoPE（旋转位置编码）← LLaMA / Qwen 等主流 LLM 采用
    │ qm·kn 只依赖相对位置 (m-n)，外推性大幅改善
    ↓
YaRN / LongRoPE（长上下文扩展，支持 128K+ context）
```

> 💡 RoPE 核心：将 q 和 k 各自乘以旋转矩阵 `Rθ,pos`，则内积结果只含相对位置 `(m-n)` 的信息。

---

### 3.3　Feed-Forward Network（前馈网络）

每个 Transformer 层中，FFN 独立作用于每个位置（position-wise）：

$$\text{FFN}(x) = \text{GELU}(x \cdot W_1 + b_1) \cdot W_2 + b_2$$

- 维度先升后降：`d_model → 4·d_model → d_model`（原始设计）
- FFN 参数量约占整个模型的 **2/3**，是模型存储知识的核心结构
- 现代 LLM（LLaMA / Qwen）使用 **SwiGLU** 变体，效果更好：
  - `SwiGLU(x) = Swish(x·W) ⊗ (x·V)`，引入门控机制
  - 中间维度改为约 `2/3 × 4d` 以保持参数量一致

---

### 3.4　残差连接 + Layer Normalization

**残差连接（Residual Connection）：**

$$x_{l+1} = x_l + F(x_l)$$

- 梯度高速公路：反传梯度可以直接跳过若干层到达浅层
- 解决深层网络退化问题，使百层以上网络可以稳定训练
- 理论上：网络至少不会比恒等映射更差

**LayerNorm vs BatchNorm：**

| | BatchNorm | LayerNorm |
|---|---|---|
| 归一化维度 | batch 维度 | 特征维度（per-sample） |
| NLP 适用性 | 差（序列变长统计量不稳） | 好（与 batch size 无关） |
| 使用场景 | CV 主流 | NLP / LLM 主流 |

**Pre-LN vs Post-LN：**
- **Post-LN**（原版 Transformer）：在残差之后归一化，梯度传播难，训练不稳定
- **Pre-LN**（现代 LLM 标配）：在 sublayer 输入前归一化，训练稳定性显著更好

---

## 第四章　从 Transformer 到现代大模型

### 4.1　三种架构范式

| 架构 | 注意力方式 | 代表模型 | 擅长任务 |
|------|-----------|---------|---------|
| **Encoder-only** | 双向（全局） | BERT、RoBERTa | 理解：分类、NER(实体识别任务)、抽取 |
| **Decoder-only** | 单向（因果） | GPT / LLaMA / Qwen / Claude | 生成：对话、代码、推理 |
| **Encoder-Decoder** | 双向编码 + 单向解码 | T5、BART | Seq2Seq：翻译、摘要 |

> [!NOTE]
>
> 为什么encoder是双向（全局），decoder是单向（因果）。主要是因为任务不同，而非结构不同
>
> - encoder: 谁做了理解整段输入的工作
> - decoder: 谁做了一步一步生成输出的工作

> 💡 **为什么主流 LLM 都选 Decoder-only？** Scaling Law 研究表明，Decoder-only 在参数量和数据量扩展时效果更为平滑，涌现能力上限更高，且预训练目标（Next Token Prediction）统一简洁。

### 4.2　现代 LLM 的标配组件

以 **LLaMA 2 / Qwen 系列**为例：

| 组件 | 选型 | 说明 |
|------|------|------|
| 位置编码 | RoPE | 相对位置，外推性好 |
| 归一化 | RMSNorm | 比 LayerNorm 更轻量，去掉均值中心化 |
| 激活函数 | SwiGLU | 比 GELU 效果更好 |
| 注意力 | GQA | 减少 KV Cache 显存占用 |
| 注意力加速 | Flash Attention | IO 感知分块计算 |
| 分词 | BPE / SentencePiece | 词表大小约 32K~128K |

---

## 第五章　工程优化与架构进阶

### 5.1　KV Cache——推理加速的核心

**背景**：自回归生成时，每一步都要对所有历史 token 重新计算 K、V。

**解决方案**：缓存历史 token 的 K、V 矩阵，避免重复计算。

- 每生成一个新 token，只需计算当前 token 的 Q，与缓存的 K、V 做注意力
- 计算量从 `O(n²·d)` 降至每步 `O(n·d)`，推理速度大幅提升
- 代价：KV Cache 显存随序列长度**线性增长**，长上下文场景内存压力大
- 对于 LLaMA-7B：单层 KV Cache ≈ `2 × n × d_model × 2 Bytes`（fp16）
- 由此引出下一个问题：如何压缩 KV Cache 体积？

---

### 5.2　MQA / GQA——KV Cache 压缩

KV Cache 就是推理阶段对历史 Key 和 Value 的缓存

传统 MHA 每个 head 都有独立的 K、V：

- **MQA（Multi-Query Attention）**：所有 Query head 共享同一组 K、V，KV Cache 压缩至 `1/h`
- **GQA（Grouped-Query Attention）**：将 h 个 Query head 分成 g 组，每组共享一对 K、V
- GQA 是 MHA 与 MQA 的折中，**LLaMA 2 / Qwen** 等均采用
- 效果：KV Cache 显存减少 **4~8 倍**，同时保持接近 MHA 的模型质量

```
MHA:  Q heads=8,  K heads=8,  V heads=8   ← 显存最大
GQA:  Q heads=8,  K heads=2,  V heads=2   ← 折中（每组4个Q共享1对KV）
MQA:  Q heads=8,  K heads=1,  V heads=1   ← 显存最小，但质量略降
```

---

### 5.3　Flash Attention——IO 感知的注意力加速

**根本问题**：标准注意力实现会将 `n×n` 的注意力矩阵写入 HBM（显存），然后再读回来做 softmax，IO 极其耗费时间。

**GPU 存储层次：**

| 存储层 | 容量 | 带宽 |
|--------|------|------|
| HBM（显存） | 大（40~80GB） | ~2 TB/s |
| SRAM（片上缓存） | 小（~20MB） | ~19 TB/s |

**Flash Attention 核心思路**：**Tiling（分块）+ Online Softmax**

- 将 Q、K、V 切成小块，逐块在 SRAM 内完成注意力计算
- 用 Online Softmax 算法，分块迭代维护全局 softmax，**无需物化完整注意力矩阵**
- 数学结果与标准 Attention **完全等价**，纯工程 IO 优化

**效果：**
- 显存占用：`O(n²)` → `O(n)`
- 速度提升：**2~4 倍**
- Flash Attention 2 / 3 进一步优化并行策略，已成为所有主流 LLM 推理框架的标配

> 💡 Flash Attention 是在算法正确性不变的前提下，通过重新排列计算顺序减少 HBM 访问次数，属于**算法层面的工程优化**，不是近似计算。

---

### 5.4　LoRA——高效参数微调

**背景**：全参数微调一个 70B 模型需要数百 GB 显存，成本极高。

**核心假设**：预训练模型的参数更新矩阵 `ΔW` 具有**低秩结构**（intrinsic rank 很小）。

$$W' = W_0 + \Delta W = W_0 + B \cdot A$$

$$B \in \mathbb{R}^{d \times r},\quad A \in \mathbb{R}^{r \times k},\quad r \ll \min(d, k)$$

- 冻结原始权重 `W₀`，只训练低秩矩阵 `A` 和 `B`
  - `B` 初始化为全零，保证训练开始时 `ΔW = 0`
  - `A` 随机初始化（如高斯分布）
- 参数量从 `d×k` 降至 `r×(d+k)`，`r=8` 时参数量降低约 **100 倍**
- **合并推理**：`W' = W₀ + BA`，推理阶段**零额外开销**
- **QLoRA**：在 LoRA 基础上将 `W₀` 量化为 4-bit，显存进一步压缩至 `1/4`
- rank `r` 选择：通常 8~64，任务越复杂需要越大的 `r`

---

### 5.5　MoE（Mixture of Experts，混合专家）

**背景问题**：Dense 模型每个 token 都要过所有 FFN 参数，参数量与计算量同步增长，扩参成本极高。

> [!NOTE]
>
> FFN(前馈网络)：对每个 token 单独进行非线性变换的工作。
>
> 在很多 Transformer 模型里，FFN 的参数量其实比 attention 那部分还大

**核心思想**：把 FFN 替换成 N 个并行的"专家"子网络，每个 token 只激活其中 top-k 个，让一个router决定token激活哪些专家，**参数量增大但计算量不变**。

$$y = \sum_{i=1}^{N} G(x)_i \cdot \text{Expert}_i(x), \quad G(x) = \text{Softmax}(\text{TopK}(x \cdot W_g))$$

**关键组件：Router（路由器）**

```
输入 token x
    ↓
Router: x · Wg → logits ∈ ℝᴺ
    ↓
TopK(logits, k=2) → 选出 2 个专家索引 + 权重（softmax 归一化）
    ↓
Expert_i(x) × gate_weight_i  （只计算被选中的 k 个专家）
    ↓
加权求和 → 输出
```

**负载均衡损失（Load Balancing Loss）**：

不加约束时 Router 会退化为永远选同几个专家（专家坍塌）。辅助损失强制让每个专家被均匀使用：

$$\mathcal{L}_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

- `fᵢ`：专家 i 在当前 batch 中被选中的 token 比例
- `Pᵢ`：Router 给专家 i 的平均概率
- `α`：辅助损失系数（通常 0.01~0.001）

**代表模型对比：**

| 模型 | 总参数 | 激活参数 | 专家数 | top-k |
|------|--------|---------|-------|-------|
| Mixtral 8×7B | ~47B | ~13B | 8 | 2 |
| DeepSeek-MoE-16B | 16B | ~2.8B | 64 | 6（细粒度） |
| Qwen1.5-MoE-A2.7B | 14.3B | 2.7B | 64 | 4 |

**DeepSeek-MoE 的细粒度创新**：将每个专家拆成更小的"细粒度专家"，top-k 从 2 增大到 6，在同等激活参数下知识分工更精细，避免不同知识被强制塞进同一个专家。

> 💡 MoE 的本质是**稀疏激活**：模型"拥有"大量知识（大参数），但每次推理只"调取"少量专家——像一个大型咨询团队，每个问题只叫几个领域专家进来开会。

**工程挑战：**
- **All-to-All 通信**：分布式训练中不同 token 路由到不同 GPU 的专家，跨设备通信开销大
- **Expert Parallelism（EP）**：专家分布在不同 GPU，需要专门的并行策略
- **推理延迟**：top-k 路由是动态的，不同 batch 激活不同专家，难以做静态图优化

---

### 5.6　Gated Attention（门控注意力）

**动机**：标准 Attention 的输出是 Value 的加权和，表达能力受限于 softmax 的归一化约束；引入门控机制可以让模型**动态控制注意力信息的流通量**。

#### 方案一：Gated Attention Unit（GAU）

来自论文《Transformer Quality in Linear Time》（2022），将注意力与门控深度融合：

$$\text{GAU}(X) = \phi_u(XW_u) \odot \text{Attention}(X)$$

- `φ_u(XW_u)` 是一个门控向量，逐元素与注意力输出相乘
- 门控决定"哪些注意力结果被通过"，哪些被抑制
- 同时将 Q、K 维度大幅压缩（共享低秩投影），换取线性时间复杂度

#### 方案二：Attention 输出门控（现代 LLM 变体）

部分架构在多头注意力输出后加一个 sigmoid 门：

$$\text{out} = \sigma(x W_{gate}) \odot \text{MultiHead}(x)$$

- 门控值在 `[0,1]` 之间，可以完全关闭某些注意力通道
- **Hawk / Griffin（Google DeepMind 2024）** 将门控机制引入混合架构（SSM + Attention），用门控决定何时依赖局部循环记忆、何时依赖全局注意力

#### 方案三：RWKV 的时间门控（Time-mixing Gate）

RWKV 系列在线性注意力框架中引入时间衰减门控：

$$\text{output}_t = \sigma(r_t) \odot \sum_{\tau \leq t} e^{-(t-\tau)w} \cdot (k_\tau \odot v_\tau)$$

- `σ(r_t)` 是接收门（receptance gate），控制历史信息的采纳比例
- `w` 是可学习的时间衰减向量，控制不同通道的记忆衰减速率
- 实现 RNN 式 `O(1)` 推理，同时训练可并行

**三种方案对比：**

| 方案 | 门控位置 | 复杂度 | 代表 |
|------|---------|-------|------|
| GAU | 注意力输出 × 线性门 | `O(n)` 近似 | 月之暗面早期架构 |
| Output Gate | MHA 输出后 sigmoid | `O(n²)` | Griffin、部分 Gemma 变体 |
| Time Gate | 线性递推中衰减门 | `O(n)` | RWKV-4/5/6 |

> 💡 门控的本质是给模型增加一个"开关"：不只是「关注哪些位置」（Attention 本身的功能），还能控制「注意力的结果要不要被采用，采用多少」。

---

### 5.7　Attention Residuals（AttnRes）——Kimi，2026

> 📄 论文：*Attention Residuals*，Kimi Team，arXiv 2603.15031，2026 年 3 月 16 日

#### 问题根源：标准残差连接的深度稀释

标准残差连接从 2015 年沿用至今，从未被根本性质疑：

$$h_{l+1} = h_l + F_l(h_l)$$

每一层的输出以**固定单位权重 1** 累加到主干。随着深度增加，这带来一个严重问题：

- **隐状态无限增长**：每层等权叠加，深层的 `h_L` 量级远大于浅层输出
- **PreNorm 稀释（PreNorm Dilution）**：Pre-LN 在归一化后将 sublayer 输出加回原始量级的主干，浅层贡献被深层积累的庞大残差信号**淹没**
- **梯度分布不均**：深层梯度更大，浅层梯度极小，不同层的学习效率严重失衡
- **深度利用率低**：后层为了"盖过"前层的累积信号，被迫学习量级更大的输出，而非真正有用的信息

**类比洞察**：这个问题在**深度维度**上与 RNN 在**序列维度**上的信息稀释完全对偶——RNN 把所有历史压缩进一个隐状态，Transformer 的残差流把所有层的输出以等权叠加。Transformer 用 Attention 解决了序列维度的问题；Kimi 提出用同样的思路解决深度维度的问题。

---

#### 核心方案：AttnRes

将固定累加替换为**跨层的 softmax 注意力**，让每一层动态地、以内容感知的方式从之前所有层中选择性地聚合信息：

$$h_l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot v_i, \quad \alpha_{i \to l} = \text{softmax}_i(q_l^\top k_i)$$

其中：
- `q_l`：**可学习的伪查询向量**（per-layer 参数，不依赖当前输入，轻量）
- `k_i, v_i`：前序各层的隐状态（作为键和值）
- `α_{i→l}`：softmax 归一化后的深度注意力权重，**输入感知、跨层可变**

与标准残差的对比：

```
标准残差：h_l = h_{l-1} + F_{l-1}(h_{l-1})   ← 只看上一层，固定权重 1
AttnRes：  h_l = Σ α_{i→l} · v_i              ← 看所有前序层，权重动态学习
```

直观理解：每一层拥有一个"档案检索员"，不再被动接收上一层的全部遗留，而是主动向历史层档案提问，按需提取最有用的信息。

---

#### 工程挑战与 Block AttnRes

**Full AttnRes 的问题**：对所有前序层做注意力，内存复杂度为 `O(L·d)`，L=48 层时需要缓存全部中间状态，大规模训练不可行。

**Block AttnRes**：将 L 层分成 N 个 Block，**Block 内**仍用标准残差，**Block 间**用 AttnRes 做跨 Block 的注意力聚合：

```
Layer 1  ─┐
Layer 2  ─┤  Block 1 内：标准残差累加
...      ─┤
Layer k  ─┘ → Block 1 代表向量 b₁
              ↓
Layer k+1 ─┐
...        ─┤  Block 2 内：标准残差 + AttnRes(b₁, b_partial)
Layer 2k  ─┘ → Block 2 代表向量 b₂
              ↓
              ...
              ↓
Layer L：AttnRes(b₁, b₂, ..., b_{N-1}, b_partial_N)
```

内存从 `O(L·d)` 降至 `O(N·d)`，N≈8 时可恢复 Full AttnRes 约 **95% 的收益**。

**两阶段计算策略**（控制通信开销）：
- **Phase 1**：`q_l` 是纯参数（不依赖输入），可提前并行计算所有层对 Block 代表向量的注意力，每个 Block 向量只读一次，无冗余 IO
- **Phase 2**：Block 内的局部累积顺序计算，与 Phase 1 结果合并

最终开销：训练额外计算量 **< 4%**，推理延迟增量 **< 2%**。

---

#### 实验结果

**Scaling Law 验证**：在多个模型量级下，AttnRes 的收益一致，不依赖特定规模：

| 方法 | 效果等价 |
|------|---------|
| 标准残差（baseline） | — |
| Block AttnRes（N≈8） | ≈ 标准残差训练 **1.25× 计算量** 的效果 |
| Full AttnRes | 最优，但大规模不可行 |

**Kimi Linear 集成**：AttnRes 被集成进 Kimi Linear（**48B 总参数 / 3B 激活参数** 的 MoE 模型），在 1.4T tokens 上预训练：
- 输出幅度更均匀（缓解 PreNorm Dilution）
- 梯度分布更均匀（跨层学习效率提升）
- **所有评测任务上均有提升**，无负优化

**消融实验**关键结论：
- N=1（等价 Full AttnRes）→ N=8 收益基本等价，N=16 以上开始退化
- 内容感知的深度注意力权重（学习的 q_l）优于固定均匀权重（对照实验验证了动态选择的必要性）

---

> 💡 **一句话理解**：AttnRes 对残差连接做了与 Attention 对 RNN 同等量级的改造——从"盲目等权叠加"升级为"按需内容感知聚合"。标准残差是固定写死的高速公路，AttnRes 是带智能调度的立交桥。

---

## 第六章　知识点串联与总结

### 6.1　技术演进全景图

```
序列建模
  ├── RNN / LSTM（2014 前）：串行，长程依赖差
  └── Transformer（2017）：并行，全局注意力
         │
         ├── 位置编码演进
         │     Sinusoidal → Learned PE → RoPE → YaRN
         │
         ├── 架构分支
         │     Encoder-only（BERT）              ← 理解任务
         │     Encoder-Decoder（T5）             ← Seq2Seq
         │     Decoder-only（GPT → LLaMA → Qwen）  ← 主流生成
         │
         ├── FFN 架构演进
         │     Dense FFN → MoE（稀疏激活，参数↑ 计算量不变）
         │     GELU → SwiGLU → 门控 FFN 变体
         │
         ├── 注意力优化
         │     MHA → GQA / MQA（压缩 KV Cache）
         │     标准 Attention → Flash Attention（IO 优化）
         │     Standard Attn → Gated Attention（GAU / Output Gate / RWKV）
         │
         ├── 残差连接演进
         │     标准残差（固定权重叠加）
         │       → Deep Norm（缩放残差，1000层稳定训练）
         │       → AttnRes（Kimi 2026，跨层 softmax 注意力，≈1.25× 算力提升）
         │         → Block AttnRes（N≈8 分块，开销 <4%，工程可行）
         │
         └── 训练 / 微调优化
               全参微调 → LoRA → QLoRA
```

---

### 6.2　现代 LLM 一个 Transformer Block 的完整流程

```python
# 输入 x，shape: [batch, seq_len, d_model]
# 以下为含 MoE + Gated Attention 的现代 LLM Block 示意

x_norm = RMSNorm(x)                                      # Pre-LN

Q = x_norm @ Wq
K = x_norm @ Wk
V = x_norm @ Wv                                          # 线性投影

Q, K = RoPE(Q, pos), RoPE(K, pos)                        # 旋转位置编码

K, V = KVCache.update(K, V)                              # 追加缓存（推理时）

attn_out = FlashAttention(Q, K, V, causal_mask)          # 因果注意力

# 可选：Gated Attention（Output Gate 变体）
gate = sigmoid(x_norm @ W_gate)
attn_out = gate * attn_out                               # 门控调制注意力输出

x = x + attn_out @ Wo                                    # 残差写入主干（残差流）

x_norm = RMSNorm(x)                                      # Pre-LN

# Dense FFN → MoE FFN
router_logits = x_norm @ W_router                        # Router 打分
expert_weights, expert_ids = TopK(softmax(router_logits), k=2)
ffn_out = sum(expert_weights[i] * Expert_i(x_norm)       # 只激活 top-2 专家
              for i in expert_ids)

x = x + ffn_out                                          # FFN 残差写入主干

# 输出 x，shape: [batch, seq_len, d_model]
# 残差流视角：x = x₀ + Σ(各层 Attn 贡献) + Σ(各层 FFN/MoE 贡献)
```

---

### 6.3　一句话总结

> **Transformer = 注意力机制 + 残差流 + 归一化 + 位置编码**
>
> MoE 让模型「能力广博但按需调用」；Gated Attention 让信息流通「可控」；残差流视角让黑盒变得「可解释」。

---

### 6.4　常见技术问题 Q&A

**Q：Attention 的 O(n²) 复杂度怎么破解？**

A：Sparse Attention（只关注局部 + 全局 token）、Linear Attention（核函数近似 softmax）、Mamba（状态空间模型，用选择性 SSM 替代 Attention，复杂度降至 `O(n)`）。

---

**Q：RoPE 怎么支持更长序列？**

A：位置插值（PI）将原始位置下采样到训练范围内；YaRN 对不同频率分量使用不同缩放因子，低频分量（全局信息）缩放更少，效果比 PI 更好，目前是长上下文扩展的主流方案。

---

**Q：LoRA 的 rank 怎么选？**

A：经验上 8~64，复杂任务（代码、推理）用大 rank；DoRA 在 LoRA 基础上将权重分解为幅度与方向，分别更新，效果更稳定；AdaLoRA 动态分配不同层的 rank 预算。

---

**Q：为什么 Decoder-only 成为主流而不是 Encoder-Decoder？**

A：Scaling Law 表明 Decoder-only 在同等参数量下涌现能力更强；预训练目标（Next Token Prediction）统一简洁，无需设计复杂的 Seq2Seq 任务；工程实现也更简单，推理时只需维护一套 KV Cache。

---

**Q：Pre-LN 和 Post-LN 的区别？**

A：Post-LN 在残差之后归一化，梯度更难传播到浅层，深层模型容易训练不稳定；Pre-LN 在 sublayer 输入前归一化，梯度传播更顺畅，但输出层没有被归一化（有时用 RMSNorm 在最终输出前补一次）。现代所有主流 LLM 均采用 Pre-LN。

---

**Q：MoE 和 Dense 模型推理延迟谁更低？**

A：不一定。激活参数量相同时 MoE 理论上延迟相近，但 All-to-All 专家路由通信、动态激活导致的 GPU 利用率不均，实际延迟往往比等激活参数的 Dense 模型高 20~50%。MoE 的优势在于**同等推理计算量下具备更大的模型容量和知识存储量**，适合追求效果的离线场景。

---

**Q：Gated Attention 和标准 Attention 的根本区别是什么？**

A：标准 Attention 输出是 softmax 加权的 Value 之和，softmax 归一化约束导致输出幅度相对固定。门控注意力在此基础上叠加一个输入依赖的缩放因子，使模型能**动态决定当前注意力结果对残差流的贡献强度**——信息不仅可以被"选择性关注"，还可以被"选择性采纳"。

---

**Q：AttnRes 和标准残差在工程上兼容吗？开销大吗？**

A：兼容性极强，Block AttnRes 是标准残差的**即插即用替代品**，不改变 Attention、FFN、MoE Router 等任何其他组件。工程开销上，Block 数 N≈8 时训练额外计算量 <4%，推理延迟增量 <2%。Kimi 在 48B/3B MoE 模型（Kimi Linear）上 1.4T tokens 预训练验证，所有任务均有提升，Scaling Law 实验证明收益跨模型规模一致——等价于用同样数据多训练 1.25× 计算量的效果。
