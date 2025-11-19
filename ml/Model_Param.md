# 常见层的参数量公式（不含优化器开销）

## Conv2d（含偏置）

$$
\#\text{params} = C_{\text{out}}\times \frac{C_{\text{in}}}{\text{groups}}\times k_h\times k_w
+
\begin{cases}
C_{\text{out}}, & \text{bias=True}\\
0, & \text{bias=False}
\end{cases}
$$

---

## Linear（含偏置）

$$
\#\text{params} = C_{\text{out}}\times C_{\text{in}}
+
\begin{cases}
C_{\text{out}}, & \text{bias=True}\\
0, & \text{bias=False}
\end{cases}
$$

---

## BatchNorm2d（affine=True）

可学习参数：
$$
\#\text{params} = 2C
$$
对应两个通道维参数 $\gamma$（缩放）和 $\beta$（平移）；
运行均值/方差为 buffer，不计入 `parameters()`。

---

## LayerNorm / GroupNorm（affine=True）

$$
\#\text{params} = 2\times (\text{归一化的特征维度})
$$

---

## Embedding

$$
\#\text{params} = \text{num\_embeddings} \times \text{embedding\_dim}
$$

---

## 无可学习参数

ReLU、MaxPool、AvgPool、AdaptiveAvgPool 等：
$$
\#\text{params} = 0
$$

---

## 数据类型换算（仅参数本体

float32 = 4B，float16/bfloat16 = 2B，float64 = 8B，int8 = 1B……

# 定义完模型后：在 PyTorch 中查看参数量与大小

```python
# 定义完模型后：在 PyTorch 中查看参数量与大小（可直接粘贴使用）
import torch
from collections import defaultdict

# 1) 总参数与可训练参数个数
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# 2) 仅按 dtype 粗略估算参数内存占用（MB），不含梯度/优化器/激活
BYTES_PER_ELEM = {
    torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
    torch.float64: 8, torch.int8: 1, torch.int16: 2, torch.int32: 4
}
def params_size_in_mb(model):
    size_bytes = 0
    for p in model.parameters():
        bpe = BYTES_PER_ELEM.get(p.dtype, 4)
        size_bytes += p.numel() * bpe
    return size_bytes / (1024**2)

# 3) 分层报告（每个参数张量一行）
def per_param_report(model):
    rows = []
    for name, p in model.named_parameters():
        rows.append((name, tuple(p.shape), p.numel(), str(p.dtype), p.requires_grad))
    # 打印
    print(f"{'name':40s} {'shape':20s} {'numel':>12s} {'dtype':>12s} {'trainable':>10s}")
    print("-"*100)
    for n, shape, num, dt, req in rows:
        print(f"{n:40s} {str(shape):20s} {num:12d} {dt:12s} {str(req):>10s}")
    print("-"*100)
    print(f"Total params: {sum(r[2] for r in rows):,}")

# 4) 示例：统计并打印
def report_model(model, title="Model"):
    total, trainable = count_params(model)
    size_mb = params_size_in_mb(model)
    print(f"[{title}]")
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print(f"Param size (MB)  : {size_mb:.2f} MB\n")
    per_param_report(model)

# 5) 可选：使用 torchinfo 打印层级摘要（需 pip install torchinfo）
# from torchinfo import summary
# summary(model, input_size=(1, C, H, W))  # 给出示例输入形状即可

```
