# Model Indicators

Introduce indicators for measuring the complexity of a model.

## Summary

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

## Calculation

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

  

## Model Compression

When comes to model compression. We usually care about the three parameters:

- Model Size
- Runtime Memory
- Number of computing operations: two ways to calculate: 
  - FLOPS
  - MACs

