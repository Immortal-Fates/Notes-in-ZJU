

# 28天速通计划

## Plan Table

> [!TIP]
>
> 对应的学习资料见后面的链接

| 时间计划                           | 学习内容                                                    | 学习资料          | 每日自检                                                     |
| ---------------------------------- | ----------------------------------------------------------- | ----------------- | ------------------------------------------------------------ |
| Day 1，2h 阅读，2h 整理，1h 画图   | 建立 AI infra 总图，搞清训练栈、推理栈、平台栈分别做什么    | B1 B2 B3          | 能手画一张总图，讲清训练、推理、平台三层分别负责什么         |
| Day 2，2h 阅读，2h 查概念，1h 复盘 | 补齐 GPU 通信、网络、带宽、延迟、all-reduce 的工程视角      | B4 D9             | 能解释为什么多机训练很吃网络，all-reduce 为什么会卡住扩展性  |
| Day 3，2h 阅读，2h 实验，1h 记录   | 学 DDP，弄清单机多卡最基础的数据并行                        | D1 D9             | 能说出 DDP 的执行流程，能解释梯度同步发生在什么地方          |
| Day 4，2h 阅读，2h 画流程，1h 复盘 | 学 FSDP，理解参数分片、梯度分片、显存节省来自哪里           | D2                | 能说清 DDP 和 FSDP 的本质区别，能画出 forward 和 backward 时参数聚合的大概流程 |
| Day 5，2h 阅读，2h 对比，1h 总结   | 学 ZeRO，理解 ZeRO-1、2、3 的差别                           | D3 P2 R2          | 能不看资料讲出三种 ZeRO 各切掉了什么冗余                     |
| Day 6，2h 阅读，2h 画图，1h 复盘   | 学 Megatron，理解 tensor parallel、pipeline parallel 的作用 | P1 D4 R3          | 能解释 TP 和 PP 分别在切什么，为什么二者常常一起用           |
| Day 7，3h 输出，2h 回顾            | 做第一周总复盘，把训练主线从 DDP 讲到 FSDP、ZeRO、Megatron  | B2 D1 D2 D3 P1 P2 | 不看笔记，做 10 分钟口述，说明训练为什么要从复制走向分片和并行 |

| 时间计划                             | 学习内容                                                     | 学习资料          | 每日自检                                                     |
| ------------------------------------ | ------------------------------------------------------------ | ----------------- | ------------------------------------------------------------ |
| Day 8，2h 阅读，2h 跑仓库，1h 记录   | 认识 TorchTitan，理解它为什么像 PyTorch 训练侧的样板工程     | P4 R1             | 能说清 TorchTitan 想解决什么工程问题，它和单独拼装 FSDP、TP、PP 有什么差别 |
| Day 9，2h 阅读，2h 配置阅读，1h 复盘 | 学训练工程中的 checkpoint、resume、日志、容错                | P4 R1 B5          | 能说清大训练任务为什么必须把可靠性放进系统设计里             |
| Day 10，2h 阅读，2h 实操，1h 总结    | 从一个训练仓库里读配置，理清 tokenizer、dataset、scheduler、checkpoint 的入口 | R1 R3             | 能快速定位一个训练项目里的模型配置、数据配置、并行配置       |
| Day 11，2h 阅读，2h 对比，1h 总结    | 用一个表整理 DDP、FSDP、ZeRO、TP、PP 的适用场景              | D2 D3 D4 P4       | 能回答什么场景优先上 FSDP，什么场景优先上 TP 和 PP           |
| Day 12，2h 阅读，2h 实操，1h 记录    | 进入 Kubernetes 训练世界，理解 Kubeflow Trainer 在管什么     | D8                | 能说清 Kubeflow Trainer 和自己手写分布式启动脚本的区别       |
| Day 13，2h 阅读，2h 实操，1h 复盘    | 用一个小模型完成一次从单机到分布式的配置阅读或启动演练       | R1 R2 D8          | 能写出一个最小训练任务从本地到平台的迁移步骤                 |
| Day 14，3h 输出，2h 查漏             | 做第二周复盘，形成训练侧心智模型图                           | B2 D2 D3 D4 D8 P4 | 能画出训练链路里模型、数据、并行、网络、存储、监控的关系图   |

| 时间计划                            | 学习内容                                                     | 学习资料           | 每日自检                                                     |
| ----------------------------------- | ------------------------------------------------------------ | ------------------ | ------------------------------------------------------------ |
| Day 15，2h 阅读，2h 画流程，1h 复盘 | 进入推理引擎，理解 vLLM 的总体架构                           | D5 P3 R4           | 能说清 vLLM 为什么火，核心不只是快，而是内存管理和调度做得更好 |
| Day 16，2h 阅读，2h 记笔记，1h 复盘 | 专攻 PagedAttention 和 KV Cache                              | P3 D5              | 能解释 KV Cache 为什么会成为瓶颈，PagedAttention 在解决什么问题 |
| Day 17，2h 阅读，2h 对比，1h 输出   | 学 continuous batching、chunked prefill 这一套吞吐优化思路   | B6 D6              | 能讲清静态 batching 和 continuous batching 的差别            |
| Day 18，2h 阅读，2h 跑服务，1h 记录 | 看 TGI 架构，理解它的服务组件拆分方式                        | D6 R5              | 能对比 TGI 和 vLLM 的共同点与不同点，至少说出 3 条           |
| Day 19，2h 阅读，2h 画图，1h 总结   | 学 KServe 的 LLM runtime，理解 vLLM 怎么接进 Kubernetes      | D7                 | 能说清为什么很多团队会把推理服务放到 KServe 这种平台层来管   |
| Day 20，2h 阅读，2h 理解，1h 复盘   | 学 prefill 和 decode 解耦的思路，认识 disaggregated inference | B7 D7              | 能解释为什么长 prompt 和长输出会把一个统一服务拖慢           |
| Day 21，2h 实操，2h 压测，1h 总结   | 本地部署一个推理服务，观测吞吐、延迟、显存占用               | R4 或 R5，外加 D10 | 至少拿到一组 TTFT、吞吐、显存数据，并能解释哪个指标最先恶化  |

| 时间计划                              | 学习内容                                                     | 学习资料                   | 每日自检                                                     |
| ------------------------------------- | ------------------------------------------------------------ | -------------------------- | ------------------------------------------------------------ |
| Day 22，2h 阅读，2h 画平台图，1h 复盘 | 把训练和推理放回平台层，理解 Kubeflow 和 KServe 的分工       | D7 D8 B1                   | 能画出一个平台视角的 AI infra 结构图                         |
| Day 23，2h 阅读，2h 实操，1h 记录     | 学 Prometheus 和 OpenTelemetry，理解 metrics、traces、logs 怎么接起来 | D10 D11                    | 能列出推理服务最关键的 8 个监控指标                          |
| Day 24，2h 阅读，2h 实操，1h 总结     | 学模型评测和回归验证，把 lm-eval-harness 接到你的服务视角里  | R6                         | 能说清离线评测、线上监控、回归测试分别解决什么问题           |
| Day 25，2h 阅读，2h 输出，1h 复盘     | 学可靠性，理解硬件故障、训练中断、服务抖动为什么是 AI infra 核心问题 | B5 B4 D9                   | 能举出 3 类 AI 基础设施里最常见的故障点                      |
| Day 26，2h 阅读，2h 对比，1h 总结     | 学性能和成本优化，理解扩容、缓存、批处理、分层部署的思路     | D6 D7 B6 B7                | 能回答为什么单纯加 GPU 不一定是最优解                        |
| Day 27，3h 输出，2h 修订              | 做一个你自己的 mini AI infra 方案，包含训练、推理、监控、评测 | B1 B2 D7 D8 D10 D11        | 能把你的方案讲给别人听，并说清楚每一层为什么这样选           |
| Day 28，3h 模拟面试，2h 查漏          | 做总复盘，按面试口径整理高频题和项目表述                     | 全部资料里你标过重点的内容 | 能回答 10 个核心问题，例如 FSDP 和 ZeRO 的关系，vLLM 为什么快，KServe 在管什么，Prometheus 和 OTel 有什么分工 |

## 工程博客 / 文章

- B1 [Building Meta's GenAI Infrastructure](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/)
- B2 [How Meta trains large language models at scale](https://engineering.fb.com/2024/06/12/data-infrastructure/training-large-language-models-at-scale-meta/)
- B3 [Bringing Llama 3 to life](https://engineering.fb.com/2024/08/21/production-engineering/bringing-llama-3-to-life/)
- B4 [RoCE networks for distributed AI training at scale](https://engineering.fb.com/2024/08/05/data-center-engineering/roce-network-distributed-ai-training-at-scale/)
- B5 [How Meta keeps its AI hardware reliable](https://engineering.fb.com/2025/07/22/data-infrastructure/how-meta-keeps-its-ai-hardware-reliable/)
- B6 [Continuous batching from first principles](https://huggingface.co/blog/continuous_batching)
- B7 [Deploying Disaggregated LLM Inference Workloads on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)

## 官方文档

- D1 [PyTorch DDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- D2 [PyTorch FSDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- D3 [DeepSpeed ZeRO Docs](https://deepspeed.readthedocs.io/en/stable/zero3.html)
- D4 [Megatron Bridge Docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/)
- D5 [vLLM Docs](https://docs.vllm.ai/en/latest/)
- D6 [Hugging Face TGI Docs](https://huggingface.co/docs/text-generation-inference/en/index)
- D7 [KServe Docs](https://kserve.github.io/website/docs/intro)
- D8 [Kubeflow Trainer Overview](https://www.kubeflow.org/docs/components/trainer/overview/)
- D9 [NCCL Overview](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)
- D10 [Prometheus Getting Started](https://prometheus.io/docs/prometheus/latest/getting_started/)
- D11 [OpenTelemetry Docs](https://opentelemetry.io/docs/)

## Paper

- P1 [Megatron-LM](https://arxiv.org/abs/1909.08053)
- P2 [ZeRO](https://arxiv.org/abs/1910.02054)
- P3 [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)
- P4 [TorchTitan](https://arxiv.org/abs/2410.06511)

## 项目

- R1 [TorchTitan](https://github.com/pytorch/torchtitan)
- R2 [DeepSpeed](https://github.com/deepspeedai/deepspeed)
- R3 [Megatron-LM](https://github.com/nvidia/megatron-lm)
- R4 [vLLM](https://github.com/vllm-project/vllm)
- R5 [Text Generation Inference](https://github.com/huggingface/text-generation-inference)
- R6 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)