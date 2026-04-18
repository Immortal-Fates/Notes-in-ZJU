# DDP

学 DDP，弄清单机多卡最基础的数据并行，能说出 DDP 的执行流程，能解释梯度同步发生在什么地方

梯度同步：到底如何实现的？？？

## Takeaway

DDP(Distributed Data Parallel)是 PyTorch 里的数据并行方案。它的做法是让每个进程各自持有一份完整模型副本，各自处理不同数据，反向传播时用分布式通信把各个进程的梯度同步起来，然后每个进程各自执行同样的参数更新。

PyTorch 官方建议一张 GPU 对应一个 DDP 进程，GPU 之间的梯度同步依赖 `torch.distributed`，而在 GPU 场景下最快且强烈推荐的后端是 NCCL。

我们可以将DDP完整地理解为两层：

1. 上层pytorch DDP: 负责训练语义，也就是谁持有模型、什么时候同步、哪些张量要同步、训练代码怎么写。它会给参数注册 autograd hook，在 backward 过程中触发梯度同步。它还会在构造阶段做参数和 buffer 的初始同步。
2. 下层NCLL：负责 GPU 之间真正的数据通信。它不是完整并行框架，而是一个专门做 GPU 间通信的库，提供 AllReduce、Broadcast、AllGather、ReduceScatter、AlltoAll，还有点对点 send 和 receive。它是 topology-aware 的，会根据底层互联去做优化，并且把通信和本地规约放进单个 kernel 里执行。

## Motivation

在大致知道DDP是什么后，我们就要思考：为什么不直接用 DataParallel，而要学 DDP？ 反正我们的目的就是多卡一起加速训练我们的模型，下面给出这个问题的三个方面回答

1. DDP能多机：DataParallel 是单进程多线程，只能单机使用。DDP 是多进程，既能单机也能多机（一般一张GPU对应一个进程）

   > 可能咱们个人不太能感受，但是大的机房进行大规模模型训练，多机就会显得非常重要

2. DDP一般更快：DataParallel因为 GIL、每轮复制模型、scatter 和 gather 的额外开销往往更慢，及时在单机上

   > 因为多进程可以绕开单进程多线程带来的解释器竞争和 GIL 问题

3. DDP能和模型并行一起使用，DataParallel 不能：这是因为有些模型太大，我们需要对split model，因此model parallel在大模型训练也非常重要

在进入DDP之前，我们需要先了解几个名词

- `world_size` 是总进程数
- `rank` 是全局进程序号
- `LOCAL_RANK` 是当前节点内的本地序号
- `process_group` 是一组要一起通信的进程

## Core Mechanism

这里我们来看到底`torch.nn.parallel.DistributedDataParallel`在每个iteration中是如何step by step的：check here [pytorch ddp](https://docs.pytorch.org/docs/main/notes/ddp.html)

- Prerequisite: DDP relies on c10d `ProcessGroup` for communications.因此在DDP创建之前需要先构造`ProcessGroup`实例，（见pipeline的流程）

- Construction: 构造阶段干了两件事

  1. 先把 rank 0 的 `state_dict()` 广播给其他进程，保证每个进程初始化相同。

  2. 然后每个process创建一个自己的`Reducer`，专门负责后续反向传播中的梯度同步（这是DDP中最重要的部件）。

     为了提高通信效率，Reducer将参数梯度组织成`bucket`，若干“梯度包裹”，后面按包裹来通信，而不是每出一个梯度就立刻单独通信。这样更省通信开销

     除了分桶之外，Reducer 还在构造过程中注册 autograd hook，one hook per parameter。当梯度准备好时，这些hook将在向后传递过程中被触发。

- forward：正常跑和普通单卡差不多，但是当`find_unused_parameters=True` 的时候。此时 DDP 会从输出往回检查 autograd 图，看看哪些参数这次根本没参与反向传播。然后它会把这些没用到的参数提前标记成 ready，避免 backward 时一直傻等一个永远不会出现的梯度

  > [!NOTE]
  >
  > 这个选项不能乱开。因为遍历整张 autograd 图本身会带来额外开销。只有你的模型确实存在某些轮次不用到的参数时，才值得开它

- backward: `loss.backward()`表面上是你在调用 autograd，但 DDP 会借助前面注册好的 hook 插进来工作。某个参数的梯度一准备好，对应的 hook 就会被触发，DDP 就把这个参数标记为“可以同步了”。

  等到某个 bucket 里的梯度全都准备好了，`Reducer` 就会对这一桶发起一次异步 `allreduce`。文档里说，这一步是为了计算所有进程之间梯度的平均值。等所有 bucket 都完成以后，DDP 会等待这些 `allreduce` 结束，然后把平均后的结果写回每个参数的 `param.grad`。所以 backward 结束以后，不同进程上同一个参数的梯度应该是一样的

   ![72401724-d296d880-371a-11ea-90ab-737f86543df9](./assets/03-DDP.assets/72401724-d296d880-371a-11ea-90ab-737f86543df9.png)

## Pipeline

1. `torchrun` 或 `mp.spawn` 拉起多个进程。

   > [!NOTE]
   >
   > 真实项目里，为什么很多人不用 `mp.spawn`，而更常见 `torchrun`

2. 每个进程绑定自己的设备，并调用 `init_process_group` 加入同一个 process group。

3. 每个进程各自创建模型副本，并用 `DistributedDataParallel` 包起来。DDP 在构造时会把模型状态从 rank 0 广播到其他进程。

4. 每个进程喂自己的那份数据。

   > [!TIP]
   >
   > DDP 不会自动帮你切分输入，数据怎么分是你自己负责，典型做法是配 `DistributedSampler`。

5. forward 计算本地 loss，backward 计算本地梯度。参数上的 autograd hook 会触发梯度同步。

6. 这些同步通常会落到 NCCL 的 collective 上，最常见就是 AllReduce。各进程拿到一致梯度以后，各自做同样的 `optimizer.step()`。

### setup 和 cleanup

在DDP任务开始前，我们需要准备进程之间的连接

```python
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
    # such as CUDA, MPS, MTIA, or XPU.
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
```

教程里的 `setup(rank, world_size)` 做了三件事。

1. 设置 `MASTER_ADDR` 和 `MASTER_PORT`。这相当于告诉所有进程，它们要去哪里合并。
2. 根据当前 accelerator 选择默认 backend: such as CUDA, MPS, MTIA, or XPU
3. 调用 `dist.init_process_group(backend, rank=rank, world_size=world_size)` 初始化进程组。

对应地，`cleanup()` 调 `destroy_process_group()` 做收尾

> [!NOTE]
>
> 所以这里我们就知道了不是写了`DDP(model)`就自动分布式了，而是进程组已经提前建好

### a demo

a demo: 先从demo中大致了解整个DDP的框架

```python
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
```

我们可以看见ddp_model用DDP包装一个model后就像使用本地模型一样

反向梯度传播会自动同步：When the `backward()` returns, `param.grad` already contains the synchronized gradient tensor.

在了解了大致框架后，我们还有几个问题需要解决

### Skewed Processing Speeds: 处理速度不均匀

In DDP, the constructor, the forward pass, and the backward pass are distributed synchronization points. 我们希望不同进程能够同步，同时到达每个需要同步的点，但是实际上这是不可能实现的，因此快的进程常常会早到然后开始等待落后的进程，所以我们需要平衡不同进行的工作负载分配。但是有时也是因为一些其他因素：e.g., network delays, resource contentions, or unpredictable workload spikes. 因此在pass a sufficiently large `timeout` value when calling init_process_group.是一个较好的方法

### Save and Load Checkpoints

我们经常用`torch.save` and `torch.load`来处理checkpoints，在DDP中常仅让一个 rank 保存 checkpoint，因为理论上所有 rank 的参数是一致的。加载时要加载到所有rank，处理好 `map_location`（告诉每个进程应该把模型加载到哪张卡上）

> [!TIP]
>
> 没有map_location可能会出现多个进程都把参数加载到同一组设备上的问题：不写 `map_location`，`torch.load` 会先把模型读到 CPU，然后再按照保存时的设备位置，把参数拷回对应的 GPU。而保存checkpoint时对应的device只是其中一张卡。

```python
def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])


    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
    # such as CUDA, MPS, MTIA, or XPU.
    acc = torch.accelerator.current_accelerator()
    # configure map_location properly
    map_location = {f'{acc}:0': f'{acc}:{rank}'}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location, weights_only=True))

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
    print(f"Finished running DDP checkpoint example on rank {rank}.")
```

### Combining DDP with Model Parallelism

这部分我暂时不需要

### Initialize DDP with torch.distributed.run/torchrun

我们可以利用 PyTorch Elastic 简化 DDP 代码并更轻松地初始化工作

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic():
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend)
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    # create model and move it to GPU with id rank
    device_id = rank % torch.accelerator.device_count()
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_id)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    dist.destroy_process_group()
    print(f"Finished running basic DDP example on rank {rank}.")

if __name__ == "__main__":
    demo_basic()
```

然后就可以使用torchrun来让所有节点初始化DDP

```bash
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 elastic_ddp.py
```

这里torchrun将启动2台机器8个进程，一共16个GPU，并在启动它的节点上的每个进程上调用elastic_ddp.py，但用户还需要应用slurm等集群管理工具才能在2个节点上实际运行此命令。

```
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
```

## NCCL

- DDP 负责“该在什么时候同步什么”

- NCCL 负责“这些数据到底怎么高效传过去”

NCCL(NVIDIA Collective Communications Library)是专门面向GPU通信的库

- topology-aware: 会考虑硬件连接关系
- 就是为了加速GPU 之间的数据交换和规约
- 和 CUDA 结合紧密

提供了很多操作，这里就不详细展开了，具体查看References中的官方文档

> [!NOTE]
>
> reduce 再接一个 Broadcast，效果等价于 AllReduce

## Q&A

1. 进程视角，而不是线程视角

   DDP 不是单进程多线程的 `DataParallel`。它是多进程方案，支持单机和多机。单机多卡时 DDP 往往比 `DataParallel` 更快，每个rank都是一份独立的训练进程

   > [!NOTE]
   >
   > DDP 为什么比 `DataParallel` 更常用，它和 FSDP、模型并行分别解决什么问题?
   >
   > 模型能装进单卡但想横向扩展，就优先用 DDP。模型装不进单卡，再考虑 FSDP

2. 为什么训练代码里写的是 `DDP(model)`，真正跨卡传梯度却经常是 NCCL 在干活

## References

- [pytorch DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)

- [NCCL(NVIDIA Collective Communications Library)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)