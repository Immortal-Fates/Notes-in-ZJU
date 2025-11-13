# Computational Performance

This chapter will focus on the major factors that affect computational performance: imperative programming, symbolic programming, asynchronous computing, automatic parallelism, and multi-GPU computation.

# Compilers and Interpreters

- Imperative programming is easier. But the single-threaded Python interpreter becomes the bottleneck.借 JIT/AOT/原生扩展踩到性能油
- Symbolic programming is more efficient and easier to port.(更易移植是)

## Hybrid Programming

- Hybrid Programming combines Imperative programming and Symbolic programming.

- This seems almost too good to be true: write the same code as before and simply convert the model using `torch.jit.script`. Once this happens the network is optimized.

  ```
  net = torch.jit.script(net)
  ```

### Serialization

One of the benefits of compiling the models is that we can serialize (save) the model and its parameters to disk. U

# Asynchronous Computation

Understanding how asynchronous programming works helps us to develop more efficient programs, by proactively reducing computational requirements and mutual dependencies. This allows us to reduce memory overhead and increase processor utilization.

## Barriers and Blockers



# Automatic Parallelism

- Modern systems have a variety of devices, such as multiple GPUs and CPUs. They can be used in parallel, asynchronously.
- Modern systems also have a variety of resources for communication, such as PCI Express, storage (typically solid-state drives or via networks), and network bandwidth. They can be used in parallel for peak efficiency.
- The backend can improve performance through automatic parallel computation and communication.

![DM_20251110211534_001](./assets/DL7-Computational_Performance.assets/DM_20251110211534_001.svg)

