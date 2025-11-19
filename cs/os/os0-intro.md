# Main Takeaway

南大操作系统[操作系统原理 (2025 春季学期)](https://jyywiki.cn/OS/2025/)

<!--more-->

# Intro

## 课程相关

- [操作系统原理 (2025 春季学期)](https://jyywiki.cn/OS/2025/)
- lab：[2025 《操作系统》实验须知](https://jyywiki.cn/OS/2025/labs/Labs.md)
- 参考书：[Operating Systems: Three Easy Pieces](https://pages.cs.wisc.edu/~remzi/OSTEP/)

## 什么是操作系统

操作系统是一个典型的 “system software”——它完成对计算机硬件系统的抽象，为应用程序提供运行的支持。我们从三个视角观察操作系统：

> 操作系统本质是**硬件与软件的中间层**，学会用“抽象-实现”框架分析问题

- 从应用程序的视角看，操作系统定义了一系列的对象 (进程/线程、地址空间、文件、设备……) 和**操纵它们的 API** (系统调用)。这组强大的 API 把一台计算机的硬件资源分享给操作系统中的所有应用程序。我们看到的一切程序，不限于浏览器、游戏，甚至还包括容器、虚拟机、调试器、游戏外挂，都是在系统调用 API 上实现的。
- 从硬件的视角看，操作系统是一个**拥有访问全部硬件功能的程序** (操作系统就是个 C 程序，不用怕)。硬件会帮助操作系统完成最初的初始化和加载，之后，操作系统加载完第一个程序后，从此作为 “中断处理程序” 在后台管理整个计算机系统。
- 从数学的视角看，一切计算机系统都是如同 “1 + 1 = 2” 一样 “well-defined” 的**数学对象**，这包括机器、程序，当然也包括操作系统。计算机系统的行为是可以用数学函数 (当然，也可以用代码) 表示的。

常见的操作系统有：

- Windows、Linux、macOS、Android、iOS 等

## Why do we study OS?

更好地使用 Linux / 工具 / 性能调优

- `top`, `htop`, `ps`, `strace`, `lsof`, `vmstat`, `iotop` 等系统工具
- 知道高 CPU / 高内存 / I/O 瓶颈的根本原因
- 知道为什么要用多进程、多线程、协程（Coroutine），以及他们的优缺点
- 能看懂一些内核参数、调度策略、内存占用等信息

简单说：**从“瞎用工具”变成“理解背后原理的人”**。



## 操作系统提供哪些服务？（OS Services）

对应用程序（Applications）和用户（Users)来说，OS 提供的常见服务有：

1. **程序执行（Program Execution）**
   
   - 装载程序到内存、创建进程、执行、结束

2. **I/O 操作（I/O Operations）**
   
   - 读取/写入文件、网络通信、终端输入输出等

3. **文件系统操作（File System Manipulation）**
   
   - 创建、删除、打开、关闭、读写文件和目录

4. **进程管理（Process Management）**
   
   - 创建/撤销进程、进程间通信（IPC）、进程同步、调度

5. **内存管理（Memory Management）**
   
   - 为进程分配和回收内存、虚拟内存、页表等

6. **安全与保护（Security & Protection）**
   
   - 用户权限、访问控制、进程隔离

7. **错误检测与处理（Error Detection & Handling）**

这些服务，绝大多数是通过 **系统调用（System Calls）** 暴露给应用程序的。

## 用户态和内核态（User Mode & Kernel Mode）

为了安全与保护，现代 CPU 支持至少两种执行特权级：

1. **内核态（Kernel Mode / Supervisor Mode）**
   
   - 可以执行特权指令（Privileged Instructions）
   
   - 可以访问任意内存、配置设备、修改页表等
   
   - 操作系统内核（Kernel）运行在内核态

2. **用户态（User Mode）**
   
   - 不能执行特权指令
   
   - 不能随便访问敏感的硬件资源
   
   - 普通应用程序运行在用户态

**为什么要区分？**

- 防止普通程序破坏系统，比如：
  
  - 随便改页表
  
  - 随便重启设备
  
  - 访问不属于自己的内存



## 系统调用（System Calls）与 OS 的“入口”

应用程序要用 OS 的服务（比如读文件、创建进程），不能直接“跳进内核代码”，  
而是通过**系统调用（System Call）**来请求内核帮忙。

大致流程：

1. 在用户程序中调用库函数（如 C 的 `read()`、`write()`、`fork()` 等）

2. 库函数内部使用特殊指令（如 **trap / syscall / int**）

3. CPU 从用户态切换到内核态（User → Kernel），控制权交给 OS 内核

4. 内核根据系统调用号（System Call Number）找到对应的内核服务函数

5. 内核完成操作后，把结果返回用户态（Kernel → User）

## 操作系统的整体结构（OS Structure）

不同 OS 的内部结构设计不一样，但经典结构大致有几类，考试和理解都很重要。

### 1. 单体内核（Monolithic Kernel）

**概念：**

- 内核的大部分功能（进程管理、内存管理、文件系统、设备驱动等）都在一个大的内核地址空间中运行。

- 模块之间调用普通函数即可，性能高，但耦合度大。

- 代表：早期 UNIX、Linux（本质是单体 + 模块化）

---

### 2. 分层结构（Layered Architecture）

**思想：**

- 把操作系统分为多层（Layer 0, Layer 1, …），每一层只依赖下层提供的服务。

- 比如：
  
  - 最底层是硬件抽象
  
  - 上面是内存管理
  
  - 再上面是进程管理、文件系统
  
  - 最上面是系统调用接口

好处：

- 结构清晰，容易理解和维护  

缺点：

- 设计层次划分比较困难；可能会牺牲一些性能。

**关键词：**

- 分层结构 — **Layered Design / Layered Architecture**

---

### 3. 微内核（Microkernel）

**思想：**

- 内核中只保留最基础的功能（例如：进程间通信 IPC、基本调度、最基础的内存管理）。

- 其他如文件系统、设备驱动、协议栈等都尽量放到用户态的服务进程中运行。

优点：

- 更好的模块化与可靠性：某个服务挂了不会直接导致整个内核崩溃。  
  缺点：

- 用户态 ↔ 内核态切换、进程间通信（IPC）开销更大，性能挑战较大。

**关键词：**

- 微内核 — **Microkernel**

- 进程间通信 — **IPC (Inter-Process Communication)**

代表：Minix 3、QNX，一些现代系统内核设计理念也深受影响。

---

### 4. 模块化内核（Modular Kernel）

**思想：**

- 内核本身支持**动态加载模块（Loadable Kernel Modules）**，  
  比如设备驱动可以在运行时插入/移除，而不需要重编译整个内核。

- 结合了“单体内核性能好”和“微内核易扩展”的某些优点。

代表：Linux 内核（Monolithic + Modules）。

# Contents

### Step 0：预备知识 Check（Prerequisites）

**主题**：

- C 语言 / 指针（C Programming, Pointers）
- 计算机组成基础（Computer Organization）

**你要能做到：**

- 会写基本的 C 程序（函数、数组、指针、结构体）
- 知道 CPU、内存（Memory）、磁盘（Disk）大概干嘛的
- 至少听说过：指令执行、寄存器（Register）、总线（Bus）这些词

**建议**：

- 如果 C 很生疏，先用几天“回炉”一下，否则后面调系统调用 / 看内存会很痛苦。

------

### Step 1：操作系统概念与结构（OS Overview & Structure）

**主题（中英）**：

- 操作系统基本概念 — **Operating System Basics**
- OS 在计算机中的位置 — **OS as an Abstraction Layer**
- OS 的目标：可靠性、效率、公平性、可扩展性

**关键概念**：

- 系统调用 — **System Call**
- 用户态 / 内核态 — **User Mode / Kernel Mode**
- 内核 — **Kernel**（宏内核 Monolithic Kernel / 微内核 Microkernel）

**你要能回答：**

- 操作系统（OS）在硬件和应用程序之间扮演什么角色？
- 为什么应用程序不能直接操作硬件？
- 什么是系统调用（System Call），为什么需要它？
- 用户态和内核态的区别是什么？为什么要区分？

------

### Step 2：进程（Process）与程序执行

**主题**：

- 进程概念 — **Process Concept**
- 进程状态 — **Process States**（Ready, Running, Blocked 等）
- 进程控制块 — **PCB: Process Control Block**
- 进程创建与销毁 — **Process Creation & Termination**

**你要能回答：**

- 什么是进程（Process）？和“程序 Program”有什么区别？
- 进程有哪些状态（就绪 / 运行 / 阻塞），状态转换如何发生？
- PCB（进程控制块）里大概包含哪些信息？
- 在类 Unix 系统中，`fork()` 和 `exec()` 在做什么？

------

### Step 3：CPU 调度（CPU Scheduling）

**主题**：

- 调度目标 — **Scheduling Criteria**（吞吐量、响应时间、公平性）
- 调度算法 — **Scheduling Algorithms**
  - 先来先服务：**FCFS (First-Come, First-Served)**
  - 最短作业优先：**SJF (Shortest Job First)**
  - 时间片轮转：**Round Robin (RR)**
  - 优先级调度：**Priority Scheduling** 等

**你要能回答：**

- 为什么需要 CPU 调度（Scheduling）？
- 不同调度算法的特点、优缺点？
- 周转时间（Turnaround Time）、等待时间（Waiting Time）等指标如何计算？
- RR 为什么对交互式系统友好？

------

### Step 4：线程（Thread）、并发（Concurrency）与上下文切换

**主题**：

- 线程概念 — **Thread**（轻量级进程 Light-weight Process）
- 多线程模型 — **Multithreading Models**
- 上下文切换 — **Context Switch**

**你要能回答：**

- 进程（Process）和线程（Thread）的区别是什么？
- 一个进程内多个线程共享哪些资源？哪些不共享？
- 什么是上下文切换（Context Switch），为什么会有开销？

------

### Step 5：同步与互斥（Synchronization & Mutual Exclusion）

**主题**：

- 临界区 — **Critical Section**
- 互斥锁 — **Mutex / Lock**
- 信号量 — **Semaphore**（计数信号量 Counting Semaphore / 二进制信号量 Binary Semaphore）
- 条件变量 — **Condition Variable**
- 经典同步问题：生产者-消费者、读者-写者、哲学家就餐等

**你要能回答：**

- 并发程序为什么会出现**竞态条件（Race Condition）**？
- 什么是临界区（Critical Section），如何实现互斥（Mutual Exclusion）？
- 信号量（Semaphore）怎么用？`P` 和 `V` 操作含义是什么？
- 用伪代码或 C 写出一个简单的同步方案（比如生产者-消费者）。

------

### Step 6：死锁（Deadlock）

**主题**：

- 死锁 — **Deadlock**
- 死锁的四个必要条件 — **Coffman Conditions**
- 死锁预防 / 避免 / 检测与恢复 — **Prevention / Avoidance / Detection & Recovery**
- 银行家算法 — **Banker’s Algorithm**

**你要能回答：**

- 死锁的定义是什么？和“饥饿（Starvation）”区别？
- 死锁发生的四个必要条件分别是什么？
- 大致理解银行家算法的思想（不用太纠结公式细节）。

------

### Step 7：内存管理（Memory Management）与虚拟内存（Virtual Memory）

**主题**：

- 逻辑地址 / 物理地址 — **Logical Address / Physical Address**
- 分区管理 — **Contiguous Allocation**（首次适配 First Fit 等）
- 分段 — **Segmentation**
- 分页 — **Paging**（页 Page、页帧 Frame）
- 多级页表 — **Multi-level Page Table**
- 虚拟内存 — **Virtual Memory**
- 请求分页、缺页中断 — **Demand Paging, Page Fault**
- 页面置换算法 — **Page Replacement Algorithms**
  - FIFO, LRU, Optimal 等

**你要能回答：**

- 虚拟内存（Virtual Memory）解决了什么问题？
- 页（Page）、页帧（Frame）、页表（Page Table）分别是什么？
- 发生缺页中断（Page Fault）时大致会经历哪些步骤？
- 置换算法（FIFO/ LRU）的思想和优缺点？

------

### Step 8：文件系统（File System）

**主题**：

- 文件 — **File**（属性、操作）
- 目录结构 — **Directory Structure**
- 索引节点 — **Inode**
- 文件分配方式 — **File Allocation Methods**
  - 连续分配（Contiguous）
  - 链接分配（Linked）
  - 索引分配（Indexed）
- 空闲空间管理 — **Free Space Management**

**你要能回答：**

- 文件（File）在操作系统眼中是什么？
- 路径（Path）、目录（Directory）是怎么组织的？
- 为什么需要 Inode 这种结构？
- 大致理解文件是如何映射到磁盘块（Disk Blocks）上的。

------

### Step 9：I/O 与设备管理、安全与虚拟化（I/O, Device Management, Security & Virtualization）

**主题**：

- I/O 系统 — **I/O System**
- 缓冲、缓存 — **Buffering, Caching**
- 设备驱动 — **Device Driver**
- 权限与保护 — **Protection & Security**
- 简单了解：虚拟机 / 容器 — **Virtual Machine, Container**

**你要能回答：**

- OS 如何管理 I/O 设备？
- 缓冲（Buffer）和缓存（Cache）各解决什么问题？
- 用户权限（User Privilege）大概怎样控制？

------

### Step 10：综合与实践（Practice & Integration）

**最后阶段建议**：

- 在 Linux 下动手做一些小实验：
  - 用 `fork()` 写一个创建子进程的小程序
  - 用 `pthread` 写多线程和互斥锁、条件变量
  - 写一个简单的生产者-消费者模型
- 尝试看一看一个真实 OS（比如 xv6 / Linux）的部分代码：
  - 了解系统调用实现流程
  - 看看调度或内存管理的一小部分实现

# References

- [操作系统原理 (2025 春季学期)](https://jyywiki.cn/OS/2025/)
