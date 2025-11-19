# Process and Program

## Program vs Process

- 进程process
  
  - 存在于**磁盘上的一段静态代码和数据**，没有运行，只是一个可执行文件

- 程序program
  
  - 一次“正在执行的程序”的**动态实例（a program in execution）**
  
  - 当你运行一个程序时，操作系统会为它创建一个进程：
    
    - 分配内存（address space）
    
    - 分配必要的资源（打开的文件、I/O、内核数据结构等）
    
    - 初始化其运行状态（寄存器、程序计数器等）

## Process States

OS 不只是“让进程跑”，而是要管理很多进程，所以需要给进程定义不同的**状态（State）**

- **新建（New）**
  
  - 进程正在被创建，还没准备好运行。

- **就绪（Ready）**
  
  - 进程已经具备运行条件，**等待 CPU** 分配时间片。
  
  - 在内存中，随时可以上 CPU。

- **运行（Running）**
  - 进程当前占用 CPU，正在执行指令。
  
- **阻塞 / 等待（Blocked / Waiting）**
  
  - 进程暂时不能继续执行，比如在等待 I/O（读磁盘、等网络、等键盘输入）。
  
  - 即使有 CPU 也跑不动，只能等事件发生。

- **结束（Terminated）**
  - 进程执行完或被杀死（killed），资源被回收。

## 进程控制块 PCB（Process Control Block）

PCB 是 OS 用来“记录一个进程所有关键信息”的结构体。

典型 PCB 中包含：

1. **进程标识信息（Process Identification）**
   - 进程 ID：**PID (Process ID)**
   
   - 父进程 ID：**PPID (Parent Process ID)**
   
   - 用户 ID（UID）、组 ID（GID）等
   
2. **处理器状态信息（CPU State）**
   
   - 通用寄存器（Registers）
   
   - 程序计数器（Program Counter, PC）：下一条要执行的指令地址
   
   - 程序状态字（PSW / Flags）

3. **调度相关信息（Scheduling Info）**
   
   - 进程状态（State）
   
   - 优先级（Priority）
   
   - 所在队列等

4. **内存管理信息（Memory Management Info）**
   
   - 基址寄存器 / 界限寄存器（Base & Limit Registers）
   
   - 页表指针（Page Table Base Register）等

5. **I/O 信息和文件信息（I/O and File Info）**
   
   - 已打开文件列表（Open Files）
   
   - I/O 设备分配情况等

当发生**上下文切换（Context Switch）**时，OS 会：

1. 把当前进程的 CPU 状态保存到它的 PCB 里

2. 从另一个进程的 PCB 中取出之前保存的 CPU 状态

3. 切换到另一个进程继续运行

## 进程的地址空间（Process Address Space）

从进程内部的角度看，它“看到”的是一个属于自己的**虚拟地址空间（Virtual Address Space）**，典型划分：

由低地址到高地址，大致分为：

1. **代码段（Text Segment / Code Segment）**
   
   - 存放程序的机器指令。

2. **数据段（Data Segment）**
   
   - 已初始化的全局变量、静态变量。

3. **BSS 段（BSS Segment）**
   
   - 未初始化的全局/静态变量。

4. **堆（Heap）**
   
   - 动态分配内存（如 `malloc` / `new`），向高地址增长。

5. **栈（Stack）**
   
   - 函数调用栈帧（Stack Frame）：局部变量、返回地址等，一般向低地址增长。

大概示意（逻辑上）：

```
高地址
+-------------------+
|       Stack       |
|   (function call) |
+-------------------+
|        ...        |
+-------------------+
|        Heap       |
|  (malloc/new)     |
+-------------------+
|   BSS + Data      |
| (global/static)   |
+-------------------+
|       Text        |
|     (code)        |
+-------------------+
低地址
```

> Tips: 每个进程觉得自己有一整块连贯的地址空间，这就是**虚拟内存（Virtual Memory）**的抽象。

## 进程的创建与终止（Process Creation & Termination）

创建、销毁进程的系统调用 API，以及如何用它们构建最基本的多进程程序。

### 1. 进程的创建（Process Creation）

典型操作系统中，创建新进程通常包含：

1. 分配一个新的 PID

2. 为其创建 PCB，并初始化各种字段

3. 分配地址空间、设置页表等

4. 初始化代码段、数据段、堆栈等

5. 把进程放到就绪队列（Ready Queue）

#### Unix / Linux 中典型的创建方式：`fork()` + `execve()`

- `fork()`：Create a child process
  - 创建一个**几乎完全相同的子进程（Child Process）**，子进程得到父进程地址空间的“复制”
  - `fork()` 在父进程中返回子进程 PID，在子进程中返回 0。
  - 如果失败（比如进程数太多、资源不足），在父进程中返回 **-1**，且不会创建子进程
- `execve()`：在当前进程中执行新程序
  - 在当前进程中**装入一个新的程序映像（Program Image）**，替换原来的代码和数据。
  - PID 不变，但“肚子里的程序”换了。
  - 正常情况无返回值，只有在出错时才返回 **-1**，并设置 `errno`（例如文件不存在、无执行权限、格式错误等）。

在 Unix/Linux 里，“**创建一个新进程并执行另一个程序**”的常见模式是：

1. 父进程调用 `fork()`：
   - 得到一个几乎一样的子进程
2. 在子进程中调用 `execve()`：
   - 把子进程的程序映像替换为目标可执行文件
3. 父进程调用 `wait()` / `waitpid()` 等等待子进程结束（如果需要）

典型写法：

```
pid_t pid = fork(); 
if (pid == 0) {    
    // 子进程     
    execve("/bin/ls", "ls", NULL); 
} else if (pid > 0) {
    // 父进程     
    wait(NULL); // 等子进程结束 
}
```

这就是为什么使用`pstree`可以发现所有进程都是从1(systemd)开始的

> Fork Bomb
>
> ```
> :(){:|:&};:   # bash: 允许冒号作为 identifier
> 
> :() {         # 格式化一下
>   : | : &
> }; :
> ```



### 2. 进程的终止（Process Termination）

进程可能通过多种方式结束：

- 正常结束（Normal Exit）

- 出错退出（Error Exit）

- 被其他进程杀死（Killed by Another Process）
  
  - 比如 `kill` 命令，`SIGKILL`, `SIGTERM` 等信号

- 被操作系统因为某些错误强制终止（如非法内存访问）

结束时 OS 需要：

1. 回收该进程占用的所有资源（内存、I/O、内核对象）

2. 更新进程状态为 Terminated

3. 通知父进程（在 Unix 中，父进程可以用 `wait()`/`waitpid()` 获取子进程的退出状态）

# Test

多做测试 
