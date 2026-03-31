---
title: 00-Intro
date: 2026-03-02
tags:
course: AI
status: draft
---
# Main Takeaway
[TOC]

在振宇指导下的入门RL第一步，训练一个倒立摆

<!--more-->

## Intro



## 入门资料

- 入门文章：[open AI](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#key-concepts-and-terminology)
- 深度强化学习视频（牛逼）：https://www.youtube.com/playlist?list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0
- rl，模仿学习和vla
- isaac sim仿真

## Quick Start

环境配置，使用conda环境装好（这里我是用的是python 3.10，基本什么都支持）

1. 安装Gymnasium

   ```
   pip install Gymnasium
   ```

2. 安装pytorch，这里需要直接去`torch.org`官网上去找符合你CUDA Version的下载版本

   使用`nvidia-smi`查看显卡各种信息

   > 这种强依赖一般在前面安装

3. 安装Stable-Baseline3

   ```
   pip install stable-baselines3
   ```

4. 安装`tensorflow`

   > 不适用也可以跳过

5. 其他安装，例如pygame等

### Envs

下面对使用RL的环境进行一下介绍

- **Gymnasium** 是 OpenAI Gym 的官方继承者，由 Farama Foundation 维护的**强化学习环境**标准化库。
  - **标准化环境接口**: 提供统一的环境API，让不同算法可以无缝切换环境
  - **丰富的预置环境**: 包含经典控制、Atari游戏、机器人控制等多种任务
  - **环境注册系统**: 方便创建和管理自定义环境
- **Stable Baselines3** 是一个高质量的强化学习算法实现库，基于PyTorch构建

### Train

用随便一个LLM写好代码就可以训练，建议先用Gymnasium内置的几个环境训练

使用tensorboard可视化训练过程：

```
tensorboard --logdir=`your trained model path`
```

## References

- [Stable-Baselines3 Docs - Reliable Reinforcement Learning Implementations — Stable Baselines3 2.6.1a1 documentation](https://stable-baselines3.readthedocs.io/en/master/index.html)
- [Basic Usage - Gymnasium Documentation (farama.org)](https://gymnasium.farama.org/introduction/basic_usage/)
- [PyTorch](https://pytorch.org/)
- [open AI spinning up](https://spinningup.openai.com/en/latest/?utm_source=chatgpt.com)
