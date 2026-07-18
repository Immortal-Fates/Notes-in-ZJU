# Code Format

在看了和自己已经写了这么多代码之后，总结一下code format，形成一个良好的代码习惯是必要的

## styles

首先肯定是google style，希望以后都遵循其中的代码风格

- [Google Style Guides | styleguide](https://google.github.io/styleguide/)



## Format

大家对于自己的代码都有不同的组织形式，而且会根据项目的大小以及维护方式而各异，因此这里主要放一些典型的好的实现，并且说明为什么他们要这么做

### Simple

在一些算法的轻量化的实现中，简单易读是重要的标准

- [openai/spinningup: An educational resource to help anyone learn deep reinforcement learning.](https://github.com/openai/spinningup)

  这是openai关于deep rl的实现库，所有的算法实现都遵循统一的模板结构。这些实现被分为两个文件：一个是算法文件，其中包含算法的核心逻辑；另一个是核心文件，其中包含运行该算法所需的各种辅助功能

- 模型训练：常常分为：数据层，模型层，训练层来分别实现



