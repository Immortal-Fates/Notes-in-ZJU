# Metaprogramming Introduction

Metaprogramming 的意思是用程序来操作程序本身，或者用工具自动生成、构建、测试、发布你的代码与产物。

- 自动构建
- 自动测试
- 自动运行实验
- 自动生成产物与依赖关系
- 自动发布与版本管理

<!--more-->

## 构建系统 Build systems

构建系统解决的问题是

- 你的项目如何从源代码变成可运行产物
- 当源文件变化时，哪些步骤需要重新执行
- 如何避免每次都全量重跑

核心概念

- 目标 target
   你想得到的东西，比如可执行文件、报告、模型权重、论文 PDF
- 依赖 dependencies
   目标依赖的输入，比如源代码、数据、配置、生成脚本
- 规则 rule
   如何从依赖生成目标
- 增量构建
   只有依赖发生变化时才重建

## Make 与 Makefile

Make 是最经典的构建系统之一，特点是简单、通用、依赖图清晰。即便你主要写 Python，也非常适合用 Make 把常用命令规范化。

### Makefile 基本形态

```
target: dep1 dep2
<TAB> command
```

重点

- command 行必须用 Tab 开头
- make 默认构建第一个 target
- make 用文件时间戳判断是否需要重建

### 最常用的 Python 项目 Makefile 模板

```
.PHONY: help run test lint format clean

help:
	@echo "make run | test | lint | format | clean"

run:
	python -m your_pkg

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache dist build
```

用法

```
make test
make format
```

好处

- 把团队约定写成规则
- 统一入口，少记命令
- CI 里直接复用

### 变量与复用

```
PY=python
PKG=your_pkg

run:
	$(PY) -m $(PKG)
```

### 自动推导规则

Make 支持模式规则，适合批量生成。
 例如把 markdown 编译成 pdf

```
%.pdf: %.md
	pandoc $< -o $@
```



## Dependency Management

### Semantic Versioning

> [!TIP]
>
> check the [details](https://semver.org/)

With semantic versioning, every version number is of the form: major.minor.patch. The rules are:

- If a new release does not change the API, increase the patch version.
- If you *add* to your API in a backwards-compatible way, increase the minor version.
- If you change the API in a non-backwards-compatible way, increase the major version.

## 持续集成 Continuous integration

CI 的意义是把你的 Makefile 目标在干净环境里自动跑一遍，保证

- 每次提交都能通过测试
- 代码风格一致
- 构建流程可重复

典型 CI 流程做的事

- 安装依赖
- 运行 `make test`
- 运行 `make lint`
- 产物构建与上传

核心思想

- 让机器做重复的验证
- 让失败尽早暴露

------

## 自动化的工程习惯

这一讲的落点通常是这些习惯

### 把常用命令收敛成固定入口

- 终端里少敲长命令
- 不靠记忆靠规则
- 新同事也能一眼看懂怎么跑项目

Makefile 是最常见的收敛手段。

### 用规则表达依赖关系

如果某个产物依赖数据与脚本，就应该写成依赖图，而不是手动记步骤。

### 增量执行

不要每次全量重跑，尤其是训练、渲染、编译这类慢任务。

## TODO

如果你的目标是工程效率与日常调试能力，这一讲最值得马上用起来的是这三件事

1. 每个项目放一个 Makefile，至少有 run test lint format clean
2. 把耗时任务做成可增量的目标，生成到明确的输出目录
3. 让 CI 只做两件事，跑测试与静态检查，保证主干健康

## References

- [missing semester metaprogramming](https://missing.csail.mit.edu/2020/metaprogramming/)