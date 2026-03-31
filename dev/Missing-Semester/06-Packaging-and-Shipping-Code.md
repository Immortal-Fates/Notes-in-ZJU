# Packaging and Shipping Code

Source: <https://missing.csail.mit.edu/2026/shipping-code/> (The Missing Semester, 2026)

## Why package and ship code

- 打包（Packaging）：把源码变成可分发的交付物（artifact）。
- 交付（Shipping）：让别人能在“不是你这台机器”的环境里安装和运行。
- 核心问题不是“代码能跑”，而是“代码能否在目标环境稳定重现”。

## Core idea: package != app image

- Package 常见形态：Python wheel、npm package、Rust crate。
- 运行交付物还可能是：
  - 可执行二进制
  - 容器镜像（image）
  - 虚拟机镜像（VM image）
- 先定义 artifact，再定义它对环境的假设（OS、架构、系统库、运行时版本）。

## Dependencies & Environments

### 依赖的本质

- 直接依赖：代码里直接使用的库。
- 传递依赖：依赖的依赖。
- 依赖冲突（dependency hell）：不同包对同一库要求不兼容版本。

### 环境隔离

- 每个项目单独虚拟环境（venv）是默认最佳实践。
- 不要污染系统 Python（操作系统本身可能依赖）。
- 环境不仅隔离包，也可隔离运行时版本（如 Python 3.11/3.12）。

### 工具与规范（Python）

- 规范层：PEP 517（build interface）、PEP 621（`pyproject.toml` metadata）。
- 工具层：`pip` / `uv`，接口相近，`uv` 通常更快。

## Package metadata: what to define

`pyproject.toml` 至少应覆盖：

- `name`、`version`、`description`
- `dependencies`
- `project.scripts`（CLI 入口）
- `build-system`（backend 与构建依赖）
- Python 版本约束（`requires-python`）

经验规则：元数据应足够让工具“无需口头说明”就能 build/install/run。

## Artifacts & Packaging

### 源码 vs Artifact

- 源码是给开发者读写的。
- Artifact 是给安装器/部署系统消费的。

### Python 常见 artifact

- `sdist`：源码分发包（`.tar.gz`）
- `wheel`：二进制/可安装包（`.whl`）

### 最小打包流程（Python）

1. 准备 `pyproject.toml` 和包代码。
2. 运行 `uv build` 产出 `dist/*.whl` 与 `dist/*.tar.gz`。
3. 用干净环境安装并验证：`uv pip install dist/*.whl`。
4. 测 CLI/import 是否可用。

### From source vs prebuilt

- From source：本机编译，慢且依赖编译工具链。
- Prebuilt：安装快，但必须匹配平台和架构。
- 轮子文件名会编码兼容信息（如 `py3-none-any` / `cp312-...-arm64`）。

## Versioning and releases

### Semantic Versioning (SemVer)

- `MAJOR.MINOR.PATCH`
- PATCH：向后兼容 bugfix
- MINOR：向后兼容新功能
- MAJOR：可能破坏兼容

### 版本约束（Python 示例）

- 精确：`requests==2.32.3`
- 下界：`click>=8.0`
- 区间：`numpy>=1.24,<2.0`
- 兼容发布：`pandas~=2.1.0`（约等于 `>=2.1.0,<2.2.0`）

### CalVer

- 另一种常见方式是日历版本（如 Ubuntu `24.04`）。
- 优点是直观看新旧，缺点是不表达兼容语义。

## Dependency management

### Manifest 与 Lockfile

- Manifest（`pyproject.toml`）：声明“允许范围”。
- Lockfile（如 `uv.lock`）：记录“本次解析出的精确版本”。

### 库 vs 应用的策略差异

- Library：建议用合理范围，减少与下游冲突。
- Application/Service：倾向锁定精确版本，保证可重现。

## Reproducibility

- 可重现不只看包版本，还取决于：运行时、系统库、OS、架构。
- `uv lock` 固定依赖图是基础。
- 更强级别是 hermetic build（如 Nix/Bazel），连编译器和系统输入都固定。
- 始终准备回滚方案：升级失败时快速回到已验证版本。

## Build and distribution pipeline

标准流水线：

1. 静态检查/测试通过。
2. 构建 artifact（wheel/binary/image）。
3. 在干净环境做 artifact 级验证。
4. 打 tag、更新 changelog。
5. 发布到目标渠道（registry/release）。

关键点：测试“你交付的东西”，而不是只测试源码目录。

## Python packaging (modern direction)

- 以 `pyproject.toml` 为中心。
- backend 负责如何构建，frontend 负责调用构建。
- 开发期可用 editable install；发布前必须验证正式 artifact 安装路径。

## Virtual environments and isolation

- 每项目一个 venv。
- 项目切换时切换 venv，避免全局包污染。
- 需要跨 Python 版本验证时，分别建环境测试。

## VMs & Containers

### 对比

- VM：隔离强，包含完整 OS，启动和资源开销更大。
- Container：共享宿主机内核，轻量快速，但隔离弱于 VM。

### Docker 实务要点

- Image 是模板，Container 是运行实例。
- Dockerfile 每条指令是一个 layer，影响构建缓存与体积。
- 常见优化：
  - 用 `*-slim` 基础镜像
  - 合并 `RUN`，并清理 apt 缓存
  - 尽量固定依赖版本
  - 避免把 secrets 烘焙进镜像层

## Configuration and secrets

- 同一套代码应通过配置切换 dev/staging/prod，不改业务代码。
- 配置来源：环境变量、配置文件。
- Secrets（API key、密码）不进仓库、不进镜像层。

## Services & Orchestration

- 现代应用通常是多服务组合（web、db、cache、queue）。
- 不建议把所有组件硬塞进一个容器。
- `docker compose` 用一个声明文件管理多容器网络、依赖和卷。
- 单机生产可配合 `systemd` 做开机自启与故障恢复。
- 规模上去后再考虑 Kubernetes，别过早引入复杂度。

## Publishing

### 分发渠道

- GitHub Releases：附带二进制/wheel。
- 语言官方 registry：如 Python 的 PyPI。
- 测试发布可先上 TestPyPI 验证流程。

### 常见发布命令（Python）

1. `uv build`
2. `uv publish --publish-url https://test.pypi.org/legacy/`（先测）
3. 确认安装可行后再发正式 PyPI。

## Publishing checklist

- 版本号和 tag 已更新
- changelog 已记录用户可见改动
- CI 全绿
- artifact 从干净提交构建
- 干净环境安装 + smoke test 已通过
- 回滚路径已准备

## Common failure modes

- 本地可运行，artifact 安装后缺文件
- 依赖约束过松，升级后运行时崩溃
- 应用未锁版本，环境漂移
- 容器镜像过大、层缓存差、含敏感信息
- 发布流程未在测试仓库演练

## Quick mental model

- Define artifact
- Pin assumptions
- Build deterministically
- Test artifact in clean env
- Publish immutably
- Keep rollback ready

## Reference

- <https://missing.csail.mit.edu/2026/shipping-code/>
- <https://semver.org/>
- <https://peps.python.org/pep-0517/>
- <https://peps.python.org/pep-0621/>
- <https://pypi.org/>
- <https://test.pypi.org/>
