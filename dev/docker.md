# Docker

just check the References

## 概念速记

- Docker：轻量级虚拟化工具，把应用及其运行环境打包为镜像，在容器中运行，隔离但共享宿主机内核。
- 镜像（Image）：只读分层文件系统，包含运行所需程序/配置，不含动态数据。
- 容器（Container）：镜像的运行实例，本质是隔离进程，容器存储层随容器生命周期消亡。
- 仓库（Registry/Repository/Tag）：仓库存多标签镜像；`ubuntu:24.04` 为仓库+标签。

## 基本命令

- `docker run <options> <image> <command>`：创建并运行容器；常用 `-d/-e/--rm/--name/-p/-it`。
- `docker build -f <Dockerfile> -t <name:tag> <context>`：构建镜像。
- `docker images`：列出本地镜像。
- `docker ps [-a]`：列出运行中/全部容器。
- 其他常用：`exec/attach/start/stop/rm/rmi/cp`。

## 单镜像基本流程

- 创建容器并进入终端：`docker run --name demo -it <image> /bin/bash`
- 从终端退出容器：`exit` 或 `Ctrl+D`
- 再次进入已有容器：`docker start -ai demo`
- 删除容器：`docker rm demo`

## 退出与保持运行

- `docker attach`：`Ctrl+p` 然后 `Ctrl+q` 仅脱离，不停止容器。
- `docker exec -it <name> /bin/bash`：退出 shell 不影响容器主进程。
- `docker run -d <image>` 后再 `docker exec -it` 进入，避免 `exit` 停止容器。

## Dockerfile 常用指令

- `FROM` 指定基础镜像；`RUN` 运行命令（可 exec 形式）。
- `WORKDIR` 设置工作目录；区别于 `RUN cd` 仅作用于当前层。
- `COPY` 复制构建上下文内文件；支持 JSON 数组写法。
- `ENV` 设置环境变量；`EXPOSE` 仅声明端口；`CMD` 为容器启动命令，可被 `docker run` 覆盖。

## 镜像构建要点

- Dockerfile 每条指令都会新建一层，层数过多会导致镜像臃肿；可用 `\` 合并命令减少层数。
- `docker build` 的最后参数是构建上下文目录，只有上下文内文件可被 `COPY`。
- 多阶段构建：用多个 `FROM` 分阶段构建，只在最终阶段保留产物；可用 `COPY --from` 取前一阶段文件。

## 数据管理

- 容器易失，持久化可用数据卷或绑定宿主机目录。
- 数据卷：可共享、即时生效、不影响镜像，容器删除后仍保留；`docker volume create/ls/inspect`。
- `docker run --mount source=<vol>,target=<path>` 挂载数据卷。
- 绑定主机目录：`--mount type=bind,source=<abs>,target=<path>[,readonly]`，必须绝对路径。

## Docker Compose

- Compose 用 YAML 编排多容器项目，`docker compose up/down/logs`。
- 配置与 `docker run/build` 选项对应，支持 `services/volumes`、端口、环境变量、重启策略等。

## References

- [Tsinghua](https://summer25.net9.org/backend/docker/note/)
