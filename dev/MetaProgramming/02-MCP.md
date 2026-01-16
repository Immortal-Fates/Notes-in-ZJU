# MCP

## Intro
MCP（Model Context Protocol）是用于 AI 应用对接外部能力的协议标准，重点解决“能力如何被描述、发现、调用”。它让不同工具与数据源以一致方式接入，降低集成成本。

> [!TIP]
>
> Just like a USB-C port for AI applications

check the [introduction](https://modelcontextprotocol.io/docs/getting-started/intro)

### Concepts of MCP

MCP follows a client-server architecture where an MCP host — an AI application like [Claude Code](https://www.anthropic.com/claude-code) or [Claude Desktop](https://www.claude.ai/download) — establishes connections to one or more MCP servers.

- MCP Host（AI 应用/IDE/Agent）：协调和管理一个或多个 MCP 客户端

- MCP Client：连接/发现 MCP server

- MCP Server：为 MCP 客户端提供上下文的程序

  Servers provide functionality through three building blocks:

  | Feature       | Explanation                                                  | Examples                                                 | Who controls it |
  | :------------ | :----------------------------------------------------------- | :------------------------------------------------------- | :-------------- |
  | **Tools**     | Functions that your LLM can actively call, and decides when to use them based on user requests. Tools can write to databases, call external APIs, modify files, or trigger other logic. | Search flights Send messages Create calendar events      | Model           |
  | **Resources** | Passive data sources that provide read-only access to information for context, such as file contents, database schemas, or API documentation. | Retrieve documents Access knowledge bases Read calendars | Application     |
  | **Prompts**   | Pre-built instruction templates that tell the model to work with specific tools and resources. | Plan a vacation Summarize my meetings Draft an email     | User            |

MCP server 常见实现与启动方式：

- Python: `uvx` (e.g. `uv tool run ...`)
- Node: `npx`

### Layers

MCP consists of two layers:

- **Data layer**: Defines the JSON-RPC based protocol for client-server communication, including lifecycle management, and core primitives, such as tools, resources, prompts and notifications.

- **Transport layer**: Defines the communication mechanisms and channels that enable data exchange between clients and servers, including transport-specific connection establishment, message framing, and authorization.

  MCP supports two transport mechanisms:

  - Stdio transport：适合本机集成：Host 启动 Server 子进程，通过 stdin/stdout 传 JSON-RPC 消息；日志可写到 stderr。连接是 1:1，本地最稳、开销小

  - Streamable HTTP transport: Uses HTTP POST for client-to-server messages with optional Server-Sent Events for streaming capabilities

    SSE：基于 HTTP 的服务端推送，适合远程/多机访问，易于穿过常见网络环境

Conceptually the data layer is the inner layer, while the transport layer is the outer layer.

### 典型流程
1. Host 通过 Client 连接一个或多个 Server
2. Server 暴露可用的 tools/resources/prompts
3. Host 选择并调用 tools 或读取 resources
4. 结果回到 Host，用于后续推理或执行

### 与 Function Calling 的关系
- Function Calling 是“模型内置的调用机制”
- MCP 是“跨应用/跨进程的标准协议”，可接入多工具与数据源

## 常用 MCP Server

- Agent-tracker: 跟踪我的codex tasks

- PlayWrite: 浏览器自动化 [github](https://github.com/microsoft/playwright-mcp?utm_source=openai)

- Context7：文档/知识检索（查最新的）[github](https://github.com/upstash/context7?utm_source=openai)

- OpenSpec：规范写代码（专用于$1\to n$）[github](https://github.com/Fission-AI/OpenSpec) [intro](https://www.aivi.fyi/llms/introduce-OpenSpec)

  ```
  ┌────────────────────┐
  │ Draft Change       │
  │ Proposal           │
  └────────┬───────────┘
           │ share intent with your AI
           ▼
  ┌────────────────────┐
  │ Review & Align     │
  │ (edit specs/tasks) │◀──── feedback loop ──────┐
  └────────┬───────────┘                          │
           │ approved plan                        │
           ▼                                      │
  ┌────────────────────┐                          │
  │ Implement Tasks    │──────────────────────────┘
  │ (AI writes code)   │
  └────────┬───────────┘
           │ ship the change
           ▼
  ┌────────────────────┐
  │ Archive & Update   │
  │ Specs (source)     │
  └────────────────────┘
  
  1. Draft a change proposal that captures the spec updates you want.
  2. Review the proposal with your AI assistant until everyone agrees.
  3. Implement tasks that reference the agreed specs.
  4. Archive the change to merge the approved updates back into the source-of-truth specs.
  ```

  下面给出使用OpenSpec的pipeline

  1. Preparation stage

     1. `openspec init` 初始化OpenSpec，选择你要用的LLM

     2. 填充项目信息（遵循openspec init的输出）

        ```
        Populate your project context:
             "Please read openspec/project.md and help me fill it out
              with details about my project, tech stack, and conventions"
        ```

  2. Draft a change proposal

     ```
     /openspec:proposal [the feature you want to do] add-custom-focus-duration // 这里我们用这个feature进行说明
     ```

     在你回答完问题后，会创建`proposal/design/tasks.md/ and specs/`

  3. 审查和验证

     ```
     # 查看活跃的变更
     openspec list
     
     # 验证提案格式
     openspec validate add-custom-focus-duration
     
     # 查看提案详情
     openspec show add-custom-focus-duration
     ```

  4. 执行

     ```
     /openspec:apply add-custom-focus-duration
     ```

     测试+找bug直到满意

  5. 归档变更

     ```
     /openspec:archive add-custom-focus-duration
     ```

- Next AI Draw.io [github](https://github.com/DayuanJiang/next-ai-draw-io/)，用于画流程图，除了mcp server还能用docker本地部署

not use:

- Everything：参考/测试服务器，包含 tools/resources/prompts
- Fetch：抓取网页内容并转为更适合 LLM 的格式
- Filesystem：受控的本地文件读写与目录访问
- Git：读取/搜索/操作本地 Git 仓库
- Memory：持久化知识图谱式记忆
- Sequential Thinking：分步骤推理与思考序列
- Time：时间与时区转换
- Brave Search：通过 Brave Search API 做网页/本地/图片/视频/新闻搜索





## References

- MCP intro: https://modelcontextprotocol.io/docs/getting-started/intro
- Model Context Protocol Spec (2024-11-05): https://modelcontextprotocol.io/specification/2024-11-05/basic/transports
- Model Context Protocol Spec (2025-11-25): https://modelcontextprotocol.io/specification/2025-11-25/basic/transports
- MCP Changelog (2025-03-26): https://modelcontextprotocol.io/specification/2025-03-26/changelog
