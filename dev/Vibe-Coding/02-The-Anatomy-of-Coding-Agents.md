# The Anatomy of Coding Agents

[TOC]

## Main Takeaway

- Agent architecture and components
- Tool use and function calling
- MCP (Model Context Protocol)

如果把一个 coding agent 比作一个实习程序员，它至少要有这几个部分：

1. **大脑（LLM）**
    负责理解任务、生成计划、决定下一步做什么。
2. **眼睛和耳朵（Context）**
    负责读取代码、文档、终端输出、错误日志、用户需求。
3. **手（Tools）**
    负责真正执行动作，比如读文件、改文件、跑测试、调用 API、查询数据库。
4. **短期记忆（Working Memory / State）**
    负责记录当前任务进展，例如已经看过哪些文件、刚才执行了什么命令、结果是什么。
5. **规则与约束（Policies / Instructions）**
    负责限制 agent 的行为，比如“不要乱删文件”“先跑测试再提交”“输出必须是 JSON”。
6. **反馈回路（Observation Loop）**
    执行动作后要看结果，再决定下一步，而不是一次性瞎生成到底。