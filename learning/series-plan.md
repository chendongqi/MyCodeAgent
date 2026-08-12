# Code Agent 解剖：系列规划

**副标题**：通过阅读一个真实 agent 项目，理解 LLM agent 的设计与工程实现

**定位**：解剖 MyCodeAgent 项目，面向有 LLM 基础但没完整实现过 agent 的开发者

---

## 系列结构

### Part 1：核心机制（从数据流到 ReAct 循环）

| 篇号 | 标题（问题视角） | 核心代码 |
|------|----------------|---------|
| 01 | 用户输入一句话，agent 内部发生了什么？ | `cli.py` → `bootstrap.py` → `loop.py` 全链路 |
| 02 | agent 是怎么一轮一轮"思考→动作→观察"的？ | `runtime/loop.py` + `runtime/state.py` + `runtime/completion.py` |
| 03 | 各家 LLM 格式不一样，agent 怎么统一对接？ | `core/llm.py` + `core/openai_compat.py` |
| 04 | 系统提示词是怎么组装的，agent 的"人格"从哪来？ | `runtime/prompt_builder.py` + `prompts/agents_prompts/` |

### Part 2：工具与能力扩展

| 篇号 | 标题（问题视角） | 核心代码 |
|------|----------------|---------|
| 05 | 模型怎么知道有哪些工具可以用？Function Calling 如何实现？ | `tools/base.py` + `tools/registry.py` + `tools/executor.py` |
| 06 | agent 的能力怎么用 Skills 动态扩展？ | `extensions/skills/` + `prompt_builder.py` skills 注入 |
| 07 | 外部工具怎么接进来？MCP 集成是怎么做的？ | `extensions/mcp/bootstrap.py` + `mcp_servers.json` |
| 08 | 一个任务太复杂，怎么拆给子 agent 做？ | `tools/builtin/task.py` + `runtime/subagents.py` |

### Part 3：上下文与持久化

| 篇号 | 标题（问题视角） | 核心代码 |
|------|----------------|---------|
| 09 | 对话越来越长，token 超了怎么办？ | `runtime/history.py` + `runtime/context/` |
| 10 | agent 崩了怎么恢复，对话历史存在哪？ | `runtime/transcript.py` + `runtime/session_memory.py` |

### Part 4：Harness Engineering

| 篇号 | 标题（问题视角） | 覆盖机制 |
|------|----------------|---------|
| 11 | agent 的主循环是怎么推进和终止的？ | 控制流：单一循环 / 不可变状态机 / 完成门 / 反馈注入 |
| 12 | token 超限怎么办？agent 怎么决定给模型看什么？ | 上下文工程：History/ModelView 分离 / 压缩 / Session Memory 前馈 |
| 13 | 工具执行管道是怎么设计的？安全边界怎么保障？ | 工具管道：协议 / 权限 / 沙箱 / 读写并发 / 熔断 |
| 14 | 出错了怎么办？agent 有哪些容错和恢复机制？ | 容错恢复：模型错误分类重试 / Transcript / Resume / Uncertain Actions |
| 15 | agent 在干什么，怎么知道？行为如何观测？ | 可观测性：事件驱动 / 双 sink / JSONL Trace / TransitionReason |

---

## 项目信息

- **源码**：https://github.com/chendongqi/MyCodeAgent
- **说明**：源码中已按系列讲解顺序在关键位置加入配套注释，读文章时可对照代码，也可直接克隆跑起来、基于它开发自己的 agent

---

## 文章规范

- **风格**：直接，无废话，代码结合讲解
- **结构**：提出问题 → 先给结论 → 逐段读代码 → 设计亮点 → 小结
- **代码量**：每篇不超过 4 段核心代码，每段不超过 30 行，其余用流程图或伪代码
- **长度**：每篇 2000-3000 字（Tutorial 级）

---

## 输出目录

文章写到 `learning/blog/` 目录，命名规则：`code-agent-series-XX-slug.md`
