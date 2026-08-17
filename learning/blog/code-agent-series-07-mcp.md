---
title: "Code Agent 解剖（07）：外部工具怎么接进来？MCP 集成是怎么做的？"
date: 2026-08-17
description: "深入 MyCodeAgent 的 MCP 集成：可选依赖、配置发现、stdio/HTTP 传输、Adapter 把远端工具伪装成本地 Tool、结果收成通用协议。理解外部能力如何进入同一条工具管道，以及「注册进目录」和「允许执行」为什么不是一回事。"
tags: [Code Agent, MCP, Tool Adapter, Function Calling, LLM Agent]
category: AI Engineering
draft: false
---

## Skills 解决不了的那一类能力

上一篇的 Skills 是本地 Markdown 指令：不写 Python、不启新进程，模型按需把正文读进来照着做。适合「团队 code review 规范」这种流程知识。

外部能力不是这个形状。网页搜索、文档库、浏览器、公司内部 API，跑在**另一个进程**里，有自己的参数和生命周期。为每个服务手写一个 `Tool` 子类能用，但接一家就改一次代码。

MCP（Model Context Protocol）的工程含义很具体：让外部进程按标准协议暴露工具，agent 在启动时发现它们，**伪装成已经注册过的本地 Tool**，之后走第 05 篇同一条管道——schema、编排、权限、观测写回。

---

## 结论先放这

MCP 的启动入口不在 `extensions/mcp/` 里，而在 **agent 组装上下文** 时：

```
CodeAgent._initialize_runtime_components()
        │
        ▼
build_runtime_context()              ← runtime/factory.py
        │
        ├─ _register_builtin_tools()   内置工具先进 Registry
        │
        ├─ if host.enable_mcp:         ← 默认 False，没开则整段跳过
        │       host._register_mcp_tools()
        │           │
        │           ▼
        │   register_mcp_servers()     ← extensions/mcp/bootstrap.py
        │           │
        │           ├─ load_mcp_servers()   读 mcp.json / MCP_SERVERS
        │           ├─ MCPClient（stdio / http）
        │           ├─ list_tools() 发现远端工具
        │           └─ MCPToolAdapter → ToolRegistry
        │
        └─ ContextBuilder(...)         ← 把 _mcp_tools_prompt 写进 Tool Contracts
                │
                ▼
        之后每步 ReAct 和 Read/Bash 一样：
        tools= schema / Orchestrator → Executor → adapter.run()
                │
                ▼
        protocol.py 把 MCP content 收成通用信封
```

**为什么从 factory 开始看？** 第 01 篇讲过 `build_runtime_context()` 是启动期组装上下文引擎的地方。Skills 和 MCP 都挂在这里，但顺序固定：**先内置工具，再 MCP，最后 ContextBuilder**。MCP 工具必须在 `ContextBuilder` 创建之前注册完，这样 `tool_prompt_allowlist=frozenset(host.tool_registry.list_tools())` 才能包含外部工具，`mcp_tools_prompt` 也能一并传入。

和 Skills 的分界：

| | Skills | MCP |
|--|--------|-----|
| 载体 | `SKILL.md` | 外部进程 / HTTP 服务 |
| 进 Registry 的是什么 | 一个本地 `Skill` 工具 | **每个**远端工具各一个 Adapter |
| 默认开不开 | 有文件就加载 | 关；要 `--enable-mcp` |
| 依赖 | 无 | 可选 extra：`mcp` SDK |

---

## 第一步：从 factory 的 `if host.enable_mcp` 开始

读 MCP，建议先打开 `runtime/factory.py` 的 `build_runtime_context()`。Skills 和内置工具处理完之后，才是 MCP 分支：

```python
# runtime/factory.py — build_runtime_context()
host._register_builtin_tools()   # ① 内置工具先进 Registry

host._mcp_clients = []
host._mcp_tools_prompt = ""
if host.enable_mcp:              # ② 默认 False，没开则整段跳过
    host._register_mcp_tools()

host.context_builder = ContextBuilder(
    tool_registry=host.tool_registry,
    mcp_tools_prompt=host._mcp_tools_prompt,   # ③ MCP 说明书进 Tool Contracts
    tool_prompt_allowlist=frozenset(host.tool_registry.list_tools()) | {"Task"},
    ...
)
```

这三行就是 MCP 启动的**总开关**：

1. **`enable_mcp` 为 False** → 下面什么都不发生，`_mcp_clients` 保持空列表，Registry 里只有内置工具
2. **为 True** → 调 `host._register_mcp_tools()`，连外部 server、发现工具、注册 Adapter
3. **无论开不开**，接着都会建 `ContextBuilder`；开了 MCP 时，`_mcp_tools_prompt` 已经填好，会拼进 system 的 Tool Contracts

`enable_mcp` 从哪来？默认 `Config.enable_mcp = False`；CLI `--enable-mcp` 或环境变量 `ENABLE_MCP=true` 打开。这不是嫌麻烦——stdio 会拉起子进程，SDK（`mcp`、`anyio`）也不打进核心安装。

---

## 第二步：`_register_mcp_tools()` 委托给 bootstrap

factory 只做判断和调用，具体逻辑在 `host._register_mcp_tools()`：

```python
# runtime/host.py — _register_mcp_tools()（简化）
clients, tools_meta = register_mcp_servers(self.tool_registry, self.project_root)
self._mcp_clients = clients
self._mcp_tools_prompt = format_mcp_tools_prompt(tools_meta)
```

- `register_mcp_servers()` 负责连 server、发现工具、往 **已有** `tool_registry` 里塞 Adapter
- 返回的 `clients` 存到 `self._mcp_clients`，`CodeAgent.close()` 时逐个 `close_sync()`，避免子进程泄漏
- `tools_meta` 格式化成自然语言，供 ContextBuilder 追加 `## MCP Tools`

没装 MCP extra 却开了开关，这里会抛 `MCPExtraRequiredError`，提示 `pip install 'mycodeagent[mcp]'`。懒加载在 bootstrap 里——只有走进 `_register_mcp_tools()` 才 import SDK：

```python
# extensions/mcp/bootstrap.py — 只有真正注册时才碰 SDK
def _load_mcp_runtime():
    try:
        from extensions.mcp.adapter import register_mcp_tools
        from extensions.mcp.client import MCPClient, MCPClientConfig
    except ImportError as exc:
        raise MCPExtraRequiredError(...) from exc
    return MCPClient, MCPClientConfig, register_mcp_tools
```

核心安装测过：没开 `enable_mcp`，factory 不会走到这里，主循环照样能跑。

---

## 第三步：配置从哪来

`load_mcp_servers()` 按优先级读：

1. 环境变量 `MCP_SERVERS`（一段 JSON）
2. 项目根下 `mcp_servers.json` / `.mcp.json` / `mcp.json`

兼容 Claude 常见的包一层 `mcpServers`：

```json
{
  "mcpServers": {
    "docs": { "command": "uvx", "args": ["mcp-server-fetch"] },
    "search": { "url": "https://example.com/mcp" }
  }
}
```

没有 `url` 就当 stdio：用 `command` + `args` 拉起进程。有 `url`（或 `transport=http`）走 HTTP。`uvx`/`uv` 会额外把缓存目录钉到项目下的 `.uv_cache`，避免污染用户全局环境。

`MCP_CONNECT_MODE` 默认 `startup`：启动时连接并 `list_tools`。设成 `disabled` 等于配置在、但不连。

---

## 第四步：发现工具，塞进 Registry

`register_mcp_servers()` 对每个 server 建一个 `MCPClient`，再 `list_tools_sync()`。每个远端工具变成一个 `MCPToolAdapter`：

```python
# extensions/mcp/adapter.py — 发现 + 命名（简化）
raw_public_name = f"{namespace}:{remote_name}"   # docs:search
public_name = sanitize_tool_name(raw_public_name)  # docs_search
public_name = ensure_unique(public_name)           # 与内置名冲突则 _2
adapter = MCPToolAdapter(client, public_name, remote_name, description, schema)
tool_registry.register_tool(adapter)
```

三个细节：

1. **命名空间**：公开名带 server 前缀，避免两家都叫 `search`
2. **清洗**：Function Calling 的名字只允许 `[a-zA-Z0-9_-]`，冒号会变成下划线
3. **schema 投影**：远端 `inputSchema` 转成本地 `ToolParameter` 列表，于是 `get_openai_tools()` 自动带上这些工具——模型看见它们，和看见 `Read` 的方式相同

Adapter 的 `run()` 不再读本地文件，而是 `mcp_client.call_tool_sync(remote_name, parameters)`。对 Orchestrator / Executor 来说，这就是又一个 `Tool`。

**`list_tools` 的底层**：`session.list_tools()` 发出一条 JSON-RPC 请求：

```json
{"method": "tools/list", "params": {}}
```

server 返回它暴露的所有工具，每个带 `name`、`description`、`inputSchema`。`register_mcp_tools` 把这个列表遍历一遍，每个工具变成一个 `MCPToolAdapter` 进 Registry。

**`call_tool` 的底层**：模型触发工具调用后，Adapter 的 `run()` 发出：

```json
{"method": "tools/call", "params": {"name": "search", "arguments": {"query": "..."}}}
```

server 执行后返回 MCP `content` 块，`protocol.py` 把它投影成本项目的通用信封（第六步详述）。

---

## 第五步：异步 SDK，同步工具管道

MCP 官方 SDK 全是 `async`，但 `Tool.run()` 是同步的。这是核心矛盾：主循环是同步代码，工具执行不能突然变成 `await`。

`MCPClient` 自己持有一个私有 event loop，用 `_run_sync` 把协程堵成同步调用：

```python
# extensions/mcp/client.py
def _run_sync(self, coro):
    try:
        asyncio.get_running_loop()
        # 已在别的 loop 里，无法再 run_until_complete，直接报错
        # 避免「在 running loop 里 run_until_complete」这种死锁
        raise RuntimeError("cannot run inside an active event loop")
    except RuntimeError:
        pass  # 不在 loop 里，安全

    if self._loop is None or self._loop.is_closed():
        self._loop = asyncio.new_event_loop()
    return self._loop.run_until_complete(coro)

# 对外暴露同步版本，Tool.run() 调这些
def list_tools_sync(self):   return self._run_sync(self.list_tools())
def call_tool_sync(self, name, arguments): return self._run_sync(self.call_tool(name, arguments))
```

**连接建立**：两种 transport 的连接方式不同，但建完后对上层完全一样：

```
stdio（本地子进程）：
    StdioServerParameters(command="uvx", args=[...])
        → 拉起子进程，建立 stdin/stdout 管道
        → ClientSession(read, write)
        → session.initialize()   ← MCP 握手，协商协议版本

http（远程服务）：
    streamablehttp_client(url)
        → 建立 HTTP 连接
        → ClientSession(read, write)
        → session.initialize()
```

连接是懒的：第一次 `list_tools` 或 `call_tool` 才真正 `connect()`。会话被对端关掉（`ClosedResourceError`）时，先 `close()` 再重连一次。`CodeAgent.close()` 会把所有 `_mcp_clients` 逐个 `close_sync()`，避免 stdio 子进程泄漏。

---

## 第六步：MCP 结果收成通用信封

远端返回的是 MCP 的 `content` 块（text / resource / 二进制），不是本项目的 `{status, data, text, ...}`。`protocol.py` 做投影：

- 抽出 text 块拼成 `text`
- `structuredContent` 放进 `data.structured`
- 非文本变成 `[binary content ...]` / `[resource uri]` 摘要
- `isError=true` 走 error 信封

错误按原因分类，方便模型（和熔断器）区分「参数写错了」还是「对面挂了」：

| 情况 | error.code |
|------|------------|
| schema 校验失败 | `MCP_PARAM_ERROR` |
| 解析/包装失败 | `MCP_PARSE_ERROR` |
| 超时 / 连接失败 | `MCP_TIMEOUT` / `MCP_NETWORK_ERROR` |
| 远端执行失败 | `MCP_EXECUTION_ERROR` |

这和第 05 篇的协议是同一套顶层字段。模型不需要知道工具是本地 Python 还是 MCP 进程。

---

## 第七步：模型怎么「看见」它们

两条通道，和第 04 / 05 篇一致：

1. **`tools=` schema**：Adapter 已在 Registry 里，`get_openai_tools()` 自然带上
2. **Tool Contracts 文本**：`format_mcp_tools_prompt()` 生成 `- name: desc` + `params: ...`，`ContextBuilder` 追加 `## MCP Tools`

单 server 注册失败只打 warning、跳过，不把整个 agent 启动打挂。这是外部进程的正确姿态：对面随时可能没装好。

---

## 一个需要注意的 MVP 限制

权限分类器的白名单是硬编码的：

```python
# tools/permissions.py — RiskClassifier.classify()
READ_ONLY_TOOLS = {"Read", "Grep", "Glob"}   # → ALLOW
WRITE_TOOLS     = {"Edit"}                   # → ALLOW（有路径才行）
"TodoWrite"                                  # → ALLOW
"Bash"                                       # → 黑/灰/白名单逐条匹配
"Skill"                                      # → ALLOW
# 其他所有名字 → DENY（fail-closed）
```

MCP 工具名是动态的（`docs_search`、`server_name:tool_name`），不在这个白名单里，走到最后一条：

```python
return PermissionDecision(
    action=PermissionAction.DENY,
    risk=RiskLevel.UNKNOWN,
    reason=f"unknown tool '{tool_name}' fails closed",
)
```

`Executor` 拿到 DENY 直接短路返回 `PERMISSION_DENIED` 错误，工具不会实际执行。

这是当前实现的**已知局限**，不是最终设计。权限系统还没扩展到动态工具名。要让 MCP 工具真正可用，需要在分类器里加"已注册的 MCP 工具 → ALLOW"分支，或改成可配置白名单。

这里有一个重要的设计原则值得保留：**进 Registry ≠ 允许执行**。Registry 控制"模型能在 schema 里看到什么"，权限门控制"什么能真正落地执行"——两者分离，即使 Registry 扩大，执行权仍可以独立收紧。MCP 工具一旦加入白名单，编排侧同样保守：`ToolOrchestrator` 只把 `Read`/`Grep`/`Glob` 当并发安全，MCP 工具走串行批次。

---

## 设计亮点

1. **可选依赖**：核心安装零 MCP；开关 + extra 同时满足才加载 SDK
2. **Adapter 而不是平行管道**：外部工具复用 Registry / schema / 编排，不在 loop 里写 `if mcp`
3. **发现失败可降级**：单个 server 挂了，其余工具和主循环继续
4. **注册与授权分离**：目录可以变大，执行权仍 fail-closed
5. **结果投影**：MCP content 块不泄漏进模型上下文的「另一种 JSON」

---

## 小结

| 机制 | 作用 |
|------|------|
| `factory.py` `if host.enable_mcp` | **启动入口**：内置工具注册后才走 MCP |
| `_register_mcp_tools()` | 委托 bootstrap，保存 clients / tools_prompt |
| `--enable-mcp` / `ENABLE_MCP` | 默认关闭，避免无谓拉起子进程 |
| `mcp.json` / `MCP_SERVERS` | 声明 stdio 或 HTTP server |
| `MCPClient` | 懒连接、同步封装、断线重连 |
| `MCPToolAdapter` | 远端工具 → 本地 `Tool` |
| `protocol.py` | MCP 结果 → 通用工具信封 |
| 权限 fail-closed | 未知 MCP 工具名默认拒绝执行 |

下一篇进入子 agent：一个任务太复杂时，`Task` 工具如何把工作拆给另一个轻量循环。

---

## 关于本系列的源码

本系列所有分析均基于开源项目 [MyCodeAgent](https://github.com/chendongqi/MyCodeAgent)。

源码里已经按照本系列文章的讲解顺序，在关键位置加入了配套注释——读文章时可以对照代码，也可以直接克隆下来自己跑、改、扩展，基于它开发你自己的 agent。

```bash
git clone https://github.com/chendongqi/MyCodeAgent
cd MyCodeAgent
cp .env.example .env   # 填入你的 LLM API key
uv sync
uv run python main.py
```

---

*欢迎访问 [PrimeSkills](https://primeskills.store) —— 一个精心策划的 AI Agent 与技能市场，所有内容均经过真实企业级工作流验证。没有噱头，只有真正有效的东西。*

*更多实用知识和有趣产品，欢迎访问我的[个人主页](https://home.wonlab.top)*
