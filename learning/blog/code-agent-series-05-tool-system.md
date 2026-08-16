---
title: "Code Agent 解剖（05）：模型怎么知道有哪些工具可以用？Function Calling 如何实现？"
date: 2026-08-16
description: "跟着一次 tool_call 从头走到尾：工具如何注册进 Registry、Registry 如何生成 function schema 告知模型、模型触发 tool_calls 后 Orchestrator 如何分批执行、ToolExecutor 的权限/乐观锁/熔断管道、ToolResult 协议为什么要内外分离、以及观测结果怎么截断写回 history。"
tags: [Code Agent, Function Calling, Tool System, OpenAI Tools, LLM Agent]
category: AI Engineering
draft: false
---

## 从一次 tool_call 开始

假设模型决定读一个文件，它的响应里会出现：

```json
{
  "tool_calls": [
    { "id": "call_abc", "name": "Read", "arguments": "{\"path\": \"tools/base.py\"}" }
  ]
}
```

两个问题：

1. **模型怎么知道 Read 工具存在、它接受什么参数？**
2. **harness 收到这条响应后，怎么把工具真正跑起来、把结果喂回模型？**

要回答这两个问题，得从工具的起点讲起——注册。

---

## 第一步：工具注册——Python 类进入 Registry

所有工具在 agent 启动时注册进 `ToolRegistry`。Registry 内部维护两张表，对应两种注册方式：

```python
# 方式 1：Tool 对象（推荐）——继承 Tool 基类，有完整的结构化参数定义
registry.register_tool(ReadFileTool(name="Read", description="...", project_root=...))
# 存入 self._tools: dict[str, Tool]

# 方式 2：函数注册（快速）——只需要 name、description 和一个可调用函数
registry.register_function("my_func", description="...", func=my_func)
# 存入 self._functions: dict[str, dict]
```

两种方式的核心区别在于**参数定义**：Tool 对象通过 `get_parameters()` 返回结构化的参数列表（每个参数有名称、类型、描述、是否必填）；函数注册没有参数定义，后续生成 schema 时只能固定为单个 `input` 字符串。

注册完成后，Registry 就是工具系统的"总目录"。接下来发每一次 LLM 请求时，都要从这里拿 schema。

---

## 第二步：API 请求发出前——Registry 生成 schema 告知模型

模型不会凭空知道工具。每次进入 ReAct 的一个 step，`runtime/loop.py` 先从 Registry 取出 tools schema，再随请求一起发给模型：

```python
# runtime/loop.py — _prepare_step_context()（loop.py:1007）
tools_schema = host._get_openai_tools_for_current_mode()
# 等价于 self.tool_registry.get_openai_tools()

# runtime/loop.py — _react_loop()（loop.py:374）
raw_response = host.llm.invoke_raw(messages, tools=tools_schema, tool_choice=tool_choice)
```

`get_openai_tools()` 把两张注册表都转成 OpenAI function schema，所以有两个 for 循环，生成的 schema 结构也不同：

```python
# tools/registry.py — get_openai_tools()
def get_openai_tools(self) -> list[dict]:
    tools = []

    # 循环 1：处理 Tool 对象（self._tools）
    # tool.get_parameters() 返回结构化参数列表，_parameters_to_schema 转成完整 JSON Schema
    for tool in sorted(self._tools.values(), key=lambda t: t.name):
        if not self._circuit_breaker.is_available(tool.name):
            continue  # 熔断的工具不暴露给模型（第三步会讲）
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": self._parameters_to_schema(tool.get_parameters()),
                # 结果示例：{"path": string(required), "offset": integer, "limit": integer}
            },
        })

    # 循环 2：处理函数（self._functions）
    # 函数注册时没有参数定义，schema 固定写死为单个 input 字符串
    for name, info in sorted(self._functions.items(), key=lambda item: item[0]):
        if not self._circuit_breaker.is_available(name):
            continue
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": info.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string", "description": "raw input string"}},
                    "required": ["input"],
                    "additionalProperties": False,
                },
                # 无论函数实际签名是什么，模型看到的始终只有 input 一个参数
            },
        })

    return tools
```

两种注册在模型视角的差异：

| | Tool 对象注册 | 函数注册 |
|---|---|---|
| 模型看到的参数 | 完整（name/type/description/required） | 固定只有 `input: string` |
| 适合场景 | 需要多个参数、有校验逻辑 | 快速接入、单输入 |

这一步的结果：模型收到完整的 function schema 列表，知道有哪些工具、每个工具接受什么参数，于是可以决定调哪个、填什么值。

---

## 第三步：模型返回 tool_calls——Orchestrator 接管

模型决定调工具，响应里带 `tool_calls`。一步 ReAct 循环里，模型可能同时请求多个工具（比如并行读三个文件）。

接管入口是 `ToolOrchestrator.run()`，它要解决两件事：**哪些能并发跑、输出多大算超标**。

### 解析参数：plan_tool_calls

第一件事是把原始 `tool_calls` 列表解析成可执行的计划：

```python
# tools/orchestrator.py — plan_tool_calls()
def plan_tool_calls(self, tool_calls: list[dict]) -> list[ToolCallPlan]:
    plans = []
    for call in tool_calls:
        tool_name = call.get("name") or "unknown_tool"
        tool_call_id = call.get("id") or f"call_{uuid.uuid4().hex}"
        raw_args = call.get("arguments") or {}
        # arguments 可能是 JSON 字符串，也可能已经是 dict，parse_tool_input 都能处理
        parsed_input, parse_error = parse_tool_input(raw_args)
        plans.append(ToolCallPlan(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            parsed_input=parsed_input if isinstance(parsed_input, dict) else {},
            parse_error=parse_error,          # 解析失败时不为 None，后续直接短路
            concurrency_safe=self.is_concurrency_safe(tool_name, parse_error),
        ))
    return plans
```

### 分批：partition_tool_calls

解析完后，按并发安全性把计划分成批次：

```python
SAFE_TOOL_NAMES   = {"Read", "Grep", "Glob"}       # 无副作用，可并发
UNSAFE_TOOL_NAMES = {"Edit", "Bash", "Task", ...}  # 有副作用，串行

def partition_tool_calls(self, plans) -> list[ToolBatch]:
    batches = []
    for plan in plans:
        # 当前 plan 安全 且 末尾批次也安全 → 合并进同一并发批
        if batches and plan.concurrency_safe and batches[-1].concurrency_safe:
            batches[-1].calls.append(plan)
            continue
        # 否则新建批次
        batches.append(ToolBatch(concurrency_safe=plan.concurrency_safe, calls=[plan]))
    return batches
```

举例：模型同时请求 `[Read, Read, Edit, Grep]`，分批结果是：

```
[并发批(Read, Read)]  →  [串行批(Edit)]  →  [并发批(Grep)]
```

并发批用 `ThreadPoolExecutor` 并行跑，串行批逐个顺序跑。无论执行顺序怎样，**写回 history 的顺序永远与模型请求顺序一致**（并发结果按 offset 重排）。

---

## 第四步：每个工具的执行管道——ToolExecutor

每个计划最终交给 `ToolExecutor.execute()` 执行。这里是工具真正落地的地方，依次经过四道关卡：

```python
# tools/executor.py — 执行管道
def execute(self, name: str, input_text: Any) -> ToolResult:
    parameters = self.registry.prepare_parameters(input_text)

    # 关卡 1：权限检查
    if (denied := self._decide_permission(name, parameters)):
        return denied

    # 关卡 2：乐观锁注入（仅 Edit）
    if name == "Edit":
        parameters = self.registry.inject_optimistic_lock_params(name, parameters)

    # 关卡 3：熔断检查
    if not self.registry.is_available(name):
        return self.registry.create_circuit_open_result(name, parameters)

    # 关卡 4：实际执行
    result = tool.run(parameters)

    # 执行后：更新熔断器；Read 结果缓存乐观锁元信息
    self.registry.record_execution_result(name, result)
    if name == "Read":
        self.registry.cache_read_result(result, parameters)

    return result
```

### 关卡 1：权限检查

`RiskClassifier` 按工具类型和命令内容给出 `ALLOW / DENY / ASK` 三种决策：

- **Read / Grep / Glob**：直接 ALLOW（只读，无副作用）
- **Edit**：检查 runtime_mode，只读子 agent 里 DENY
- **Bash**：正则匹配——`sudo`、`rm`、`git reset`、嵌套 shell 直接 DENY；`mv`、`pip install` 等需要 ASK；低风险读操作 ALLOW
- **未知工具**：默认 DENY（fail-closed 策略）

### 关卡 2：乐观锁注入（Edit 专用）

agent 通常的模式是先 Read 再 Edit。如果两次调用之间有人改了文件，静默覆盖会造成数据丢失。

框架的解法：Read 成功后把文件的 `mtime` 和 `size` 缓存起来；Edit 执行前自动注入为期望值：

```python
# Read 成功 → 缓存元信息（tools/registry.py）
meta = {"file_mtime_ms": stats["file_mtime_ms"], "file_size_bytes": stats["file_size_bytes"]}
self._read_cache[path_resolved] = meta

# Edit 前 → 从缓存自动注入（tools/registry.py）
def _inject_optimistic_lock_params(self, tool_name, parameters):
    if "expected_mtime_ms" in parameters:
        return parameters       # 模型已提供，不覆盖
    meta = self._read_cache.get(parameters.get("path"))
    if meta:
        parameters["expected_mtime_ms"] = meta["file_mtime_ms"]
        parameters["expected_size_bytes"] = meta["file_size_bytes"]
    return parameters
```

Edit 工具内部拿到期望值后与文件当前状态对比，冲突时返回 `CONFLICT` 错误，而不是静默覆盖。

### 关卡 3：熔断检查

工具连续失败 N 次（默认 3 次）后进入 OPEN 状态，在 recovery_timeout（默认 300 秒）内拒绝所有调用：

```
CLOSED（正常）→ 连续失败 N 次 → OPEN（禁用）
                                  ↓ 冷却期结束
                              HALF_OPEN（放行一次）
                                  ↓ 成功 → CLOSED / 失败 → OPEN（重置计时）
```

OPEN 状态下，工具同时从第二步的 `get_openai_tools()` 输出里消失——不只是拒绝执行，连 schema 都不再暴露给模型，下一步模型就不会再尝试调它。

---

## 第五步：tool.run() 返回什么——ToolResult 协议

通过了三道关卡，`tool.run(parameters)` 被调用。每个工具必须返回 `ToolResult`，不允许返回裸字符串：

```python
# tools/base.py
@dataclass(frozen=True)  # 不可变：管道流转中不会被意外修改
class ToolResult:
    status: ToolStatus       # success / partial / error
    text: str                # 给 LLM 阅读的摘要
    data: Dict[str, Any]     # 核心载荷（永不为 None）
    error_code: ...          # 仅 error 状态有意义
    stats: Dict[str, Any]    # time_ms 等指标
    context: Dict[str, Any]  # cwd、params_input 等上下文
```

`ToolResult` 是**内部对象**，在管道各环节之间以 Python 对象形式流转，不做序列化。只有在最后写入 history 时才转成 JSON 字符串：

```python
# tools/base.py — 仅在模型边界才序列化
def tool_result_payload(result: ToolResult) -> dict:
    payload = {"status": ..., "data": ..., "text": ..., "stats": ..., "context": ...}
    # error 字段仅在 status=error 时出现，避免模型对成功结果产生误判
    if result.status is ToolStatus.ERROR:
        payload["error"] = {"code": result.error_code.value, "message": ...}
    return payload
```

内外分离的原因：管道中间如果序列化成字符串，下一个环节用数据时得再 parse 一次 JSON，且字符串无法做类型检查。`ToolResult` 是结构化对象，管道里每个环节都能按字段名取值、按状态判断。

---

## 第六步：结果回到 Orchestrator——预算截断后写入 history

`_execute_one()` 拿到 `ToolResult` 后，Orchestrator 把它包成 `ToolObservation`，构造时自动完成序列化：

```python
@dataclass(frozen=True)
class ToolObservation:
    result: ToolResult      # 供管道内部使用
    raw_result: ToolResult  # 截断前的原始结果（调试用）
    observation: str        # 序列化后的 JSON，直接写 history

    def __post_init__(self):
        object.__setattr__(self, "observation", serialize_tool_result(self.result))
```

所有批次执行完后，还会做**两层预算控制**，防止工具输出撑爆 context window：

```
层 1（单工具）：result > 50KB → 强制截断，完整内容 spill 到磁盘，result 中附文件路径
层 2（总量）：  所有 result 之和 > 200KB → 从最大的未截断项开始截，直到总量达标
```

最终，`observation` 字符串按 `tool_call_id` 写入 history：

```json
{ "role": "tool", "tool_call_id": "call_abc", "content": "{...ToolResult JSON...}" }
```

模型下一步读到这条记录，继续推理。

---

## 全程回顾

```
① 启动时
   register_tool / register_function → self._tools / self._functions

② 每个 ReAct step 发请求前
   get_openai_tools() → tools schema → llm.invoke_raw(tools=...) → 模型知道有哪些工具

③ 模型返回 tool_calls
   ToolOrchestrator.run()
     plan_tool_calls()      解析 arguments，标注并发安全性
     partition_tool_calls() 分批（只读→并发，写→串行）

④ 逐批执行
   _execute_plan() → ToolExecutor.execute()
     权限检查（ALLOW/DENY/ASK）
     乐观锁注入（Edit：自动补 expected_mtime_ms）
     熔断检查（OPEN 状态直接拒绝）
     tool.run(parameters) → ToolResult

⑤ 结果回流
   ToolResult → ToolObservation（构造时序列化）
   两层预算截断（50KB 单工具 + 200KB 总量）
   → history tool 消息（role=tool, content=JSON）
```

注册决定了工具存在；schema 生成决定了模型能看到什么；模型触发 tool_calls；Orchestrator 分批；Executor 过关卡；ToolResult 内部流转，边界才序列化；最终写回 history 完成一次 ReAct 的动作-观察环。

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
