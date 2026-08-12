# 模块 0：鸟瞰图——整体架构与数据流

> **一句话定义**：Agent harness = LLM + 工具 + 循环，其余都是工程保障。

---

## 为什么从鸟瞰图开始？

你用 LangChain 时调用的是高层 API，底层的 LLM 调用、工具执行、历史管理都被框架隐藏了。MyCodeAgent 把这些全部暴露出来——没有框架魔法，每一个零件都清晰可见。

在深入每个零件之前，先在脑子里建一张地图，后续学习不会迷路。

---

## 整体数据流

```
用户输入消息（字符串）
        │
        ▼
┌─────────────────────────────────────────┐
│  app/cli.py                             │
│  解析命令行参数，决定运行模式             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  runtime/factory.py                     │
│  根据 Config 创建 CodeAgent 实例         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│  runtime/host.py  CodeAgent（依赖容器）                      │
│                                                             │
│  持有所有零件：                                              │
│    llm             ← core/llm.py（发请求、解析响应）          │
│    tool_registry   ← tools/registry.py（注册和查找工具）      │
│    history_manager ← runtime/history.py（完整消息存储）       │
│    context_engine  ← runtime/context/engine.py（投影视图）    │
│    transcript      ← runtime/transcript.py（持久化）         │
│    trace_logger    ← extensions/tracing/（可观测性）         │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│  runtime/loop.py  RuntimeRunner（主循环）                    │
│                                                             │
│  step 1: 把用户消息追加到 history                            │
│  step 2: context_engine → 生成 model_view（不是全量历史！）   │
│  step 3: llm.invoke_raw(model_view + tools_schema)          │
│                                                             │
│   ┌── 有 tool_calls？────────────────────┐                  │
│   │                                     │                  │
│   ▼                                     ▼                  │
│  执行工具（orchestrator）           completion_verifier      │
│  把结果追加到 history                判断是否真的完成了        │
│  回到 step 2（下一步）                                        │
│                                    PASS → 输出，终止         │
│                                    FAIL → 继续循环           │
│                                                             │
│  终止条件：TerminalReason                                    │
│    COMPLETED / MAX_STEPS / TOKEN_BUDGET / MODEL_ERROR       │
└─────────────────────────────────────────────────────────────┘
               │
               ▼
        输出最终结果（字符串）
```

---

## 项目目录结构速查

```
MyCodeAgent/
├── core/              LLM 接口 + 配置（模块 1 重点）
│   ├── llm.py         统一 LLM 客户端，响应归一化
│   ├── config.py      所有配置项的类型定义与默认值
│   └── openai_compat.py  实际 HTTP 请求层
│
├── tools/             工具系统（模块 2 重点）
│   ├── base.py        Tool 抽象类 + 响应协议
│   ├── registry.py    工具注册表
│   ├── executor.py    执行管道（权限 + 熔断）
│   ├── orchestrator.py 批量执行协调
│   └── builtin/       内置工具（Read/Edit/Glob/Grep/Bash/Task/TodoWrite）
│
├── runtime/           运行时核心（模块 3、4 重点）
│   ├── loop.py        ★ 主循环 RuntimeRunner（1000+ 行，最核心）
│   ├── state.py       LoopState 不可变状态机
│   ├── completion.py  完成门判断逻辑
│   ├── host.py        CodeAgent 依赖容器
│   ├── history.py     消息历史存储
│   ├── context/       上下文投影引擎
│   └── transcript.py  持久化恢复
│
├── app/               启动层（模块 5 重点）
│   ├── cli.py         命令行解析
│   └── bootstrap.py   内置工具注册
│
├── extensions/        可选扩展（默认不启用）
│   ├── mcp/           MCP 工具集成
│   ├── skills/        用户自定义技能
│   └── tracing/       JSONL trace 日志
│
└── prompts/           系统提示词 + 工具提示词
```

---

## 状态机设计：LoopState

这是理解整个循环的基础数据结构，位于 `runtime/state.py`。

```python
# runtime/state.py

class TransitionReason(str, Enum):
    """每次状态转移的原因——相当于状态机的"事件"标签"""
    USER_INPUT = "user_input"                    # 用户刚输入消息
    MODEL_RETURNED_TOOL_CALLS = "..."            # 模型要求调用工具
    TOOLS_EXECUTED = "..."                       # 工具执行完毕
    MODEL_RETURNED_FINAL = "..."                 # 模型返回最终回答
    CONTEXT_COMPACTED = "..."                    # 上下文被压缩
    MODEL_RECOVERY_RETRY = "..."                 # 模型出错，正在重试
    MAX_STEPS_EXCEEDED = "..."                   # 超过最大步数

class TerminalReason(str, Enum):
    """循环终止的原因——相当于状态机的"终态""""
    COMPLETED = "completed"                      # 正常完成
    COMPLETED_UNVERIFIED = "completed_unverified"# 完成但缺乏验证证据
    MAX_STEPS = "max_steps"                      # 超步强制终止
    TOKEN_BUDGET = "token_budget"                # token 预算耗尽
    MODEL_ERROR = "model_error"                  # 模型错误无法恢复
    USER_ABORT = "user_abort"                    # 用户中断

@dataclass(frozen=True)           # ← 关键：不可变！
class LoopState:
    messages: list[dict]          # 当前给模型看的消息（投影视图）
    step: int                     # 当前步数（从 1 开始）
    turn_count: int               # 第几轮对话
    tool_choice: str              # "auto" = 让模型决定是否用工具
    last_tool_calls: list         # 上一步用了哪些工具
    last_response_meta: dict      # 上次模型返回的元信息（finish_reason 等）
    model_recovery_counts: dict   # 各类错误的重试计数
    completion_block_count: int   # 完成门被阻挡的次数

    def next(self, reason: TransitionReason, **changes) -> "LoopState":
        """产生下一个状态——注意：返回新对象，不修改当前对象"""
        return replace(self, transition=Transition(reason), **changes)
```

### 设计亮点：为什么 LoopState 是不可变的？

`frozen=True` 意味着每次状态转移都必须通过 `.next()` 创建新对象，而不能原地修改。

好处：
1. **任意时刻的状态都可以快照**——调试时可以打印任何一步的完整状态
2. **trace 日志天然准确**——每次 `.next()` 都对应一条 trace 事件，不会遗漏
3. **错误隔离**——某步出错不会污染之前的状态，恢复更容易

---

## 主循环入口：run() 函数

`runtime/loop.py` 中 `RuntimeRunner.run()` 是整个 agent 的入口：

```python
# runtime/loop.py（节选，约 236-248 行）

def run(self, input_text: str, **kwargs) -> str:
    show_raw = kwargs.pop("show_raw", False)
    
    # 第一步：初始化本次运行（记录 run_id，处理输入，emit run_start 事件）
    processed_input, trace_logger, run_id = self._prepare_run(input_text, show_raw)
    
    response_text = ""
    try:
        # 第二步：进入 ReAct 主循环
        response_text = self._react_loop(
            pending_input=processed_input,
            show_raw=show_raw,
            trace_logger=trace_logger,
        )
    finally:
        # 第三步：无论成功还是异常，都记录运行结束
        self._finish_run(trace_logger, run_id, response_text)
    
    return response_text
```

注意 `try/finally` 结构——即使 `_react_loop` 抛异常，`_finish_run` 也一定会执行，确保 trace 记录完整。

---

## 主循环骨架：_react_loop()

```python
# runtime/loop.py（节选，约 321-347 行）

def _react_loop(self, pending_input: str, show_raw: bool, trace_logger) -> str:
    host = self.host
    
    # 初始化状态
    state = LoopState(messages=[], step=1, turn_count=1, tool_choice="auto")
    state = self._transition(state, TransitionReason.USER_INPUT, ...)
    
    # 主循环：最多跑 max_steps 步
    for step in range(1, host.max_steps + 1):
        
        # ① 构建给模型看的消息视图（不是全量历史）
        state, tools_schema, messages = self._prepare_step_context(...)
        
        # ② 调用 LLM
        raw_response = host.llm.invoke_raw(messages, tools=tools_schema, ...)
        
        # ③ 提取响应内容
        tool_calls = extract_tool_calls(raw_response)
        response_text = extract_response_content(raw_response)
        
        if tool_calls:
            # ④a 有工具调用：执行工具，结果追加到历史，继续循环
            ...
            state = self._transition(state, TransitionReason.MODEL_RETURNED_TOOL_CALLS, ...)
            # 执行工具...
            state = self._transition(state, TransitionReason.TOOLS_EXECUTED, ...)
            continue
        
        else:
            # ④b 无工具调用：检查是否真的完成了
            verdict = completion_verifier.evaluate(...)
            
            if verdict == CompletionGateVerdict.PASS:
                self._terminal(TerminalReason.COMPLETED, ...)
                return response_text           # ← 正常退出
            
            elif verdict == CompletionGateVerdict.FAIL:
                # 完成门拒绝：把"你还没完成"的提示追加进历史，继续循环
                ...
                continue
    
    # 超过 max_steps 强制终止
    self._terminal(TerminalReason.MAX_STEPS, ...)
    return response_text
```

---

## 三个关键边界（架构原则）

AGENT.md 里写得很明确，这是理解整个项目的纲领：

| 边界 | 意思 |
|------|------|
| `tools/` 不导入 `runtime/` | 工具只负责执行，不参与 loop 控制 |
| `runtime/context/` 只管投影 | 上下文策略在这里，不在 history 里 |
| `Edit` 是唯一写操作工具 | 所有文件变更集中在一个工具，便于权限和回滚 |

**最重要的一句话**（来自 AGENT.md）：

> "Keep MyCodeAgent a lean, local-first single-agent coding harness. The stable runtime has one loop, `runtime.loop.RuntimeRunner`; do not introduce another agent loop."

只有一个 loop，不能有第二个——这是整个架构的核心约束。

---

## 可观测性：每次循环都有 trace

harness 在每次状态转移时都会 emit 一个事件：

```python
# 事件类型举例（runtime/events.py）
"run_start"                    # 开始运行
"user_input"                   # 用户输入
"context_build"                # 构建了 model view（含消息数量）
"model_request"                # 即将调用 LLM
"model_returned_tool_calls"    # 模型要求调用工具
"tools_executed"               # 工具执行完毕
"model_returned_final"         # 模型返回最终答案
"terminal"                     # 循环终止（含 TerminalReason）
```

这些事件写入 `memory/traces/*.jsonl`，是调试和理解 agent 行为的最好工具。

---

## 小结

| 概念 | 对应文件 | 一句话 |
|------|---------|--------|
| 依赖容器 | `runtime/host.py` | 把所有零件 wire 在一起 |
| 主循环 | `runtime/loop.py` | ReAct 的工程实现 |
| 状态机 | `runtime/state.py` | 不可变状态，每步可追踪 |
| 完成门 | `runtime/completion.py` | 防止模型"假装完成" |
| LLM 接口 | `core/llm.py` | 屏蔽多 provider 差异 |
| 工具注册 | `tools/registry.py` | 工具的统一入口 |
| 上下文投影 | `runtime/context/` | 控制模型能看到多少历史 |

**下一步**：进入模块 1，打开 `core/llm.py`，看 harness 如何屏蔽多 provider 差异，统一拿到 `text` 和 `tool_calls`。

---

*对应源文件：`runtime/state.py`、`runtime/loop.py`（236-347 行）、`runtime/host.py`、`AGENT.md`*
