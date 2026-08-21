---
title: "Code Agent 解剖（10）：agent 崩了怎么恢复，对话历史存在哪？"
date: 2026-08-19
description: "深入 MyCodeAgent 的持久化与恢复机制：TranscriptStore append-only JSONL 事件流、五种事件类型、ResumeLoader 从事件重建运行时状态、UncertainAction 的工具中断处理、SessionMemory 跨 run 的前馈记忆。理解 agent 为什么不能靠内存存历史，以及崩溃后确定性恢复的完整链路。"
tags: [Code Agent, Transcript, Persistence, Resume, Session Memory, LLM Agent]
category: AI Engineering
draft: false
---

## 内存不可靠

agent 运行时，所有对话历史都在 `HistoryManager` 的内存列表里。一旦进程崩溃——网络超时、OOM、Ctrl+C——这些历史全部丢失。用户重启后只能从头开始，之前几十步的探索结果全部消失。

更麻烦的是工具调用的中断问题：agent 在执行 `Edit` 工具修改文件时崩了，文件可能改了一半，也可能没改，进程日志里什么都没有。重启后 agent 看不到这次 Edit 的结果，不知道文件改没改。

MyCodeAgent 用 **Transcript** 解决这两个问题：把所有关键事实持续写入 append-only JSONL 文件，崩溃后从文件重建运行时状态。

---

## 核心设计：事件流而不是快照

持久化有两种常见方案：

- **快照**：定期把整个状态序列化保存（类似数据库备份）
- **事件流**：每个操作完成后追加一条事件记录（类似数据库 WAL）

Transcript 用的是事件流。原因：快照需要保证原子性（不能写一半），而且重启后需要选择恢复哪个快照。事件流只追加，天然原子（每行独立），重启后重放所有事件就能还原状态。

```
memory/transcripts/transcript-{session_id}.jsonl

{"event_id":"evt-abc","timestamp":"...","session_id":"s-123","run_id":"run-1","step":0,"event_type":"message","payload":{"role":"user","content":"帮我重构认证模块"}}
{"event_id":"evt-def","timestamp":"...","session_id":"s-123","run_id":"run-1","step":1,"event_type":"state_transition","payload":{"reason":"model_returned_tool_calls",...}}
{"event_id":"evt-ghi","timestamp":"...","session_id":"s-123","run_id":"run-1","step":1,"event_type":"tool_lifecycle","payload":{"tool_name":"Read","tool_call_id":"call-xyz","status":"requested"}}
{"event_id":"evt-jkl","timestamp":"...","session_id":"s-123","run_id":"run-1","step":1,"event_type":"tool_lifecycle","payload":{"tool_name":"Read","tool_call_id":"call-xyz","status":"completed","result":"..."}}
...
{"event_id":"evt-zzz","timestamp":"...","session_id":"s-123","run-1","step":28,"event_type":"terminal","payload":{"reason":"completed"}}
```

---

## 五种事件类型

每条事件记录 loop 里的一个具体事实：

| 事件类型 | 记录什么 | 何时写入 |
|---------|---------|---------|
| `message` | user/assistant/tool 消息内容 | 每次 `history_manager.append_*()` 后 |
| `state_transition` | loop 状态转移原因 + 详情 | 每次 `_transition()` 后 |
| `tool_lifecycle` | 工具的 requested/started/completed/failed | 每个工具执行阶段 |
| `checkpoint` | 上下文压缩检查点（摘要 + 分割点） | 触发压缩后 |
| `terminal` | loop 终止原因 | loop 结束时 |

**写入路径**：

```
RuntimeRunner 调 self._emit(event_type, payload, step=step)
    ↓
RuntimeEventSink.emit(RuntimeEvent)
    ↓  [CompositeRuntimeEventSink]
    ├─ TraceRuntimeEventSink → trace_logger（JSONL trace 文件，调试用）
    └─ TranscriptRuntimeEventSink → TranscriptRecorder → TranscriptStore.append_event()
```

Transcript 和 Trace 是两个独立的 sink，订阅同一个事件流。Trace 是调试用的详细日志，Transcript 是恢复用的事实日志，两者内容有重叠但用途不同。

---

## TranscriptStore：append-only 写入

```python
# runtime/transcript.py
class TranscriptStore:
    def append_event(self, event: TranscriptEvent) -> TranscriptEvent:
        line = json.dumps(event.to_dict(), ensure_ascii=False)
        with self._lock:                      # 文件锁，防止并发写入损坏
            self._repair_trailing_record()    # 修复末尾不完整行（进程崩溃遗留）
            with self.path.open("a") as f:
                f.write(line)
                f.write("\n")
                f.flush()                     # 立刻刷到磁盘，不依赖 OS 缓冲
```

`_repair_trailing_record()` 在每次写入前检查文件末尾：如果有不完整的行（没有换行符，或者 JSON 解析失败），说明上次写入在中途被打断，把那一行截掉。这保证文件里每一行都是完整的合法 JSON。

---

## 事件如何流入 Transcript

写入不是直接调 TranscriptStore，中间有两层抽象：

```
loop 里的 self._emit("message", {...}, step=step)
    ↓
RuntimeRunner._emit_runtime_event(run_id, step, event_type, payload)
    ↓
host.runtime_event_sink.emit(RuntimeEvent)
    ↓  CompositeRuntimeEventSink 同时转发给两个 sink
    ├─ TraceRuntimeEventSink.emit()      → extensions/tracing/logger.py（调试 trace）
    └─ TranscriptRuntimeEventSink.emit() → TranscriptRecorder.record_*()
                                               ↓
                                           TranscriptStore.append_*()
                                               ↓
                                           JSONL 文件追加一行
```

`TranscriptRecorder` 是 `TranscriptStore` 的门面（Facade）：把"写 message"、"写 state_transition"、"写 tool_lifecycle"这些高层操作各自封装成 `record_message()`、`record_state_transition()`、`record_tool_lifecycle()`，隐藏底层的 JSON 序列化细节。同时它持有一个 `on_recorded` 回调，每次写入后调用 `SessionMemoryManager.ingest_event()`，增量更新 Session Memory。

---

## 恢复：从事件流重建运行时状态

用户重启后调 `agent.resume_transcript()`，或 CLI 用 `--resume` 参数，走到 `ResumeLoader.load_session()`：

```python
# runtime/transcript.py — ResumeLoader._load_events()（简化）
def _load_events(self, events, *, run_id):
    history_messages = []    # 重建历史消息列表
    checkpoint = None        # 最后一个压缩检查点
    terminal = None          # 终止事件
    tool_events = {}         # 每个工具调用的完整生命周期

    for event in events:
        if event.event_type is MESSAGE:
            history_messages.append({role, content, metadata})

        elif event.event_type is CHECKPOINT:
            checkpoint = event.payload   # 保留最后一个 checkpoint

        elif event.event_type is TERMINAL:
            terminal = event.payload

        elif event.event_type is TOOL_LIFECYCLE:
            # 把同一个 tool_call_id 的所有状态聚合在一起
            tool_events[(run_id, tool_call_id)]["statuses"].append(status)
```

然后分析 `tool_events`，把每个工具调用归类：

```
completed   → 已完成，不需要重放
failed      → 已失败，不需要重放
requested 但未 started → 未执行，pending（可重新计划）
started 但无 completed/failed → uncertain（中断，状态未知）
```

**UncertainAction** 是恢复里最微妙的概念：工具已经开始执行，但在结果写回之前 agent 崩了。结果可能是成功、失败或任何中间状态。

```python
uncertain_actions.append(UncertainAction(
    tool_name=tool_name,
    tool_call_id=tool_call_id,
    replay_allowed=tool_name not in {"Edit", "Bash", "Task"},
    # Read/Grep/Glob：幂等，可以重放
    # Edit/Bash/Task：有副作用，不能盲目重放，需要用户判断
))
```

CLI 恢复时会打印 uncertain actions，让用户知道"这些工具可能执行了也可能没执行，请自行确认"。

---

## 重建完成后 apply_to_host

`ResumeState.apply_to_host(host)` 把重建的状态注入运行中的 agent：

```python
# runtime/transcript.py — ResumeState.apply_to_host()
def apply_to_host(self, host):
    # 1. 重置上下文引擎，清除压缩检查点
    host.context_engine.reset()

    # 2. 把重建的历史消息写入 HistoryManager
    host.history_manager.load_messages(self.history_messages)

    # 3. 如果有压缩检查点，重新激活（ProjectionBuilder 读它时会折叠旧历史）
    if self.checkpoint:
        host.context_engine.compact_store.set_active(CompactCheckpoint(...))

    # 4. 恢复 Read 工具的乐观锁缓存（mtime 快照，避免 Edit 冲突误报）
    if read_cache := self.runtime_state.get("read_cache"):
        host.tool_registry.import_read_cache(read_cache)
```

恢复后 agent 就像从未崩溃过——历史完整、压缩状态恢复、乐观锁缓存有效。

---

## Session Memory：跨 run 的前馈记忆

Transcript 存的是完整事实流，Session Memory 是从事实流派生的**有界摘要**。

`SessionMemoryDeriver` 扫描所有 transcript 事件，提取高层信息：

```python
@dataclass(frozen=True)
class SessionMemory:
    current_goal: SessionMemoryItem | None  # 用户最近的目标（最新 user 消息）
    completed_work: tuple[...]              # 完成了什么（final 类型的 assistant 消息）
    key_decisions: tuple[...]               # 关键决策（压缩 checkpoint、重要状态转移）
    failed_attempts: tuple[...]             # 失败过什么（model_recovery_failed 等）
    todo_items: tuple[...]                  # 未完成的 TodoWrite 条目
    verification_status: tuple[...]         # 验证状态（完成门信息）
```

Session Memory 在 `build_model_view()` 里注入成一条 system 消息，放在系统提示词之后、历史消息之前：

```
[系统提示词]
[Session Memory]  ← "你之前完成了 X，失败过 Y，当前目标是 Z"
[历史消息]
```

**为什么要 Session Memory，不直接读 transcript？**

Transcript 可能有几千行事件，全部放进 model view 会超出 token 预算。Session Memory 是一个有界的高层摘要，控制在几百行字以内，让模型跨 run 知道背景而不需要读全量事件。

Session Memory 是增量维护的：`TranscriptRecorder` 每写入一个事件就调一次 `SessionMemoryManager.ingest_event()`，`SessionMemoryDeriver.update()` 增量追加新事件，不需要每次全量重建。

---

## 实际文件结构

```
memory/
├── transcripts/
│   ├── transcript-session-abc123.jsonl   ← 主 agent session
│   └── transcript-subagent-child-xyz.jsonl ← 子 agent session（每次 Task 调用独立）
└── traces/
    ├── session-abc123.jsonl              ← 调试 trace（详细）
    └── session-abc123.html               ← 可视化报告（可选）
```

Transcript 和 Trace 文件并列存放，但用途不同：
- **Transcript**：崩溃恢复的真相来源，只存关键事实
- **Trace**：调试分析用，记录所有细节（包括 token 用量、每步耗时、模型原始输出）

---

## 设计亮点

**事件流而非快照**：每个事件写入即持久化，不需要等待 agent 完成，崩溃后从最后一个完整事件开始恢复。

**uncertain actions 显式标记**：工具中断不是静默失败，而是显式标记为 uncertain，由用户判断是否重放，而不是 agent 自动猜测。

**恢复是确定性的**：从同一份 transcript 读取，每次重建得到相同的 history_messages 和 loop_state，不依赖任何随机状态。

**Session Memory 增量维护**：不在恢复时全量重建，每次事件写入后增量更新，把重建成本均摊到每次写入。

---

## 小结

| 组件 | 职责 |
|------|------|
| `TranscriptStore` | append-only JSONL 写入，有文件锁和末尾修复 |
| `TranscriptRecorder` | 门面层，封装高层写入操作，回调 SessionMemory 更新 |
| `TranscriptRuntimeEventSink` | 把 RuntimeEvent 路由到 TranscriptRecorder |
| `ResumeLoader` | 从事件流重建 history/checkpoint/tool_states/uncertain_actions |
| `ResumeState.apply_to_host` | 把重建状态注入 agent，完成恢复 |
| `SessionMemoryDeriver` | 从事件流派生有界工作记忆，供前馈注入 model view |
| `UncertainAction` | 显式标记中断的工具调用，保护用户不被盲目重放副作用 |

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
