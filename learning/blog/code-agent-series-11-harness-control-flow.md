---
title: "Code Agent 解剖（11）：Harness 设计之一——控制流"
date: 2026-08-21
description: "从 harness 工程设计视角审视 MyCodeAgent 的控制流：为什么选择单一主循环而不是多循环、不可变状态机的工程价值、完成门的反馈回路设计、终止路径的完备性。这是 Part 4 Harness Engineering 的第一篇，关注的是「为什么这样设计」而不是「是什么」。"
tags: [Code Agent, Harness Engineering, Control Flow, ReAct, LLM Agent]
category: AI Engineering
draft: false
---

## 什么是 Harness

前面十篇介绍了 agent 的各个功能：LLM 调用、工具系统、上下文压缩、持久化恢复。Part 4 换个角度——不问"这个功能怎么用"，而是问"这个 agent 框架是怎么设计的，做了哪些工程选择"。

Harness 这个词来自赛马，指套在马身上控制方向的装置。在 agent 语境里，harness 指框架本身的工程结构——控制流怎么组织、状态怎么管理、副作用怎么约束、错误怎么恢复。

这一篇聚焦控制流。

---

## 设计选择一：单一主循环

MyCodeAgent 只有一个 `RuntimeRunner`，整个 agent 的行为都在 `_react_loop()` 这一个循环里发生。

对比一下另一种常见设计——多循环嵌套：

```
# 常见的多循环设计
while not done:
    plan = planner.plan(goal)           # 外层：规划循环
    for step in plan.steps:
        result = executor.execute(step) # 内层：执行循环
        if result.needs_replan:
            break
```

MyCodeAgent 的选择：

```python
# 单一循环
for step in range(1, max_steps + 1):
    model_view = build_model_view(history)
    response = llm.invoke(model_view)
    if tool_calls:
        observations = execute_tools(tool_calls)
        append_to_history(observations)
        continue
    if completion_gate.pass(response):
        return response
    inject_feedback(gate.blocking_reason)
```

**单一循环的工程价值**：

1. **状态只有一份**：`LoopState` 是唯一的运行时状态，没有"规划状态"和"执行状态"两套需要同步。调试时只需要看一个状态对象。

2. **控制流可观测**：每次循环迭代对应一个 step，每个 step 的所有事件都有相同的 `step` 编号。trace 文件里按 step 过滤就能还原任意时刻发生了什么。

3. **没有隐式状态机**：多循环设计里"当前在哪个循环、到了哪一步"通常靠变量标记，出 bug 时难以追查。单一循环的进度就是 `step` 计数器，直接。

代价是：单一循环意味着规划和执行都由模型在同一个循环里完成，不能在"规划"阶段用不同的策略（如强制规划优先）。这个项目选择把规划权完全交给模型，循环只提供执行框架。

---

## 设计选择二：不可变状态机

`LoopState` 是 `frozen=True` 的 dataclass：

```python
@dataclass(frozen=True)
class LoopState:
    step: int
    tool_choice: str
    completion_block_count: int
    model_recovery_counts: dict[str, int]
    last_error: str | None
    transition: Transition | None  # 最近一次转移的原因
    # ...

    def next(self, reason: TransitionReason, **changes) -> "LoopState":
        return replace(self, transition=Transition(reason), **changes)
```

每次状态变化产生新对象，原对象不变。

**为什么不可变？**

可变状态的问题：任何代码都可能悄悄修改状态，bug 发生时很难定位是哪一步改的。

```python
# 可变状态的问题
state.step = 2
state.last_error = "timeout"  # 这行在哪个分支里改的？
```

不可变状态把每次变化变成显式操作：

```python
# 不可变状态：每次修改都是一次新对象创建，原因记录在 transition 字段里
state = state.next(TransitionReason.TOOLS_EXECUTED, step=step+1)
#                   ↑ 为什么改变：工具执行完了
```

`TransitionReason` 枚举是关键——它把"为什么走到这一步"编码成可查的枚举值，而不是靠代码注释或调用栈推断。trace 文件里每个 state_transition 事件都有 reason，可以按 reason 过滤出"所有模型空响应重试"或"所有完成门拦截"事件。

**不可变的额外好处**：每个 `LoopState` 对象是一个时间点的快照。如果需要在某个 step 设置断点或者回放，直接从 transcript 重建那个时刻的 LoopState 就行，不需要重新运行整个循环。

---

## 设计选择三：完成门的反馈回路

大多数 agent 框架的退出逻辑是："模型没有返回工具调用就退出"。MyCodeAgent 多了一层：

```
模型没有 tool_calls
    ↓
完成门判决（不是直接退出）
    ├─ PASS      → return final_text
    ├─ UNVERIFIED → return final_text（带标记）
    └─ FAIL      → 注入反馈消息 → continue（继续循环）
```

FAIL 时注入的反馈消息是结构化的：

```python
# runtime/completion.py — _build_blocking_feedback()
lines = ["<system-reminder>Completion blocked by runtime gate.</system-reminder>"]
if "incomplete_todos" in reasons:
    lines.append("Incomplete todos remain: " + "; ".join(incomplete_todos))
if "missing_verification_evidence:tests" in reasons:
    lines.append("Missing verification evidence for tests. Run the required verification tool.")
```

用 `<system-reminder>` 标签包裹是有意图的：系统提示词里有指令"遇到 system-reminder 必须执行"，这样模型更可能真的去执行反馈里要求的操作（跑测试、清空 todo），而不是忽略。

**反馈回路的工程价值**：

没有完成门，模型可能在 todo 列表还有未完成项时说"我完成了"，或者用户要求跑测试而模型忘了执行。完成门让 harness 层面有一个独立于模型的验证机制——不信任模型的自我声明，用客观的规则检查（是否有未完成 todo、是否有测试执行证据）。

反馈注入而不是直接终止是另一个设计选择：报错终止会让用户重新输入，注入反馈让 agent 有机会自我修正，用户体验更好，且在大多数情况下模型确实能根据反馈完成剩余工作。

上限是 `completion_gate_retry_limit`（默认 2 次），防止无限反馈注入把上下文撑爆。

---

## 设计选择四：终止路径的完备性

loop 所有可能的出口：

```
正常出口：
  完成门 PASS              → return final_text
  完成门 UNVERIFIED        → return final_text（带 completed_unverified 标记）

异常出口（都有对应的 TerminalReason）：
  超出 max_steps           → TerminalReason.MAX_STEPS
  token 累计超出预算        → TerminalReason.TOKEN_BUDGET
  模型调用异常无法恢复      → TerminalReason.MODEL_ERROR
  空响应重试耗尽           → TerminalReason.EMPTY_RESPONSE_FAILED
  完成门反馈重试耗尽       → TerminalReason.COMPLETION_GATE_BLOCKED
```

每个异常出口都写入一个 `terminal` 事件到 transcript，原因明确。这确保了 transcript 里每个 run 都有一个 terminal 事件，`ResumeLoader` 可以靠 terminal 事件判断"这个 run 是正常完成还是异常终止"，从而决定恢复后是否需要重新运行。

**为什么要穷举终止原因？**

如果只有"完成"和"出错"两种终止，调试时很难知道 agent 为什么停了。穷举的 TerminalReason 枚举让每次终止都有明确的解释，配合 trace 文件可以直接查"这次 agent 因为什么停止了"。

---

## 双层循环的职责分离

第 2 篇介绍过双层循环结构，从 harness 设计角度看，它解决了一个具体问题：**步数消耗和错误重试的分离**。

```
外层 for step in range(max_steps):   ← 控制"走了多少步"
    内层 while True:                  ← 控制"这步里重试了几次"
        try: llm.invoke()
        except PROMPT_TOO_LONG: compact → continue   # 内层 continue，step 不变
        if empty_response: inject_hint → continue    # 内层 continue，step 不变
        break                                         # 正常响应，跳出内层
    # 外层继续，step + 1
```

如果把错误重试放在外层循环里，每次重试都消耗一个 step，用户设置 `max_steps=50` 实际上因为错误重试只跑了 30 步真正的 ReAct 迭代。双层循环把重试隔离在内层，错误恢复不占用 step 配额。

---

## 小结

| 设计选择 | 方案 | 工程价值 |
|---------|------|---------|
| 循环结构 | 单一主循环 | 状态唯一、控制流可观测、无隐式状态机 |
| 状态管理 | 不可变 + TransitionReason | 每次变化显式，reason 可查，支持快照重放 |
| 完成条件 | 完成门三态 + 反馈注入 | 独立于模型的验证，自动修正而不是报错终止 |
| 终止路径 | 穷举 TerminalReason | 每次终止有明确原因，方便调试和恢复判断 |
| 错误重试 | 内层循环隔离 | 重试不消耗 step 配额，step 计数真实反映进度 |

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
