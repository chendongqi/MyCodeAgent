---
title: "Code Agent 解剖（02）：agent 是怎么一轮一轮思考和行动的？"
date: 2026-08-13
description: "深入 MyCodeAgent 的 ReAct 主循环：不可变状态机、双层循环结构、完成门三态判决、模型错误恢复机制。理解 agent 为什么不是「调一次模型拿结果」，而是一个有反馈、有保障、有明确终止条件的控制系统。"
tags: [Code Agent, ReAct, 状态机, 完成门, LLM Agent]
category: AI Engineering
draft: false
---

## agent 为什么需要循环

最简单的 LLM 调用是这样的：

```python
response = llm.invoke(messages)
print(response)
```

这能完成简单问答，但不能完成"帮我重构这个函数并跑一遍测试"这类任务——因为它涉及多个步骤：读文件、改代码、执行命令、根据结果决定下一步。每一步的输入依赖上一步的输出，而模型不能一次性预知所有步骤。

ReAct（Reasoning + Acting）是解决这个问题的模式：让模型在"思考→执行工具→观察结果→再思考"的循环里迭代推进，直到任务完成。

MyCodeAgent 的 `RuntimeRunner._react_loop()` 是这个模式的具体工程实现。这篇文章把它拆开来看。

---

## 循环开始前：_prepare_run

`_react_loop` 不是 `RuntimeRunner.run()` 直接调的第一件事，前面还有一步 `_prepare_run()`：

```python
# runtime/loop.py — RuntimeRunner.run()
def run(self, input_text, **kwargs):
    processed_input, trace_logger, run_id = self._prepare_run(input_text, show_raw)
    response_text = self._react_loop(pending_input=processed_input, ...)
```

`_prepare_run` 做了五件事，每件事都指向不同的子系统：

```python
def _prepare_run(self, input_text, show_raw):
    # 1. 刷新 Skills 提示词（如果 skills/ 目录有变化）
    host._refresh_skills_prompt()
    host.context_builder.set_skills_prompt(host._skills_prompt)

    # 2. 预处理用户输入：识别并展开 @file 引用，注入 system-reminder 提示模型去读
    preprocess_result = preprocess_input(input_text)
    processed_input = preprocess_result.processed_input

    # 3. 清空上一轮的 trace 事件，为本轮初始化 run_id 和 transcript 记录
    trace_logger.clear_current_run_events()
    host._run_id += 1
    host._active_transcript_run_id = f"run-{host._run_id}"

    # 4. 把预处理后的用户消息写入 history_manager
    #    ← 这是用户输入进入历史的唯一时机
    self._append_user_message(processed_input)

    # 5. 发射 run_start / user_input 事件给 trace 和 transcript
    self._emit("run_start", {...}, step=0)
    self._emit("user_input", {...}, step=0)

    return processed_input, trace_logger, run_id
```

**`pending_input` 的来历**：`processed_input` 就是 `pending_input`，是预处理后的用户输入字符串。它在 `_prepare_run` 里已经写入了历史，传给 `_react_loop` 之后不会再次追加，只用于两件事：给 `build_model_view()` 估算本轮需要保留多少 token 空间，以及给完成门推断"用户这次要求了什么"。

**`@file` 展开是什么**：用户输入 `帮我看看 @src/main.py` 时，`preprocess_input()` 检测到 `@file` 引用，在消息末尾追加一条 `<system-reminder>` 告诉模型"你必须先用 Read 工具读这个文件再回答"，这样模型不会凭想象回答。

---

## 整体结构：双层循环

```
_react_loop()
│
├─ 外层 for（step 1 → max_steps）     每次迭代 = 一个 ReAct step
│   │
│   ├─ _prepare_step_context()         构建本步的 Model View
│   │
│   ├─ 内层 while True                 单步内的模型调用与错误恢复
│   │   ├─ llm.invoke_raw()            调模型
│   │   ├─ 模型调用异常？               分类处理（压缩/重试/终止）
│   │   ├─ 响应解析（text + tool_calls）
│   │   ├─ 空响应？                     注入提示重试
│   │   └─ break                       正常响应，跳出内层
│   │
│   ├─ 有 tool_calls？                  Acting 分支
│   │   ├─ 执行工具（ToolOrchestrator）
│   │   ├─ 结果写入历史
│   │   └─ continue（下一个 step）
│   │
│   └─ 没有 tool_calls？                Reasoning 分支
│       ├─ 完成门判决（PASS/FAIL/UNVERIFIED）
│       ├─ PASS → return final_text    正常出口
│       ├─ FAIL → 注入反馈 + continue
│       └─ UNVERIFIED → return         带不确定性退出
│
└─ 超出 max_steps → return             兜底出口
```

两层循环职责不同：外层 `for` 推进 ReAct 步骤，内层 `while` 处理单步内的模型错误。把错误处理收进内层是关键设计，否则外层的步数计数会因为重试被消耗掉。

---

## 不可变状态机

每个 step 开始时，状态是这样的：

```python
# runtime/state.py
@dataclass(frozen=True)          # frozen=True：不可修改，任何"修改"都产生新对象
class LoopState:
    messages: list[dict]         # 当前 model view（不是完整历史）
    step: int                    # 当前步数
    tool_choice: str             # "auto" | "none" | 指定工具
    transition: Transition|None  # 最近一次状态转移记录
    completion_block_count: int  # 完成门拦截次数
    model_recovery_counts: dict  # 各类错误的重试计数
    last_error: str|None         # 最近一次错误信息
    # ...其余诊断字段

    def next(self, reason: TransitionReason, **changes) -> "LoopState":
        # 产生新对象，记录转移原因——不修改 self
        return replace(self, transition=Transition(reason=reason), **changes)
```

`frozen=True` 意味着：

```python
state.step = 2      # ❌ 抛出 FrozenInstanceError
state = state.next(TransitionReason.TOOLS_EXECUTED, step=2)  # ✅ 产生新对象
```

**为什么要不可变？** 可变状态在 debug 时最难追查——"这个字段是在哪步被改成这个值的？"不可变状态配合 `TransitionReason` 枚举，每次状态变化都有明确的原因记录，trace 系统能还原完整的执行路径。这是函数式编程思想在工程里的具体应用。

`TransitionReason` 枚举记录了 loop 的每一种走向：

```python
class TransitionReason(str, Enum):
    USER_INPUT = "user_input"                    # 新一轮开始
    MODEL_RETURNED_TOOL_CALLS = "..."            # 模型要调工具（Acting）
    TOOLS_EXECUTED = "..."                       # 工具执行完
    MODEL_RETURNED_FINAL = "..."                 # 模型给出最终回答（Reasoning）
    STOP_HOOK_BLOCKING = "..."                   # 完成门拦截，注入反馈后继续
    MODEL_RECOVERY_RETRY = "..."                 # 模型出错，恢复后重试
    MAX_STEPS_EXCEEDED = "..."                   # 超步终止
    TOKEN_BUDGET_EXCEEDED = "..."                # token 预算耗尽终止
```

---

## Model View：给模型看的不是完整历史

每步开始时调 `_prepare_step_context()`，它返回的 `messages` 不是 `history_manager` 里的全量消息，而是经过 `context_engine.build_model_view()` 投影出来的有界子集。

```
history_manager（完整历史，append-only，永不删除）
         ↓
   build_model_view()
         ↓
   model view（token 预算内的投影，发给 LLM）
```

完整历史里可能有 200 条消息，但 token 预算只允许发 50 条。`build_model_view()` 负责决定"发哪 50 条"，超出预算时触发压缩（对旧轮次做 LLM 摘要）。

**为什么要分离？** 历史是事实，不能删；给模型看的是视图，可以裁剪。这两件事混在一起会导致历史被破坏，崩溃后就无法恢复。第 09 篇会详细讲这套上下文工程。

---

## Acting 分支：模型返回了工具调用

```python
# loop.py — Acting 分支（简化）
if tool_calls:
    # 1. 确保每个 tool_call 有 id（部分模型不返回）
    for call in tool_calls:
        if not call.get("id"):
            call["id"] = f"call_{uuid.uuid4().hex}"

    # 2. 把 assistant 消息（含 tool_calls 列表）写入历史
    host.history_manager.append_assistant(
        content=response_text,
        metadata={"action_type": "tool_call", "tool_calls": tool_calls},
    )

    # 3. 执行工具（只读工具可并发，写操作强制串行）
    observations = host.tool_orchestrator.run(tool_calls, step=step)

    # 4. 把每个工具的执行结果写入历史，下一步模型能看到
    for obs in observations:
        host.history_manager.append_tool(
            tool_name=obs.tool_name,
            observation=obs.observation,
        )

    continue  # 外层 for 继续下一个 step
```

工具执行结果通过 `history_manager.append_tool()` 写入历史，下一步 `build_model_view()` 构建 model view 时这些结果就在消息里。模型"看到"了工具执行结果，这就是"Observation"。

---

## Reasoning 分支：完成门

没有 `tool_calls` 时，模型给出了文字回答。但 loop 不会直接 `return`，先过完成门：

```python
# runtime/completion.py — 完成门三步走

# 步骤 1：推断要求（从用户输入里识别"需要验证"的关键词）
requirements = infer_completion_requirements(
    user_input=pending_input,          # 扫 "pytest" / "跑测试" 等关键词
    history_messages=history_messages, # 读最新 TodoWrite 记录检查未完成项
)

# 步骤 2：收集证据（从历史 Bash 工具调用里找验证行为）
evidence = collect_verification_evidence(history_messages)
# 注意：Edit 之后的验证证据才算有效，Edit 之前的被标记为 invalid

# 步骤 3：判决
verdict = verifier.evaluate(candidate, requirements, evidence, ...)
# PASS：所有要求满足 → return final_text
# FAIL：有未完成 todo 或缺验证证据 → 注入反馈消息，继续循环
# UNVERIFIED：用户说"尽量"，证据缺失但允许跳过 → return（带标记）
```

**完成门解决什么问题？** 模型可能在 todo 列表还没清空时就说"我完成了"，也可能在用户要求跑测试后忘记执行就回答。完成门通过规则检查拦截这两种情况，把"为什么没完成"注入成 user 消息，让模型重新来过。

验证证据有一个反直觉的细节：**Edit 之后再执行的 pytest 才算有效**。如果先跑了测试（通过），再修改了代码（Edit），之前的测试结果就被作废了——因为修改后的代码还没经过验证。

```python
# completion.py — collect_verification_evidence()
# 找到最近一次 Edit 的 step
latest_mutation_step = max(step for edit in history if edit.tool == "Edit")

# Edit 之前的验证证据标记为无效
for evidence in evidences:
    if evidence.step < latest_mutation_step:
        evidence.valid = False  # 已被后续 Edit 作废
```

---

## 模型错误恢复

内层 `while True` 处理两类模型问题：

**调用异常（PROMPT_TOO_LONG）**：

```
invoke_raw() 抛出异常
    ↓
classify_model_error() 判断是 PROMPT_TOO_LONG
    ↓
context_engine.reactive_compact() 压缩历史
    ↓
重建 model_view，内层 continue 重试
    ↓
还是失败？→ 终止
```

**响应异常（EMPTY_RESPONSE）**：

```
调用成功，但 response_text 为空且没有 tool_calls
    ↓
注入提示："请在 content 中回复最终答案，或使用工具调用"
    ↓
内层 continue 重试（最多 1 次）
    ↓
还是空？→ 终止
```

两种恢复都有次数上限（`_get_model_recovery_limit()`），超限就走终止路径，不会无限重试。

---

## 终止条件一览

loop 有且仅有以下几种出口：

```
正常出口：
  完成门 PASS             → return final_text
  完成门 UNVERIFIED       → return final_text（带标记）

异常出口：
  超出 max_steps          → return 错误提示
  token 预算耗尽          → return 错误提示
  模型错误无法恢复         → return 错误提示
  完成门反馈重试耗尽       → return 错误提示
```

注意没有"空响应直接退出"——空响应会重试，重试耗尽才终止。这保证了 loop 的行为是可预期的，不会因为一次偶发的空响应就悄悄失败。

---

## 小结

| 机制 | 作用 |
|------|------|
| 双层循环 | 外层推进 ReAct step，内层处理模型错误恢复，职责分离 |
| 不可变状态机 | 每次转移记录原因，trace 能还原完整执行路径 |
| Model View 分离 | 历史完整保存，给模型看的是有界投影，两者不混淆 |
| 完成门三态 | PASS/FAIL/UNVERIFIED，拦截模型"虚假完成"，注入反馈继续 |
| 验证证据时效 | Edit 之后的验证才算数，防止"改了代码但测试结果是旧的" |
| 错误恢复有上限 | 每类错误最多重试 N 次，超限终止，行为可预期 |

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
