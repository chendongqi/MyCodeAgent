---
title: "Code Agent 解剖（09）：对话越来越长，token 超了怎么办？"
date: 2026-08-19
description: "深入 MyCodeAgent 的上下文工程：HistoryManager append-only 事实日志、ProjectionBuilder 读时投影、ContextBudgetPolicy 触发判断、ContextCompactor LLM 摘要压缩。理解为什么「历史永不删除」和「给模型看有界视图」是两件完全不同的事，以及 agent 如何在不丢失事实的前提下把上下文压进预算。"
tags: [Code Agent, Context Engineering, Token Budget, LLM Agent, History Compression]
category: AI Engineering
draft: false
---

## 问题从哪里来

一个 coding agent 跑起来，每个 ReAct step 都往历史里追加消息：用户输入、模型回答、工具调用、工具结果。跑二三十步之后，历史里可能有几百条消息、几万个 token。

LLM 的上下文窗口是有限的（默认 128k token）。把全量历史原样发给模型有两个问题：

1. **超限报错**：超过上下文窗口，API 返回 `PROMPT_TOO_LONG` 错误
2. **质量下降**：即使没超，太长的历史会稀释近期信息，模型注意力分散

解决方案有两类：截断（丢掉旧消息）和摘要（把旧消息压缩成摘要）。MyCodeAgent 用的是摘要方案，且设计上有一个关键原则：**历史永不删除，压缩只影响给模型看的视图**。

---

## 核心设计：History 和 Model View 分离

这是理解整套机制的前提，先把两个概念分清楚：

```
HistoryManager（完整事实日志，append-only，永不删除）
    所有消息原文：user / assistant / tool / summary
    是"真相"，transcript 的写入依据，崩溃恢复的来源

         ↓  build_model_view() 在每个 step 调用一次

Model View（给 LLM 看的有界投影）
    系统消息 + 历史的子集（可能是压缩后的）
    是"视图"，只影响这一次 LLM 调用，不修改历史
```

两者分离的好处：压缩是一个可逆的读时操作。原始历史永远完整，可以随时重建，崩溃恢复不依赖压缩状态。

---

## 整体流程

每个 ReAct step 开始时，loop 调 `_prepare_step_context()`，里面依次走三步：

```
step N 开始
    ↓
compact_if_needed()        检查是否需要压缩，需要则触发
    ↓
build_model_view()         把历史投影成本 step 发给模型的消息列表
    ↓
llm.invoke_raw(messages, tools=...)   发给 LLM
```

压缩和投影是分离的两步，互不依赖：没触发压缩时，投影就是全量历史；触发过压缩后，投影就是摘要 + 最近 N 轮原文。

---

## 第一步：判断要不要压缩

`ContextBudgetPolicy.should_compact()` 做这个判断：

```python
# runtime/context/budget.py
def should_compact(self, *, messages, pending_input, last_usage_tokens):
    threshold = int(self.config.context_window * self.config.compression_threshold)
    #           默认：128000 × 0.8 = 102400 tokens

    # 两个估算来源，取较大值（用最悲观的估计，避免漏触发）
    estimated_from_messages = self.estimate_tokens(messages, pending_input)
    estimated_from_usage = last_usage_tokens + len(pending_input) // 3
    estimated = max(estimated_from_messages, estimated_from_usage)

    if estimated >= threshold:
        return CompactDecision(True, "threshold_exceeded", ...)
```

`estimate_tokens()` 用 `字符数 // 3` 近似 token 数——不精确，但足够保守。用两个来源取 max 是为了覆盖"字符估算偏低"的情况：如果上一轮 LLM 实际消耗了很多 token，就以实际用量为准。

**`context_window` 和 `compression_threshold` 从哪来：**

```python
# core/config.py（从 .env 读取）
context_window: int = 128000          # CONTEXT_WINDOW 环境变量
compression_threshold: float = 0.8   # COMPRESSION_THRESHOLD 环境变量
min_retain_rounds: int = 10          # MIN_RETAIN_ROUNDS 环境变量
```

触发阈值是 `context_window × compression_threshold`，不是等到 100% 才压缩——在 80% 时提前压缩，留出空间给模型的输出 token。

---

## 第二步：压缩——产生 Checkpoint

触发压缩后，`ContextCompactor.compact()` 执行：

```python
# runtime/context/compact.py
def compact(self, messages):
    # 1. 按 user 消息边界切分轮次
    rounds = self.round_segmenter.identify(messages)
    #    rounds = [Round(0,5), Round(6,12), Round(13,18), ...]
    #    每个 Round 从一条 user 消息开始到下一条 user 消息之前

    # 2. 保留最近 min_retain_rounds 轮（默认 10 轮）的原文
    retain_start_round = len(rounds) - min_retain_rounds
    retain_start_idx = rounds[retain_start_round].start_idx
    messages_to_compact = messages[:retain_start_idx]   # 旧的部分

    # 3. 把旧消息交给 summary_generator（LLM 调用），生成摘要
    summary = self.summary_generator(messages_to_compact)
    # summary_generator 有超时保护（默认 120s）；超时则返回 None，不压缩

    # 4. 创建 checkpoint：摘要文本 + retain_start_idx
    checkpoint = self.compact_store.create_checkpoint(
        summary=summary,
        retain_start_idx=retain_start_idx,
        ...
    )
```

**关键：`compact()` 不修改 `HistoryManager` 里的任何消息**。它只是在 `CompactStore` 里存了一个 checkpoint（摘要 + 分割点）。原始消息完整保留。

`RoundSegmenter` 的分割逻辑很简单：每遇到 `role="user"` 就开启新轮：

```
messages: [user₁][assistant][tool][tool][user₂][assistant][tool][user₃]...
rounds:   |────── Round 1 ────────|──── Round 2 ─────|── Round 3 ──...
```

保留最近 10 轮，之前的全部送去压缩。

**`summary_generator` 是什么：** 由 `create_summary_generator(llm, config)` 在 `factory.py` 里创建，返回一个闭包。调用时把旧消息序列化成文本，发给 LLM，提示词要求生成结构化摘要（已完成的目标、关键决策、修改过的文件等）。

---

## 第三步：投影——决定模型看到什么

每次 `build_model_view()` 都调 `ProjectionBuilder.project()`：

```python
# runtime/context/projection.py
def project(self, source_messages):
    checkpoint = self.compact_store.active_checkpoint

    if not checkpoint:
        # 没有压缩过：模型看到全量历史
        return ProjectionResult(messages=source_messages, mode="full_history")

    # 有 checkpoint：折叠旧历史
    summary_msg = Message(content=checkpoint.summary, role="summary")
    recent_messages = source_messages[checkpoint.retain_start_idx:]

    return ProjectionResult(
        messages=[summary_msg] + recent_messages,
        mode="compact_checkpoint",
    )
```

压缩后模型看到的消息结构：

```
[系统消息×N]
[summary 消息]   ← "之前发生了：用户要求重构认证模块，已完成X、Y、Z..."
[最近10轮原文]   ← 完整的 user/assistant/tool 消息
```

**这就是"读时投影"**：`project()` 每次都从原始 `source_messages` 里读，动态生成视图，不修改任何数据。checkpoint 里的 `retain_start_idx` 是分割点，告诉投影"从哪里开始保留原文"。

---

## 两种触发方式

压缩有两个触发时机，对应不同场景：

**主动触发（每步开始时）**：`compact_if_needed()`，在 `_prepare_step_context()` 里调用。估算超阈值就触发，属于预防性压缩。

**被动触发（PROMPT_TOO_LONG 时）**：`reactive_compact()`，在 loop 的内层 while 里，当 LLM 调用因上下文太长抛异常时触发。压缩成功后重建 model view 重试。

```
# 被动触发路径（loop.py）
try:
    raw_response = llm.invoke_raw(messages, ...)
except Exception as exc:
    if classify_model_error(exc).kind is ModelErrorKind.PROMPT_TOO_LONG:
        compact_info = host.context_engine.reactive_compact(...)
        if compact_info.get("compacted"):
            # 重建 model view，内层 continue 重试
            messages = host.context_engine.build_model_view(...).messages
            continue
```

---

## 压缩失败怎么办

`summary_generator` 失败（LLM 超时、网络错误）时，`compact()` 返回 `{"compacted": False, "reason": "summary_unavailable"}`，不做任何修改。

主动触发时：跳过压缩，这步继续用没压缩的历史。下一步再判断要不要压缩。

被动触发时（PROMPT_TOO_LONG）：压缩失败 → 无法恢复 → loop 终止，返回错误消息。这是最坏情况，实际中很少发生，因为主动触发通常会提前介入。

---

## 设计亮点

**历史永不删除**：`HistoryManager` append-only，压缩不修改历史，崩溃后可从 transcript 完整重建。

**读时投影而非写时截断**：截断是破坏性的（信息丢失），投影是可逆的（checkpoint 仍在，随时可以丢弃重新投影）。

**两个估算来源取 max**：既用字符数估算（当前消息内容），又用上轮实际 token 用量，取较大值，保证估算偏保守不会遗漏。

**压缩超时不崩溃**：`summary_generator` 内部用 `ThreadPoolExecutor` 做超时控制，LLM 摘要调用超时直接返回 `None`，不影响主循环，只是跳过这次压缩。

---

## 小结

| 组件 | 职责 |
|------|------|
| `HistoryManager` | append-only 事实日志，永不删除，所有消息写入的地方 |
| `ContextBudgetPolicy` | 估算 token 用量，判断是否超过阈值需要压缩 |
| `RoundSegmenter` | 把历史按 user 消息边界切割成轮次 |
| `ContextCompactor` | 调 LLM 生成旧历史摘要，创建 checkpoint（不修改原始历史） |
| `CompactStore` | 存储 checkpoint（摘要 + 分割点） |
| `ProjectionBuilder` | 读时投影：有 checkpoint 则返回摘要 + 最近 N 轮，否则返回全量 |
| `ContextEngine` | 以上所有组件的协调者，对外暴露 `compact_if_needed()` 和 `build_model_view()` |

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
