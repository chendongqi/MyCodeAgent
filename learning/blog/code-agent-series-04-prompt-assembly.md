---
title: "Code Agent 解剖（04）：系统提示词是怎么组装的，agent 的「人格」从哪来？"
date: 2026-08-13
description: "深入 MyCodeAgent 的提示词组装层：Constitution / Tool Contracts / Project Rules / Runtime Signals 四层 system 消息，以及它们如何与历史投影拼成发给模型的 Model View。理解 agent 的行为约束不是「一段超长 prompt」，而是可缓存、可指纹、可热更新的分层装配。"
tags: [Code Agent, System Prompt, Prompt Engineering, Context Builder, LLM Agent]
category: AI Engineering
draft: false
---

## 「人格」不是一段字符串

很多人第一次写 agent，会把身份、工具说明、项目约定揉进一个巨大的 `system_prompt`。能跑，但难维护：改一个工具用法要翻整段文本；加项目规则会污染全局人设；Skills / MCP 一变，整段缓存失效。

MyCodeAgent 把「发给模型的 system 部分」拆成可独立演进的几层，由 `runtime/prompt_builder.py` 的 `ContextBuilder` 组装。上一篇讲了模型响应怎么归一；这篇讲请求另一侧——**模型还没开口前，它的行为边界从哪来**。

---

## 结论先放这

```
发给 LLM 的 messages =
  [system] Constitution          ← L1：身份、语气、安全、工作方式
  [system] Tool Contracts        ← 各工具的自然语言用法说明
  [system] Project Rules         ← 仓库里的 code_law.md（若有）
  [system] Runtime Signals       ← 运行时通知块（可变）
  [system] Session Memory        ← 跨轮摘要前馈（可选，在 ContextEngine 注入）
  + history 投影后的 user/assistant/tool/...
```

四层（加可选的 session memory）合在一起，才是 agent 的「人格 + 能力说明书 + 项目记忆」。

和 Function Calling 的关系要先分清：

| 通道 | 内容 | 走哪 |
|------|------|------|
| Tool Contracts（本文） | 自然语言：怎么用、什么时候用、响应长什么样 | `prompts/tools_prompts/*.py` → system 消息 |
| tools schema（下一篇） | JSON Schema：参数名、类型、required | `ToolRegistry.get_openai_tools()` → API 的 `tools=` |

模型既读「说明书」，又拿「机器可校验的参数表」。两套信息互补，不是重复粘贴。

---

## 组装入口：从 Model View 回推

每一步 ReAct 构建上下文时，`ContextEngine.build_model_view()` 做的事可以简化成：

```python
# runtime/context/engine.py — 拼装顺序（简化）
system_messages = self.context_builder.get_system_messages()
# 可选：session memory 再插一条 system
messages = system_messages + dynamic_messages + history_messages
```

`history_messages` 是投影后的对话事实；**人格层全部来自 `context_builder`**。所以读 prompt 组装，入口就是 `ContextBuilder.get_prompt_assembly()`。

---

## 四层 system 消息

```python
# runtime/prompt_builder.py — get_prompt_assembly() 核心分层

# ① Constitution：身份与行为宪法（L1_system_prompt.py）
constitution_messages = [{"role": "system", "content": constitution_text}]

# ② Tool Contracts：扫 prompts/tools_prompts/*.py，拼成一篇工具说明书
tool_contract_messages = [{"role": "system", "content": "# Tool Contracts\n" + ...}]

# ③ Project Rules：读项目根目录 code_law.md / CODE_LAW.md
project_rule_messages = [{"role": "system", "content": "# Project Rules (CODE_LAW)\n" + ...}]

# ④ Runtime Signals：set_runtime_system_blocks() 注入的临时通知
runtime_signal_messages = [{"role": "system", "content": block}, ...]

stable = constitution + tool_contracts + project_rules
all_system = stable + runtime_signals
```

### ① Constitution：默认人设

来源：`prompts/agents_prompts/L1_system_prompt.py` 里的 `system_prompt`。

它规定的是**跨项目稳定**的东西：你是 CLI coding agent、用 Function Calling 而不是明文 Action、怎么用 Todo / Skills / Task、语气要短、安全拒绝恶意代码、改代码前先摸清约定……

注意一个演进痕迹：L1 文本末尾还有 `{tools}` 占位符，组装时会被**刻意清空**：

```python
constitution_text = self._load_system_prompt().replace("{tools}", "").strip()
```

工具细节已迁到独立的 Tool Contracts 层，避免人设文件和工具列表耦死。CLI 的 `--system` 仍可通过 `system_prompt_override` 整段替换 Constitution，方便实验。

### ② Tool Contracts：工具的「说明书」

`_load_tool_prompts()` 扫描 `prompts/tools_prompts/`，加载每个 `*_prompt` 字符串，按文件名排序拼接。例如 Read 的 prompt 会告诉模型：用行号格式、怎么分页、`status=partial` 意味着什么、别用 `cat`。

两个动态插槽：

- **Skills**：Skill 工具 prompt 里有 `{{available_skills}}`，运行时替换成当前技能目录摘要（每轮 `_prepare_run` 可能刷新）
- **MCP / 熔断**：MCP 工具说明追加为 `## MCP Tools`；被熔断的工具追加 `## Disabled Tools`，减少无效调用

`tool_prompt_allowlist` 只加载已注册工具对应的 prompt，避免「注册表里没有的工具还出现在说明书里」。

### ③ Project Rules：这个仓库特有的记忆

若项目根有 `code_law.md` 或 `CODE_LAW.md`，整文件注入为 Project Rules。典型内容：目录结构、常用测试/构建命令、本仓库不变量。

Constitution 教「怎么当 coding agent」；CODE_LAW 教「在这个仓库里具体怎么干」。换项目只换这一层，人设不用改。

加载按 **mtime + 内容 hash** 缓存：你改了 `code_law.md`，下一轮组装会自动失效重读。

### ④ Runtime Signals：不污染 user 轮次的通知

通过 `set_runtime_system_blocks()` 注入。设计意图是：运行时提醒走 system，而不是伪装成用户又说了一句话——历史轮次语义更干净。

Session Memory 不在 `PromptAssembly` 的四层里，而在 `build_model_view` 里按字符预算再插一条 system。它是「跨 run 的摘要前馈」，和稳定人设分层存放。

---

## 指纹与缓存：改哪一层，一眼能看出来

每层算一个 SHA256 fingerprint，稳定三层再合成 `system_fingerprint`：

```
constitution_fp
tool_contracts_fp
project_rules_fp
        ↓
 system_fingerprint   ← 稳定层是否变化
runtime_signals_fp    ← 可变层是否变化
```

用途有二：

1. **缓存**：fingerprint 没变就复用 `_cached_assembly`，避免每步重扫磁盘、重拼大段文本
2. **可观测**：`trace_model_request_state()` 把各层 fingerprint 打进 trace，并标出 `changed_layers`——调试「这轮模型怎么突然换风格了」时，先看是人设、工具说明还是项目规则变了

Skills / MCP / runtime blocks 一更新，对应 setter 会把 `_cached_assembly = None`，强制下一轮重装。

---

## 最终长什么样

一次典型请求的 messages 头部类似：

```
[0] system  Constitution（身份与政策）
[1] system  Tool Contracts（Read/Edit/Bash/... 说明书）
[2] system  Project Rules（code_law.md）
[3] system  （可选）Session Memory / Runtime Signals
[4] user    用户问题
[5] assistant + tool_calls
[6] tool    观测结果
...
```

这是 **Message List 自然累积**，不是旧式「把 Thought/Action 拼进一个 scratchpad 字符串」。对话事实在 history；行为约束在 system 层；两者在 `build_model_view` 才汇合。

---

## 设计亮点

1. **稳定 vs 可变分离**：Constitution / Contracts / CODE_LAW 可缓存；Skills、runtime、session memory 可热更新
2. **人设与项目解耦**：换仓库主要换 CODE_LAW，不改 L1
3. **说明书 ≠ schema**：自然语言教策略，JSON Schema 管参数——下一篇讲后者怎么进 `tools=`
4. **fingerprint 可审计**：prompt 漂移变成可对比的哈希，而不是「感觉模型变了」

---

## 小结

| 层 | 来源 | 变不变 |
|----|------|--------|
| Constitution | `L1_system_prompt.py` 或 `--system` | 相对稳定 |
| Tool Contracts | `tools_prompts/*.py` + Skills/MCP/熔断 | 工具或技能变化时变 |
| Project Rules | `code_law.md` | 随仓库文档变 |
| Runtime Signals / Session Memory | 运行时注入 | 高频可变 |

agent 的「人格」= Constitution 的语气与政策 + Tool Contracts 的能力边界 + Project Rules 的本地记忆。组装器的工作，是把这三样（外加少量运行时信号）稳定地放进每一步的 Model View 头部。

下一篇进入工具系统：模型怎么通过 Function Calling 真正「看见」可调用的工具，以及参数如何走进执行管道。

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
