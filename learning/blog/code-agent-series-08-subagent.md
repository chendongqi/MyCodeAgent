---
title: "Code Agent 解剖（08）：一个任务太复杂，怎么拆给子 agent 做？"
date: 2026-08-18
description: "深入 MyCodeAgent 的子 agent 机制：Task 工具、RuntimeProfile 沙箱约束、SubagentLauncher 完整执行链路、结构化结果合约。理解主 agent 如何把探索性任务委派给只读轻量子 agent，以及为什么子 agent 不是「简化版主 agent」而是受严格约束的执行单元。"
tags: [Code Agent, Subagent, Task Tool, ReAct, LLM Agent]
category: AI Engineering
draft: false
---

## 主 agent 为什么需要子 agent

主 agent 的上下文是有限的。一次复杂任务——比如"找出项目里所有涉及认证的代码，整理成报告"——可能需要读几十个文件、跑很多次搜索。这些探索步骤会快速消耗上下文，而且和最终的"生成报告"任务混在一起，增加了压缩和失真的风险。

更根本的问题是：探索性工作（只读、找信息）和生成性工作（写代码、修改文件）的风险等级完全不同。如果能把探索任务单独剥离，放在一个只读沙箱里跑，主 agent 只拿回结果，好处很明显：

- 探索过程不污染主 agent 的上下文
- 子 agent 用轻量模型，降低成本
- 子 agent 不能写文件，即使被诱导也无法造成副作用

MyCodeAgent 的 Task 工具就是这个机制的实现。

---

## 结论先放这

```
主 agent 调 Task 工具
    ↓
TaskTool.run()                     校验参数，构造 TaskRequest
    ↓
_DeferredSubagentLauncher.launch() 代理层，惰性初始化真正的 launcher
    ↓
SubagentLauncher.launch()          真正的执行入口（subagents.py）
    ├─ 1. 查找 RuntimeProfile（工具白名单、步数、token 预算、模型）
    ├─ 2. _select_llm()：优先用 LIGHT_LLM，没配则回退到主模型
    ├─ 3. _create_child_trace()：创建独立 trace 日志
    ├─ 4. 向父 trace 发射 subagent_requested/started 事件
    ├─ 5. _SubagentRuntimeHost：创建独立沙箱（history/context/tool 全新）
    ├─ 6. _render_request()：把任务描述 + 结构化上下文拼成 prompt
    ├─ 7. RuntimeRunner(host).run(prompt) ← 完整 ReAct 循环
    ├─ 8. _child_metrics()：从 trace 事件里提取 terminal_reason/tool_usage/token_usage
    └─ 9. ExploreResult.from_json()：解析并校验子 agent 输出的 JSON
    ↓
SubagentLaunchResult → TaskTool 打包成标准信封，主 agent 拿到摘要继续工作
```

子 agent 跑的是**完整的 ReAct 循环**，和主 agent 用的是同一个 `RuntimeRunner`。不同的是它运行在一个受约束的沙箱里，工具、步数、token 预算都被 `RuntimeProfile` 严格限定。

**调用链中的三层 launch**：代码里有三个 `launch` 方法，容易混淆：

| 位置 | 类型 | 作用 |
|------|------|------|
| `task.py` `TaskLauncher.launch` | Protocol 接口 | 只是接口声明，无实现（`...`） |
| `host.py` `_DeferredSubagentLauncher.launch` | 代理层 | 惰性初始化，转发给真正的 launcher |
| `subagents.py` `SubagentLauncher.launch` | 真正实现 | 创建沙箱、运行子循环、解析结果 |

这三层分离的原因：`TaskTool` 注册时 `SubagentLauncher` 还不能创建（`CodeAgent` 未初始化完），代理层解决了这个时序问题；`Protocol` 接口让 `TaskTool` 不直接依赖 `SubagentLauncher`，两者解耦。

**TaskRequest 和 SubagentRequest 的关系**：两者是字段完全相同的镜像 dataclass。

```python
# task.py                          # subagents.py
class TaskRequest:                  class SubagentRequest:
    profile_name: str                   profile_name: str
    task: str                           task: str
    model_choice: str | None            model_choice: str | None
    structured_context: dict            structured_context: dict
    parent_session_id: str | None       parent_session_id: str | None
    parent_run_id: str | None           parent_run_id: str | None
```

`task.py` 不能直接 import `subagents.py`（会循环依赖），所以在本文件定义了一个镜像。`_DeferredSubagentLauncher.launch(request)` 把 `TaskRequest` 直接传给 `SubagentLauncher.launch(request: SubagentRequest)`，Python 运行时只看字段值不做类型校验，两者完全兼容。

`SubagentLauncher.launch()` 第一件事是 `RUNTIME_PROFILES.get(request.profile_name)`，`profile_name="explore"` 就对应 `EXPLORE_PROFILE`，沙箱约束从这里开始生效。

---

## Task 工具：入口、参数校验、完整调用链

```python
# tools/builtin/task.py — TaskTool.run()（简化）
def run(self, parameters):
    description = parameters.get("description")  # 任务的简短描述（给主 agent 看的标签）
    prompt = parameters.get("prompt")            # 自包含的探索指令（子 agent 执行的完整任务）
    profile = parameters.get("subagent_type")    # 当前只支持 "explore"
    model = parameters.get("model", "light")     # "light"（轻量模型）或 "main"

    # 参数校验通过后，构造 TaskRequest 委派出去
    launched = self._launcher.launch(
        TaskRequest(
            profile_name="explore",   # ← 这个字符串决定后面走哪条路
            task=f"{description}\n\n{prompt}",
            model_choice=model,
        )
    )
```

`self._launcher` 是什么？在 `host.py` 里注册 TaskTool 时传入的：

```python
# runtime/host.py — _initialize_runtime_components()
self.tool_registry.register_tool(
    TaskTool(
        project_root=...,
        launcher=self._DeferredSubagentLauncher(self._get_subagent_launcher),
        #                ↑ 这就是 self._launcher 的真实身份
    )
)
```

所以 `self._launcher.launch(request)` 实际走的完整调用链是：

```
TaskTool.run()
  self._launcher.launch(TaskRequest(profile_name="explore", ...))
    ↓  [host.py _DeferredSubagentLauncher.launch]
    self._get_launcher()          ← 惰性初始化，首次调用才真正创建 SubagentLauncher
      ↓  [host.py _get_subagent_launcher]
      create_subagent_launcher(host)   ← factory.py，把主 agent 的 llm/registry 传进去
        → SubagentLauncher(main_llm, tool_registry, ...)
    SubagentLauncher.launch(request)   ← subagents.py，真正的实现
      ↓
      profile = RUNTIME_PROFILES.get(request.profile_name)
      #         ↑ "explore" → EXPLORE_PROFILE（工具白名单、步数、token 预算全在这里）
      ↓
      创建沙箱、跑 ReAct 循环、解析结果
```

`profile_name="explore"` 这个字符串就是从 TaskTool 到 `EXPLORE_PROFILE` 的连接点——`SubagentLauncher.launch()` 第一件事就是用它在 `RUNTIME_PROFILES` 字典里查对应的 profile。

`prompt` 必须是自包含的——子 agent 看不到主 agent 的历史，只有这一段指令。模型写 Task 调用时，必须把所有背景信息显式写进 `prompt`，不能依赖隐式上下文传递。

---

## RuntimeProfile：沙箱约束的定义

子 agent 的所有约束都在 `RuntimeProfile` 里声明：

```python
# runtime/subagents.py
EXPLORE_PROFILE = RuntimeProfile(
    name="explore",
    system_prompt=EXPLORE_SYSTEM_PROMPT,         # 固定系统提示词，要求返回 JSON
    tool_allowlist={"Read", "Grep", "Glob"},     # 只读工具，不能写文件、不能执行命令
    max_steps=12,                                # 最多 12 步，防止无限循环
    context_token_budget=16_000,                 # 单次上下文窗口上限（主 agent 是 128k）
    total_token_budget=32_000,                   # 累计 token 上限，超出强制终止
    model_choice="light",                        # 默认用轻量模型降低成本
    result_contract="ExploreResult",             # 必须返回符合此合约的 JSON
)
```

`RuntimeProfile.__post_init__` 有硬约束，Task 工具和 Edit/Bash 不允许出现在 `tool_allowlist` 里：

```python
if self.recursive_subagents or "Task" in self.tool_allowlist:
    raise ValueError("formal subagent profiles cannot recurse")
forbidden = {"Edit", "Bash"}
if forbidden & self.tool_allowlist:
    raise ValueError("formal subagent profiles must be strictly read-only")
```

**为什么要在 profile 里写死，而不是运行时动态控制？**

Profile 是声明性的约束，不依赖运行时状态。主 agent 的历史、当前步数、context 压缩状态都不会影响子 agent 能用哪些工具——子 agent 的边界在代码里固定好了，无法被提示词诱导绕过。

---

## SubagentLauncher.launch()：沙箱创建与运行细节

### 第一步：选模型和 trace

```python
# subagents.py — SubagentLauncher.launch()
llm, model_choice = self._select_llm(requested_model)
# _select_llm：优先用 light_llm（从 LIGHT_LLM_* 环境变量创建）
# 没配 LIGHT_LLM_MODEL_ID 则回退到主 agent 的 main_llm

child_trace = self._create_child_trace()
# 独立的 trace JSONL 文件，session_id 以 "child-" 开头
# 和主 agent trace 分离，但 launch() 里发射的父子事件通过 parent_session_id 关联
```

### 第二步：创建沙箱 _SubagentRuntimeHost

`_SubagentRuntimeHost` 和主 agent 的 `CodeAgent` 结构完全对齐——都有 `history_manager`、`context_engine`、`tool_executor`、`tool_orchestrator`——但所有状态全部新建，主 agent 的历史不会流入子 agent。

`RuntimeRunner` 通过鸭子类型复用：它只依赖 host 上的属性，不要求继承任何基类，所以 `_SubagentRuntimeHost` 不需要继承 `CodeAgent`，只要有同名属性就能跑同一套循环。

沙箱里的关键约束：

```python
# 1. config：用 profile 预算覆盖关键字段
self.config = Config.from_env().model_copy(update={
    "context_window": profile.context_token_budget,  # explore=16000，远小于主 agent 的 128k
})
self.max_steps = profile.max_steps          # explore=12，硬性步数上限
self.max_total_tokens = profile.total_token_budget  # explore=32000，超出强制终止

# 2. registry：只含白名单工具（build_registry 过滤）
# 主 agent 的所有工具中，只有 Read/Grep/Glob 会进子 agent 的 registry

# 3. ContextBuilder：系统提示词固定为 profile.system_prompt（要求返回 JSON）
#    不加载 MCP 工具提示词和 Skills，不暴露非白名单工具的 schema

# 4. 上下文压缩：用 _summarize_child_messages（纯截断拼接，不启 LLM）
#    原因：子 agent 预算小，启 LLM 压缩太贵，简单截断足够

# 5. 权限双重保障：
permission_context = PermissionContext(runtime_mode="readonly_subagent")
# RiskClassifier 在此模式下对 Edit/Bash/Task 直接 DENY
# 即使这些工具意外出现在 registry 里也执行不了——白名单 + 权限门双重拦截

# 6. completion_verifier：_StructuredResultCompletionVerifier（子 agent 专用）
#    检查输出是否符合 result_contract JSON 格式，不符合则触发 FAIL 反馈让子 agent 重输
```

### 第三步：拼 prompt 并运行

```python
# _render_request：把任务文本和结构化上下文拼成 prompt
# 输出格式：task 文本 + "\n\nStructured context:\n" + JSON
# 这是子 agent 唯一能看到的上下文——它看不到主 agent 的历史
prompt = _render_request(request)

# 和主 agent 完全相同的 RuntimeRunner，跑完整 ReAct 循环
# 子 agent 的系统提示词要求它最终输出一个 JSON 对象
raw_result = RuntimeRunner(host).run(prompt)
```

### 第四步：提取指标，解析结果

```python
# _child_metrics：遍历 child_trace.events，统计三类信息：
# terminal_reason：子 agent 怎么结束的（completed/max_steps/token_budget 等）
# tool_usage：每个工具调用了几次
# token_usage：累计消耗 token 数
terminal_reason, tool_usage, token_usage = _child_metrics(child_trace.events)

# 子 agent 必须以 completed/completed_unverified 结束，其他原因都当失败
if terminal_reason not in {"completed", "completed_unverified"}:
    raise ValueError(f"child terminal reason: {terminal_reason}")

# ExploreResult.from_json 严格校验：
# - status 只能是 completed/partial
# - summary 不能为空
# - 有 Markdown fence（```）直接抛异常
# 不合法则 launch() 捕获异常，返回 status=FAILED
structured = ExploreResult.from_json(raw_result, tool_usage=tool_usage, ...)
```

---

## 结构化结果合约

子 agent 的系统提示词要求它**只返回 JSON，不返回 Markdown**：

```
You are an Explore Agent.
Inspect the repository with read-only tools and return exactly one JSON object:
{"status":"completed|partial","summary":"...","findings":["..."],
"evidence":["relative/path.py:line"],"unresolved_questions":["..."]}.
Do not use markdown fences.
```

`RuntimeRunner` 运行结束后，`launch()` 解析这段 JSON：

```python
raw_result = RuntimeRunner(host).run(prompt)

structured = ExploreResult.from_json(
    raw_result,
    tool_usage=tool_usage,
    terminal_reason=terminal_reason,
)
```

`ExploreResult.from_json()` 严格校验：`status` 只能是 `completed` 或 `partial`，`summary` 必须有内容，否则抛异常，`launch()` 捕获后返回 `status=FAILED`。

**为什么要结构化 JSON 而不是自然语言？**

主 agent 拿到子 agent 的结果后，需要机器可靠地提取 `summary`（给模型看的摘要）、`findings`（具体发现）、`evidence`（代码位置证据）。自然语言需要主 agent 再解析一次，增加了失真的可能性，也没法做格式校验。JSON 合约把"子 agent 必须提供什么"写死在代码里。

---

## 轻量模型与 LIGHT_LLM

子 agent 默认 `model_choice="light"`，`_select_llm()` 会尝试用 `light_llm`：

```python
def _select_llm(self, requested_model):
    if requested_model == "light":
        if self.light_llm is None:
            self.light_llm = _create_light_llm()   # 从 LIGHT_LLM_* 环境变量创建
        if self.light_llm is not None:
            return self.light_llm, "light"
    return self.main_llm, "main"   # 没配轻量模型就回退到主模型
```

`LIGHT_LLM_*` 环境变量在 `.env` 里配置（`LIGHT_LLM_PROVIDER`、`LIGHT_LLM_MODEL_ID` 等），没配就回退到主 agent 的模型。探索任务通常只需要读代码、搜索、总结，轻量模型足够胜任，成本可以降一个量级。

---

## 主 agent 拿到结果后看到什么

`TaskTool.run()` 最终返回标准信封：

```python
return self.success_result(
    data={
        "status": "completed",
        "profile": "explore",
        "result": {
            "summary": "找到了 3 处认证相关代码...",
            "findings": ["auth/login.py:45 — JWT 验证", ...],
            "evidence": ["auth/login.py:45", ...],
        },
    },
    text=result.summary,   # ← 给模型看的摘要，直接进 observation
    extra_stats={
        "tool_calls": 8,
        "token_usage": 12000,
        "model": "light",
    },
)
```

`text` 字段是摘要，直接作为 observation 追加到主 agent 的历史里。模型看到这段摘要，决定下一步该做什么。完整的 `data` 里有 `findings` 和 `evidence`，模型可以在后续步骤里引用具体的代码位置。

---

## 当前的限制

Task 工具目前只支持 `subagent_type="explore"`，传入其他值直接返回参数错误。代码里有 `VERIFICATION_PROFILE`，但没有对应的 Task 入口——验证子 agent 只由主循环的完成门直接调用（`--enable-verification-agent` 开启后）。

多轮对话中每次 Task 调用都是独立的——子 agent 没有跨次调用的记忆，每次都是新的 `_SubagentRuntimeHost`。如果需要分多次探索、逐步积累，需要主 agent 自己把上次的发现写进下次调用的 `prompt`。

---

## 设计亮点

1. **复用同一个 RuntimeRunner**：子 agent 和主 agent 跑完全一样的 ReAct 循环，不是简化版，是用 profile 约束了能力边界
2. **沙箱是声明性的**：profile 里写死工具白名单，运行时无法绕过，不依赖提示词的守规矩
3. **结构化合约**：JSON 结果让主 agent 可靠地提取信息，不需要再解析自然语言
4. **轻量模型降成本**：探索任务不需要最强模型，配置 LIGHT_LLM 可以省一大笔开销
5. **独立 trace**：子 agent 有自己的 trace 文件，父子事件通过 `parent_session_id` 关联，可以单独分析子任务执行情况

---

## 小结

| 机制 | 作用 |
|------|------|
| `TaskTool` | 参数校验 + 委派入口 |
| `RuntimeProfile` | 声明沙箱约束（工具、步数、token、模型） |
| `SubagentLauncher` | 创建沙箱、选模型、运行子循环、解析结果 |
| `_SubagentRuntimeHost` | 独立的 history/context/tool，与主 agent 完全隔离 |
| `ExploreResult` | JSON 结构化合约，保证结果可机器解析 |
| `readonly_subagent` 权限模式 | 双重保障：白名单 + 权限分类器双重拦截写操作 |

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
