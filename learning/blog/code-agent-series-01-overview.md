---
title: "Code Agent 解剖（01）：用户输入一句话，agent 内部发生了什么？"
date: 2026-08-12
description: "从 python main.py 到回答输出，完整追踪一条用户输入在 MyCodeAgent 中的数据流。覆盖 CLI 入口、依赖组装、ReAct 主循环三个阶段，建立后续深入各模块的整体地图。"
tags: [Code Agent, LLM Agent, ReAct, Agent Harness, Python]
category: AI Engineering
draft: false
---

## 框架把问题藏起来了

用过 LangChain 或 LlamaIndex 的人都有同感：文档说三行代码跑起来一个 agent，代码确实跑起来了，但出问题时完全不知道往哪调。工具调用失败了？上下文截断了？模型没按预期停止？框架把这些全包掉了，你只能猜。

MyCodeAgent 是一个没有框架魔法的本地 coding agent，大约 14000 行 Python，所有核心逻辑都暴露在源码里。本系列用它作为解剖对象，逐层看清 agent 是怎么跑起来的。

第一篇先建整体地图：一句用户输入，从键盘到回答，经过了哪几道关口。

---

## 三个阶段，一张地图

```
python main.py
    │
    ▼
┌──────────────────────────────┐
│  阶段 1：CLI 入口             │  app/cli.py
│  解析参数，决定运行模式        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  阶段 2：依赖组装             │  app/bootstrap.py
│  Config → LLM → 工具 → Agent │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│  阶段 3：ReAct 主循环                                  │  runtime/loop.py
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │  用户输入 → 构建 Model View → 调 LLM          │    │
│  │      ↓                                      │    │
│  │  有 tool_calls？→ 执行工具 → 追加观测结果      │    │
│  │  没有 tool_calls？→ 完成门检查                │    │
│  │      ↓                                      │    │
│  │  通过 → 输出     失败 → 注入反馈 → 继续循环   │    │
│  └─────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

三个阶段各有一个核心职责：CLI 决定"怎么跑"，bootstrap 决定"用什么跑"，loop 决定"跑出来什么"。

---

## 阶段 1：CLI 入口

`main.py` 只有一行：

```python
# main.py
from app.cli import main

if __name__ == "__main__":
    main()
```

真正的逻辑在 `app/cli.py` 的 `main()` 函数里。它做两件事：判断运行模式，然后把控制权交给 bootstrap。

```python
# app/cli.py — main() 的核心判断
args = parser.parse_args()

# -p 参数存在 → 一次性模式（跑完即退出，适合脚本调用）
# 没有 -p    → 交互模式（while True 循环，等待用户输入）
one_shot = getattr(args, "print_prompt", None) is not None

runtime = build_runtime(args, agent_class=...)
```

两种模式的区别只是外壳：一次性模式跑完就 `raise SystemExit`，交互模式进入 `while True` 循环读用户输入。核心的 agent 逻辑在两种模式下完全一样。

交互模式用 `prompt_toolkit` 读输入：

```python
# cli.py
session = PromptSession(history=FileHistory(".chat_history"))

user_input = session.prompt(
    HTML("<user>user</user> <arrow>➜</arrow> "),
    style=prompt_style,
).strip()
```

`session.prompt()` 是阻塞调用，底层把终端切换到 raw mode（逐字符读，不等回车），渲染彩色提示符，并处理光标移动、退格、上下键翻历史。按回车后返回完整字符串，恢复终端正常模式。`FileHistory` 把每次输入持久化到 `.chat_history` 文件，重启后按上键仍能翻出历史记录。

这个项目同时用了 `prompt_toolkit` 和 `rich` 两个库，分工明确：`prompt_toolkit` 负责输入，`rich` 负责输出（Panel、Markdown 渲染、spinner）。这是 Python CLI 工具的常见组合，Claude Code 本身的交互界面也是这个技术栈。

交互模式有一个值得注意的细节——`agent_kwargs_factory`：

```python
# 只在交互模式下才需要 EnhancedUI
# 但 EnhancedUI 依赖 llm.model/llm.provider，而 llm 对象在 bootstrap 里才创建
# 解决方法：把"创建 UI"打包成 lambda，等 bootstrap 把 llm 准备好后再调用
runtime_kwargs["agent_kwargs_factory"] = lambda config, llm, project_root: {
    "ui": EnhancedUI(model=llm.model, provider=llm.provider, ...)
}
```

这是依赖注入的标准做法：调用方不知道被注入物的细节，只提供一个工厂函数，让 bootstrap 在合适的时机调用。

---

## 阶段 2：依赖组装

`app/bootstrap.py` 的 `build_runtime()` 按固定顺序组装所有依赖，顺序不能乱，因为后面的步骤依赖前面的结果：

```python
# app/bootstrap.py — build_runtime() 核心步骤

resolved_project_root = resolve_project_root(selected_project_root)  # --cwd 或当前目录
config = Config.from_env()                                            # 从 .env 加载
llm = HelloAgentsLLM(model=args.model, api_key=args.api_key, ...)    # 命令行参数优先
tool_registry = ToolRegistry()                                        # 空注册表，CodeAgent 内部填充
agent = CodeAgent(llm=llm, tool_registry=tool_registry, config=config, ...)
```

"命令行参数优先"这句话值得展开，不同参数的兜底逻辑并不一样。

argparse 没有收到某个参数时，`args.xxx` 的值是 `None`（因为 `default=None`）。`None` 传进 `HelloAgentsLLM` 后，每类参数各有不同的处理方式：

**`model`、`timeout`：`or` 直接读环境变量**

```python
# core/llm.py
self.model   = model   or self._get_env("LLM_MODEL_ID")
self.timeout = timeout or int(self._get_env("LLM_TIMEOUT", "120"))
```

`None or xxx` 在 Python 里走右边，传 `None` 等于"去环境变量找"。

**`api_key`、`provider`、`base_url`：专门的 resolve 方法**

```python
self.provider = self._resolve_provider(provider, api_key, base_url)
self.api_key, resolved_base_url = self._resolve_credentials(api_key, base_url)
```

这三个涉及多 provider 自动探测和 profile 表查找，逻辑更复杂，单独抽了方法，最终也是 `None` → 查环境变量 → 查 `PROVIDER_PROFILES` 默认表。

**`temperature`：在 bootstrap 层提前决定好**

```python
# bootstrap.py
temperature=(
    getattr(args, "temperature", None)
    if getattr(args, "temperature", None) is not None
    else config.temperature   # config 已经从 .env 读好了
),
```

`temperature` 的类型签名是 `float`，不是 `Optional[float]`。如果把 `None` 传进去，后续 `float(None)` 会抛异常。另外它不从环境变量读，来源只有命令行和 `config`，所以在 bootstrap 层就必须决定好传哪个值。

三种模式汇总：

```
model / timeout       → None 传入，LLM 内部 or 读环境变量
api_key / provider    → None 传入，LLM 内部 resolve 方法处理
temperature           → bootstrap 层提前用三元表达式决定好，不传 None
```

`CodeAgent` 拿到这些依赖后，在 `_initialize_runtime_components()` 里继续把内部子组件装配起来：

```python
# runtime/host.py — CodeAgent._initialize_runtime_components()

# ① 历史管理 + 上下文引擎（给模型看多少历史）
build_runtime_context(self)

# ② 持久化（trace 日志 + transcript 崩溃恢复）
build_runtime_persistence(self, ...)

# ③ 工具执行器（权限检查 + 实际调用）
self.tool_executor = ToolExecutor(self.tool_registry, ...)

# ④ 工具编排器（只读工具并发，写操作串行）
self.tool_orchestrator = ToolOrchestrator(self)

# ⑤ ReAct 主循环驱动器 ← 真正干活的地方
self.runner = RuntimeRunner(self)
```

注意最后一行：`RuntimeRunner(self)` 把整个 `CodeAgent` 作为参数传进去。Runner 需要访问 agent 上的所有属性（llm、tool_registry、history_manager...），`CodeAgent` 在这里是一个**依赖容器**，不是业务逻辑的执行者。

---

## 阶段 3：ReAct 主循环

用户输入从键盘到 `RuntimeRunner` 经过了四层：

```
session.prompt()                   # cli.py — 阻塞读用户输入
  → run_interactive_turn()         # cli.py — 捕获 Ctrl+C，不让中断泄漏到外层
    → RichConsoleCodeAgent.run()   # cli.py — 控制 UI spinner 的开关
      → CodeAgent.run()            # host.py — 只有一行，转发给 runner
        → RuntimeRunner.run()      # loop.py — 真正开始 ReAct 循环
```

`RichConsoleCodeAgent` 是 `CodeAgent` 的子类，只在交互模式下使用，重写了 `run()` 和 `_execute_tool()` 来插入 spinner、工具调用树等 UI 渲染逻辑。一次性模式（`-p` 参数）直接用裸 `CodeAgent`，没有这层开销。这种设计让 `CodeAgent` 本身保持干净，不耦合任何 UI 代码。

`CodeAgent.run()` 本身只有一行：

```python
# runtime/host.py
def run(self, input_text, **kwargs):
    return self.runner.run(input_text, **kwargs)
```

真正的逻辑全在 `RuntimeRunner` 里，`CodeAgent` 只是依赖容器，不执行业务逻辑。

然后进入 `_react_loop()`：

```python
# runtime/loop.py — _react_loop() 简化版

for step in range(1, host.max_steps + 1):

    # 1. 构建本轮的 Model View（完整历史的有界投影）
    state, tools_schema, messages = self._prepare_step_context(...)

    # 2. 调 LLM，拿回原始响应
    raw_response = host.llm.invoke_raw(messages, tools=tools_schema)

    # 3. 从响应里提取 tool_calls 和文字内容
    tool_calls = extract_tool_calls(raw_response)
    response_text = extract_response_content(raw_response)

    # 4. 有工具调用 → 执行工具 → 把结果追加到历史 → 继续循环
    if tool_calls:
        observations = host.tool_orchestrator.run(tool_calls, step=step, ...)
        # 追加 assistant 消息 + tool result 消息到历史
        continue

    # 5. 没有工具调用 → 交给完成门判断
    verdict = host.completion_verifier.evaluate(response_text, ...)
    if verdict.verdict == CompletionGateVerdict.PASS:
        return response_text          # ← 正常出口

    # 6. 完成门拦截 → 把反馈注入历史 → 让模型再跑一轮
    self._append_user_message(verdict.blocking_feedback, ...)
    continue
```

ReAct 是"Reasoning + Acting"的缩写，对应 loop 里的两个分支：有 `tool_calls` 是 Acting（执行动作），没有 `tool_calls` 是 Reasoning（输出思考/回答）。

**几个容易被忽视的设计决策**：

1. **Model View 不等于历史**：每轮传给 LLM 的不是全量 `history_manager` 里的消息，而是经过 `build_model_view()` 投影出来的有界子集。历史永远完整，给模型看的是经过 token 预算控制的视图。

2. **完成门有反馈回路**：loop 不是"没有 tool_calls 就退出"，而是先过完成门。门拦下来时，会把"todo 列表还有未完成项"这类信息注入成 user 消息，让模型重新思考，最多重试 2 次。

3. **状态机是不可变的**：`LoopState` 是 `frozen=True` 的 dataclass，每次状态转移都调 `.next()` 返回新对象。任何时刻的状态都是独立的快照，方便 trace 和崩溃恢复。

---

## 三层职责划分

```
cli.py          决定怎么跑（交互 vs 一次性，UI 层）
bootstrap.py    决定用什么跑（依赖组装，工厂层）
loop.py         决定跑出来什么（ReAct 逻辑，执行层）
```

这三层之间的边界非常清晰：cli 不知道 LLM 怎么调用，bootstrap 不知道循环怎么跑，loop 不知道 UI 怎么渲染。每层只做自己的事。

后续几篇会逐层深入：LLM 接口层怎么统一对接多个 provider，工具协议是怎么设计的，ReAct 循环里的完成门和状态机具体是怎么工作的，上下文压缩是怎么触发的。这篇建立的这张地图，就是定位这些内容的坐标系。

---

## 小结

| 阶段 | 核心文件 | 做了什么 |
|------|---------|---------|
| CLI 入口 | `app/cli.py` | 解析参数，决定交互/一次性模式，把依赖组装委托给 bootstrap |
| 依赖组装 | `app/bootstrap.py` | 按序创建 Config → LLM → ToolRegistry → CodeAgent |
| ReAct 主循环 | `runtime/loop.py` | 构建 Model View → 调 LLM → 执行工具 → 完成门 → 输出 |

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
