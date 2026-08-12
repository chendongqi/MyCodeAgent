# 学习计划：用 MyCodeAgent 掌握 Agent Harness 工程

> **目标读者**：有一定 agent 开发经验（用过 LangChain 等框架），想理解底层是怎么跑的  
> **学习方式**：直接看代码边看边讲，配合流程图建立直觉  
> **预计总时长**：4～6 小时（分多次学习）

---

## 核心认知：什么是 Agent Harness？

**Agent Harness = LLM + 工具 + 循环，其余都是工程保障。**

你用 LangChain 时调用的是框架的高层 API，底层的 LLM 调用、工具执行、历史管理都被隐藏了。MyCodeAgent 把这些全部暴露出来，代码清晰、无魔法，是理解 agent 工程的绝佳教材。

> "harness"这个词来自赛马——把马力约束在正确的方向上。Agent harness 的作用类似：把 LLM 的能力约束在工具执行、安全边界、上下文预算之内，让它能干实际的事。

---

## 学习路线（6 个模块）

```
模块 0：鸟瞰图          ← 先建地图，不迷路
    ↓
模块 1：LLM 接口层      ← 最底层，LLM 调用与响应归一化
    ↓
模块 2：工具系统        ← 工具协议、注册、执行管道
    ↓
模块 3：主循环 ★        ← 核心！ReAct 循环的工程实现
    ↓
模块 4：上下文工程      ← 为什么不能直接喂全量历史
    ↓
模块 5：组装与启动      ← 所有零件怎么 wire 在一起
```

---

## 各模块详细说明

### 模块 0：鸟瞰图
**文档**：[01_overview.md](./01_overview.md)  
**目标**：在脑子里建立整体地图，后续学习不迷路  
**关键文件**：`AGENT.md`、`runtime/state.py`  
**核心内容**：
- 整体数据流：用户输入 → host 组装 → loop 驱动 → 输出
- `LoopState` 不可变状态机设计
- `TransitionReason` + `TerminalReason` 枚举的意义

---

### 模块 1：LLM 接口层
**文档**：02_llm_interface.md（待写）  
**目标**：理解 harness 如何屏蔽多 provider 差异，统一拿到 text 和 tool_calls  
**关键文件**：`core/llm.py`、`core/openai_compat.py`、`core/config.py`  
**核心内容**：
- `HelloAgentsLLM` 类的职责边界
- 5 个响应提取函数（content / tool_calls / usage / reasoning / meta）
- 为什么要做响应归一化：DeepSeek/Qwen/Kimi 格式差异

---

### 模块 2：工具系统
**文档**：03_tool_system.md（待写）  
**目标**：理解工具协议、注册机制、执行管道，以及熔断器  
**关键文件**：`tools/base.py`、`tools/registry.py`、`tools/executor.py`、`tools/orchestrator.py`  
**核心内容**：
- 标准信封结构 `{status, data, text, stats, context}`
- ToolRegistry：注册 → 生成 OpenAI function schema → 查找工具
- 执行管道：权限检查 → `tool.run()` → 熔断器记录
- 乐观锁冲突检测（Read mtime → Edit 前验证）

---

### 模块 3：主循环 ★
**文档**：04_main_loop.md（待写）  
**目标**：把 loop 的每一步走一遍，彻底理解 ReAct 模式的工程实现  
**关键文件**：`runtime/loop.py`、`runtime/completion.py`、`runtime/state.py`  
**核心内容**：
- `RuntimeRunner.run()` → `_react_loop()` 完整执行序列
- 有工具调用 vs 无工具调用的分支逻辑
- 完成门三态：PASS / FAIL / UNVERIFIED
- 终止条件枚举：COMPLETED / MAX_STEPS / TOKEN_BUDGET / MODEL_ERROR

---

### 模块 4：上下文工程
**文档**：05_context_engineering.md（待写）  
**目标**：理解为什么不能直接把全部历史喂给模型，以及 harness 怎么处理  
**关键文件**：`runtime/history.py`、`runtime/context/engine.py`、`runtime/context/compact.py`、`runtime/transcript.py`  
**核心内容**：
- History（完整事实）vs Model View（给 LLM 看的投影）
- token 预算计算与超阈值压缩
- 压缩不破坏历史：只压投影，原始消息永远保留
- transcript 持久化：崩溃恢复的真相来源

---

### 模块 5：组装与启动
**文档**：06_bootstrap_assembly.md（待写）  
**目标**：看清楚一个 agent 实例是怎么从零组装出来的  
**关键文件**：`runtime/host.py`、`runtime/factory.py`、`app/bootstrap.py`、`app/cli.py`  
**核心内容**：
- `CodeAgent` 作为依赖容器（不是通过继承，而是组合）
- factory 模式：根据 Config 决定启用哪些扩展
- 内置工具的注册过程
- CLI → factory → RuntimeRunner 的完整启动链

---

## 每个模块的讲解模式

1. **一句话说清楚这层干嘛的**
2. **贴核心代码段（不超过 50 行）边读边讲**
3. **指出一个"反直觉"或"设计亮点"**
4. **流程图**——可视化数据或控制流

---

## 最终验证：你真的掌握了吗？

```bash
cd /path/to/MyCodeAgent
# 配好 .env，然后跑起来
python main.py

# 查看 trace 文件（每次循环都有一条记录）
cat memory/traces/*.jsonl | python -m json.tool | head -100
```

**检验标准**：能说清楚每一条 trace event 对应 `runtime/loop.py` 哪一行，就算真的掌握了。

---

## 关键设计原则（贯穿所有模块）

| 原则 | 体现 |
|------|------|
| 单一循环 | 只有一个 `RuntimeRunner`，没有第二个 agent loop |
| 边界清晰 | tools 不导入 runtime，runtime 不直接操作文件 |
| 组合优于继承 | `CodeAgent` 是依赖容器，不是基类 |
| 不可变状态 | `LoopState` frozen dataclass，每次转移产生新对象 |
| 历史永不删除 | 上下文压缩只影响投影视图，不破坏原始历史 |
| 持久化优先 | transcript append-only，崩溃后可恢复 |
