# Agent Harness 工程学习笔记

> 通过 MyCodeAgent 项目系统掌握 agent harness 的设计与实现。

## 学习目标

掌握从 LLM 调用到 agent 完整跑起来的全链路——不依赖框架魔法，理解每一个零件的设计意图。

## 文档索引

### 教材（我写）

| 文件 | 内容 | 状态 |
|------|------|------|
| [00_learning_plan.md](./00_learning_plan.md) | 完整学习计划（路线图 + 模块说明） | ✅ 已完成 |
| [01_overview.md](./01_overview.md) | 模块 0：鸟瞰图——整体架构与数据流 | ✅ 已完成 |
| 02_llm_interface.md | 模块 1：LLM 接口层 | 🔜 待写 |
| 03_tool_system.md | 模块 2：工具系统 | 🔜 待写 |
| 04_main_loop.md | 模块 3：主循环（ReAct 核心） | 🔜 待写 |
| 05_context_engineering.md | 模块 4：上下文工程 | 🔜 待写 |
| 06_bootstrap_assembly.md | 模块 5：组装与启动 | 🔜 待写 |

### 笔记（你写）

> 复制 [_notes_template.md](./_notes_template.md) 开始写，文件名加 `_notes` 后缀。

| 文件 | 对应模块 | 状态 |
|------|---------|------|
| 01_overview_notes.md | 模块 0：鸟瞰图 | 📝 待填写 |
| 02_llm_interface_notes.md | 模块 1：LLM 接口层 | 📝 待填写 |
| 03_tool_system_notes.md | 模块 2：工具系统 | 📝 待填写 |
| 04_main_loop_notes.md | 模块 3：主循环 | 📝 待填写 |
| 05_context_engineering_notes.md | 模块 4：上下文工程 | 📝 待填写 |
| 06_bootstrap_assembly_notes.md | 模块 5：组装与启动 | 📝 待填写 |

## 阅读建议

按编号顺序阅读。每篇文档结构一致：
1. **一句话定义**——这层干什么的
2. **核心代码**——直接看关键代码段，边读边理解
3. **设计亮点**——反直觉或值得记忆的设计决策
4. **流程图**——帮助建立直觉的可视化

## 项目结构速查

```
core/         LLM 接口 + 配置
runtime/      主循环 + 上下文工程 + 状态机
tools/        工具协议 + 注册表 + 执行管道
app/          CLI 入口 + bootstrap
extensions/   可选扩展（MCP / Skills / Tracing）
prompts/      系统提示词
```
