---
title: "Code Agent 解剖（03）：各家 LLM 格式不一样，agent 怎么统一对接？"
date: 2026-08-13
description: "深入 MyCodeAgent 的 LLM 接口层：零依赖 HTTP 传输、provider 路由表、五元组响应归一化。理解 agent 为什么不直接绑某一家 SDK，而是把「发请求」和「读响应」拆成稳定边界，让 ReAct 循环只看到统一结构。"
tags: [Code Agent, LLM, OpenAI Compatible, Function Calling, Agent Harness]
category: AI Engineering
draft: false
---

## 问题不在「会不会调 API」

换一家模型，往往不是改一行 `base_url` 就完事：

- DeepSeek、智谱、Kimi、Qwen 都说自己「兼容 OpenAI」，但 key 环境变量名、默认端点、tool_calls 字段位置并不完全一样
- 有的模型还在用旧版 `function_call`，有的把 content 做成多段 list
- 个别后端还限制 temperature、禁止多个 system 消息、不认 `tool_choice=auto`

如果这些差异散落在 `loop.py` 里，主循环会变成「if provider == xxx」的沼泽。MyCodeAgent 的做法是：把差异关进 `core/llm.py` + `core/openai_compat.py`，让 ReAct 循环只看见统一的 `text` 和 `tool_calls`。

上一篇讲完「一轮一轮怎么跑」。这篇讲循环真正调模型时，那一层边界长什么样。

---

## 结论先放这

```
RuntimeRunner._react_loop()
        │
        ▼
HelloAgentsLLM.invoke_raw(messages, tools=...)
        │  ① provider 路由：选端点 / key / 默认模型
        │  ② 请求归一化：temperature、多 system、tool_choice 等坑
        ▼
OpenAICompatibleClient          ← 标准库 urllib，无 openai SDK
        │  POST {base_url}/chat/completions
        ▼
raw response（对象或 dict，各家略有差异）
        │
        ▼
extract_* 五元组                 ← 响应归一化
  content / tool_calls / usage / meta / reasoning
        │
        ▼
loop 只消费统一结构，继续 Acting / Reasoning
```

两层分工很清楚：

| 层 | 文件 | 干什么 |
|----|------|--------|
| 传输 | `openai_compat.py` | 发 HTTP、解析 JSON/SSE，伪装成 `client.chat.completions.create` |
| 适配 | `llm.py` | provider 路由 + 请求兼容 + 响应提取 |

loop **故意**调 `invoke_raw()`，而不是 `invoke()`——前者返回完整响应对象，后面才能拆出 tool_calls、usage、finish_reason；后者直接吐字符串，主循环用不了。

---

## 第一层：自己实现一个「假 SDK」

项目没有依赖官方 `openai` 包。`openai_compat.py` 用 `urllib` 实现了 harness 真正用到的最小子集：

```python
# core/openai_compat.py — 核心调用路径
class OpenAICompatibleClient:
    def __init__(self, api_key, base_url, timeout):
        self.chat = _Chat(self)   # 露出 client.chat.completions.create(...)

    def _create_completion(self, payload):
        if payload.get("stream"):
            return self._stream(payload)
        with self._request(payload) as response:
            return ResponseObject(json.loads(response.read().decode("utf-8")))
```

请求就是一次普通 POST：`{base_url}/chat/completions`，Header 带 `Bearer` token。流式则按 SSE 协议读 `data:` 行，遇到 `[DONE]` 结束。

关键是 `ResponseObject`：把 JSON dict 包成可以 `.choices[0].message` 点出来的对象，同时提供 `model_dump()` 还原字典。

```python
class ResponseObject:
    def __getattr__(self, name):
        return _to_object(self._value[name])  # dict → 可点访问

    def model_dump(self):
        return self._value
```

上层提取函数因此不关心响应来自官方 SDK 还是自制客户端——属性访问和字典访问都能走通。

**为什么不用官方 SDK？** 依赖少，核心路径全在仓库里可见；调不通时不用猜第三方封装吞了什么。代价是只实现 harness 需要的子集，不是完整 SDK。

---

## 第二层：provider 怎么选出来

`HelloAgentsLLM.__init__` 按固定优先级解析配置：

```
显式参数 provider
    → 环境变量 LLM_PROVIDER
        → 自动探测（detect_envs / URL marker）
            → 落成 "auto"（泛 OpenAI 兼容）
```

凭证和默认端点不写死在 if/else 里，而是查 `PROVIDER_PROFILES` 表：

```python
# core/llm.py — 表驱动路由（示意）
PROVIDER_PROFILES = {
    "deepseek": {
        "key_envs": ("DEEPSEEK_API_KEY", "LLM_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "url_markers": ("api.deepseek.com",),
        ...
    },
    "zhipu": { ... },
    "kimi": { ... },
    # ...
}

# 解析时：
self.provider = self._resolve_provider(provider, api_key, base_url)
self.api_key, resolved_base_url = self._resolve_credentials(api_key, base_url)
```

配了 `LLM_PROVIDER=zhipu`，代码直接从表拿到智谱的默认 `base_url` 和该查哪些 key 环境变量，不必在业务代码里写死域名。

自动探测还有一条安全阀：若同时检测到多个 provider 的专用 key，直接报错要求显式指定——避免「你以为在调 A，实际命中了 B」。

---

## 第三层：响应归一化（这一层最值钱）

loop 在拿到 `raw_response` 之后，只做这件事：

```python
# runtime/loop.py — 主循环消费统一字段
response_text = extract_response_content(raw_response) or ""
reasoning_content = extract_reasoning_content(raw_response)
usage = extract_usage(raw_response)
response_meta = extract_response_meta(raw_response)
tool_calls = extract_tool_calls(raw_response)
```

五个函数把「各家略有不同的 JSON」投影成 harness 的稳定形状。其中 loop 最依赖的是 `extract_tool_calls`：

```python
# core/llm.py — 新旧 Function Calling 格式归一
def extract_tool_calls(response):
    message = _response_message(response)

    # 新格式：message.tool_calls[]（主流）
    calls = response_attr(message, "tool_calls") or []
    if calls:
        return [{"id": ..., "name": ..., "arguments": ...} for call in calls]

    # 旧格式：message.function_call（单次调用）
    function_call = response_attr(message, "function_call")
    if function_call:
        return [{"id": None, "name": ..., "arguments": ...}]

    return []
```

无论模型吐的是 `tool_calls` 还是 `function_call`，loop 拿到的都是 `[{id, name, arguments}]`。上一篇 Acting 分支里「有 tool_calls → 执行工具」的分支，建立在这个归一化之上。

其余几个函数各管一块：

| 函数 | 抽出什么 | loop 拿来干什么 |
|------|---------|----------------|
| `extract_response_content` | 正文（兼容 content 为 list） | 写入历史 / 完成门候选答案 |
| `extract_tool_calls` | 工具调用列表 | Acting 分支 |
| `extract_usage` | prompt/completion/total tokens | 记用量、查 token 预算 |
| `extract_response_meta` | finish_reason、长度、refusal 等 | 空响应重试、截断判定、trace |
| `extract_reasoning_content` | 思维链（可选字段） | 调试展示，不改变控制流 |

通用读取器只有一行，却撑起整套提取逻辑：

```python
def response_attr(value, key):
    # dict 走 .get；对象走 getattr —— ResponseObject 与官方 SDK 同一套提取代码
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)
```

---

## 请求侧也有「兼容补丁」

归一化不只发生在响应。发请求前，`_build_request()` 还会抹平几类已知坑：

1. **Kimi K2 / 2.5**：只接受 `temperature=1`，其它值会被自动改掉并打一次 warning
2. **MiniMax**：合并多条 system 消息；丢掉 `tool_choice=auto`（对方不认）
3. **误填完整路径**：若 `base_url` 写成了 `.../chat/completions`，构造客户端前先剥掉后缀

这些逻辑集中在适配层，loop 仍然只传 `messages / tools / tool_choice`。新增一家「大体兼容但有怪癖」的后端时，优先在这里加规则，而不是改主循环。

重试也在这一层：`_invoke_with_retries` 按 `LLM_MAX_RETRIES` + 指数退避包住非流式调用。注意这和上一篇讲的「内层 while 模型错误恢复」不是同一层——那边处理的是空响应、PROMPT_TOO_LONG 等**语义级**恢复；这里处理的是瞬时网络/HTTP 失败。

---

## 三个调用入口，loop 只用一个

| 方法 | 返回 | 谁在用 |
|------|------|--------|
| `invoke_raw()` | 原始响应对象 | **主循环**（再交给 extract_*） |
| `invoke()` | 纯文本字符串 | 摘要压缩等「只要一段话」的场景 |
| `think()` / `stream_invoke()` | 流式文本片段 | 面向用户的流式展示 |

主循环选 `invoke_raw` 是刻意的：Agent 需要的不只是「模型说了啥」，还要「有没有工具调用、为什么停、花了多少 token」。字符串入口会把这些信号丢掉。

---

## 设计亮点

1. **传输与适配分离**：HTTP 细节在 `openai_compat`，业务兼容在 `llm`，loop 两边都不碰
2. **表驱动 provider**：加一家模型主要是加 `PROVIDER_PROFILES` 行，而不是改调用链
3. **响应投影而不是改写**：extract_* 只读 raw response，不改 provider 原始载荷；trace 仍可存完整快照
4. **零 SDK 依赖**：核心路径可见、可移植；用「假 SDK 形状」保持调用习惯不变

---

## 小结

| 机制 | 作用 |
|------|------|
| `OpenAICompatibleClient` | 标准库实现最小 OpenAI 兼容传输 |
| `PROVIDER_PROFILES` | 端点 / key / 默认模型表驱动路由 |
| `invoke_raw` + `extract_*` | 主循环只消费统一 text / tool_calls / meta |
| 新旧 tool 格式兼容 | `tool_calls` 与 `function_call` 归一成同一列表 |
| 请求侧兼容补丁 | temperature、多 system、误填 URL 等坑集中处理 |

下一篇会顺着「模型怎么知道有哪些工具」往下挖：工具 schema 从哪来、Function Calling 参数怎么进执行管道。

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
