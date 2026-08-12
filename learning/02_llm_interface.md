# 模块 1：LLM 接口层

> **对应文件**：`core/llm.py`、`core/openai_compat.py`  
> **一句话**：把各家 LLM 的响应格式，统一变成 loop 能用的标准结构。

---

## 整体架构

```
loop 层调用
    ↓
HelloAgentsLLM.invoke_raw(messages, tools=...)
    ↓
OpenAICompatibleClient._create_completion()   ← 真正发 HTTP 请求
    ↓
raw response（各家格式略有不同）
    ↓
extract_tool_calls() / extract_response_content() / ...   ← 归一化
    ↓
loop 层拿到统一结构
```

---

## 第一层：openai_compat.py（92 行，零外部依赖）

用标准库 `urllib` 自己实现了 OpenAI SDK 的最小子集。

### 关键代码：ResponseObject

```python
class ResponseObject:
    def __init__(self, value: dict):
        self._value = value

    def __getattr__(self, name):
        return _to_object(self._value[name])  # dict → 可点访问

    def model_dump(self):
        return self._value
```

**作用**：把 JSON dict 包装成可以 `.field` 点出来的对象。

这样上层代码 `response_attr(response, "choices")` 既能处理真正的 openai SDK 返回的对象，也能处理这个自制的轻量对象——同一套代码，两种来源无感切换。

### 关键代码：流式响应

```python
def _stream(self, payload):
    with self._request(payload) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue           # 跳过空行/注释行
            data = line[5:].strip()
            if data == "[DONE]":
                return             # 流结束标志
            yield ResponseObject(json.loads(data))
```

这是 SSE（Server-Sent Events）协议的标准解析方式：每行以 `data:` 开头，`[DONE]` 表示结束。

---

## 第二层：llm.py 的两大职责

### 职责一：provider 路由

```python
# __init__ 中的优先级链：
self.provider = self._resolve_provider(provider, api_key, base_url)
#  → 1) 传参 provider
#  → 2) 环境变量 LLM_PROVIDER
#  → 3) _auto_detect_provider()（查环境变量 + URL marker）

self.api_key, base_url = self._resolve_credentials(api_key, base_url)
#  → 查 PROVIDER_PROFILES[provider] 拿默认 base_url 和 key 环境变量名
```

`PROVIDER_PROFILES` 是一个大字典，每个 provider 记录了：

| 字段 | 含义 |
|------|------|
| `key_envs` | 按优先级查哪些环境变量找 API key |
| `detect_envs` | 自动探测时看哪些环境变量 |
| `base_url` | 默认端点 |
| `url_markers` | URL 里有哪些关键词可以识别这个 provider |

配了 `LLM_PROVIDER=zhipu`，代码直接查表拿到 `https://open.bigmodel.cn/api/paas/v4`，不需要手动填。

### 职责二：响应归一化（5 个 extract_* 函数）

这是**这一层最核心的设计**：loop 层只调这 5 个函数，不关心底层用哪家模型：

```python
text  = extract_response_content(response)   # 拿文字回复
calls = extract_tool_calls(response)          # 拿工具调用列表（重点）
usage = extract_usage(response)               # 拿 token 统计
meta  = extract_response_meta(response)       # 拿 finish_reason 等
think = extract_reasoning_content(response)  # 拿思维链（可选）
```

### extract_tool_calls 处理了两种历史格式

```python
def extract_tool_calls(response):
    message = _response_message(response)

    # 新格式（OpenAI 1.x，大多数现代模型）
    calls = response_attr(message, "tool_calls") or []
    if calls:
        return [{"id": ..., "name": ..., "arguments": ...} for call in calls]

    # 旧格式（OpenAI 0.x，部分国产模型还在用）
    function_call = response_attr(message, "function_call")
    if function_call:
        return [{"id": None, "name": ..., "arguments": ...}]

    return []
```

两种格式最终归一化成同一个 `[{id, name, arguments}]` 列表，loop 层完全不感知差异。

---

## 对 loop 暴露的三个入口

| 方法 | loop 中的用途 | 返回 |
|------|-------------|------|
| `invoke_raw()` | **主循环用**，拿完整 response 交给 extract_* 函数 | 原始 response 对象 |
| `invoke()` | 摘要压缩等一次性场景 | 直接返回文字字符串 |
| `think()` | 流式输出给用户看（streaming） | `Iterator[str]` 逐 token |

---

## 设计亮点：response_attr 通用读取器

```python
def response_attr(value, key):
    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)
```

这个小函数让所有 extract_* 函数既能处理 dict（自制 ResponseObject 的 `model_dump()`），也能处理真正的对象（官方 SDK 返回的 Pydantic model），一套代码两用。

---

## 反直觉的设计点

**项目没有用 openai 官方 SDK**，而是用 `urllib` 自己实现了 92 行的最小替代品。

原因：减少外部依赖，让所有核心逻辑都在项目源码里可见，方便学习和移植。openai SDK 本身封装了大量细节，一旦用了，就有东西藏在第三方包里看不到。

---

## 三个检验问题

1. 你用的是 `zhipu` provider，`base_url` 是怎么确定的？代码走了哪几步？
2. `extract_tool_calls` 为什么要兼容 `function_call` 旧格式？
3. `invoke_raw()` 和 `invoke()` 的区别是什么？loop 为什么用前者而不是后者？
