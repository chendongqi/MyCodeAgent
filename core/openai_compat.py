"""Small synchronous transport for OpenAI-compatible chat-completions APIs.

系列 03：LLM 传输层（无官方 openai SDK）。
职责只有两件——发 HTTP、把 JSON/SSE 包成可点属性的对象。
provider 路由与响应归一化在 core/llm.py，不在这里。
"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ResponseObject:
    """把 JSON dict 伪装成 SDK 风格对象：response.choices[0].message 可点访问。

    同时提供 model_dump() 还原原始 dict，供 trace / serialize_response 使用。
    上层 extract_* 因此不关心响应来自制客户端还是官方 SDK。
    """

    def __init__(self, value: dict[str, Any]):
        self._value = value

    def __getattr__(self, name: str) -> Any:
        try:
            return _to_object(self._value[name])
        except KeyError as exc:
            raise AttributeError(name) from exc

    def model_dump(self) -> dict[str, Any]:
        return self._value


def _to_object(value: Any) -> Any:
    # 递归包装：嵌套 dict/list 也变成可点访问，保持与 OpenAI SDK 返回形状一致
    if isinstance(value, dict):
        return ResponseObject(value)
    if isinstance(value, list):
        return [_to_object(item) for item in value]
    return value


class _Completions:
    def __init__(self, client: "OpenAICompatibleClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_completion(kwargs)


class _Chat:
    def __init__(self, client: "OpenAICompatibleClient"):
        self.completions = _Completions(client)


class OpenAICompatibleClient:
    """官方 SDK 的最小子集：只实现 harness 用到的 chat.completions.create。

    调用习惯保持 client.chat.completions.create(**request)，
    底层是标准库 urllib，零第三方依赖，核心路径全部可见。
    """

    def __init__(self, api_key: str, base_url: str, timeout: int):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.chat = _Chat(self)

    def _request(self, payload: dict[str, Any]):
        # OpenAI 兼容协议：POST {base_url}/chat/completions + Bearer token
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            return urlopen(request, timeout=self.timeout)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible API returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenAI-compatible API request failed: {exc.reason}") from exc

    def _create_completion(self, payload: dict[str, Any]) -> Any:
        if payload.get("stream"):
            return self._stream(payload)
        with self._request(payload) as response:
            return ResponseObject(json.loads(response.read().decode("utf-8")))

    def _stream(self, payload: dict[str, Any]):
        # SSE：每行 data: {...}，data: [DONE] 表示流结束
        with self._request(payload) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                if data:
                    yield ResponseObject(json.loads(data))
