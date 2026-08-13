"""HelloAgents统一LLM接口 - 基于OpenAI原生API

系列 03：LLM 适配层。
两件事：① provider 路由（选端点/key）② 响应归一化（extract_* 五元组）。
传输细节在 openai_compat.py；ReAct 循环只消费本文件导出的统一字段。
"""

import json
import logging
import os
import time
from types import MappingProxyType
from typing import Any, Callable, Iterator, Literal, Optional

from .openai_compat import OpenAICompatibleClient

from .exceptions import HelloAgentsException

logger = logging.getLogger(__name__)


def response_attr(value: Any, key: str) -> Any:
    """通用字段读取：dict 走 .get，对象走 getattr。

    ResponseObject（自制）与官方 SDK 的 Pydantic 对象用同一套 extract_* 代码。
    """

    return value.get(key) if isinstance(value, dict) else getattr(value, key, None)


def parse_tool_input(raw: Any) -> tuple[Any, str | None]:
    """Normalize a model tool argument payload without host-specific parsing."""

    if raw is None:
        return {}, None
    if isinstance(raw, (dict, list)):
        return raw, None
    try:
        return json.loads(str(raw).strip() or "{}"), None
    except (TypeError, ValueError) as error:
        return None, str(error)


def _response_message(response: Any) -> Any:
    # 所有 extract_* 的公共入口：choices[0].message
    choices = response_attr(response, "choices") or []
    return response_attr(choices[0], "message") if choices else None


# ── 响应归一化五元组：loop 只依赖这些函数，不直接解析各家 JSON ──────────────

def extract_response_content(response: Any) -> str | None:
    """抽出正文。部分模型把 content 做成多段 list，这里拼成字符串。"""

    content = response_attr(_response_message(response), "content")
    if isinstance(content, list):
        return "".join(
            str(response_attr(part, "text") or "") for part in content
        )
    return content


def extract_reasoning_content(response: Any) -> Any:
    """抽出可选思维链字段（reasoning_content / reasoning），不影响控制流。"""

    message = _response_message(response)
    reasoning = response_attr(message, "reasoning_content") or response_attr(message, "reasoning")
    if reasoning:
        return reasoning
    extra = response_attr(message, "model_extra") or response_attr(message, "additional_kwargs")
    return response_attr(extra, "reasoning_content") or response_attr(extra, "reasoning")


def extract_usage(response: Any) -> dict[str, Any] | None:
    """投影为稳定的 token 用量形状，供 context_engine 记账与预算检查。"""

    usage = response_attr(response, "usage")
    if not usage:
        return None
    return {
        "prompt_tokens": response_attr(usage, "prompt_tokens"),
        "completion_tokens": response_attr(usage, "completion_tokens"),
        "total_tokens": response_attr(usage, "total_tokens"),
    }


def extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    """新旧 Function Calling 格式 → 统一 [{id, name, arguments}]。

    新格式：message.tool_calls[]（主流 OpenAI 兼容）
    旧格式：message.function_call（单次调用，部分国产模型仍在用）
    loop 的 Acting 分支只认归一化后的列表。
    """

    message = _response_message(response)
    calls = response_attr(message, "tool_calls") or []
    if calls:
        return [
            {
                "id": response_attr(call, "id"),
                "name": response_attr(response_attr(call, "function") or {}, "name")
                or response_attr(call, "name")
                or "unknown_tool",
                "arguments": response_attr(response_attr(call, "function") or {}, "arguments")
                or response_attr(call, "arguments")
                or {},
            }
            for call in calls
        ]
    function_call = response_attr(message, "function_call")
    if function_call:
        return [{
            "id": None,
            "name": response_attr(function_call, "name") or "unknown_tool",
            "arguments": response_attr(function_call, "arguments") or {},
        }]
    return []


def extract_response_meta(response: Any) -> dict[str, Any]:
    """抽出控制流信号（finish_reason / 长度 / refusal），不含正文本身。

    与 extract_response_content 互补：text 给人看，meta 给空响应重试、
    截断判定、完成门与 trace 用。
    """

    choices = response_attr(response, "choices") or []
    choice = choices[0] if choices else None
    message = response_attr(choice, "message")
    content = response_attr(message, "content")
    reasoning = response_attr(message, "reasoning_content") or response_attr(message, "reasoning")
    tool_calls = response_attr(message, "tool_calls")
    return {
        "finish_reason": response_attr(choice, "finish_reason"),
        "role": response_attr(message, "role"),
        "content_len": len(str(content)) if content is not None else 0,
        "reasoning_len": len(str(reasoning)) if reasoning is not None else 0,
        "refusal_present": response_attr(message, "refusal") is not None,
        "tool_calls_count": len(tool_calls) if isinstance(tool_calls, list) else int(bool(tool_calls)),
        "function_call_present": response_attr(message, "function_call") is not None,
    }


def serialize_response(response: Any) -> Any:
    """Keep raw model diagnostics JSON-compatible where supported."""

    try:
        candidate = response.model_dump() if hasattr(response, "model_dump") else response
        json.dumps(candidate, ensure_ascii=False)
    except Exception:
        return {"raw": str(response)}
    return candidate

# 支持的LLM提供商
SUPPORTED_PROVIDERS = Literal[
    "openai",
    "deepseek",
    "qwen",
    "modelscope",
    "kimi",
    "zhipu",
    "siliconflow",
    "ollama",
    "vllm",
    "local",
    "auto",
]

PROVIDER_ALIASES = MappingProxyType(
    {
        "silicon-flow": "siliconflow",
        "silicon_flow": "siliconflow",
    }
)

# 表驱动 provider 路由：加一家模型主要是加一行 profile，而不是改调用链。
# key_envs     = 按优先级查 API key 的环境变量
# detect_envs  = 自动探测时看哪些变量（命中多个会报错，要求显式指定）
# base_url     = 该 provider 的默认端点
# url_markers  = 从 base_url 字符串反推 provider 的关键词
PROVIDER_PROFILES = MappingProxyType(
    {
        "openai": MappingProxyType(
            {
                "key_envs": ("OPENAI_API_KEY", "LLM_API_KEY"),
                "detect_envs": ("OPENAI_API_KEY",),
                "base_url_envs": ("LLM_BASE_URL",),
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-3.5-turbo",
                "url_markers": ("api.openai.com",),
            }
        ),
        "deepseek": MappingProxyType(
            {
                "key_envs": ("DEEPSEEK_API_KEY", "LLM_API_KEY"),
                "detect_envs": ("DEEPSEEK_API_KEY",),
                "base_url_envs": ("LLM_BASE_URL",),
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-chat",
                "url_markers": ("api.deepseek.com",),
            }
        ),
        "qwen": MappingProxyType(
            {
                "key_envs": ("DASHSCOPE_API_KEY", "LLM_API_KEY"),
                "detect_envs": ("DASHSCOPE_API_KEY",),
                "base_url_envs": ("LLM_BASE_URL",),
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model": "qwen-plus",
                "url_markers": ("dashscope.aliyuncs.com",),
            }
        ),
        "modelscope": MappingProxyType(
            {
                "key_envs": ("MODELSCOPE_API_KEY", "LLM_API_KEY"),
                "detect_envs": ("MODELSCOPE_API_KEY",),
                "base_url_envs": ("LLM_BASE_URL",),
                "base_url": "https://api-inference.modelscope.cn/v1/",
                "model": "Qwen/Qwen2.5-72B-Instruct",
                "url_markers": ("api-inference.modelscope.cn",),
            }
        ),
        "kimi": MappingProxyType(
            {
                "key_envs": ("KIMI_API_KEY", "MOONSHOT_API_KEY", "LLM_API_KEY"),
                "detect_envs": ("KIMI_API_KEY", "MOONSHOT_API_KEY"),
                "base_url_envs": ("LLM_BASE_URL",),
                "base_url": "https://api.moonshot.cn/v1",
                "model": "moonshot-v1-8k",
                "url_markers": ("api.moonshot.cn",),
            }
        ),
        "zhipu": MappingProxyType(
            {
                "key_envs": ("ZHIPU_API_KEY", "GLM_API_KEY", "LLM_API_KEY"),
                "detect_envs": ("ZHIPU_API_KEY", "GLM_API_KEY"),
                "base_url_envs": ("LLM_BASE_URL",),
                "base_url": "https://open.bigmodel.cn/api/paas/v4",
                "model": "glm-4",
                "url_markers": ("open.bigmodel.cn",),
            }
        ),
        "siliconflow": MappingProxyType(
            {
                "key_envs": ("SILICONFLOW_API_KEY", "LLM_API_KEY"),
                "detect_envs": ("SILICONFLOW_API_KEY",),
                "base_url_envs": ("SILICONFLOW_BASE_URL", "LLM_BASE_URL"),
                "base_url": "https://api.siliconflow.cn/v1",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "url_markers": ("api.siliconflow.cn",),
            }
        ),
        "ollama": MappingProxyType(
            {
                "key_envs": ("OLLAMA_API_KEY", "LLM_API_KEY"),
                "detect_envs": ("OLLAMA_API_KEY", "OLLAMA_HOST"),
                "base_url_envs": ("OLLAMA_HOST", "LLM_BASE_URL"),
                "base_url": "http://localhost:11434/v1",
                "model": "llama3.2",
                "default_key": "ollama",
                "url_markers": ("ollama",),
            }
        ),
        "vllm": MappingProxyType(
            {
                "key_envs": ("VLLM_API_KEY", "LLM_API_KEY"),
                "detect_envs": ("VLLM_API_KEY", "VLLM_HOST"),
                "base_url_envs": ("VLLM_HOST", "LLM_BASE_URL"),
                "base_url": "http://localhost:8000/v1",
                "model": "meta-llama/Llama-2-7b-chat-hf",
                "default_key": "vllm",
                "url_markers": ("vllm",),
            }
        ),
        "local": MappingProxyType(
            {
                "key_envs": ("LLM_API_KEY",),
                "detect_envs": (),
                "base_url_envs": ("LLM_BASE_URL",),
                "base_url": "http://localhost:8000/v1",
                "model": "local-model",
                "default_key": "local",
                "url_markers": (),
            }
        ),
        "auto": MappingProxyType(
            {
                "key_envs": ("LLM_API_KEY",),
                "detect_envs": (),
                "base_url_envs": ("LLM_BASE_URL",),
                "base_url": None,
                "model": "gpt-3.5-turbo",
                "url_markers": (),
            }
        ),
    }
)


class HelloAgentsLLM:
    """
    OpenAI 兼容服务的统一客户端（适配层，不是传输层）。

    设计要点：
    - 参数优先，环境变量兜底；provider 用 PROVIDER_PROFILES 表驱动
    - 请求侧抹平各家怪癖（temperature / 多 system / tool_choice）
    - 响应侧由模块级 extract_* 归一化；本类负责把 raw response 拿回来
    - 主循环用 invoke_raw；invoke/think 给「只要文本」的场景
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[SUPPORTED_PROVIDERS] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        支持自动检测provider或使用统一的LLM_*环境变量配置。

        Args:
            model: 模型名称，如果未提供则从环境变量LLM_MODEL_ID读取
            api_key: API密钥，如果未提供则从环境变量读取
            base_url: 服务地址，如果未提供则从环境变量LLM_BASE_URL读取
            provider: LLM提供商，如果未提供则自动检测
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 超时时间，从环境变量LLM_TIMEOUT读取，默认60秒
        """
        # 优先使用传入参数，如果未提供，则从环境变量加载
        self.model = model or self._get_env("LLM_MODEL_ID")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout or int(self._get_env("LLM_TIMEOUT", "120"))
        self.max_retries = int(self._get_env("LLM_MAX_RETRIES", "2"))
        self.retry_backoff = float(self._get_env("LLM_RETRY_BACKOFF", "1.0"))
        self.kwargs = kwargs
        self._temperature_policy_notice_emitted = False

        # provider 解析优先级：显式参数 → LLM_PROVIDER → 自动探测 → "auto"
        self.provider = self._resolve_provider(provider, api_key, base_url)

        # 按 profile 表解析 key / base_url（各家环境变量名不同）
        self.api_key, resolved_base_url = self._resolve_credentials(api_key, base_url)
        self.base_url = self._normalize_base_url(resolved_base_url)

        # 验证必要参数
        if not self.model:
            self.model = self._get_default_model()
        if not all([self.api_key, self.base_url]):
            raise HelloAgentsException("API密钥和服务地址必须被提供或在.env文件中定义。")

        # 传输层客户端（urllib），对外仍是 chat.completions.create 形状
        self._client = self._create_client()

    def _get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Read process environment after the application-owned dotenv load."""
        return os.getenv(key, default)

    def _resolve_provider(self, provider: Optional[str], api_key: Optional[str], base_url: Optional[str]) -> str:
        """
        解析 provider：
        1) 显式参数 provider
        2) 环境变量/ .env 中的 LLM_PROVIDER
        3) 自动探测
        """
        if provider:
            return self._normalize_provider(provider)
        env_provider = self._get_env("LLM_PROVIDER")
        if env_provider:
            return self._normalize_provider(env_provider)
        return self._auto_detect_provider(api_key, base_url)

    def _normalize_provider(self, provider: str) -> str:
        """标准化 provider 名称，兼容大小写和常见别名。"""
        normalized = provider.strip().lower()
        return PROVIDER_ALIASES.get(normalized, normalized)

    def _normalize_base_url(self, base_url: Optional[str]) -> Optional[str]:
        """将误填的完整接口路径归一化为 OpenAI 客户端所需的 base_url。"""
        if not base_url:
            return base_url
        normalized = base_url.strip().rstrip("/")
        for suffix in ("/chat/completions", "/completions"):
            if normalized.lower().endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
        return normalized

    def _auto_detect_provider(self, api_key: Optional[str], base_url: Optional[str]) -> str:
        """Resolve declared provider environment names and URL markers."""
        hits = [
            provider
            for provider, profile in PROVIDER_PROFILES.items()
            if any(self._get_env(key) for key in profile["detect_envs"])
        ]
        if len(hits) > 1:
            providers = ", ".join(sorted(set(hits)))
            raise HelloAgentsException(
                f"检测到多个 provider 配置: {providers}。请显式设置 provider 或 LLM_PROVIDER。"
            )
        if len(hits) == 1:
            return hits[0]

        # URL detection remains a convenience only for declared provider URLs.
        actual_base_url = base_url or self._get_env("LLM_BASE_URL")
        if actual_base_url:
            base_url_lower = actual_base_url.lower()
            for provider, profile in PROVIDER_PROFILES.items():
                if any(marker in base_url_lower for marker in profile["url_markers"]):
                    return provider

        # Generic OpenAI-compatible configuration uses LLM_* values.
        return "auto"

    def _resolve_credentials(self, api_key: Optional[str], base_url: Optional[str]) -> tuple[str, str]:
        """Resolve credentials from the selected provider's metadata."""
        profile = PROVIDER_PROFILES.get(self.provider, PROVIDER_PROFILES["auto"])
        resolved_api_key = api_key or self._first_env(profile["key_envs"]) or profile.get("default_key")
        resolved_base_url = base_url or self._first_env(profile["base_url_envs"]) or profile["base_url"]
        return resolved_api_key, resolved_base_url

    def _first_env(self, keys: tuple[str, ...]) -> Optional[str]:
        """Return the first configured process environment value for *keys*."""
        return next((value for key in keys if (value := self._get_env(key))), None)

    def _create_client(self) -> OpenAICompatibleClient:
        """创建OpenAI客户端"""
        return OpenAICompatibleClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    @staticmethod
    def _compact_request_kwargs(kwargs: dict) -> dict:
        """Drop None-valued fields before provider request dispatch."""
        return {k: v for k, v in kwargs.items() if v is not None}

    def _is_minimax_backend(self) -> bool:
        base = (self.base_url or "").lower()
        return "minimaxi.com" in base or "minimax.io" in base

    def _apply_provider_compat(self, request_kwargs: dict) -> dict:
        """Apply backend-specific request normalization."""
        normalized = dict(request_kwargs)
        if self._is_minimax_backend():
            normalized["n"] = 1
            if normalized.get("tool_choice") == "auto":
                normalized.pop("tool_choice", None)
        return normalized

    def _normalize_messages_for_provider(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Normalize message list for backend-specific constraints."""
        if not self._is_minimax_backend():
            return messages
        if not isinstance(messages, list) or len(messages) <= 1:
            return messages
        system_messages = [m for m in messages if isinstance(m, dict) and str(m.get("role") or "") == "system"]
        if len(system_messages) <= 1:
            return messages

        merged_parts: list[str] = []
        for m in system_messages:
            content = m.get("content")
            if content is None:
                continue
            text = str(content).strip()
            if text:
                merged_parts.append(text)
        merged_system = {"role": "system", "content": "\n\n".join(merged_parts)}
        non_system = [m for m in messages if not (isinstance(m, dict) and str(m.get("role") or "") == "system")]
        if merged_system["content"]:
            return [merged_system] + non_system
        return non_system

    def _requires_temperature_one(self) -> bool:
        """Kimi 2.5 / K2 系列模型仅接受 temperature=1。"""
        if self.provider != "kimi":
            return False
        model = (self.model or "").strip().lower()
        if not model:
            return False
        strict_markers = (
            "kimi2.5",
            "kimi-2.5",
            "kimi_2.5",
            "kimi-k2",
            "kimi k2",
            "k2-",
            "-k2",
            "/k2",
            "k2",
        )
        return any(marker in model for marker in strict_markers)

    def _resolve_temperature(self, requested: Optional[float]) -> float:
        """根据模型约束解析 temperature。"""
        value = self.temperature if requested is None else requested
        try:
            temp = float(value)
        except Exception:
            temp = float(self.temperature)

        if self._requires_temperature_one() and temp != 1.0:
            if not self._temperature_policy_notice_emitted:
                logger.warning(
                    "模型 %s 仅支持 temperature=1，已自动从 %.3f 调整为 1。",
                    self.model,
                    temp,
                )
                self._temperature_policy_notice_emitted = True
            return 1
        return temp

    def _get_default_model(self) -> str:
        """获取默认模型"""
        profile = PROVIDER_PROFILES.get(self.provider, PROVIDER_PROFILES["auto"])
        return profile["model"]

    def _build_request(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        **overrides: Any,
    ) -> dict[str, Any]:
        """组装一次请求：消息归一化 → temperature 约束 → provider 兼容补丁。

        请求侧怪癖（Kimi temperature、MiniMax 多 system 等）集中在这里处理，
        避免泄漏进 RuntimeRunner。
        """
        request = {
            "model": self.model,
            "messages": self._normalize_messages_for_provider(messages),
            "temperature": self._resolve_temperature(overrides.pop("temperature", None)),
            "max_tokens": overrides.pop("max_tokens", self.max_tokens),
            "stream": True if stream else None,
            **overrides,
        }
        return self._apply_provider_compat(self._compact_request_kwargs(request))

    def _invoke_with_retries(
        self,
        messages: list[dict[str, str]],
        project_response: Callable[[Any], Any] | None = None,
        **overrides: Any,
    ) -> Any:
        """非流式调用 + 瞬时失败重试（网络/HTTP）。

        注意：这与 loop 内层 while 的「语义级」恢复（空响应 / PROMPT_TOO_LONG）
        不是同一层——这里只处理传输失败。
        """
        for attempt in range(self.max_retries + 1):
            try:
                request = self._build_request(messages, **overrides)
                response = self._client.chat.completions.create(**request)
                return project_response(response) if project_response else response
            except Exception as error:
                if attempt >= self.max_retries:
                    raise HelloAgentsException(f"LLM调用失败: {str(error)}")
                wait_s = self.retry_backoff * (2**attempt)
                logger.warning(
                    "LLM调用失败，%.1fs后重试（%d/%d）: %s",
                    wait_s,
                    attempt + 1,
                    self.max_retries,
                    error,
                )
                time.sleep(wait_s)

    def think(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[object] = None,
    ) -> Iterator[str]:
        """
        调用大语言模型进行思考，并返回流式响应。
        这是主要的调用方法，默认使用流式响应以获得更好的用户体验。

        Args:
            messages: 消息列表
            temperature: 温度参数，如果未提供则使用初始化时的值

        Yields:
            str: 流式响应的文本片段
        """
        logger.info("正在调用 %s 模型...", self.model)
        try:
            overrides: dict[str, Any] = {"temperature": temperature}
            if tools:
                overrides["tools"] = tools
                if tool_choice is not None:
                    overrides["tool_choice"] = tool_choice
            response = self._client.chat.completions.create(
                **self._build_request(messages, stream=True, **overrides)
            )

            # 处理流式响应
            logger.debug("大语言模型响应成功（streaming）")
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content

        except Exception as e:
            logger.error("调用LLM API时发生错误: %s", e)
            raise HelloAgentsException(f"LLM调用失败: {str(e)}")

    def invoke(self, messages: list[dict[str, str]], **kwargs) -> str:
        """非流式调用，直接返回正文字符串（摘要压缩等「只要一段话」的场景）。"""
        return self._invoke_with_retries(
            messages, lambda response: response.choices[0].message.content, **kwargs
        )

    def invoke_raw(self, messages: list[dict[str, str]], **kwargs):
        """非流式调用，返回原始响应对象——主循环专用入口。

        RuntimeRunner 拿到 raw 后再用 extract_* 拆出 text / tool_calls / usage / meta。
        若用 invoke()，这些信号会在取 .content 时丢掉。
        """
        return self._invoke_with_retries(messages, **kwargs)

    def stream_invoke(self, messages: list[dict[str, str]], **kwargs) -> Iterator[str]:
        """Alias for think()."""
        temperature = kwargs.get('temperature')
        yield from self.think(messages, temperature)
