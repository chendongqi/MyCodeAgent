"""Context budget estimation and compaction decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass

from core.config import Config
from runtime.history import Message


@dataclass(frozen=True)
class CompactDecision:
    should_compact: bool
    reason: str
    estimated_tokens: int
    threshold: int
    message_count: int


class ContextBudgetPolicy:
    """决定当前上下文是否需要触发压缩。

    触发条件：估算 token 数 >= context_window × compression_threshold
    默认：128000 × 0.8 = 102400 tokens

    为什么用估算而不是精确计数？
    精确 token 计数需要调用 tokenizer（慢且依赖模型）。
    用字符数 // 3 是一个足够保守的近似（中英文平均值）：宁可提前压缩，
    不能等到真的超限才触发——那时模型调用已经报错了。
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config.from_env()

    def estimate_tokens(self, messages: list[Message], pending_input: str = "") -> int:
        """估算消息列表的 token 数：字符数 // 3（保守近似）。

        除了 content，还要计入 tool_calls JSON 和 tool_name，
        因为这两部分也会占 token 但不在 content 字段里。
        """
        total_chars = len(pending_input or "")
        for msg in messages or []:
            total_chars += len(str(msg.content or ""))
            metadata = msg.metadata or {}
            if msg.role == "assistant" and metadata.get("tool_calls"):
                try:
                    total_chars += len(json.dumps(metadata["tool_calls"], ensure_ascii=False))
                except Exception:
                    total_chars += len(str(metadata["tool_calls"]))
            if msg.role == "tool" and metadata.get("tool_name"):
                total_chars += len(str(metadata["tool_name"]))
        return total_chars // 3

    def should_compact(
        self,
        *,
        messages: list[Message],
        pending_input: str = "",
        last_usage_tokens: int = 0,
    ) -> CompactDecision:
        """综合两个来源的估算，取较大值决定是否需要压缩。

        两个估算来源：
        1. estimate_tokens(messages)：从当前消息内容估算
        2. last_usage_tokens：上一步 LLM 实际返回的 token 用量（更准确）
        取 max 是为了用最悲观的估计，避免漏触发。
        """
        message_count = len(messages or [])
        # 默认：128000 × 0.8 = 102400 tokens
        # 注意：context_window 是固定配置值，不会根据实际模型自动调整。
        # 使用上下文窗口小的模型（如 GPT-3.5 16k、Ollama 8k）时，
        # 需要手动在 .env 里设置 CONTEXT_WINDOW 匹配模型实际值，
        # 否则主动触发会失效，只能靠 PROMPT_TOO_LONG 异常被动兜底。
        threshold = int(self.config.context_window * self.config.compression_threshold)
        estimated_from_messages = self.estimate_tokens(messages, pending_input)
        # last_usage_tokens 是上一轮 LLM 实际消耗的 token 数（从响应 usage 字段读取）
        # 加上本轮新输入的字符估算，得到累计用量
        estimated_from_usage = int(last_usage_tokens or 0) + len(pending_input or "") // 3
        estimated = max(estimated_from_messages, estimated_from_usage)

        if message_count < 3:
            # 消息太少（如只有 system + 第一条 user），压缩没有意义
            return CompactDecision(False, "messages_not_enough", estimated, threshold, message_count)
        if estimated < threshold:
            return CompactDecision(False, "below_threshold", estimated, threshold, message_count)
        return CompactDecision(True, "threshold_exceeded", estimated, threshold, message_count)
