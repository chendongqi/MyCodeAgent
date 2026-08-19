"""Non-destructive context compaction."""

from __future__ import annotations

from typing import Any, Callable

from core.config import Config
from runtime.context.compact_store import CompactStore
from runtime.context.rounds import RoundSegmenter
from runtime.history import Message


class ContextCompactor:
    """Creates compact checkpoints while preserving full history."""

    def __init__(
        self,
        *,
        config: Config | None = None,
        compact_store: CompactStore | None = None,
        summary_generator: Callable[[list[Message]], str | None] | None = None,
        round_segmenter: RoundSegmenter | None = None,
    ):
        self.config = config or Config.from_env()
        self.compact_store = compact_store or CompactStore()
        self.summary_generator = summary_generator
        self.round_segmenter = round_segmenter or RoundSegmenter()

    def compact(self, messages: list[Message]) -> dict[str, Any]:
        """对历史消息执行一次压缩，产生 checkpoint，不修改原始历史。

        压缩逻辑：
        1. 用 RoundSegmenter 把历史按 user 消息切割成"轮次"
        2. 保留最近 min_retain_rounds 轮（默认 10 轮）的原文
        3. 把更早的消息交给 summary_generator（LLM 调用）生成摘要
        4. 摘要 + retain_start_idx 存入 CompactStore 作为 checkpoint

        为什么不删除原始消息？
        HistoryManager 是 append-only 的事实日志。
        压缩只影响"读时投影"（ProjectionBuilder 读 checkpoint 时生成 [summary]+最近N轮），
        原始消息永远保留，崩溃恢复时还能从 transcript 完整重建。
        """
        source_messages = list(messages or [])
        # 按 user 消息边界分割轮次
        rounds = self.round_segmenter.identify(source_messages)
        min_rounds = self.config.min_retain_rounds
        if len(rounds) <= min_rounds:
            # 轮次不够，没有可压缩的"旧历史"
            return {
                "compacted": False,
                "reason": "rounds_not_enough",
                "rounds_count": len(rounds),
                "min_retain_rounds": min_rounds,
            }

        # retain_start_idx：保留最近 N 轮的起始索引，之前的全部送去压缩
        retain_start_round = len(rounds) - min_rounds
        retain_start_idx = rounds[retain_start_round].start_idx
        messages_to_compact = source_messages[:retain_start_idx]
        if not messages_to_compact:
            return {"compacted": False, "reason": "no_messages_to_compact"}

        if not self.summary_generator:
            return {"compacted": False, "reason": "summary_unavailable"}

        try:
            # summary_generator 是 create_summary_generator() 返回的闭包
            # 内部调用 llm.invoke() 生成摘要，有 summary_timeout 超时保护
            summary = self.summary_generator(messages_to_compact)
        except Exception:
            summary = None

        if summary is None:
            # LLM 生成失败或超时：不压缩，保持原状，下一步继续尝试
            return {"compacted": False, "reason": "summary_unavailable"}

        # 创建 checkpoint：summary 文本 + retain_start_idx（投影时用这个分割点）
        checkpoint = self.compact_store.create_checkpoint(
            summary=summary,
            source_message_count=len(source_messages),
            retain_start_idx=retain_start_idx,
            messages_compacted=len(messages_to_compact),
            metadata={
                "rounds_count": len(rounds),
                "min_retain_rounds": min_rounds,
                "retain_start_round": retain_start_round,
            },
        )
        return {
            "compacted": True,
            "checkpoint_id": checkpoint.id,
            "messages_before": len(source_messages),
            "messages_compacted": len(messages_to_compact),
            "retain_start_idx": retain_start_idx,
            "summary_generated": True,
            "summary_len": len(summary),
        }
