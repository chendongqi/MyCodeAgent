"""Read-time history projection.

HistoryManager 保存完整事实日志；本模块在**读取时**生成模型可见的子集，
不修改 source。这就是 loop 里说的「有界投影」：
  - full_history：尚未压缩，投影 = 全量历史
  - compact_checkpoint：旧消息折叠成一条 summary，保留最近几轮原文
"""

from __future__ import annotations

from dataclasses import dataclass, field

from runtime.context.compact_store import CompactStore
from runtime.history import Message


@dataclass(frozen=True)
class ProjectionResult:
    messages: list[Message]
    source_message_count: int
    projection_mode: str = "full_history"
    warnings: tuple[str, ...] = field(default_factory=tuple)
    compact_checkpoint_id: str | None = None


class ProjectionBuilder:
    """Builds the active history view without mutating the runtime log."""

    def __init__(self, compact_store: CompactStore | None = None):
        self.compact_store = compact_store

    def project(self, source_messages: list[Message]) -> ProjectionResult:
        """根据是否有 checkpoint，决定给模型看什么。

        没有 checkpoint → full_history：直接返回全量历史（原文照发）
        有 checkpoint   → compact_checkpoint：
            返回 [summary 消息] + source[retain_start_idx:]
            即一条摘要 + 最近 N 轮的原文
            source 里被压缩的部分对模型不可见，但原始消息仍在 HistoryManager 里

        这是"读时投影"的核心：不修改 source，只在读取时动态折叠旧历史。
        """
        source = list(source_messages or [])
        checkpoint = self.compact_store.active_checkpoint if self.compact_store else None
        if not checkpoint:
            # 尚未压缩：模型看到完整历史
            return ProjectionResult(
                messages=source,
                source_message_count=len(source),
                projection_mode="full_history",
                warnings=(),
            )

        retain_start_idx = min(max(checkpoint.retain_start_idx, 0), len(source))
        # 把摘要包装成 summary role 消息，放在投影列表的最前面
        # 模型看到的顺序：[summary（旧历史摘要）][最近N轮原文]
        summary = Message(
            content=checkpoint.summary,
            role="summary",
            metadata={
                "checkpoint_id": checkpoint.id,
                "source_message_count": checkpoint.source_message_count,
                "messages_compacted": checkpoint.messages_compacted,
            },
        )
        return ProjectionResult(
            messages=[summary] + source[retain_start_idx:],
            source_message_count=len(source),
            projection_mode="compact_checkpoint",
            warnings=(),
            compact_checkpoint_id=checkpoint.id,
        )
