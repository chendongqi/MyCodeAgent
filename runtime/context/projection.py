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
        source = list(source_messages or [])
        checkpoint = self.compact_store.active_checkpoint if self.compact_store else None
        if not checkpoint:
            return ProjectionResult(
                messages=source,
                source_message_count=len(source),
                projection_mode="full_history",
                warnings=(),
            )

        retain_start_idx = min(max(checkpoint.retain_start_idx, 0), len(source))
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
