"""Context model-view construction."""

from __future__ import annotations

from typing import Any

from core.config import Config
from runtime.context.budget import ContextBudgetPolicy
from runtime.context.compact import ContextCompactor
from runtime.context.compact_store import CompactStore
from runtime.context.model_view import ModelView
from runtime.context.normalizer import MessageNormalizer
from runtime.context.projection import ProjectionBuilder
from runtime.session_memory import SessionMemory, render_session_memory


class ContextEngine:
    """Builds the exact model-facing context for a loop iteration."""

    def __init__(
        self,
        context_builder: Any,
        *,
        config: Config | None = None,
        summary_generator: Any = None,
        compact_store: CompactStore | None = None,
        budget_policy: ContextBudgetPolicy | None = None,
        compactor: ContextCompactor | None = None,
        projection_builder: ProjectionBuilder | None = None,
        normalizer: MessageNormalizer | None = None,
    ):
        self.context_builder = context_builder
        self.config = config or Config.from_env()
        self.compact_store = compact_store or CompactStore()
        self.budget_policy = budget_policy or ContextBudgetPolicy(self.config)
        self.compactor = compactor or ContextCompactor(
            config=self.config,
            compact_store=self.compact_store,
            summary_generator=summary_generator,
        )
        self.projection_builder = projection_builder or ProjectionBuilder(self.compact_store)
        self.normalizer = normalizer or MessageNormalizer()
        self.last_usage_tokens = 0
        self.total_usage_tokens = 0
        self.session_memory: SessionMemory | None = None

    def record_usage(self, total_tokens: int | None) -> None:
        if total_tokens is None:
            return
        self.last_usage_tokens = int(total_tokens)
        self.total_usage_tokens += int(total_tokens)

    def reset(self) -> None:
        """Reset session-scoped context state."""
        self.compact_store.clear()
        self.last_usage_tokens = 0
        self.total_usage_tokens = 0
        self.session_memory = None

    def set_session_memory(self, memory: SessionMemory | None) -> None:
        self.session_memory = memory

    def should_compact(self, *, history_manager: Any, pending_input: str) -> bool:
        source_messages = history_manager.get_messages()
        checkpoint = self.compact_store.active_checkpoint
        if checkpoint and checkpoint.source_message_count == len(source_messages):
            return False
        projection = self.projection_builder.project(source_messages)
        decision = self.budget_policy.should_compact(
            messages=projection.messages,
            pending_input=pending_input,
            last_usage_tokens=self.last_usage_tokens,
        )
        return decision.should_compact

    def compact_if_needed(
        self,
        *,
        history_manager: Any,
        pending_input: str,
        step: int = 0,
        trace_logger: Any = None,
    ) -> dict[str, Any]:
        source_messages = history_manager.get_messages()
        checkpoint = self.compact_store.active_checkpoint
        if checkpoint and checkpoint.source_message_count == len(source_messages):
            return {
                "compacted": False,
                "reason": "checkpoint_current",
                "checkpoint_id": checkpoint.id,
            }

        projection = self.projection_builder.project(source_messages)
        decision = self.budget_policy.should_compact(
            messages=projection.messages,
            pending_input=pending_input,
            last_usage_tokens=self.last_usage_tokens,
        )
        if trace_logger:
            trace_logger.log_event(
                "context_compaction_decision",
                {
                    "should_compact": decision.should_compact,
                    "reason": decision.reason,
                    "estimated_tokens": decision.estimated_tokens,
                    "threshold": decision.threshold,
                    "message_count": decision.message_count,
                },
                step=step,
            )
        if not decision.should_compact:
            return {
                "compacted": False,
                "reason": decision.reason,
                "estimated_tokens": decision.estimated_tokens,
                "threshold": decision.threshold,
            }
        info = self.compactor.compact(source_messages)
        if trace_logger:
            event_name = (
                "context_compaction_completed"
                if info.get("compacted")
                else "context_compaction_skipped"
            )
            trace_logger.log_event(event_name, info, step=step)
        return info

    def build_model_view(
        self,
        *,
        history_manager: Any,
        pending_input: str = "",
        step: int = 0,
        trace_logger: Any = None,
    ) -> ModelView:
        """组装本 step 发给 LLM 的完整 Model View（系列 04 与 09 的汇合点）。

        最终 messages 顺序（见 learning/blog/code-agent-series-04-prompt-assembly.md）：
          [system]×N  ← ContextBuilder.get_system_messages()
                        Constitution / Tool Contracts / Project Rules / Runtime Signals
          [system]×0-1 ← session memory（跨 run 摘要前馈，不在 PromptAssembly 四层里）
          [user|assistant|tool|...] ← history 的有界投影 + OpenAI 格式规范化

        注意：tools= JSON schema 不在这里，由 loop 单独传给 llm.invoke_raw(tools=...)。
        """
        # ── 1. History 有界投影（系列 09，与 prompt 分层正交）────────────────
        # source：HistoryManager 的 append-only 全量事实，永不因压缩而删除
        source_messages = history_manager.get_messages()
        # project：读时投影；有 compact checkpoint 时 → [summary] + 最近 N 轮
        projection = self.projection_builder.project(source_messages)
        # normalize：Message → OpenAI dict（assistant 补 tool_calls、summary → system 等）
        history_messages = self.normalizer.normalize(projection.messages)

        # ── 2. System 层：agent「人格 + 能力说明书 + 项目规则」（系列 04）──────
        # 来自 runtime/prompt_builder.py ContextBuilder.get_prompt_assembly()：
        #   [0] Constitution     L1_system_prompt.py（或 --system 整段替换）
        #   [1] Tool Contracts   prompts/tools_prompts/*.py + Skills/MCP/熔断插槽
        #   [2] Project Rules    code_law.md（可选）
        #   [3+] Runtime Signals set_runtime_system_blocks() 注入的可变通知
        # 稳定三层按 fingerprint 缓存；Skills/MCP/runtime 变更时 setter 清空缓存
        system_messages = self.context_builder.get_system_messages()

        # ── 3. 动态 system：Session Memory（系列 10 前馈，按字符预算裁剪）──────
        # 与 PromptAssembly 分层存放：是 transcript 衍生的跨轮摘要，不是人设文件
        dynamic_messages: list[dict[str, Any]] = []
        session_memory_chars = 0
        session_memory_message_count = 0
        dynamic_sources: list[str] = []
        if self.session_memory is not None:
            budget = max(0, int(getattr(self.config, "session_memory_char_budget", 4000) or 4000))
            rendered, session_memory_chars = render_session_memory(self.session_memory, char_budget=budget)
            if rendered:
                dynamic_messages.append({"role": "system", "content": rendered})
                session_memory_message_count = 1
                dynamic_sources.append("session_memory")

        # ── 4. 拼接：system 在前，history 在后（Message List，非 scratchpad）──
        # 典型头部：[Constitution][Tool Contracts][CODE_LAW][Session Memory?][user...]
        messages = list(system_messages) + dynamic_messages + list(history_messages)

        # ── 5. 体量估算（供 compaction 决策与 trace，非精确 token 计数）────────
        # pending_input 已在 _prepare_run 写入 history；此处计入是为首步预算估算
        estimated_chars = len(pending_input or "")
        for message in messages:
            estimated_chars += len(str(message.get("content", "")))

        # ── 6. 封装 ModelView：messages + 可观测元数据 ─────────────────────────
        # source_message_count vs history_message_count：压缩前后条数对比（trace 用）
        # projection_mode：full_history | compact_checkpoint
        view = ModelView(
            messages=messages,
            system_message_count=len(system_messages),
            history_message_count=len(history_messages),
            source_message_count=projection.source_message_count,
            estimated_chars=estimated_chars,
            projection_mode=projection.projection_mode,
            compact_checkpoint_id=projection.compact_checkpoint_id,
            warnings=projection.warnings,
            dynamic_message_count=len(dynamic_messages),
            session_memory_message_count=session_memory_message_count,
            session_memory_chars=session_memory_chars,
            dynamic_context_sources=tuple(dynamic_sources),
        )

        # trace 事件 model_view_build：调试「这轮模型看到了什么」
        # 配合 trace_model_request_state 的 prompt fingerprint，可对比 prompt 漂移
        if trace_logger:
            trace_logger.log_event(
                "model_view_build",
                {
                    "message_count": view.message_count,
                    "system_message_count": view.system_message_count,
                    "history_message_count": view.history_message_count,
                    "source_message_count": view.source_message_count,
                    "estimated_chars": view.estimated_chars,
                    "projection_mode": view.projection_mode,
                    "compact_checkpoint_id": view.compact_checkpoint_id,
                    "warnings": list(view.warnings),
                    "dynamic_message_count": view.dynamic_message_count,
                    "session_memory_message_count": view.session_memory_message_count,
                    "session_memory_chars": view.session_memory_chars,
                    "dynamic_context_sources": list(view.dynamic_context_sources),
                },
                step=step,
            )

        return view

    def reactive_compact(
        self,
        *,
        history_manager: Any,
        pending_input: str,
        step: int = 0,
        trace_logger: Any = None,
    ) -> dict[str, Any]:
        source_messages = history_manager.get_messages()
        checkpoint = self.compact_store.active_checkpoint
        if checkpoint and checkpoint.source_message_count == len(source_messages):
            info = {
                "compacted": False,
                "reason": "checkpoint_current",
                "checkpoint_id": checkpoint.id,
            }
        else:
            info = self.compactor.compact(source_messages)
            if not info.get("reason"):
                info["reason"] = "reactive_prompt_too_long"

        if trace_logger:
            event_name = (
                "context_compaction_completed"
                if info.get("compacted")
                else "context_compaction_skipped"
            )
            trace_logger.log_event(event_name, info, step=step)
        return info
