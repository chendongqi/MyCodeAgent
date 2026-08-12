"""Runtime runner for the canonical single-agent turn loop."""

from __future__ import annotations

import uuid
from typing import Any

from core.llm import (
    extract_reasoning_content,
    extract_response_content,
    extract_response_meta,
    extract_tool_calls,
    extract_usage,
    serialize_response,
)
from runtime.completion import (
    CompletionGateVerdict,
    DeterministicCompletionVerifier,
    build_completion_candidate,
    collect_verification_evidence,
    infer_completion_requirements,
)
from runtime.events import (
    RuntimeEvent,
    create_runtime_event_sink,
    record_active_checkpoint,
    trace_model_request_state,
    transition_state,
)
from runtime.input_preprocess import preprocess_input
from runtime.model_errors import ModelErrorKind, classify_model_error
from runtime.state import LoopState, TerminalReason, TransitionReason


class RuntimeRunner:
    """Canonical single-agent turn loop."""

    def __init__(self, host: Any):
        self.host = host

    def _event_sink(self):
        host = self.host
        sink = getattr(host, "runtime_event_sink", None)
        if sink is None:
            sink = create_runtime_event_sink(
                getattr(host, "trace_logger", None),
                getattr(host, "transcript_recorder", None),
            )
            host.runtime_event_sink = sink
        if not callable(getattr(host, "emit_runtime_event", None)):
            host.emit_runtime_event = self._emit_runtime_event
        return sink

    def _emit(self, event_type: str, payload: dict[str, Any], *, step: int) -> None:
        self._emit_runtime_event(
            run_id=self._get_transcript_run_id(),
            step=step,
            event_type=event_type,
            payload=payload,
        )

    def _emit_runtime_event(
        self,
        *,
        run_id: str,
        step: int,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        self._event_sink().emit(
            RuntimeEvent(
                run_id=run_id,
                step=step,
                type=event_type,
                payload=payload,
            )
        )

    def _transition(
        self,
        state: LoopState,
        reason: TransitionReason,
        trace_logger,
        *,
        step: int | None = None,
        details: dict[str, Any] | None = None,
        **changes: Any,
    ) -> LoopState:
        return transition_state(
            state,
            reason,
            emit=lambda event, payload, event_step: self._emit(event, payload, step=event_step),
            step=step,
            details=details,
            **changes,
        )

    def _terminal(
        self,
        reason: TerminalReason,
        trace_logger,
        *,
        step: int = 0,
        **details: Any,
    ) -> None:
        self._emit("terminal", {"reason": reason.value, "details": details}, step=step)

    def _get_transcript_run_id(self) -> str:
        run_id = getattr(self.host, "_active_transcript_run_id", None)
        if run_id is not None:
            return str(run_id)
        fallback = getattr(self.host, "_run_id", 0)
        return f"run-{fallback}"

    def _record_active_transcript_checkpoint(self, *, step: int) -> None:
        record_active_checkpoint(
            self.host,
            emit=lambda event, payload, event_step: self._emit(event, payload, step=event_step),
            step=step,
        )

    def _trace_model_request_state(
        self,
        trace_logger,
        *,
        tools_schema: list[dict[str, Any]],
        step: int,
    ) -> None:
        trace_model_request_state(
            self.host,
            emit=lambda event, payload, event_step: self._emit(event, payload, step=event_step),
            tools_schema=tools_schema,
            step=step,
        )

    def _append_user_message(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        append_user = self.host.history_manager.append_user
        if metadata is None:
            append_user(content)
            return
        try:
            append_user(content, metadata=metadata)
        except TypeError:
            append_user(content)

    def _get_completion_verifier(self):
        verifier = getattr(self.host, "completion_verifier", None)
        if verifier is not None:
            return verifier
        verifier = DeterministicCompletionVerifier()
        self.host.completion_verifier = verifier
        return verifier

    def _get_model_recovery_limit(self, kind: ModelErrorKind) -> int:
        host = self.host
        if kind is ModelErrorKind.EMPTY_RESPONSE:
            return int(getattr(host, "empty_response_retry_limit", 1) or 1)
        if kind is ModelErrorKind.PROMPT_TOO_LONG:
            return 1
        if kind is ModelErrorKind.MAX_OUTPUT:
            return int(getattr(host, "max_output_recovery_limit", 0) or 0)
        return 0

    def _increment_model_recovery_count(self, state: LoopState, kind: ModelErrorKind) -> dict[str, int]:
        counts = dict(state.model_recovery_counts)
        counts[kind.value] = counts.get(kind.value, 0) + 1
        return counts

    def _trace_model_error_classified(
        self,
        trace_logger,
        *,
        step: int,
        stage: str,
        kind: ModelErrorKind,
        retry_count: int,
        retry_limit: int,
        message: str,
        finish_reason: str | None = None,
    ) -> None:
        self._emit(
            "model_error_classified",
            {
                "stage": stage,
                "kind": kind.value,
                "retry_count": retry_count,
                "retry_limit": retry_limit,
                "message": message,
                "finish_reason": finish_reason,
            },
            step=step,
        )

    def _trace_model_recovery_attempted(
        self,
        trace_logger,
        *,
        step: int,
        kind: ModelErrorKind,
        retry_count: int,
        retry_limit: int,
        action: str,
    ) -> None:
        self._emit(
            "model_recovery_attempted",
            {
                "kind": kind.value,
                "retry_count": retry_count,
                "retry_limit": retry_limit,
                "action": action,
            },
            step=step,
        )

    def _trace_model_recovery_failed(
        self,
        trace_logger,
        *,
        step: int,
        kind: ModelErrorKind,
        retry_count: int,
        retry_limit: int,
        reason: str,
    ) -> None:
        self._emit(
            "model_recovery_failed",
            {
                "kind": kind.value,
                "retry_count": retry_count,
                "retry_limit": retry_limit,
                "reason": reason,
            },
            step=step,
        )

    def run(self, input_text: str, **kwargs) -> str:
        show_raw = kwargs.pop("show_raw", False)
        processed_input, trace_logger, run_id = self._prepare_run(input_text, show_raw)
        response_text = ""
        try:
            response_text = self._react_loop(
                pending_input=processed_input,
                show_raw=show_raw,
                trace_logger=trace_logger,
            )
        finally:
            self._finish_run(trace_logger, run_id, response_text)
        return response_text

    def _prepare_run(self, input_text: str, show_raw: bool) -> tuple[str, Any, int]:
        """Refresh dynamic inputs and record the start of a user run."""
        host = self.host
        if not show_raw:
            host.last_response_raw = None
        if host.console_progress:
            host._console("⏳ Agent 正在处理，请稍候...")

        host._refresh_skills_prompt()
        host.context_builder.set_skills_prompt(host._skills_prompt)
        preprocess_result = preprocess_input(input_text)
        processed_input = preprocess_result.processed_input
        if preprocess_result.mentioned_files:
            mentioned = ", ".join(preprocess_result.mentioned_files)
            if host.console_verbose:
                host._console(f"\n📎 检测到文件引用: {mentioned}")
                if preprocess_result.truncated_count > 0:
                    host._console(f"   (另有 {preprocess_result.truncated_count} 个文件被省略)")
            elif host.logger.isEnabledFor(10):
                host.logger.debug("检测到文件引用: %s", mentioned)
                if preprocess_result.truncated_count > 0:
                    host.logger.debug("另有 %d 个文件被省略", preprocess_result.truncated_count)

        trace_logger = host.trace_logger
        if hasattr(trace_logger, "clear_current_run_events"):
            trace_logger.clear_current_run_events()
        host._turn_cancelled = False
        host._run_id += 1
        run_id = host._run_id
        host._active_transcript_run_id = f"run-{run_id}"
        host._log_system_messages_if_needed(trace_logger)
        self._emit(
            "run_start",
            {"run_id": run_id, "input": input_text, "processed": processed_input},
            step=0,
        )
        self._append_user_message(processed_input)
        self._emit(
            "message",
            {"role": "user", "content": processed_input, "metadata": {}},
            step=0,
        )
        self._emit(
            "user_input",
            {"text": input_text, "processed": processed_input},
            step=0,
        )
        if host.console_verbose:
            host._console(f"\n⚙️ Engine 启动: {input_text}")
        elif host.logger.isEnabledFor(10):
            host.logger.debug("Engine 启动: %s", input_text)
        return processed_input, trace_logger, run_id

    def _finish_run(self, trace_logger, run_id: int, response_text: str) -> None:
        """Record run completion and release run-scoped state."""
        host = self.host
        self._emit(
            "run_end",
            {"run_id": run_id, "final": response_text},
            step=0,
        )
        host._active_transcript_run_id = None
        if host.console_progress:
            host._console("✅ Agent 已完成")
        host.logger.debug("response=%s", response_text)
        host.logger.info(
            "history_size=%d, rounds=%d",
            host.history_manager.get_message_count(),
            host.history_manager.get_rounds_count(),
        )

    def _react_loop(self, pending_input: str, show_raw: bool, trace_logger) -> str:
        # ReAct 主循环：每次迭代 = 一个 step（思考→动作→观察）
        # 整体结构：外层 for 控制最大步数，内层 while 处理单步内的模型错误重试
        host = self.host
        tool_choice = "auto"
        # 完成门被拦截后最多重试次数（默认 2）
        completion_retry_limit = int(getattr(host, "completion_gate_retry_limit", 2) or 2)

        # 初始化不可变状态机，每次状态转移产生新对象，不修改原对象
        state = LoopState(
            messages=[],
            step=1,
            turn_count=1,
            tool_choice=tool_choice,
        )
        state = self._transition(
            state,
            TransitionReason.USER_INPUT,
            trace_logger,
            step=0,
            pending_input_len=len(pending_input or ""),
        )

        # ── 外层循环：每次迭代是一个 ReAct step ──────────────────────────────
        for step in range(1, host.max_steps + 1):
            # 构建本轮发给模型的 Model View（≠ 完整 history）：
            #   source  = history_manager 里的 append-only 全量消息（永不因压缩而删除）
            #   project = ProjectionBuilder 在读取时做非破坏性投影：
            #             无 checkpoint → full_history（原样）
            #             有 checkpoint → compact_checkpoint（摘要 + 最近 N 轮）
            #   view    = system prompt + session memory + 投影后的 history
            # 「有界」= token 预算控制：超阈值触发 compact，模型只看到投影后的子集。
            # pending_input 已在 _prepare_run 写入 history，此处仅用于 token 估算。
            state, tools_schema, messages = self._prepare_step_context(
                state=state,
                pending_input=pending_input,
                step=step,
                trace_logger=trace_logger,
            )
            # base_messages 用于空响应重试时拼接提示，保留本步的原始消息快照
            base_messages = messages

            response_text = ""
            tool_calls: list[dict[str, Any]] = []
            reasoning_content = None
            response_meta: dict[str, Any] = {}

            # ── 内层循环：单步内的模型调用与错误恢复 ──────────────────────────
            # 正常情况一次 break 出去；出错时根据错误类型决定重试或终止
            while True:
                try:
                    # 调 LLM，拿到原始响应对象
                    raw_response = host.llm.invoke_raw(messages, tools=tools_schema, tool_choice=tool_choice)
                except Exception as exc:
                    # ── 模型调用异常处理 ──
                    # 对异常分类：PROMPT_TOO_LONG / API_ERROR / UNKNOWN 等
                    classification = classify_model_error(error=exc)
                    retry_count = state.model_recovery_counts.get(classification.kind.value, 0)
                    retry_limit = self._get_model_recovery_limit(classification.kind)
                    self._trace_model_error_classified(
                        trace_logger,
                        step=step,
                        stage="model_invoke",
                        kind=classification.kind,
                        retry_count=retry_count,
                        retry_limit=retry_limit,
                        message=classification.message,
                        finish_reason=classification.finish_reason,
                    )

                    if (
                        classification.kind is ModelErrorKind.PROMPT_TOO_LONG
                        and retry_count < retry_limit
                        and hasattr(host.context_engine, "reactive_compact")
                    ):
                        # PROMPT_TOO_LONG：触发上下文压缩后重试
                        # reactive_compact 对旧历史做 LLM 摘要，缩短 token 后重建 model view
                        next_retry_count = retry_count + 1
                        recovery_counts = self._increment_model_recovery_count(state, classification.kind)
                        self._trace_model_recovery_attempted(
                            trace_logger,
                            step=step,
                            kind=classification.kind,
                            retry_count=next_retry_count,
                            retry_limit=retry_limit,
                            action="reactive_compact",
                        )
                        compact_info = host.context_engine.reactive_compact(
                            history_manager=host.history_manager,
                            pending_input=pending_input,
                            step=step,
                            trace_logger=trace_logger,
                        )
                        if compact_info.get("compacted"):
                            # 压缩成功：记录 checkpoint，重建 model view，内层 continue 重试
                            self._record_active_transcript_checkpoint(step=step)
                            state = self._transition(
                                state,
                                TransitionReason.MODEL_RECOVERY_RETRY,
                                trace_logger,
                                step=step,
                                model_recovery_counts=recovery_counts,
                                compact_attempted=True,
                                last_model_error_kind=classification.kind.value,
                                last_model_error_stage="model_invoke",
                                last_error=classification.message,
                                details={
                                    "error_kind": classification.kind.value,
                                    "retry_count": next_retry_count,
                                    "retry_limit": retry_limit,
                                    "action": "reactive_compact",
                                    "checkpoint_id": compact_info.get("checkpoint_id"),
                                },
                            )
                            model_view = host.context_engine.build_model_view(
                                history_manager=host.history_manager,
                                pending_input=pending_input,
                                step=step,
                                trace_logger=trace_logger,
                            )
                            messages = model_view.messages
                            base_messages = messages
                            state = state.update(messages=messages)
                            self._emit(
                                "context_build",
                                {
                                    "message_count": len(messages),
                                    "history_count": model_view.history_message_count,
                                    "source_message_count": model_view.source_message_count,
                                    "projection_mode": model_view.projection_mode,
                                },
                                step=step,
                            )
                            continue

                        # 压缩失败：无法恢复，走终止路径
                        self._trace_model_recovery_failed(
                            trace_logger,
                            step=step,
                            kind=classification.kind,
                            retry_count=next_retry_count,
                            retry_limit=retry_limit,
                            reason=str(compact_info.get("reason") or "reactive_compact_failed"),
                        )
                        state = self._transition(
                            state,
                            TransitionReason.MODEL_RECOVERY_FAILED,
                            trace_logger,
                            step=step,
                            model_recovery_counts=recovery_counts,
                            compact_attempted=True,
                            last_model_error_kind=classification.kind.value,
                            last_model_error_stage="model_invoke",
                            last_error=classification.message,
                            details={
                                "error_kind": classification.kind.value,
                                "retry_count": next_retry_count,
                                "retry_limit": retry_limit,
                                "action": "reactive_compact",
                                "reason": compact_info.get("reason"),
                            },
                        )
                    else:
                        # 其他错误或重试次数已耗尽：直接标记恢复失败
                        self._trace_model_recovery_failed(
                            trace_logger,
                            step=step,
                            kind=classification.kind,
                            retry_count=retry_count,
                            retry_limit=retry_limit,
                            reason="non_recoverable" if retry_limit == 0 else "retry_exhausted",
                        )
                        state = self._transition(
                            state,
                            TransitionReason.MODEL_RECOVERY_FAILED,
                            trace_logger,
                            step=step,
                            last_model_error_kind=classification.kind.value,
                            last_model_error_stage="model_invoke",
                            last_error=classification.message,
                            details={
                                "error_kind": classification.kind.value,
                                "retry_count": retry_count,
                                "retry_limit": retry_limit,
                            },
                        )

                    # 模型调用异常无法恢复，终止整个 loop
                    self._terminal(
                        TerminalReason.MODEL_ERROR,
                        trace_logger,
                        step=step,
                        error_kind=classification.kind.value,
                        message=classification.message,
                        retry_count=retry_count,
                        retry_limit=retry_limit,
                    )
                    return "抱歉，我无法在限定步数内完成这个任务。"

                # ── 模型调用成功，解析响应 ────────────────────────────────────
                if show_raw:
                    # --show-raw 模式：把原始响应存到 host 上，供 CLI 打印调试
                    host.last_response_raw = (
                        raw_response.model_dump()
                        if hasattr(raw_response, "model_dump")
                        else raw_response
                    )

                # 从同一份 raw_response 拆出不同用途的字段（core/llm.py 统一解析）：
                # response_text     ← choices[0].message.content，模型输出的正文
                # reasoning_content ← message.reasoning_content / reasoning（部分模型有）
                # usage             ← response.usage，本轮 token 用量
                # response_meta     ← 响应的结构化信号（不含正文），供容错/完成门/trace
                # tool_calls        ← message.tool_calls，Function Calling 列表
                # raw_dump          ← 完整响应的 JSON 快照，供 trace 审计
                response_text = extract_response_content(raw_response) or ""
                reasoning_content = extract_reasoning_content(raw_response)
                usage = extract_usage(raw_response)
                if usage and usage.get("total_tokens") is not None:
                    host.context_engine.record_usage(usage["total_tokens"])
                    max_total_tokens = int(getattr(host, "max_total_tokens", 0) or 0)
                    if max_total_tokens and host.context_engine.total_usage_tokens > max_total_tokens:
                        # token 累计用量超出预算上限，强制终止
                        state = self._transition(
                            state,
                            TransitionReason.TOKEN_BUDGET_EXCEEDED,
                            trace_logger,
                            step=step,
                            details={
                                "total_tokens": host.context_engine.total_usage_tokens,
                                "token_budget": max_total_tokens,
                            },
                        )
                        self._terminal(
                            TerminalReason.TOKEN_BUDGET,
                            trace_logger,
                            step=step,
                            total_tokens=host.context_engine.total_usage_tokens,
                            token_budget=max_total_tokens,
                        )
                        return "抱歉，我无法在限定预算内完成这个任务。"

                # response_meta 与 response_text 互补：
                #   text → 写入 history、完成门候选答案、展示给用户
                #   meta → finish_reason/content_len 等，驱动空响应重试、输出截断判定、状态机与 trace
                response_meta = extract_response_meta(raw_response)
                tool_calls = extract_tool_calls(raw_response)
                raw_dump = serialize_response(raw_response)
                # 把所有信息 emit 出去，供 trace 审计、完成门候选、空响应重试等决策使用
                self._emit(
                    "model_output",
                    {
                        "raw": response_text,
                        "usage": usage,
                        "meta": response_meta,
                        "raw_response": raw_dump,
                        "tool_calls": tool_calls,
                    },
                    step=step,
                )

                if host.console_verbose and reasoning_content:
                    display_reasoning = reasoning_content
                    if len(display_reasoning) > 1200:
                        display_reasoning = display_reasoning[:1200] + "...(truncated)"
                    host._console(f"\n🧠 Reasoning: {display_reasoning}\n")

                # 检查响应内容是否有问题（空响应 / 输出超长截断）
                classification = None
                candidate_error = classify_model_error(
                    response_text=response_text,
                    tool_calls=tool_calls,
                    response_meta=response_meta,
                )
                if candidate_error.kind in {ModelErrorKind.EMPTY_RESPONSE, ModelErrorKind.MAX_OUTPUT}:
                    classification = candidate_error

                # 响应正常，跳出内层 while，进入后续的 tool_calls / 完成门判断
                if classification is None:
                    break

                # ── 响应异常处理（空响应 / 输出截断）────────────────────────────
                retry_count = state.model_recovery_counts.get(classification.kind.value, 0)
                retry_limit = self._get_model_recovery_limit(classification.kind)
                self._trace_model_error_classified(
                    trace_logger,
                    step=step,
                    stage="model_response",
                    kind=classification.kind,
                    retry_count=retry_count,
                    retry_limit=retry_limit,
                    message=classification.message,
                    finish_reason=classification.finish_reason,
                )
                # ── 空响应重试 ──────────────────────────────────────────────
                if classification.kind is ModelErrorKind.EMPTY_RESPONSE and retry_count < retry_limit:
                    # 空响应：追加一条提示 user 消息告诉模型"请给出回答"，内层 continue 重试
                    next_retry_count = retry_count + 1
                    recovery_counts = self._increment_model_recovery_count(state, classification.kind)
                    hint = "上次 content 为空且未返回 tool_calls，请在 content 中回复最终答案，或使用工具调用。"
                    messages = base_messages + [{"role": "user", "content": hint}]
                    self._trace_model_recovery_attempted(
                        trace_logger,
                        step=step,
                        kind=classification.kind,
                        retry_count=next_retry_count,
                        retry_limit=retry_limit,
                        action="retry_with_hint",
                    )
                    state = self._transition(
                        state,
                        TransitionReason.MODEL_EMPTY_RETRY,
                        trace_logger,
                        step=step,
                        model_recovery_counts=recovery_counts,
                        last_model_error_kind=classification.kind.value,
                        last_model_error_stage="model_response",
                        last_error=classification.message,
                        last_response_meta=response_meta,
                        details={
                            "error_kind": classification.kind.value,
                            "finish_reason": response_meta.get("finish_reason"),
                            "retry_count": next_retry_count,
                            "retry_limit": retry_limit,
                        },
                    )
                    self._emit(
                        "empty_response_retry",
                        {
                            "finish_reason": response_meta.get("finish_reason"),
                            "content_len": response_meta.get("content_len"),
                            "reasoning_len": response_meta.get("reasoning_len"),
                            "hint": hint,
                        },
                        step=step,
                    )
                    if host.console_verbose:
                        host._console("⚠️ LLM返回空响应，追加提示后重试一次")
                    else:
                        host.logger.warning("LLM返回空响应，追加提示后重试一次")
                    continue

                # 重试次数耗尽或不可恢复，终止
                self._trace_model_recovery_failed(
                    trace_logger,
                    step=step,
                    kind=classification.kind,
                    retry_count=retry_count,
                    retry_limit=retry_limit,
                    reason="retry_exhausted" if retry_limit else "non_recoverable",
                )
                transition_reason = (
                    TransitionReason.MODEL_EMPTY_FAILED
                    if classification.kind is ModelErrorKind.EMPTY_RESPONSE
                    else TransitionReason.MODEL_RECOVERY_FAILED
                )
                state_changes: dict[str, Any] = {
                    "last_response_meta": response_meta,
                    "last_model_error_kind": classification.kind.value,
                    "last_model_error_stage": "model_response",
                    "last_error": classification.message,
                }
                if classification.kind is ModelErrorKind.MAX_OUTPUT:
                    state_changes["max_output_recovery_count"] = state.max_output_recovery_count + 1
                state = self._transition(
                    state,
                    transition_reason,
                    trace_logger,
                    step=step,
                    details={
                        "error_kind": classification.kind.value,
                        "finish_reason": response_meta.get("finish_reason"),
                        "retry_count": retry_count,
                        "retry_limit": retry_limit,
                    },
                    **state_changes,
                )
                terminal_reason = (
                    TerminalReason.EMPTY_RESPONSE_FAILED
                    if classification.kind is ModelErrorKind.EMPTY_RESPONSE
                    else TerminalReason.MODEL_ERROR
                )
                self._terminal(
                    terminal_reason,
                    trace_logger,
                    step=step,
                    error_kind=classification.kind.value,
                    finish_reason=response_meta.get("finish_reason"),
                    retry_count=retry_count,
                    retry_limit=retry_limit,
                )
                if classification.kind is ModelErrorKind.EMPTY_RESPONSE:
                    self._emit(
                        "error",
                        {
                            "stage": "llm_response",
                            "error_code": "INTERNAL_ERROR",
                            "message": "Empty response",
                            "meta": response_meta,
                        },
                        step=step,
                    )
                return "抱歉，我无法在限定步数内完成这个任务。"

            # ── 内层 while 正常 break 出来，处理本步的响应结果 ─────────────────

            if tool_calls:
                # ── Acting 分支：模型返回了工具调用 ──────────────────────────
                # 1. 记录状态转移
                state = self._transition(
                    state,
                    TransitionReason.MODEL_RETURNED_TOOL_CALLS,
                    trace_logger,
                    step=step,
                    last_tool_calls=tool_calls,
                    last_response_meta=response_meta,
                    details={"tool_count": len(tool_calls)},
                )
                # 2. 确保每个 tool_call 有 id（部分模型不返回 id）
                for call in tool_calls:
                    if not call.get("id"):
                        call["id"] = f"call_{uuid.uuid4().hex}"
                # 3. 把 assistant 消息（含 tool_calls）写入历史
                assistant_content = str(response_text or "")
                host.history_manager.append_assistant(
                    content=assistant_content,
                    metadata={
                        "step": step,
                        "action_type": "tool_call",
                        "tool_calls": tool_calls,
                    },
                    reasoning_content=reasoning_content,
                )
                self._emit(
                    "message",
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "metadata": {"action_type": "tool_call", "tool_calls": tool_calls},
                    },
                    step=step,
                )
                # 4. 执行工具（ToolOrchestrator 负责并发/串行调度）
                observations = host.tool_orchestrator.run(
                    tool_calls,
                    step=step,
                    trace_logger=trace_logger,
                )
                # 5. 把每个工具的执行结果作为 tool 消息写入历史，供下一步模型看到
                for obs in observations:
                    obs_metadata = getattr(obs, "metadata", None) or {}
                    host.history_manager.append_tool(
                        tool_name=obs.tool_name,
                        observation=obs.observation,
                        metadata={
                            "step": step,
                            "tool_call_id": obs.tool_call_id,
                            **obs_metadata,
                        },
                    )
                    self._emit(
                        "message",
                        {
                            "role": "tool",
                            "content": obs.observation,
                            "metadata": {
                                "tool_name": obs.tool_name,
                                "tool_call_id": obs.tool_call_id,
                                **obs_metadata,
                            },
                        },
                        step=step,
                    )

                    if host.console_verbose:
                        display_obs = (
                            obs.observation[:300] + "..." if len(obs.observation) > 300 else obs.observation
                        )
                        host._console(f"\n👀 Observation: {display_obs}\n")
                    elif host.logger.isEnabledFor(10):
                        display_obs = (
                            obs.observation[:300] + "..." if len(obs.observation) > 300 else obs.observation
                        )
                        host.logger.debug("Observation: %s", display_obs)
                state = self._transition(
                    state,
                    TransitionReason.TOOLS_EXECUTED,
                    trace_logger,
                    step=step,
                    last_tool_calls=tool_calls,
                    details={"tool_count": len(tool_calls)},
                )
                # 工具执行完，外层 for 继续下一个 step（模型看到观测结果后再思考）
                continue

            # ── Reasoning 分支：模型没有返回工具调用，输出了文字 ─────────────
            # 不能直接返回，先经过完成门检查
            final_text = str(response_text).strip()

            # 收集完成检查所需的三类信息：
            # candidate：本次回答内容
            # requirements：从用户输入推断出的完成要求（如"需要跑测试"）
            # evidence：历史中的工具执行证据（如"Bash 跑了 pytest"）
            candidate = build_completion_candidate(
                final_text=final_text,
                step=step,
                response_meta=response_meta,
                history_messages=host.history_manager.get_messages(),
            )
            self._emit(
                "completion_candidate",
                candidate.to_trace_payload(),
                step=step,
            )

            requirements = infer_completion_requirements(
                user_input=pending_input,
                history_messages=host.history_manager.get_messages(),
            )
            self._emit(
                "completion_requirements",
                requirements.to_trace_payload(),
                step=step,
            )

            evidence = collect_verification_evidence(host.history_manager.get_messages())
            for item in evidence:
                self._emit("verification_evidence", item.to_trace_payload(), step=step)

            # 完成门判决：PASS / UNVERIFIED / FAIL
            verdict = self._get_completion_verifier().evaluate(
                candidate,
                requirements,
                evidence,
                host.history_manager.get_messages(),
            )
            self._emit(
                "completion_gate_verdict",
                verdict.to_trace_payload(),
                step=step,
            )

            if verdict.verdict in {CompletionGateVerdict.PASS, CompletionGateVerdict.UNVERIFIED}:
                # ── 完成门通过：正常出口 ──────────────────────────────────────
                # PASS：确认完成；UNVERIFIED：有验证要求但缺证据，以"未确认完成"状态退出
                action_type = "final" if verdict.verdict is CompletionGateVerdict.PASS else "final_unverified"
                host.history_manager.append_assistant(
                    content=final_text,
                    metadata={"step": step, "action_type": action_type},
                    reasoning_content=reasoning_content,
                )
                self._emit(
                    "message",
                    {
                        "role": "assistant",
                        "content": final_text,
                        "metadata": {"action_type": action_type},
                    },
                    step=step,
                )
                state = self._transition(
                    state,
                    TransitionReason.MODEL_RETURNED_FINAL,
                    trace_logger,
                    step=step,
                    last_response_meta={
                        "final_length": len(final_text),
                        "completion_verdict": verdict.verdict.value,
                    },
                    details={
                        "final_length": len(final_text),
                        "completion_verdict": verdict.verdict.value,
                    },
                )
                terminal_reason = (
                    TerminalReason.COMPLETED
                    if verdict.verdict is CompletionGateVerdict.PASS
                    else TerminalReason.COMPLETED_UNVERIFIED
                )
                self._terminal(
                    terminal_reason,
                    trace_logger,
                    step=step,
                    final_length=len(final_text),
                    completion_verdict=verdict.verdict.value,
                )
                self._emit(
                    "finish",
                    {"final": final_text, "completion_verdict": verdict.verdict.value},
                    step=step,
                )
                return final_text

            # ── 完成门拦截（FAIL）：注入反馈，让模型重新来过 ─────────────────
            # 把本次"失败的回答"记入历史，同时把门的拦截原因作为 user 消息注入，
            # 模型下一步能看到"为什么不通过"并做出修正
            host.history_manager.append_assistant(
                content=final_text,
                metadata={"step": step, "action_type": "final_candidate"},
                reasoning_content=reasoning_content,
            )
            self._emit(
                "message",
                {
                    "role": "assistant",
                    "content": final_text,
                    "metadata": {"action_type": "final_candidate"},
                },
                step=step,
            )
            block_count = state.completion_block_count + 1
            feedback = verdict.blocking_feedback or "Completion blocked by runtime gate."
            self._append_user_message(
                feedback,
                metadata={"step": step, "source": "completion_gate"},
            )
            self._emit(
                "message",
                {
                    "role": "user",
                    "content": feedback,
                    "metadata": {"source": "completion_gate"},
                },
                step=step,
            )
            state = self._transition(
                state,
                TransitionReason.STOP_HOOK_BLOCKING,
                trace_logger,
                step=step,
                completion_block_count=block_count,
                stop_hook_active=True,
                details={
                    "completion_verdict": verdict.verdict.value,
                    "reasons": list(verdict.reasons),
                    "retry_count": block_count,
                    "retry_limit": completion_retry_limit,
                },
            )
            if block_count >= completion_retry_limit:
                # 完成门反馈重试次数耗尽，强制终止
                self._terminal(
                    TerminalReason.COMPLETION_GATE_BLOCKED,
                    trace_logger,
                    step=step,
                    completion_verdict=verdict.verdict.value,
                    reasons=list(verdict.reasons),
                    retry_count=block_count,
                    retry_limit=completion_retry_limit,
                )
                return "抱歉，我无法在限定步数内完成这个任务。"
            # 注入反馈后外层 for 继续下一个 step，模型看到反馈后重新思考
            continue

        # ── 超出最大步数，强制终止 ───────────────────────────────────────────
        state = self._transition(
            state,
            TransitionReason.MAX_STEPS_EXCEEDED,
            trace_logger,
            step=host.max_steps,
            details={"max_steps": host.max_steps},
        )
        self._terminal(
            TerminalReason.MAX_STEPS,
            trace_logger,
            step=host.max_steps,
            max_steps=host.max_steps,
        )
        return "抱歉，我无法在限定步数内完成这个任务。"

    def _prepare_step_context(
        self,
        *,
        state: LoopState,
        pending_input: str,
        step: int,
        trace_logger,
    ) -> tuple[LoopState, list[dict[str, Any]], list[dict[str, Any]]]:
        """Refresh runtime signals, compact history, and build the model view.

        返回的 messages 是 ModelView，不是 history_manager 的原始列表。
        流程：compact_if_needed → build_model_view（内部 project + normalize）。
        """
        host = self.host
        tools_schema = host._get_openai_tools_for_current_mode()
        self._trace_model_request_state(
            trace_logger,
            tools_schema=tools_schema,
            step=step,
        )
        if host.console_verbose:
            host._console(f"\n--- Step {step}/{host.max_steps} ---")
        elif host.console_progress:
            host._console(f"… Step {step}/{host.max_steps}")
        elif host.logger.isEnabledFor(10):
            host.logger.debug("Step %d/%d", step, host.max_steps)

        compact_info = host.context_engine.compact_if_needed(
            history_manager=host.history_manager,
            pending_input=pending_input,
            step=step,
            trace_logger=trace_logger,
        )
        if compact_info.get("compacted"):
            self._record_active_transcript_checkpoint(step=step)
            state = self._transition(
                state,
                TransitionReason.CONTEXT_COMPACTED,
                trace_logger,
                step=step,
                compact_attempted=True,
                details={
                    "checkpoint_id": compact_info.get("checkpoint_id"),
                    "messages_compacted": compact_info.get("messages_compacted"),
                    "retain_start_idx": compact_info.get("retain_start_idx"),
                },
            )
            final_context = host.context_engine.build_model_view(
                history_manager=host.history_manager,
                pending_input=pending_input,
                step=step,
                trace_logger=trace_logger,
            ).messages
            self._emit(
                "history_compression_final_context",
                {"message_count": len(final_context), "messages": final_context},
                step=step,
            )
            if host.console_verbose:
                host._console("\n📦 触发历史压缩...")
                host._console(
                    "✅ 压缩完成，当前轮次数: %d"
                    % host.history_manager.get_rounds_count()
                )
                host._print_context_preview(final_context)
            elif host.logger.isEnabledFor(10):
                host.logger.debug("触发历史压缩")
                host.logger.debug(
                    "压缩完成，当前轮次数: %d",
                    host.history_manager.get_rounds_count(),
                )
                host._print_context_preview(final_context)

        model_view = host.context_engine.build_model_view(
            history_manager=host.history_manager,
            pending_input=pending_input,
            step=step,
            trace_logger=trace_logger,
        )
        messages = model_view.messages
        state = state.update(step=step, messages=messages)
        self._emit(
            "context_build",
            {
                "message_count": len(messages),
                "history_count": model_view.history_message_count,
                "source_message_count": model_view.source_message_count,
                "projection_mode": model_view.projection_mode,
            },
            step=step,
        )
        return state, tools_schema, messages
