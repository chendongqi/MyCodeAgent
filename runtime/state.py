"""Lightweight loop state for the runtime agent harness."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class TransitionReason(str, Enum):
    """每次状态转移的原因，记录 loop 的每一步"为什么走到这里"。

    设计目的：trace 和 debug 时能还原整条执行路径，
    而不是只看到"最终结果"。每个枚举值对应 loop.py 里一处 _transition() 调用。
    """
    USER_INPUT = "user_input"                          # 新一轮对话开始，用户输入已就绪
    CONTEXT_COMPACTED = "context_compacted"            # 主动触发了上下文压缩
    MODEL_EMPTY_RETRY = "model_empty_retry"            # 模型返回空响应，注入提示后重试
    MODEL_EMPTY_FAILED = "model_empty_failed"          # 空响应重试耗尽，放弃
    MODEL_RECOVERY_RETRY = "model_recovery_retry"      # PROMPT_TOO_LONG 压缩后重试
    MODEL_RECOVERY_FAILED = "model_recovery_failed"    # 模型错误恢复失败
    MODEL_RETURNED_TOOL_CALLS = "model_returned_tool_calls"  # 模型返回了工具调用（Acting）
    TOOLS_EXECUTED = "tools_executed"                  # 工具执行完毕，观测结果已写入历史
    MODEL_RETURNED_FINAL = "model_returned_final"      # 模型返回了最终文字回答（Reasoning）
    STOP_HOOK_BLOCKING = "stop_hook_blocking"          # 完成门拦截，注入反馈后继续
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"          # 超出最大步数限制
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"    # token 累计用量超出预算
    UNRECOVERABLE_ERROR = "unrecoverable_error"        # 不可恢复的错误


class TerminalReason(str, Enum):
    """loop 的终止原因，对应所有可能的出口。

    分两类：
    - 正常出口：COMPLETED / COMPLETED_UNVERIFIED
    - 异常出口：其余所有值

    每个终止事件都会写入 transcript，是崩溃恢复时判断
    "这轮是否正常完成"的依据。
    """
    COMPLETED = "completed"                            # 完成门 PASS，正常完成
    COMPLETED_UNVERIFIED = "completed_unverified"      # 完成门 UNVERIFIED，带不确定性的完成
    EMPTY_RESPONSE_FAILED = "empty_response_failed"    # 空响应重试耗尽
    COMPLETION_GATE_BLOCKED = "completion_gate_blocked"  # 完成门反馈重试耗尽
    MAX_STEPS = "max_steps"                            # 超出最大步数
    TOOL_ERROR_UNRECOVERABLE = "tool_error_unrecoverable"  # 工具不可恢复错误
    USER_ABORT = "user_abort"                          # 用户主动中断（Ctrl+C）
    MODEL_ERROR = "model_error"                        # 模型调用错误
    TOKEN_BUDGET = "token_budget"                      # token 预算耗尽


@dataclass(frozen=True)
class Transition:
    """单次状态转移的快照，记录原因和附加细节。"""
    reason: TransitionReason
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoopState:
    """ReAct 主循环的完整状态快照。

    关键设计：frozen=True（不可变）。
    每次状态变化都通过 .update() / .next() 返回新对象，
    而不是修改现有对象。这样：
    1. 任意时刻的状态都可以独立保存，便于 trace 和 debug
    2. 状态转移是显式的，不会出现"某个地方悄悄改了状态"的隐患
    3. 与 transcript 的 append-only 设计一致：历史只增不改
    """
    messages: list[dict[str, Any]]      # 当前发给模型的消息列表（model view，非完整历史）
    step: int                           # 当前步数（从 1 开始，上限 max_steps）
    turn_count: int                     # 第几轮对话（多轮会话累计）
    tool_choice: str                    # 工具调用策略："auto" | "none" | 指定工具名
    transition: Transition | None = None            # 最近一次状态转移记录
    compact_attempted: bool = False                 # 本轮是否触发过上下文压缩
    max_output_recovery_count: int = 0             # MAX_OUTPUT 错误的恢复次数
    model_recovery_counts: dict[str, int] = field(default_factory=dict)  # 各类错误的重试计数
    stop_hook_active: bool = False                  # 完成门是否处于拦截状态
    completion_block_count: int = 0                # 完成门拦截次数（达上限则终止）
    last_tool_calls: list[dict[str, Any]] = field(default_factory=list)  # 上一步的工具调用
    last_response_meta: dict[str, Any] = field(default_factory=dict)     # 上一次模型响应元信息
    last_model_error_kind: str | None = None        # 最近一次模型错误类型
    last_model_error_stage: str | None = None       # 错误发生阶段（invoke / response）
    last_error: str | None = None                   # 最近一次错误消息

    def update(self, **changes: Any) -> "LoopState":
        """产生一个修改了指定字段的新状态对象，不记录 transition。
        用于纯数据更新（如刷新 messages），不涉及语义上的状态转移。
        """
        return replace(self, **changes)

    def next(self, reason: TransitionReason, **changes: Any) -> "LoopState":
        """产生一个新状态对象，同时记录本次转移的原因。
        用于有语义意义的状态转移（如"工具执行完毕"），
        transition 字段会被 trace 系统捕获并写入日志。
        """
        details = changes.pop("details", {})
        return replace(self, transition=Transition(reason=reason, details=details), **changes)
