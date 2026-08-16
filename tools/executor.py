"""Tool execution boundary separated from registry/schema concerns."""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from .base import ErrorCode, ToolResult, ToolStatus
from .context import ToolExecutionContext
from .permissions import PermissionAction, PermissionDecision, RiskLevel


class ToolExecutor:
    """Execute registered tools with permission checks and result packaging."""

    def __init__(
        self,
        registry,
        permission_checker: Optional[Callable[[str], bool]] = None,
        context: Optional[ToolExecutionContext] = None,
    ):
        self.registry = registry
        self.context = context or ToolExecutionContext(
            permission_checker=permission_checker or (lambda _name: True)
        )

    def execute(self, name: str, input_text: Any, *, trace_logger=None, step: int = 0) -> ToolResult:
        """执行管道：权限检查 → 乐观锁注入 → 熔断检查 → tool.run()

        管道各关卡有顺序依赖：先检权限，再注入乐观锁参数，再检熔断，最后执行。
        任一关卡失败立即短路返回 ToolResult(status=ERROR)，不会继续往下走。
        """
        parameters = self.registry.prepare_parameters(input_text)

        # 关卡 1：权限检查（ALLOW / DENY / ASK），fail-closed：未知工具默认 DENY
        permission_payload = self._decide_permission(name, parameters, trace_logger=trace_logger, step=step)
        if permission_payload is not None:
            return permission_payload

        # 关卡 2：Edit 工具自动注入乐观锁参数（从 Read 缓存取 mtime/size）
        # 若模型未提供 expected_mtime_ms，框架自动补全；冲突时 Edit 返回 CONFLICT
        if name == "Edit":
            parameters = self.registry.inject_optimistic_lock_params(name, parameters)

        # 关卡 3：熔断检查（连续失败 N 次后工具临时禁用）
        if not self.registry.is_available(name):
            return self.registry.create_circuit_open_result(name, parameters)

        result_payload = None
        tool = self.registry.get_tool(name)
        func = self.registry.get_function(name) if tool is None else None

        if tool is not None:
            try:
                result = tool.run(parameters)
                result_payload = self.registry.normalize_result(name, result, parameters)
            except Exception as exc:
                result_payload = self.registry.create_internal_error_result(
                    name=name,
                    message=f"执行工具 '{name}' 时发生异常: {str(exc)}",
                    params_input=parameters,
                )
        elif func is not None:
            try:
                raw_input = input_text if not isinstance(input_text, dict) else input_text.get("input", input_text)
                result = func(raw_input)
                result_payload = self.registry.normalize_result(name, result, parameters)
            except Exception as exc:
                result_payload = self.registry.create_internal_error_result(
                    name=name,
                    message=f"执行工具 '{name}' 时发生异常: {str(exc)}",
                    params_input=parameters,
                )
        else:
            result_payload = self.registry.create_internal_error_result(
                name=name,
                message=f"未找到名为 '{name}' 的工具。",
                params_input={},
            )

        # 关卡 4 后处理：更新熔断器状态；Read 结果缓存进乐观锁 store
        self.registry.record_execution_result(name, result_payload)
        if name == "Read":
            # 缓存 mtime + size，供下一次 Edit 自动注入乐观锁参数
            self.registry.cache_read_result(result_payload, parameters)

        return result_payload

    def _decide_permission(self, name: str, parameters: dict[str, Any], *, trace_logger=None, step: int = 0):
        decider = self.context.permission_decider
        if decider is not None:
            decision = decider(name, parameters, self.context.permission_context)
            effective_action = decision.action
            if effective_action is PermissionAction.ASK and self.context.permission_context.ask_policy == "deny":
                effective_action = PermissionAction.DENY
            if (
                effective_action is PermissionAction.ALLOW
                and not self.context.permission_checker(name)
            ):
                decision = PermissionDecision(
                    action=PermissionAction.DENY,
                    risk=RiskLevel.HIGH,
                    reason="tool blocked by runtime allowlist",
                    policy_source="runtime_allowlist",
                    input_summary=json.dumps(
                        parameters,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
                effective_action = PermissionAction.DENY
            if trace_logger:
                trace_logger.log_event(
                    "permission_decision",
                    decision.as_trace_payload(
                        tool_name=name,
                        effective_action=effective_action.value,
                    ),
                    step=step,
                )
            if effective_action is not PermissionAction.ALLOW:
                return self._permission_denied_payload(name, parameters, decision, effective_action)
            return None

        if not self.context.permission_checker(name):
            message = f"Tool '{name}' is not allowed in the current mode."
            return ToolResult(
                status=ToolStatus.ERROR,
                data={},
                text=message,
                error_code=ErrorCode.PERMISSION_DENIED,
                error_message=message,
                stats={"time_ms": 0},
                context={"cwd": ".", "params_input": parameters},
            )
        return None

    def _permission_denied_payload(self, name: str, parameters: dict[str, Any], decision, effective_action):
        rendered_action = "denied" if effective_action is PermissionAction.DENY else "requires confirmation"
        message = f"Tool '{name}' {rendered_action} by permission core."
        return ToolResult(
            status=ToolStatus.ERROR,
            data={},
            text=message,
            error_code=ErrorCode.PERMISSION_DENIED,
            error_message=message,
            error_details={
                "details": {
                    "permission": {
                        **decision.as_trace_payload(tool_name=name),
                        "effective_action": effective_action.value,
                    }
                }
            },
            stats={"time_ms": 0},
            context={
                "cwd": ".",
                "params_input": parameters,
                "runtime_mode": self.context.permission_context.runtime_mode,
            },
        )
