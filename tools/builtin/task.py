"""Task tool adapter for the formal Explore subagent."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from tools.base import ErrorCode, Tool, ToolParameter, ToolResult
from prompts.tools_prompts.task_prompt import task_prompt


@dataclass(frozen=True)
class TaskRequest:
    """TaskTool 构造的请求对象，字段和 SubagentRequest 完全一致。

    为什么不直接用 SubagentRequest？
    TaskTool 不能 import runtime/subagents.py（会引入循环依赖），
    所以在 task.py 里定义了一个字段相同的镜像 dataclass。
    _DeferredSubagentLauncher.launch() 直接把 TaskRequest 传给
    SubagentLauncher.launch(request: SubagentRequest)，Python 运行时
    只看字段值，不做类型校验，两者完全兼容（鸭子类型）。
    """
    profile_name: str
    task: str
    model_choice: str | None = None
    structured_context: dict[str, Any] = field(default_factory=dict)
    parent_session_id: str | None = None
    parent_run_id: str | None = None


class TaskLauncher(Protocol):
    def launch(self, request: TaskRequest) -> Any: ...


class TaskTool(Tool):
    """Task 工具：主 agent 把探索性子任务委派给只读子 agent 执行。

    调用链：
      主 agent 调 Task 工具
        → TaskTool.run() 校验参数，构造 TaskRequest
          → _launcher.launch(request)（_DeferredSubagentLauncher 代理）
            → SubagentLauncher.launch()
              → 创建 _SubagentRuntimeHost（独立的 history/context/tool 沙箱）
              → RuntimeRunner(host).run(prompt)（完整 ReAct 循环，只读工具）
              → 解析结构化 JSON 结果 → SubagentLaunchResult
        → 把结果打包成标准信封返回给主 agent

    当前只支持 subagent_type="explore"，工具白名单 = {Read, Grep, Glob}。
    子 agent 不能调 Edit/Bash/Task，防止递归和副作用。
    """

    def __init__(
        self,
        name: str = "Task",
        project_root: Optional[Path] = None,
        working_dir: Optional[Path] = None,
        launcher: Optional[TaskLauncher] = None,
    ):
        if project_root is None:
            raise ValueError("project_root must be provided by the framework")
        if launcher is None:
            raise ValueError("launcher must be provided by the framework")
        super().__init__(
            name=name,
            description=task_prompt,
            project_root=project_root,
            working_dir=working_dir or project_root,
        )
        self._launcher = launcher

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="description",
                type="string",
                description="Short summary of the delegated search task",
                required=True,
            ),
            ToolParameter(
                name="prompt",
                type="string",
                description="Self-contained read-only exploration instructions",
                required=True,
            ),
            ToolParameter(
                name="subagent_type",
                type="string",
                description="Formal profile; only 'explore' is supported",
                required=True,
            ),
            ToolParameter(
                name="model",
                type="string",
                description="Optional model route: main or light",
                required=False,
                default="light",
            ),
        ]

    def run(self, parameters: Dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        params_input = dict(parameters)

        # ── 参数提取 ────────────────────────────────────────────────────────
        # description：任务的简短标签，供主 agent 在工具调用树里识别这次委派
        # prompt：子 agent 实际执行的完整指令，必须自包含（子 agent 看不到主 agent 的历史）
        # profile：当前只支持 "explore"，决定工具白名单和 token 预算
        # model："light" 用轻量模型降成本，"main" 用和主 agent 相同的模型
        description = parameters.get("description")
        prompt = parameters.get("prompt")
        profile = str(parameters.get("subagent_type") or "").strip().lower()
        model = str(parameters.get("model") or "light").strip().lower()

        # ── 参数校验：任何一项不合法都直接短路，不启动子 agent ──────────────
        if not isinstance(description, str) or not description.strip():
            return self.error_result(
                error_code=ErrorCode.INVALID_PARAM,
                message="Parameter 'description' is required and must be non-empty.",
                params_input=params_input,
            )
        if not isinstance(prompt, str) or not prompt.strip():
            return self.error_result(
                error_code=ErrorCode.INVALID_PARAM,
                message="Parameter 'prompt' is required and must be non-empty.",
                params_input=params_input,
            )
        # 目前只开放了 explore profile；verification profile 由完成门内部直接调用
        if profile != "explore":
            return self.error_result(
                error_code=ErrorCode.INVALID_PARAM,
                message="Parameter 'subagent_type' must be 'explore'.",
                params_input=params_input,
            )
        if model not in {"main", "light"}:
            return self.error_result(
                error_code=ErrorCode.INVALID_PARAM,
                message="Parameter 'model' must be 'main' or 'light'.",
                params_input=params_input,
            )

        # ── 委派执行 ────────────────────────────────────────────────────────
        # self._launcher 是 _DeferredSubagentLauncher（注册 TaskTool 时传入，见 host.py）
        # 调用链：
        #   self._launcher.launch(request)
        #     → _DeferredSubagentLauncher.launch(request)        [host.py:57]
        #       → self._get_launcher()                           惰性创建 SubagentLauncher
        #         → create_subagent_launcher(host)               [factory.py:89]
        #           → SubagentLauncher(main_llm, tool_registry, ...)
        #       → SubagentLauncher.launch(request)               [subagents.py:405]
        #         → RUNTIME_PROFILES.get(request.profile_name)  拿到 EXPLORE_PROFILE
        #         → 创建沙箱、跑 ReAct 循环、解析结果
        # launch() 同步阻塞：子 agent 跑完整个 ReAct 循环后才返回
        launched = self._launcher.launch(
            TaskRequest(
                profile_name="explore",   # → RUNTIME_PROFILES["explore"] = EXPLORE_PROFILE
                task=f"{description.strip()}\n\n{prompt.strip()}",
                model_choice=model,
            )
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)

        # status 从枚举值里取字符串（兼容 SubagentStatus.FAILED.value 和直接字符串两种情况）
        status = getattr(getattr(launched, "status", None), "value", getattr(launched, "status", None))

        # 子 agent 失败（运行时异常、token 超限、max_steps 等）
        # 或返回结果不符合 ExploreResult 协议（缺 summary / status 不合法），都当错误处理
        if status == "failed" or not _is_explore_result(getattr(launched, "result", None)):
            return self.error_result(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=f"Explore subagent failed: {launched.error or launched.terminal_reason}",
                params_input=params_input,
                time_ms=elapsed_ms,
            )

        # ── 结果打包 ────────────────────────────────────────────────────────
        result = launched.result
        # text=result.summary：摘要直接作为 observation 进主 agent 历史，模型能立即看到
        # data 里的 findings/evidence 供模型在后续步骤中引用具体代码位置
        # child_session_id/run_id 供 trace 系统关联父子会话
        return self.success_result(
            data={
                "status": getattr(result.status, "value", result.status),
                "profile": "explore",
                "child_session_id": launched.child_session_id,
                "child_run_id": launched.child_run_id,
                "result": result.to_dict(),
            },
            text=result.summary,
            params_input=params_input,
            time_ms=elapsed_ms,
            extra_stats={
                "tool_calls": sum(result.tool_usage.values()),  # 子 agent 总共调了几次工具
                "token_usage": launched.token_usage,
                "model": launched.model_used,
            },
        )


__all__ = ["TaskTool"]


def _is_explore_result(result: Any) -> bool:
    return (
        result is not None
        and isinstance(getattr(result, "summary", None), str)
        and callable(getattr(result, "to_dict", None))
        and hasattr(result, "status")
        and isinstance(getattr(result, "tool_usage", None), dict)
    )
