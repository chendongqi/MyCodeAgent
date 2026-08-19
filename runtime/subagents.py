"""Restricted subagents built on the canonical RuntimeRunner."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from core.config import Config
from core.llm import HelloAgentsLLM
from extensions.tracing import NullTraceLogger, create_trace_logger
from runtime.completion import (
    CompletionCandidate,
    CompletionGateResult,
    CompletionGateVerdict,
    CompletionRequirements,
    DeterministicCompletionVerifier,
    VerificationEvidence,
)
from runtime.context import ContextEngine
from runtime.history import HistoryManager
from runtime.loop import RuntimeRunner
from runtime.prompt_builder import ContextBuilder
from runtime.session_memory import SessionMemory, SessionMemoryManager
from runtime.transcript import TranscriptRecorder, TranscriptStore
from tools.context import ToolExecutionContext
from tools.executor import ToolExecutor
from tools.orchestrator import ToolOrchestrator
from tools.permissions import PermissionContext, RiskClassifier
from tools.registry import ToolRegistry


READONLY_TOOLS = frozenset({"Glob", "Grep", "Read"})

EXPLORE_SYSTEM_PROMPT = """You are an Explore Agent.
Inspect the repository with read-only tools and return exactly one JSON object:
{"status":"completed|partial","summary":"...","findings":["..."],
"evidence":["relative/path.py:line"],"unresolved_questions":["..."]}.
Do not use markdown fences. Do not modify files, execute shell commands, ask the user,
or delegate to another agent."""

VERIFICATION_SYSTEM_PROMPT = """You are a Verification Agent.
Independently assess the supplied completion candidate using read-only repository tools.
Return exactly one JSON object:
{"verdict":"PASS|FAIL|PARTIAL|UNVERIFIED","reasons":["..."],
"findings":["..."],"evidence":["relative/path.py:line"]}.
Do not use markdown fences. Never modify files or delegate to another agent."""


class SubagentStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    UNVERIFIED = "unverified"


class VerificationVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    UNVERIFIED = "UNVERIFIED"


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    system_prompt: str
    tool_allowlist: frozenset[str]
    max_steps: int
    context_token_budget: int
    total_token_budget: int
    model_choice: str
    context_source_policy: str
    completion_policy: str
    result_contract: str
    recursive_subagents: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.system_prompt:
            raise ValueError("runtime profile requires name and system prompt")
        if self.max_steps <= 0 or self.context_token_budget <= 0 or self.total_token_budget <= 0:
            raise ValueError("runtime profile budgets must be positive")
        if self.model_choice not in {"main", "light"}:
            raise ValueError("runtime profile model choice must be main or light")
        if self.recursive_subagents or "Task" in self.tool_allowlist:
            raise ValueError("formal subagent profiles cannot recurse")
        forbidden = {"Edit", "Bash"}
        if forbidden & self.tool_allowlist:
            raise ValueError("formal subagent profiles must be strictly read-only")


EXPLORE_PROFILE = RuntimeProfile(
    name="explore",
    system_prompt=EXPLORE_SYSTEM_PROMPT,
    tool_allowlist=READONLY_TOOLS,   # 只能用 Read/Grep/Glob，不能写文件、不能执行命令
    max_steps=12,                    # 最多 12 步，防止无限循环
    context_token_budget=16_000,     # 单次上下文窗口上限（远小于主 agent 的 128k）
    total_token_budget=32_000,       # 整轮累计 token 上限，超出强制终止
    model_choice="light",            # 默认用轻量模型，降低成本
    context_source_policy="self_contained_task_and_structured_context",
    completion_policy="structured_result",
    result_contract="ExploreResult", # 子 agent 必须返回符合此合约的 JSON
)

VERIFICATION_PROFILE = RuntimeProfile(
    name="verification",
    system_prompt=VERIFICATION_SYSTEM_PROMPT,
    tool_allowlist=READONLY_TOOLS,
    max_steps=10,
    context_token_budget=20_000,
    total_token_budget=40_000,
    model_choice="main",             # 验证任务用主模型，需要更强的判断能力
    context_source_policy="completion_candidate_requirements_evidence",
    completion_policy="structured_result",
    result_contract="VerificationResult",
)

RUNTIME_PROFILES = {
    EXPLORE_PROFILE.name: EXPLORE_PROFILE,
    VERIFICATION_PROFILE.name: VERIFICATION_PROFILE,
}


@dataclass(frozen=True)
class SubagentRequest:
    profile_name: str
    task: str
    structured_context: dict[str, Any] = field(default_factory=dict)
    model_choice: str | None = None
    parent_session_id: str | None = None
    parent_run_id: str | None = None


@dataclass(frozen=True)
class ExploreResult:
    status: SubagentStatus
    summary: str
    findings: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    tool_usage: dict[str, int] = field(default_factory=dict)
    terminal_reason: str = "unknown"

    @classmethod
    def from_json(
        cls,
        raw: str,
        *,
        tool_usage: dict[str, int],
        terminal_reason: str,
    ) -> "ExploreResult":
        payload = _parse_json_object(raw)
        status = SubagentStatus(str(payload.get("status", "")).lower())
        if status not in {SubagentStatus.COMPLETED, SubagentStatus.PARTIAL}:
            raise ValueError("ExploreResult status must be completed or partial")
        summary = payload.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("ExploreResult requires summary")
        return cls(
            status=status,
            summary=summary.strip(),
            findings=_string_tuple(payload.get("findings")),
            evidence=_string_tuple(payload.get("evidence")),
            unresolved_questions=_string_tuple(payload.get("unresolved_questions")),
            tool_usage=dict(tool_usage),
            terminal_reason=terminal_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True)
class VerificationResult:
    verdict: VerificationVerdict
    reasons: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    child_session_id: str | None = None
    child_run_id: str | None = None
    terminal_reason: str = "unknown"
    tool_usage: dict[str, int] = field(default_factory=dict)
    token_usage: int = 0

    @classmethod
    def from_json(
        cls,
        raw: str,
        *,
        child_session_id: str | None = None,
        child_run_id: str | None = None,
        terminal_reason: str,
        tool_usage: dict[str, int] | None = None,
        token_usage: int = 0,
    ) -> "VerificationResult":
        payload = _parse_json_object(raw)
        verdict = VerificationVerdict(str(payload.get("verdict", "")).upper())
        return cls(
            verdict=verdict,
            reasons=_string_tuple(payload.get("reasons")),
            findings=_string_tuple(payload.get("findings")),
            evidence=_string_tuple(payload.get("evidence")),
            child_session_id=child_session_id,
            child_run_id=child_run_id,
            terminal_reason=terminal_reason,
            tool_usage=dict(tool_usage or {}),
            token_usage=token_usage,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["verdict"] = self.verdict.value
        return payload


@dataclass(frozen=True)
class SubagentLaunchResult:
    status: SubagentStatus
    profile_name: str
    child_session_id: str
    child_run_id: str
    result: ExploreResult | VerificationResult | None
    model_used: str = "main"
    terminal_reason: str = "unknown"
    error: str | None = None
    elapsed_ms: int = 0
    tool_usage: dict[str, int] = field(default_factory=dict)
    token_usage: int = 0


SubagentResult = SubagentLaunchResult


class _RecordingTrace:
    def __init__(self, delegate: Any, *, session_id: str | None = None):
        self.delegate = delegate
        self.events: list[tuple[str, int, dict[str, Any]]] = []
        self.session_id = session_id or str(
            getattr(delegate, "session_id", f"child-{uuid.uuid4().hex}")
        )

    def log_event(self, name: str, payload: dict[str, Any], step: int = 0) -> None:
        self.events.append((name, step, payload))
        self.delegate.log_event(name, payload, step=step)

    def log_system_messages(self, messages: list[dict[str, Any]]) -> None:
        self.delegate.log_system_messages(messages)

    def finalize(self) -> None:
        self.delegate.finalize()


class _SubagentRuntimeHost:
    def __init__(
        self,
        *,
        profile: RuntimeProfile,
        llm: Any,
        registry: ToolRegistry,
        project_root: Path,
        trace_logger: Any,
    ):
        # _SubagentRuntimeHost 是子 agent 的独立沙箱，结构和主 agent 的 CodeAgent 完全对齐，
        # 但所有状态全部新建，不继承主 agent 的任何运行时对象。
        # RuntimeRunner 只依赖 host 上的属性，通过这种鸭子类型共用同一套循环逻辑：
        # 只要 host 有 llm/tool_registry/history_manager/context_engine 等属性，
        # RuntimeRunner 就能在上面跑，不需要继承任何基类。

        self.profile = profile
        self.llm = llm
        self.tool_registry = registry
        self.project_root = str(project_root)

        # 从环境变量加载基础配置，再用 profile 的预算覆盖关键字段：
        # context_window 限制为 profile.context_token_budget（远小于主 agent 的 128k），
        # 防止子 agent 因探索步骤太多而消耗过多 token
        self.config = Config.from_env().model_copy(
            update={
                "context_window": profile.context_token_budget,
                "show_react_steps": False,  # 子 agent 不输出 ReAct 步骤到控制台
                "show_progress": False,
            }
        )
        self.max_steps = profile.max_steps         # 硬性步数上限（explore=12）
        self.max_total_tokens = profile.total_token_budget  # 硬性 token 上限，超出强制终止
        self.console_progress = False
        self.console_verbose = False
        self.logger = logging.getLogger(f"runtime.subagent.{profile.name}")
        self.last_response_raw = None
        self._skills_prompt = ""   # 子 agent 不加载 Skills
        self._run_id = 0
        self._active_transcript_run_id = None
        self._system_messages_logged = False

        # 独立的历史管理器，从空白开始，主 agent 的历史不会流入子 agent
        self.history_manager = HistoryManager(config=self.config)

        # 系统提示词固定为 profile.system_prompt（要求返回 JSON），
        # 不加载 MCP 工具提示词和 Skills——子 agent 只需要知道白名单工具的用法
        self.context_builder = ContextBuilder(
            tool_registry=registry,
            project_root=self.project_root,
            system_prompt_override=profile.system_prompt,
            mcp_tools_prompt="",
            skills_prompt="",
            tool_prompt_allowlist=profile.tool_allowlist,  # 只暴露白名单工具的 schema
        )

        # 上下文引擎：用 _summarize_child_messages 做压缩（不启 LLM，纯截断拼接）
        # 为什么不用主 agent 那套 LLM 压缩？子 agent 预算小，启 LLM 压缩消耗太大，
        # 简单截断足够，因为子 agent 任务是探索，不需要保留很长的对话历史
        self.context_engine = ContextEngine(
            self.context_builder,
            config=self.config,
            summary_generator=_summarize_child_messages,
        )

        # 独立的 trace 日志和 transcript，session_id 以 "subagent-" 开头，
        # 写到独立的 JSONL 文件，与主 agent trace 分离但通过 parent_session_id 关联
        self.trace_logger = trace_logger
        transcript_session = f"subagent-{trace_logger.session_id}"
        transcript_path = project_root / "memory" / "transcripts" / f"transcript-{transcript_session}.jsonl"
        self.transcript_store = TranscriptStore(transcript_path, session_id=transcript_session)
        self.session_memory_manager = SessionMemoryManager(on_update=self._apply_session_memory)
        self.session_memory = SessionMemory()
        self.transcript_recorder = TranscriptRecorder(
            self.transcript_store,
            on_recorded=self.session_memory_manager.ingest_event,
        )

        # 权限模式锁定为 readonly_subagent：
        # RiskClassifier 在此模式下会对 Edit/Bash/Task 直接返回 DENY，
        # 即使这些工具意外出现在 registry 里也无法执行——双重保障（白名单 + 权限门）
        permission_context = PermissionContext(runtime_mode="readonly_subagent")
        self.tool_executor = ToolExecutor(
            registry,
            context=ToolExecutionContext(
                permission_decider=RiskClassifier().classify,
                permission_context=permission_context,
                project_root=self.project_root,
            ),
        )
        self.tool_orchestrator = ToolOrchestrator(self)

        # 子 agent 专用的完成门：检查输出是否符合 result_contract（ExploreResult JSON），
        # 不符合则触发 FAIL 反馈让子 agent 重输，而不是直接终止
        self.completion_verifier = _StructuredResultCompletionVerifier(profile)

    def _apply_session_memory(self, memory: SessionMemory) -> None:
        self.session_memory = memory
        self.context_engine.set_session_memory(memory)

    def _refresh_skills_prompt(self) -> None:
        self._skills_prompt = ""

    def _log_system_messages_if_needed(self, trace_logger) -> None:
        if not self._system_messages_logged:
            trace_logger.log_system_messages(self.context_builder.get_system_messages())
            self._system_messages_logged = True

    def _print_context_preview(self, _messages: list[dict[str, Any]]) -> None:
        return None

    def _get_openai_tools_for_current_mode(self) -> list[dict[str, Any]]:
        return self.tool_registry.get_openai_tools()


class SubagentLauncher:
    def __init__(
        self,
        *,
        project_root: Path,
        main_llm: Any,
        tool_registry: ToolRegistry,
        light_llm: Any = None,
        parent_trace_logger: Any = None,
        parent_history_manager: Any = None,
        parent_context_engine: Any = None,
        parent_host: Any = None,
    ):
        self.project_root = Path(project_root)
        self.main_llm = main_llm
        self.light_llm = light_llm
        self.tool_registry = tool_registry
        self.parent_trace_logger = parent_trace_logger
        self.parent_history_manager = parent_history_manager
        self.parent_context_engine = parent_context_engine
        self.parent_host = parent_host

    def build_registry(self, profile: RuntimeProfile) -> ToolRegistry:
        filtered = ToolRegistry()
        for tool in self.tool_registry.get_all_tools():
            if tool.name in profile.tool_allowlist:
                filtered.register_tool(tool)
        return filtered

    def launch(self, request: SubagentRequest) -> SubagentLaunchResult:
        """子 agent 的完整执行入口。整个过程同步阻塞，子 agent 跑完才返回。

        执行步骤：
        1. 根据 profile_name 查找 RuntimeProfile（决定工具白名单、预算、模型）
        2. 选择 LLM（light 优先，没配则回退到 main）
        3. 创建独立 trace 日志（child_trace），和主 agent 的 trace 分开写
        4. 向父 trace 发射 subagent_requested / subagent_started 事件（供观测）
        5. 创建 _SubagentRuntimeHost（独立沙箱：history/context/tool 全新创建）
        6. 用 _render_request 把任务文本 + 结构化上下文拼成 prompt
        7. RuntimeRunner(host).run(prompt) ← 和主 agent 完全相同的 ReAct 循环
        8. 从 child_trace.events 里提取 terminal_reason/tool_usage/token_usage
        9. 把 raw_result（子 agent 最后输出的 JSON 字符串）解析成结构化对象
        10. 返回 SubagentLaunchResult，TaskTool 拿到后打包成标准信封
        """
        started = time.monotonic()

        # 步骤 1：查找 profile（explore 或 verification），不存在则报错
        # profile 里定义了这次子任务能用哪些工具、跑多少步、花多少 token
        profile = RUNTIME_PROFILES.get(request.profile_name)
        if profile is None:
            raise ValueError(f"unsupported runtime profile: {request.profile_name}")

        # 步骤 2：选模型
        # request.model_choice 优先（TaskTool 传入的 "light"/"main"），
        # 没传则用 profile 的默认值（explore 默认 "light"）
        requested_model = request.model_choice or profile.model_choice
        if requested_model not in {"main", "light"}:
            raise ValueError(f"unsupported subagent model choice: {requested_model}")
        llm, model_choice = self._select_llm(requested_model)

        # 步骤 3：创建子 trace（独立 JSONL 文件，session_id 以 "child-" 开头）
        # 为什么独立？子 agent 的探索步骤不应混进主 agent 的 trace，
        # 但两者通过 parent_session_id 关联，可以跨文件追踪父子关系
        child_trace = self._create_child_trace()
        child_session_id = child_trace.session_id
        child_run_id = "run-1"

        # 步骤 4：向父 trace 发射事件，让主 agent 的 trace 里能看到"我启动了一个子任务"
        self._parent_event(
            "subagent_requested",
            request, profile,
            child_session_id=child_session_id, child_run_id=child_run_id, model=model_choice,
        )
        self._parent_event(
            "subagent_started",
            request, profile,
            child_session_id=child_session_id, child_run_id=child_run_id, model=model_choice,
        )
        try:
            # 步骤 5：创建子 agent 的独立沙箱
            # _SubagentRuntimeHost 和主 agent 的 CodeAgent 结构相同，
            # 但所有状态（history/context/tool_registry/transcript）全部新建，
            # 完全隔离，不共享主 agent 的任何运行时状态
            host = _SubagentRuntimeHost(
                profile=profile,
                llm=llm,
                registry=self.build_registry(profile),  # 只含白名单工具的过滤后 registry
                project_root=self.project_root,
                trace_logger=child_trace,
            )

            # 步骤 6：把任务描述和结构化上下文拼成 prompt
            # _render_request 输出：task 文本 + "\n\nStructured context:\n" + JSON
            # 这是子 agent 唯一能看到的上下文——它看不到主 agent 的历史
            prompt = _render_request(request)

            # 步骤 7：用和主 agent 完全相同的 RuntimeRunner 跑 ReAct 循环
            # 子 agent 的系统提示词要求它最终输出一个 JSON 对象（不能有 Markdown 代码块）
            # raw_result 就是子 agent 最后 return 的字符串（JSON 格式）
            raw_result = RuntimeRunner(host).run(prompt)

            # 步骤 8：从 child_trace 里统计执行指标
            # terminal_reason：子 agent 怎么结束的（completed/max_steps/token_budget 等）
            # tool_usage：每个工具调用了几次（用于统计和计费）
            # token_usage：子 agent 累计消耗的 token 数
            terminal_reason, tool_usage, token_usage = _child_metrics(child_trace.events)
            if terminal_reason == "unknown" and raw_result:
                terminal_reason = "completed"
            # 子 agent 必须正常完成——其他终止原因（max_steps/token_budget等）都当失败
            if terminal_reason not in {"completed", "completed_unverified"}:
                raise ValueError(f"child terminal reason: {terminal_reason}")

            # 步骤 9：解析 JSON 结果
            # ExploreResult.from_json / VerificationResult.from_json 严格校验格式，
            # 不合法（缺字段、status 值错误、带 Markdown fence 等）直接抛异常
            if profile.result_contract == "ExploreResult":
                structured = ExploreResult.from_json(
                    raw_result,
                    tool_usage=tool_usage,
                    terminal_reason=terminal_reason,
                )
                status = structured.status
                verdict = structured.status.value
            else:
                structured = VerificationResult.from_json(
                    raw_result,
                    child_session_id=child_session_id,
                    child_run_id=child_run_id,
                    terminal_reason=terminal_reason,
                    tool_usage=tool_usage,
                    token_usage=token_usage,
                )
                # VerificationVerdict → SubagentStatus 的映射：
                # PASS → COMPLETED，UNVERIFIED → UNVERIFIED，FAIL/PARTIAL → PARTIAL
                status = (
                    SubagentStatus.COMPLETED
                    if structured.verdict is VerificationVerdict.PASS
                    else SubagentStatus.UNVERIFIED
                    if structured.verdict is VerificationVerdict.UNVERIFIED
                    else SubagentStatus.PARTIAL
                )
                verdict = structured.verdict.value
            elapsed_ms = int((time.monotonic() - started) * 1000)
            result = SubagentLaunchResult(
                status=status,
                profile_name=profile.name,
                child_session_id=child_session_id,
                child_run_id=child_run_id,
                result=structured,
                model_used=model_choice,
                terminal_reason=terminal_reason,
                elapsed_ms=elapsed_ms,
                tool_usage=tool_usage,
                token_usage=token_usage,
            )
            self._parent_event(
                "subagent_completed",
                request,
                profile,
                child_session_id=child_session_id,
                child_run_id=child_run_id,
                model=model_choice,
                terminal_reason=terminal_reason,
                tool_usage=tool_usage,
                token_usage=token_usage,
                verdict=verdict,
                elapsed_ms=elapsed_ms,
            )
            return result
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            self._parent_event(
                "subagent_failed",
                request,
                profile,
                child_session_id=child_session_id,
                child_run_id=child_run_id,
                model=model_choice,
                terminal_reason="runtime_error",
                elapsed_ms=elapsed_ms,
                error=str(exc),
            )
            return SubagentLaunchResult(
                status=SubagentStatus.FAILED,
                profile_name=profile.name,
                child_session_id=child_session_id,
                child_run_id=child_run_id,
                result=None,
                model_used=model_choice,
                terminal_reason="runtime_error",
                error=str(exc),
                elapsed_ms=elapsed_ms,
            )
        finally:
            child_trace.finalize()

    def _select_llm(self, requested_model: str) -> tuple[Any, str]:
        if requested_model == "light":
            if self.light_llm is None:
                self.light_llm = _create_light_llm()
            if self.light_llm is not None:
                return self.light_llm, "light"
        return self.main_llm, "main"

    def _create_child_trace(self) -> _RecordingTrace:
        if (
            self.parent_trace_logger is not None
            and getattr(self.parent_trace_logger, "enabled", True) is False
        ):
            return _RecordingTrace(
                NullTraceLogger(),
                session_id=f"child-{uuid.uuid4().hex}",
            )
        return _RecordingTrace(
            create_trace_logger(
                trace_dir=str(self.project_root / "memory" / "traces"),
                project_root=self.project_root,
            )
        )

    def _parent_event(
        self,
        name: str,
        request: SubagentRequest,
        profile: RuntimeProfile,
        **payload: Any,
    ) -> None:
        if self.parent_trace_logger is None:
            return
        parent_session_id = request.parent_session_id or str(
            getattr(self.parent_trace_logger, "session_id", "")
        )
        parent_run_id = request.parent_run_id or str(
            getattr(self.parent_host, "_active_transcript_run_id", "") or ""
        )
        self.parent_trace_logger.log_event(
            name,
            {
                "parent_session_id": parent_session_id,
                "parent_run_id": parent_run_id,
                "profile": profile.name,
                "max_steps": profile.max_steps,
                "context_token_budget": profile.context_token_budget,
                "total_token_budget": profile.total_token_budget,
                **payload,
            },
            step=0,
        )


class SubagentCompletionVerifier:
    """Run deterministic checks first, then an optional readonly verifier child."""

    def __init__(self, launcher: SubagentLauncher):
        self.launcher = launcher
        self.deterministic = DeterministicCompletionVerifier()

    def evaluate(
        self,
        candidate: CompletionCandidate,
        requirements: CompletionRequirements,
        evidence: list[VerificationEvidence],
        history_messages: list[Any],
    ) -> CompletionGateResult:
        deterministic = self.deterministic.evaluate(candidate, requirements, evidence, history_messages)
        if deterministic.verdict is not CompletionGateVerdict.PASS:
            return deterministic
        if not requirements.requires_verification:
            return deterministic
        request = SubagentRequest(
            profile_name="verification",
            task="Independently verify this completion candidate.",
            structured_context={
                "candidate": candidate.to_trace_payload(),
                "requirements": requirements.to_trace_payload(),
                "evidence": [item.to_trace_payload() for item in evidence],
            },
        )
        try:
            launched = self.launcher.launch(request)
            result = launched.result
            if not isinstance(result, VerificationResult):
                raise ValueError("invalid verification result")
        except Exception:
            return CompletionGateResult(
                verdict=CompletionGateVerdict.UNVERIFIED,
                reasons=("verification_agent_error",),
                passed_evidence=deterministic.passed_evidence,
            )
        if result.verdict is VerificationVerdict.PASS:
            return deterministic
        if result.verdict is VerificationVerdict.UNVERIFIED:
            return CompletionGateResult(
                verdict=CompletionGateVerdict.UNVERIFIED,
                reasons=result.reasons or ("verification_agent_unverified",),
                passed_evidence=deterministic.passed_evidence,
            )
        reasons = result.reasons or (f"verification_agent_{result.verdict.value.lower()}",)
        return CompletionGateResult(
            verdict=CompletionGateVerdict.FAIL,
            reasons=reasons,
            blocking_feedback="Independent verification did not pass: " + "; ".join(reasons),
            passed_evidence=deterministic.passed_evidence,
        )


class _StructuredResultCompletionVerifier:
    """Validate a child contract without inheriting parent completion requirements."""

    def __init__(self, profile: RuntimeProfile):
        self.profile = profile

    def evaluate(
        self,
        candidate: CompletionCandidate,
        requirements: CompletionRequirements,
        evidence: list[VerificationEvidence],
        history_messages: list[Any],
    ) -> CompletionGateResult:
        try:
            if self.profile.result_contract == "ExploreResult":
                ExploreResult.from_json(
                    candidate.final_text,
                    tool_usage={},
                    terminal_reason="completed",
                )
            else:
                VerificationResult.from_json(
                    candidate.final_text,
                    terminal_reason="completed",
                )
        except Exception as exc:
            return CompletionGateResult(
                verdict=CompletionGateVerdict.FAIL,
                reasons=("invalid_subagent_result",),
                blocking_feedback=f"Return only a valid {self.profile.result_contract} JSON object: {exc}",
            )
        return CompletionGateResult(verdict=CompletionGateVerdict.PASS)


def _render_request(request: SubagentRequest) -> str:
    context = json.dumps(request.structured_context, ensure_ascii=False, sort_keys=True)
    return f"{request.task.strip()}\n\nStructured context:\n{context}"


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        raise ValueError("structured child result must not use markdown fences")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("structured child result must be an object")
    return payload


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("result list fields must be string arrays")
    return tuple(item for item in value if item.strip())


def _summarize_child_messages(messages: list[Any], *, char_budget: int = 8000) -> str | None:
    if not messages:
        return None
    lines = ["[Compacted child history]"]
    for message in messages:
        role = str(getattr(message, "role", "unknown") or "unknown")
        content = str(getattr(message, "content", "") or "")
        metadata = getattr(message, "metadata", {}) or {}
        label = role
        if role == "tool":
            label = f"tool:{metadata.get('tool_name', 'unknown')}"
        content = content[:1200]
        line = f"[{label}] {content}"
        remaining = char_budget - len("\n".join(lines)) - 1
        if remaining <= 0:
            break
        lines.append(line[:remaining])
    return "\n".join(lines)[:char_budget]


def _child_metrics(events: list[tuple[str, int, dict[str, Any]]]) -> tuple[str, dict[str, int], int]:
    terminal_reason = "unknown"
    tool_usage: dict[str, int] = {}
    token_usage = 0
    for name, _step, payload in events:
        if name == "tool_call":
            tool = str(payload.get("tool") or "unknown")
            tool_usage[tool] = tool_usage.get(tool, 0) + 1
        elif name == "model_output":
            usage = payload.get("usage") or {}
            token_usage += int(usage.get("total_tokens") or 0)
        elif name == "terminal":
            terminal_reason = str(payload.get("reason") or terminal_reason)
    return terminal_reason, tool_usage, token_usage


def _create_light_llm() -> HelloAgentsLLM | None:
    model = os.getenv("LIGHT_LLM_MODEL_ID")
    if not model:
        return None
    try:
        return HelloAgentsLLM(
            model=model,
            api_key=os.getenv("LIGHT_LLM_API_KEY"),
            base_url=os.getenv("LIGHT_LLM_BASE_URL"),
            provider=os.getenv("LIGHT_LLM_PROVIDER", "auto"),
            temperature=float(os.getenv("LIGHT_LLM_TEMPERATURE", "0.5")),
        )
    except Exception:
        logging.getLogger("runtime.subagent").warning(
            "Failed to initialize light subagent model; falling back to main",
            exc_info=True,
        )
        return None


__all__ = [
    "EXPLORE_PROFILE",
    "RUNTIME_PROFILES",
    "VERIFICATION_PROFILE",
    "ExploreResult",
    "RuntimeProfile",
    "SubagentCompletionVerifier",
    "SubagentLaunchResult",
    "SubagentLauncher",
    "SubagentRequest",
    "SubagentResult",
    "SubagentStatus",
    "VerificationResult",
    "VerificationVerdict",
]
