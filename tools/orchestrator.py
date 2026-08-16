"""Tool call orchestration boundary for the agent runtime."""

from __future__ import annotations

import os
import traceback as tb
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

from core.llm import parse_tool_input
from tools.base import ErrorCode, ToolResult, ToolStatus, serialize_tool_result, tool_result_payload
from tools.observation_store import force_truncate_result, truncate_result


class RuntimeEventEmitter(Protocol):
    """Neutral callback implemented by the runtime host, not by tools."""

    def emit_runtime_event(
        self,
        *,
        run_id: str,
        step: int,
        event_type: str,
        payload: dict[str, Any],
    ) -> None: ...


@dataclass(frozen=True)
class ToolObservation:
    """一次工具执行的完整观测记录，写入 history 的最终单元。

    frozen=True：不可变对象，执行后不允许修改，保证 history 写入的确定性。

    Attributes:
        tool_name:    工具名称，对应 LLM tool_call 里的 name 字段
        tool_call_id: 模型分配的唯一 ID，写 history 时用于关联 assistant → tool 消息
        result:       执行结果（可能经过截断/预算压缩）
        raw_result:   截断前的原始结果；未截断时为 None
        metadata:     预算/截断等元信息，调试用
        observation:  序列化后的 JSON 字符串，直接写入 history tool 消息的 content
    """
    tool_name: str
    tool_call_id: str
    result: ToolResult
    raw_result: ToolResult | None = None
    metadata: dict[str, Any] | None = None
    # init=False：不由调用方传入，由 __post_init__ 自动从 result 序列化生成
    observation: str = field(init=False)

    def __post_init__(self) -> None:
        # 在对象构造完成后立即序列化，后续直接用 observation 字符串，避免重复 JSON dumps
        object.__setattr__(self, "observation", serialize_tool_result(self.result))


@dataclass(frozen=True)
class ToolResultBudget:
    """工具结果的字节预算配置。

    Attributes:
        max_tool_bytes:     单个工具结果的最大字节数（超过则截断该工具）
        max_message_bytes:  本批次所有工具结果的总字节上限（超过则从最大的开始截断）
    """
    max_tool_bytes: int
    max_message_bytes: int


@dataclass(frozen=True)
class ToolCallPlan:
    """一次工具调用的解析计划，是 plan_tool_calls() 的输出单元。

    Attributes:
        call:             模型返回的原始 tool_call dict（含 name/id/arguments）
        tool_name:        工具名称（从 call 中提取）
        tool_call_id:     工具调用 ID（从 call 中提取，缺失时自动生成 uuid）
        parsed_input:     arguments 解析后的 dict（parse_tool_input 的结果）
        parse_error:      arguments 解析失败时的异常；None 表示解析成功
        concurrency_safe: 是否可以与其他工具并发执行（只读工具为 True）
    """
    call: dict[str, Any]
    tool_name: str
    tool_call_id: str
    parsed_input: dict[str, Any]
    parse_error: Exception | None
    concurrency_safe: bool


@dataclass(frozen=True)
class ToolBatch:
    """一组可以统一调度的工具调用。

    Attributes:
        concurrency_safe: True → 批内所有工具并发执行；False → 严格串行
        calls:            属于本批次的 ToolCallPlan 列表
    """
    concurrency_safe: bool
    calls: list[ToolCallPlan]


class ToolOrchestrator:
    """Execute model tool calls while preserving model order.

    核心职责：
    1. 并发分组：连续只读工具合为一个并发批次，写工具强制串行
    2. 预算控制：单工具上限 + 批次总量上限，超出时截断并 spill 到磁盘
    3. 观测顺序：无论并发执行，写回 history 的顺序与模型请求顺序一致

    执行流水线（run() 方法）：
        tool_calls（模型原始输出）
            ↓ plan_tool_calls()        解析 arguments，标注并发安全性
            ↓ partition_tool_calls()   按安全性分批
            ↓ per batch:
                并发批 → _run_batch_concurrently() → ThreadPoolExecutor
                串行批 → _run_batch_serially()     → 逐个执行
            ↓ _normalize_empty_result()  空结果补全占位文本
            ↓ _apply_observation_limit() 按行数/字节做初步截断
            ↓ _apply_result_budget()     两层字节预算（单工具+总量）
            → list[ToolObservation]      写入 history
    """

    # 只读工具可并发执行（无副作用，不修改文件系统状态）
    SAFE_TOOL_NAMES = {"Read", "Grep", "Glob"}
    # 写工具有副作用，必须串行（防止竞态：两个 Edit 同时修改同一文件会产生不确定结果）
    UNSAFE_TOOL_NAMES = {"Edit", "Bash", "Task", "Skill", "TodoWrite"}

    def __init__(self, host: Any):
        # host 是 CodeAgent 实例，提供 tool_executor、project_root、emit_runtime_event 等
        self.host = host

    # -------------------------------------------------------------------------
    # 工具生命周期事件发射（trace / transcript 可观测性）
    # -------------------------------------------------------------------------

    def _get_transcript_run_id(self) -> str:
        """获取当前 transcript run 的 ID，用于事件关联。

        优先使用 host 上的 _active_transcript_run_id（transcript 系统注入），
        回退到 _run_id（主循环步骤计数器），确保每个工具事件都能追溯到对应的 run。
        """
        run_id = getattr(self.host, "_active_transcript_run_id", None)
        if run_id is not None:
            return str(run_id)
        return f"run-{getattr(self.host, '_run_id', 0)}"

    def _emit_tool_lifecycle(
        self,
        *,
        step: int,
        tool_name: str,
        tool_call_id: str,
        status: str,
        payload: dict[str, Any] | None = None,
        trace_logger=None,
    ) -> None:
        """发射工具生命周期事件，供 trace/transcript 系统记录。

        生命周期状态序列：requested → started → completed/failed

        两条路由：
        - host 实现了 emit_runtime_event → 走 transcript 系统（完整持久化）
        - 否则 → 走 trace_logger（JSONL 日志，用于单元测试或无 transcript 场景）

        Args:
            step:         当前 ReAct 循环步骤编号
            tool_name:    工具名称
            tool_call_id: 对应的 tool_call ID（与模型输出对齐）
            status:       生命周期阶段（requested/started/completed/failed）
            payload:      阶段相关数据（args、result、error 等）
            trace_logger: 回退日志对象（无 host.emit_runtime_event 时使用）
        """
        event_payload = {
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "status": status,
            "payload": payload or {},
        }
        emit_runtime_event = getattr(self.host, "emit_runtime_event", None)
        if callable(emit_runtime_event):
            # 优先路由：transcript 系统，事件会持久化到 .jsonl 文件
            emit_runtime_event(
                run_id=self._get_transcript_run_id(),
                step=step,
                event_type="tool_lifecycle",
                payload=event_payload,
            )
            return
        # 回退路由：直接写 trace_logger
        self._log_tool_lifecycle(trace_logger, step=step, payload=event_payload)

    @staticmethod
    def _log_tool_lifecycle(trace_logger, *, step: int, payload: dict[str, Any]) -> None:
        """将工具生命周期事件写入 trace_logger（无 runtime host 时的回退路径）。

        根据 status 分发到不同的 trace 事件类型：
        - requested → log "tool_call"（记录调用意图和参数）
        - completed/failed → log "tool_result"（记录执行结果）
        - 所有状态 → 额外 log "tool_lifecycle"（完整生命周期流水账）

        这样 trace HTML 报告里可以分别看到调用记录和结果记录，以及完整生命周期。
        """
        if trace_logger is None:
            return
        lifecycle_payload = payload["payload"]
        if payload["status"] == "requested":
            # 工具被模型请求时，记录工具名和参数
            trace_logger.log_event(
                "tool_call",
                {
                    "tool": payload["tool_name"],
                    "args": lifecycle_payload.get("args") or {},
                    "tool_call_id": payload["tool_call_id"],
                },
                step=step,
            )
        elif payload["status"] in {"completed", "failed"} and "result" in lifecycle_payload:
            # 工具完成/失败时，记录结果（用于 trace HTML 中展示工具返回内容）
            trace_logger.log_event(
                "tool_result",
                {
                    "tool": payload["tool_name"],
                    "result": lifecycle_payload["result"],
                },
                step=step,
            )
        # 每个阶段都记录完整的生命周期事件（供调试分析工具耗时和状态转移）
        trace_logger.log_event("tool_lifecycle", payload, step=step)

    # -------------------------------------------------------------------------
    # 主入口
    # -------------------------------------------------------------------------

    def run(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        step: int,
        trace_logger,
    ) -> list[ToolObservation]:
        """处理模型返回的一批 tool_calls，返回有序的观测结果列表。

        完整流水线：
          1. plan_tool_calls()         → 解析每个 tool_call 的参数，生成 ToolCallPlan
          2. partition_tool_calls()    → 按安全性分批（并发批/串行批）
          3. 逐批执行：
             - 并发批 → _run_batch_concurrently()（ThreadPoolExecutor）
             - 串行批 → _run_batch_serially()（顺序执行）
          4. _normalize_empty_result() → 空输出补全占位文本，避免模型收到空 content
          5. _apply_observation_limit()→ 行数/字节初步截断（ObservationStore 策略）
          6. _apply_result_budget()    → 两层字节预算（单工具 + 批次总量），最终截断

        返回列表顺序与模型请求顺序完全一致（并发执行的结果按原始 offset 重排）。

        Args:
            tool_calls: 模型返回的 tool_calls 原始列表，每项含 name/id/arguments
            step:       当前 ReAct 循环步骤编号（用于 trace 事件关联）
            trace_logger: trace 日志对象

        Returns:
            list[ToolObservation]，每项的 .observation 字段可直接写入 history
        """
        # 步骤 1：解析参数，为每个 tool_call 生成执行计划
        plans = self.plan_tool_calls(tool_calls)

        # 步骤 2：按并发安全性将计划分批
        batches = self.partition_tool_calls(plans)

        # 记录分批计划到 trace（方便调试：哪些工具会并发，哪些串行）
        self._log_plan(trace_logger, step, batches)

        # 步骤 3：逐批执行，收集观测结果（顺序与原始 tool_calls 一致）
        observations: list[ToolObservation] = []
        for batch_index, batch in enumerate(batches):
            self._log_batch_start(trace_logger, step, batch_index, batch)
            # Python 三元表达式：先求值条件 batch.concurrency_safe，
            # 为 True  → 仅执行 _run_batch_concurrently()
            # 为 False → 仅执行 _run_batch_serially()
            # 两个函数调用只会执行其中一个，等价于 if/else 分支
            batch_observations = (
                self._run_batch_concurrently(batch, step=step, trace_logger=trace_logger)
                if batch.concurrency_safe
                else self._run_batch_serially(batch, step=step, trace_logger=trace_logger)
            )
            self._log_batch_end(trace_logger, step, batch_index, batch, batch_observations)
            observations.extend(batch_observations)

        # 步骤 4：空结果规范化（工具执行成功但无任何输出时，补充占位文本）
        observations = [self._normalize_empty_result(obs) for obs in observations]

        # 步骤 5：按行数/字节初步截断（ObservationStore 的 TOOL_OUTPUT_MAX_LINES 等配置）
        observations = [self._apply_observation_limit(obs) for obs in observations]

        # 步骤 6：两层字节预算控制（最终截断，确保写入 history 的总量不超限）
        return self._apply_result_budget(observations, step=step, trace_logger=trace_logger)

    def run_serial(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        step: int,
        trace_logger,
    ) -> list[ToolObservation]:
        """强制串行执行所有 tool_calls，跳过并发分组逻辑。

        用于子 agent（Task 工具内部）等需要严格顺序执行的场景，
        避免并发写操作在受限环境中引发竞态。
        注意：此方法不执行预算控制（_apply_result_budget），由调用方自行处理。
        """
        plans = self.plan_tool_calls(tool_calls)
        observations = self._run_batch_serially(
            ToolBatch(concurrency_safe=False, calls=plans),
            step=step,
            trace_logger=trace_logger,
        )
        return observations

    # -------------------------------------------------------------------------
    # 计划与分批
    # -------------------------------------------------------------------------

    def plan_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[ToolCallPlan]:
        """将模型原始 tool_calls 列表转换为 ToolCallPlan 列表。

        每个 plan 包含：
        - 解析后的参数 dict（parse_tool_input 处理 JSON string 或 dict）
        - 解析错误（若 arguments 是非法 JSON，parse_error 不为 None）
        - 并发安全性标注（只读工具为 True，写工具或解析失败为 False）

        解析失败的 plan 会在 _execute_plan() 里短路，返回 INVALID_PARAM 错误，
        不会进入实际的工具执行逻辑。

        Args:
            tool_calls: 模型返回的原始列表，每项格式为 {name, id, arguments}

        Returns:
            list[ToolCallPlan]，顺序与输入一致
        """
        plans: list[ToolCallPlan] = []
        for call in tool_calls:
            tool_name = call.get("name") or "unknown_tool"
            # 模型有时不提供 id，自动生成 uuid 保证 tool_call_id 唯一性
            tool_call_id = call.get("id") or f"call_{uuid.uuid4().hex}"
            raw_args = call.get("arguments") or {}
            # parse_tool_input 处理两种情况：
            # - arguments 已经是 dict（部分模型直接给 dict）
            # - arguments 是 JSON 字符串（需要 loads）
            # 返回 (parsed_dict, error)，error 不为 None 表示解析失败
            parsed_input, parse_error = parse_tool_input(raw_args)
            plans.append(
                ToolCallPlan(
                    call=call,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    # 解析失败时用空 dict，后续在 _execute_plan 里由 parse_error 触发短路
                    parsed_input=parsed_input if isinstance(parsed_input, dict) else {},
                    parse_error=parse_error,
                    concurrency_safe=self.is_concurrency_safe(tool_name, parse_error),
                )
            )
        return plans

    def is_concurrency_safe(self, tool_name: str, parse_error: Exception | None) -> bool:
        """判断一个工具调用是否可以与其他工具并发执行。

        并发安全的条件（同时满足）：
        1. 参数解析成功（parse_error is None）
        2. 工具在 SAFE_TOOL_NAMES 白名单中（Read/Grep/Glob，无副作用）

        以下情况强制串行（False）：
        - 参数解析失败：异常处理需要确定性，不能与其他工具并发
        - 工具在 UNSAFE_TOOL_NAMES 中：有文件系统副作用（Edit/Bash 等）
        - 工具名未知（不在任何名单中）：fail-safe，未知工具保守串行
        """
        if parse_error is not None:
            return False
        if tool_name in self.SAFE_TOOL_NAMES:
            return True
        if tool_name in self.UNSAFE_TOOL_NAMES:
            return False
        # 未知工具：保守串行，避免未知副作用
        return False

    def partition_tool_calls(self, plans: list[ToolCallPlan]) -> list[ToolBatch]:
        """将 ToolCallPlan 列表分批，连续安全工具合并为并发批。

        分批规则：
        - 若当前 plan 安全，且末尾批次也安全 → 合并进末尾批次（扩大并发组）
        - 否则 → 新建批次（切换并发/串行模式）

        例：[Read, Read, Edit, Grep, Grep, Edit, Read]
        → [并发(Read,Read), 串行(Edit), 并发(Grep,Grep), 串行(Edit), 并发(Read)]

        注意：ToolBatch 是 frozen dataclass，calls 字段虽然是 list 但实际上
        在合并时会直接 append，这里利用了 list 是可变对象的特性。
        """
        batches: list[ToolBatch] = []
        for plan in plans:
            # 当前 plan 安全 且 末尾批次也安全 → 合并进同一并发批
            if (
                batches
                and plan.concurrency_safe
                and batches[-1].concurrency_safe
            ):
                batches[-1].calls.append(plan)
                continue
            # 其他情况：新建批次（并发批或串行批由 plan.concurrency_safe 决定）
            batches.append(ToolBatch(concurrency_safe=plan.concurrency_safe, calls=[plan]))
        return batches

    # -------------------------------------------------------------------------
    # 批次执行
    # -------------------------------------------------------------------------

    def _run_batch_serially(
        self,
        batch: ToolBatch,
        *,
        step: int,
        trace_logger,
    ) -> list[ToolObservation]:
        """串行执行批次内的所有工具调用，保证执行顺序与 calls 列表顺序一致。

        用于写操作批次（Edit/Bash 等），避免并发修改同一文件导致竞态条件。

        串行体现在普通 for 循环：一次 _execute_plan 调用返回后才执行下一个，
        第一个工具跑完第二个才开始，时间线是顺序叠加：
            call_A → 等待 → call_B → 等待 → call_C
            总耗时 = A + B + C
        """
        observations: list[ToolObservation] = []
        for plan in batch.calls:
            # 同步调用：当前 plan 执行完毕（observation 拿到）才进入下一次循环
            observations.append(self._execute_plan(plan, step=step, trace_logger=trace_logger))
        return observations

    def _run_batch_concurrently(
        self,
        batch: ToolBatch,
        *,
        step: int,
        trace_logger,
    ) -> list[ToolObservation]:
        """并发执行批次内的所有工具调用，结果按原始顺序重排后返回。

        并发体现在 executor.submit()：submit 把任务提交给线程池后立即返回 Future，
        不等执行结果。批次内所有 plan 几乎同时提交，由线程池分配到不同线程并行跑：
            submit(A) submit(B) submit(C)  ← 几乎同时提交，不阻塞
            A、B、C 在不同线程里同时执行
            总耗时 ≈ max(A, B, C)          ← 远小于串行的 A+B+C

        结果用 dict[offset → observation] 收集而非直接 append list：
        线程完成顺序不确定（B 可能比 A 先完成），用 offset 记录每个 plan 的原始位置，
        最后按 range(len) 重建有序列表，保证返回顺序与模型请求顺序一致。

        max_workers 取「批次大小」与「环境并发上限」的较小值，
        避免批次很小时创建过多线程造成调度开销。
        """
        observations: dict[int, ToolObservation] = {}
        max_workers = min(len(batch.calls), self._get_max_concurrency())
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # submit 立即返回 Future，不阻塞——所有 plan 几乎同时开始执行
            futures = {
                offset: executor.submit(self._execute_plan, plan, step=step, trace_logger=trace_logger)
                for offset, plan in enumerate(batch.calls)
            }
            # .result() 阻塞等待该 Future 完成，按 offset 存入 dict 保留原始位置
            for offset, future in futures.items():
                observations[offset] = future.result()
        # 按原始 offset 顺序重建列表：线程完成顺序不定，这里强制恢复模型请求顺序
        return [observations[idx] for idx in range(len(batch.calls))]

    # -------------------------------------------------------------------------
    # 单次工具执行
    # -------------------------------------------------------------------------

    def _execute_plan(self, plan: ToolCallPlan, *, step: int, trace_logger) -> ToolObservation:
        """执行单个 ToolCallPlan，返回 ToolObservation。

        两条执行路径：
        1. parse_error 不为 None → 参数解析失败，直接构造 INVALID_PARAM 错误结果
           （不进入工具执行逻辑，避免带着坏参数调用工具）
        2. parse_error 为 None  → 正常执行，委托给 _execute_one()

        无论哪条路径，都会通过 _emit_tool_lifecycle() 发射 requested/failed 事件，
        确保 trace/transcript 里有完整的生命周期记录。

        Args:
            plan: 已解析的工具调用计划
            step: 当前 ReAct 循环步骤编号
            trace_logger: trace 日志对象

        Returns:
            ToolObservation（含序列化后的 observation 字符串）
        """
        # 阶段 1：发射「工具被请求」事件（无论成功与否都先记录调用意图）
        self._emit_tool_lifecycle(
            step=step,
            tool_name=plan.tool_name,
            tool_call_id=plan.tool_call_id,
            status="requested",
            payload={"args": plan.parsed_input},
            trace_logger=trace_logger,
        )

        if plan.parse_error is not None:
            # 路径 A：参数解析失败，构造错误结果并短路返回
            result = self._parse_error_result(plan.parse_error)
            # 将解析错误单独记录到 trace（与生命周期事件区分，便于分析错误类型）
            self._log_parse_error(trace_logger, step, plan.tool_name, plan.tool_call_id, plan.parse_error)
            # 发射「工具失败」事件
            self._emit_tool_lifecycle(
                step=step,
                tool_name=plan.tool_name,
                tool_call_id=plan.tool_call_id,
                status="failed",
                payload={"error": str(plan.parse_error), "args": plan.parsed_input},
                trace_logger=trace_logger,
            )
        else:
            # 路径 B：参数解析成功，进入实际执行逻辑
            result = self._execute_one(
                plan.tool_name,
                plan.parsed_input,
                plan.tool_call_id,
                trace_logger,
                step,
            )

        # 构造 ToolObservation：observation 字段在 __post_init__ 中自动序列化
        return ToolObservation(
            tool_name=plan.tool_name,
            tool_call_id=plan.tool_call_id,
            result=result,
        )

    def _execute_one(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_call_id: str,
        trace_logger,
        step: int,
    ) -> ToolResult:
        """调用 host 的 tool_executor 执行单个工具，处理异常并发射生命周期事件。

        执行路径：
        - host.tool_executor 存在 → 走 ToolExecutor.execute()（含权限/乐观锁/熔断管道）
        - 否则 → 走 host._execute_tool()（遗留接口，用于向后兼容）

        异常处理：
        - tool.run() 抛出的任何未捕获异常都在这里兜底，转为 EXECUTION_ERROR ToolResult
        - 同时发射 "failed" 生命周期事件，并将完整 traceback 写入 trace

        Args:
            tool_name:    工具名称
            tool_input:   已解析的参数 dict
            tool_call_id: 对应的 tool_call ID
            trace_logger: trace 日志对象
            step:         当前 ReAct 循环步骤编号

        Returns:
            ToolResult（成功/部分成功/错误，不会抛出异常）
        """
        host = self.host
        # 发射「工具开始执行」事件（区别于 requested：started 表示已通过权限检查，进入执行）
        self._emit_tool_lifecycle(
            step=step,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            status="started",
            payload={"args": tool_input},
            trace_logger=trace_logger,
        )
        try:
            if hasattr(host, "tool_executor") and host.tool_executor is not None:
                # 主路径：通过 ToolExecutor 执行（含权限检查 → 乐观锁注入 → 熔断检查 → tool.run()）
                result = host.tool_executor.execute(
                    tool_name,
                    tool_input,
                    trace_logger=trace_logger,
                    step=step,
                )
            else:
                # 回退路径：直接调用 host._execute_tool()（旧版兼容接口）
                result = host._execute_tool(tool_name, tool_input)

            # 校验返回类型：所有工具必须返回 ToolResult（协议强制）
            if not isinstance(result, ToolResult):
                raise TypeError(f"Tool '{tool_name}' returned unsupported result type.")

            # 根据执行结果决定生命周期状态：error → failed，其余 → completed
            lifecycle_status, lifecycle_payload = self._tool_lifecycle_result_payload(result)
            self._emit_tool_lifecycle(
                step=step,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                status=lifecycle_status,
                payload=lifecycle_payload,
                trace_logger=trace_logger,
            )
            return result

        except Exception as exc:
            # 兜底异常处理：构造 EXECUTION_ERROR 结果，保证不向上抛出
            error_result = ToolResult(
                status=ToolStatus.ERROR,
                data={},
                text=str(exc),
                error_code=ErrorCode.EXECUTION_ERROR,
                error_message=str(exc),
                stats={"time_ms": 0},
                context={"cwd": ".", "params_input": tool_input},
            )
            # 将完整 traceback 写入 trace，方便调试
            trace_logger.log_event(
                "error",
                {
                    "stage": "tool_execution",
                    "error_code": "EXECUTION_ERROR",
                    "message": str(exc),
                    "tool": tool_name,
                    "traceback": tb.format_exc(),
                },
                step=step,
            )
            # 发射「工具失败」事件
            self._emit_tool_lifecycle(
                step=step,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                status="failed",
                payload={"error": str(exc), "args": tool_input},
                trace_logger=trace_logger,
            )
            return error_result

    def _tool_lifecycle_result_payload(self, result: ToolResult) -> tuple[str, dict[str, Any]]:
        """将 ToolResult 映射到生命周期状态和事件 payload。

        Returns:
            (lifecycle_status, payload) 其中：
            - lifecycle_status: "completed" 或 "failed"
            - payload: 含序列化后的 result dict（写入 trace）
        """
        lifecycle_status = "failed" if result.status is ToolStatus.ERROR else "completed"
        return lifecycle_status, {"result": tool_result_payload(result)}

    def _parse_error_result(self, parse_err: Exception) -> ToolResult:
        """为参数解析失败构造标准错误结果。

        使用 INVALID_PARAM 错误码，让模型知道是参数格式问题，
        而不是工具本身的执行错误。
        """
        message = f"Tool arguments parse error: {parse_err}"
        return ToolResult(
            status=ToolStatus.ERROR,
            data={},
            text=message,
            error_code=ErrorCode.INVALID_PARAM,
            error_message=message,
            stats={"time_ms": 0},
            context={"cwd": ".", "params_input": {}},
        )

    def _log_parse_error(
        self,
        trace_logger,
        step: int,
        tool_name: str,
        tool_call_id: str,
        parse_err: Exception,
    ) -> None:
        """将参数解析错误写入 trace（独立于生命周期事件，便于按错误类型检索）。"""
        trace_logger.log_event(
            "error",
            {
                "stage": "tool_call_parse",
                "error_code": "INVALID_PARAM",
                "message": str(parse_err),
                "tool": tool_name,
                "tool_call_id": tool_call_id,
            },
            step=step,
        )

    # -------------------------------------------------------------------------
    # 并发控制与预算配置
    # -------------------------------------------------------------------------

    def _get_max_concurrency(self) -> int:
        """读取并发工具数上限，默认 4。

        通过环境变量 MYCODEAGENT_MAX_TOOL_CONCURRENCY 配置。
        取 max(1, ...) 保证至少有 1 个线程（避免 ThreadPoolExecutor(max_workers=0) 报错）。
        """
        raw_value = os.getenv("MYCODEAGENT_MAX_TOOL_CONCURRENCY", "4")
        try:
            return max(1, int(raw_value))
        except ValueError:
            return 4

    def _get_result_budget(self) -> ToolResultBudget:
        """从环境变量读取两层字节预算配置。

        环境变量：
        - MYCODEAGENT_MAX_TOOL_RESULT_BYTES:   单工具上限，默认 50000 字节（≈50KB）
        - MYCODEAGENT_MAX_TOOL_MESSAGE_BYTES:  批次总量上限，默认 200000 字节（≈200KB）
        """
        def _read_env(name: str, default: int) -> int:
            try:
                return max(1, int(os.getenv(name, str(default))))
            except ValueError:
                return default

        return ToolResultBudget(
            max_tool_bytes=_read_env("MYCODEAGENT_MAX_TOOL_RESULT_BYTES", 50000),
            max_message_bytes=_read_env("MYCODEAGENT_MAX_TOOL_MESSAGE_BYTES", 200000),
        )

    @staticmethod
    def _result_bytes(result: ToolResult) -> int:
        """计算 ToolResult 序列化后的 UTF-8 字节数（用于预算比较）。"""
        return len(serialize_tool_result(result).encode("utf-8"))

    # -------------------------------------------------------------------------
    # 观测结果后处理（空值规范化 → 初步截断 → 预算截断）
    # -------------------------------------------------------------------------

    def _normalize_empty_result(self, obs: ToolObservation) -> ToolObservation:
        """将空输出的工具结果替换为含占位文本的结果。

        工具执行成功但没有任何输出（text 为空且 data 为空）时，
        模型收到空 content 可能产生困惑或重复调用同一工具。
        此方法补充一个说明性占位文本，让模型知道工具已完成但无输出。

        原始结果保存在 raw_result 字段，元数据里标记 replaced=False（区别于截断替换）。
        """
        if not self._is_empty_result(obs.result):
            return obs  # 非空结果直接透传

        metadata = {**(obs.metadata or {}), "budgeted": True, "reason": "empty_result", "replaced": False}
        return ToolObservation(
            tool_name=obs.tool_name,
            tool_call_id=obs.tool_call_id,
            result=ToolResult(
                status=ToolStatus.SUCCESS,
                data={},
                text=f"{obs.tool_name} completed with no output.",
                stats={"time_ms": 0},
                context={"cwd": ".", "params_input": {}},
            ),
            raw_result=obs.result,  # 保留原始空结果，供调试
            metadata=metadata,
        )

    def _apply_observation_limit(self, obs: ToolObservation) -> ToolObservation:
        """按行数/字节做初步截断（ObservationStore 策略，早于预算截断）。

        调用 observation_store.truncate_result()，按环境变量配置
        （TOOL_OUTPUT_MAX_LINES、TOOL_OUTPUT_MAX_BYTES 等）决定是否截断。

        若 truncate_result 返回同一个对象（is 比较），说明无需截断，直接透传。
        截断时保留原始 result 到 raw_result，metadata 记录 reason 和字节变化。
        """
        truncated = truncate_result(
            obs.tool_name,
            obs.result,
            getattr(self.host, "project_root", None),
        )
        if truncated is obs.result:
            return obs  # 未截断，直接返回原对象

        raw_result = obs.raw_result if obs.raw_result is not None else obs.result
        return ToolObservation(
            tool_name=obs.tool_name,
            tool_call_id=obs.tool_call_id,
            result=truncated,
            raw_result=raw_result,
            metadata={
                **(obs.metadata or {}),
                "reason": "observation_limit",
                "raw_bytes": self._result_bytes(raw_result),
                "visible_bytes": self._result_bytes(truncated),
            },
        )

    def _apply_result_budget(
        self,
        observations: list[ToolObservation],
        *,
        step: int,
        trace_logger,
    ) -> list[ToolObservation]:
        """两层字节预算控制，防止工具输出撑爆 context window。

        层 1（单工具上限）：
            对每个观测结果，若序列化字节数 > max_tool_bytes（默认 50KB），
            调用 force_truncate_result() 强制截断，完整输出 spill 到磁盘文件，
            结果中附 truncation.full_output_path 供模型引用。

        层 2（批次总量上限）：
            层 1 处理后，若所有结果的总字节数仍 > max_message_bytes（默认 200KB），
            按结果大小降序找最大的未截断结果，逐个强制截断，直到总量达标。
            优先截断大的（贪心策略），已被层 1 截断的跳过（避免二次截断）。

        两层都通过 force_truncate_result() 实现截断，截断策略由 ObservationStore 决定。
        截断后元数据里 replaced=True 标记，用于层 2 的跳过逻辑。
        """
        budget = self._get_result_budget()
        trace_logger.log_event(
            "tool_result_budget_start",
            {
                "tool_count": len(observations),
                "max_tool_bytes": budget.max_tool_bytes,
                "max_message_bytes": budget.max_message_bytes,
            },
            step=step,
        )
        budgeted: list[ToolObservation] = []
        replaced_count = 0
        raw_total_bytes = 0    # 所有原始结果字节总和（统计用）
        visible_total_bytes = 0  # 当前可见（截断后）结果字节总和（预算计算用）

        # ── 层 1：逐个检查单工具上限 ──
        for obs in observations:
            # 以 raw_result 为基准计算原始大小（已被 _apply_observation_limit 截断的用其原始值）
            raw_result = obs.raw_result if obs.raw_result is not None else obs.result
            raw_bytes = self._result_bytes(raw_result)
            raw_total_bytes += raw_bytes
            next_obs = obs  # 默认不替换

            if raw_bytes > budget.max_tool_bytes:
                # 超过单工具上限：强制截断，完整内容 spill 到磁盘
                compressed = force_truncate_result(
                    obs.tool_name,
                    raw_result,
                    self.host.project_root,
                )
                visible_bytes = self._result_bytes(compressed)
                metadata = {
                    **(obs.metadata or {}),
                    "budgeted": True,
                    "replaced": True,           # 标记为已替换，层 2 跳过此项
                    "reason": "single_tool_budget",
                    "raw_bytes": raw_bytes,
                    "visible_bytes": visible_bytes,
                }
                # 记录 spill 文件路径到 metadata（供调试查看完整输出）
                full_output_path = compressed.data.get("truncation", {}).get("full_output_path")
                if full_output_path:
                    metadata["full_output_path"] = full_output_path
                next_obs = ToolObservation(
                    tool_name=obs.tool_name,
                    tool_call_id=obs.tool_call_id,
                    result=compressed,
                    raw_result=raw_result,
                    metadata=metadata,
                )
                replaced_count += 1
                trace_logger.log_event(
                    "tool_result_budget_item",
                    {
                        "tool_call_id": obs.tool_call_id,
                        "reason": "single_tool_budget",
                        "replaced": True,
                        "raw_bytes": raw_bytes,
                        "visible_bytes": visible_bytes,
                    },
                    step=step,
                )
            budgeted.append(next_obs)
            visible_total_bytes += self._result_bytes(next_obs.result)

        # ── 层 2：检查批次总量上限 ──
        if visible_total_bytes > budget.max_message_bytes:
            # 按当前可见大小降序排列，优先截断最大的（贪心策略减少截断次数）
            indexed = list(enumerate(budgeted))
            indexed.sort(key=lambda item: self._result_bytes(item[1].result), reverse=True)

            for idx, obs in indexed:
                if visible_total_bytes <= budget.max_message_bytes:
                    break  # 总量已达标，提前结束
                if (obs.metadata or {}).get("replaced") is True:
                    continue  # 层 1 已截断的跳过，避免对同一结果二次截断

                # 从 raw_result 重新截断（确保截断基于原始内容，而非已截断内容）
                source_result = obs.raw_result if obs.raw_result is not None else obs.result
                previous_visible = self._result_bytes(obs.result)
                compressed = force_truncate_result(
                    obs.tool_name,
                    source_result,
                    self.host.project_root,
                )
                visible_bytes = self._result_bytes(compressed)
                metadata = {
                    **(obs.metadata or {}),
                    "budgeted": True,
                    "replaced": True,
                    "reason": "aggregate_message_budget",
                    "raw_bytes": self._result_bytes(source_result),
                    "visible_bytes": visible_bytes,
                }
                full_output_path = compressed.data.get("truncation", {}).get("full_output_path")
                if full_output_path:
                    metadata["full_output_path"] = full_output_path
                # 原地替换 budgeted 列表中的元素（idx 是 enumerate 的原始位置）
                budgeted[idx] = ToolObservation(
                    tool_name=obs.tool_name,
                    tool_call_id=obs.tool_call_id,
                    result=compressed,
                    raw_result=source_result,
                    metadata=metadata,
                )
                # 更新总量计数：减去替换前的可见大小，加上截断后的大小
                visible_total_bytes = visible_total_bytes - previous_visible + visible_bytes
                replaced_count += 1
                trace_logger.log_event(
                    "tool_result_budget_item",
                    {
                        "tool_call_id": obs.tool_call_id,
                        "reason": "aggregate_message_budget",
                        "replaced": True,
                        "raw_bytes": self._result_bytes(source_result),
                        "visible_bytes": visible_bytes,
                    },
                    step=step,
                )

        # 记录预算处理结束摘要
        trace_logger.log_event(
            "tool_result_budget_end",
            {
                "tool_count": len(observations),
                "max_tool_bytes": budget.max_tool_bytes,
                "max_message_bytes": budget.max_message_bytes,
                "raw_total_bytes": raw_total_bytes,
                "visible_total_bytes": visible_total_bytes,
                "replaced_count": replaced_count,
            },
            step=step,
        )
        return budgeted

    @staticmethod
    def _is_empty_result(result: ToolResult) -> bool:
        """判断工具结果是否为「有效空输出」。

        空输出条件（同时满足）：
        1. 非错误状态（error 本身就有文字信息，不算「空」）
        2. text 为空白
        3. data 为空 dict 或 falsy 值
        """
        return (
            result.status is not ToolStatus.ERROR
            and not result.text.strip()
            and not result.data
        )

    # -------------------------------------------------------------------------
    # Trace 日志辅助
    # -------------------------------------------------------------------------

    def _log_plan(self, trace_logger, step: int, batches: list[ToolBatch]) -> None:
        """将分批计划写入 trace，记录每批的并发模式和工具名列表。

        用途：调试时可以看到哪些工具被分到同一并发批，哪些被拆分为串行批。
        """
        trace_logger.log_event(
            "tool_orchestration_plan",
            {
                "batch_count": len(batches),
                "batches": [
                    {
                        "concurrency_safe": batch.concurrency_safe,
                        "tool_names": [plan.tool_name for plan in batch.calls],
                    }
                    for batch in batches
                ],
            },
            step=step,
        )

    def _log_batch_start(self, trace_logger, step: int, batch_index: int, batch: ToolBatch) -> None:
        """记录批次开始事件（包含并发模式和工具列表）。"""
        trace_logger.log_event(
            "tool_batch_start",
            {
                "batch_index": batch_index,
                "concurrency_safe": batch.concurrency_safe,
                "tool_count": len(batch.calls),
                "tool_names": [plan.tool_name for plan in batch.calls],
            },
            step=step,
        )

    def _log_batch_end(
        self,
        trace_logger,
        step: int,
        batch_index: int,
        batch: ToolBatch,
        observations: list[ToolObservation],
    ) -> None:
        """记录批次结束事件（包含完成数量，可与 tool_count 对比检测漏执行）。"""
        trace_logger.log_event(
            "tool_batch_end",
            {
                "batch_index": batch_index,
                "concurrency_safe": batch.concurrency_safe,
                "tool_count": len(batch.calls),
                "completed_count": len(observations),
            },
            step=step,
        )
