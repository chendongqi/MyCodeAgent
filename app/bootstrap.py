"""Application bootstrap helpers for the canonical single-agent CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from core.config import Config
from core.llm import HelloAgentsLLM
from runtime.host import CodeAgent
from tools.registry import ToolRegistry

PACKAGE_RESOURCE_ROOT = str(Path(__file__).resolve().parent.parent)


@dataclass
class RuntimeBootstrap:
    """Constructed runtime dependencies for the CLI entrypoint."""

    config: Config
    llm: Any
    tool_registry: Any
    agent: Any
    project_root: str


def resolve_project_root(project_root: Optional[str] = None) -> str:
    """Resolve an existing target project directory from the invocation context."""

    target = Path(project_root).expanduser() if project_root is not None else Path.cwd()
    resolved_target = target.resolve()
    if not resolved_target.is_dir():
        raise ValueError(f"Project root must be an existing directory: {target}")
    return str(resolved_target)


def build_runtime(
    args: Any,
    *,
    project_root: Optional[str] = None,
    extension_flags: Optional[dict[str, bool]] = None,
    config_class: type[Config] = Config,
    llm_class: type[HelloAgentsLLM] = HelloAgentsLLM,
    tool_registry_factory: Callable[[], ToolRegistry] = ToolRegistry,
    agent_class: type[CodeAgent] = CodeAgent,
    agent_kwargs_factory: Optional[Callable[[Config, Any, str], dict[str, Any]]] = None,
) -> RuntimeBootstrap:
    """Assemble config, llm, registry, and agent without starting the UI loop."""

    # 步骤 1：确定 project_root（优先 --cwd 参数，否则用当前目录）
    selected_project_root = project_root if project_root is not None else getattr(args, "cwd", None)
    resolved_project_root = resolve_project_root(selected_project_root)

    # 步骤 2：加载配置（从 .env 读取所有环境变量，命令行参数可覆盖部分字段）
    config = config_class.from_env()
    for argument, attribute in (
        ("enable_mcp", "enable_mcp"),
        ("enable_verification_agent", "enable_verification_agent"),
    ):
        if getattr(args, argument, False):
            setattr(config, attribute, True)

    # 步骤 3：创建 LLM 客户端（命令行参数优先，其次 config，最后环境变量）
    llm = llm_class(
        model=getattr(args, "model", None),
        api_key=getattr(args, "api_key", None),
        base_url=getattr(args, "base_url", None),
        provider=getattr(args, "provider", None),
        temperature=(
            getattr(args, "temperature", None)
            if getattr(args, "temperature", None) is not None
            else config.temperature
        ),
    )

    # 步骤 4：创建空工具注册表（内置工具由 CodeAgent.__init__ 负责注册）
    tool_registry = tool_registry_factory()

    # 步骤 5：组装 agent 的构造参数（依赖注入：LLM + 工具注册表 + 配置全在这里传入）
    agent_kwargs = {
        "name": getattr(args, "name", "code"),
        "llm": llm,
        "tool_registry": tool_registry,
        "project_root": resolved_project_root,
        "package_resource_root": PACKAGE_RESOURCE_ROOT,
        "system_prompt": getattr(args, "system", None),
        "config": config,
        "enable_mcp": config.enable_mcp,
        "enable_skills": config.enable_skills,
        "enable_tracing": config.enable_tracing,
    }
    if extension_flags:
        if "mcp" in extension_flags:
            agent_kwargs["enable_mcp"] = bool(extension_flags["mcp"])
        if "skills" in extension_flags:
            agent_kwargs["enable_skills"] = bool(extension_flags["skills"])
        if "tracing" in extension_flags:
            agent_kwargs["enable_tracing"] = bool(extension_flags["tracing"])
    # 交互模式通过 agent_kwargs_factory 额外注入 EnhancedUI 对象
    if agent_kwargs_factory is not None:
        agent_kwargs.update(agent_kwargs_factory(config, llm, resolved_project_root))

    # 步骤 6：实例化 agent（内部会注册所有内置工具并初始化上下文引擎）
    agent = agent_class(**agent_kwargs)

    return RuntimeBootstrap(
        config=config,
        llm=llm,
        tool_registry=tool_registry,
        agent=agent,
        project_root=resolved_project_root,
    )
