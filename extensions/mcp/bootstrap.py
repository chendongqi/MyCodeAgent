"""Register MCP servers and tools in ToolRegistry.

系列 07：MCP 组装入口。默认不走这里——ENABLE_MCP / --enable-mcp 打开后，
才懒加载 SDK、读配置、为每个 server 建 Client，并把远端工具 Adapter 进 Registry。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from extensions.mcp.config import load_mcp_servers, connect_mode
from extensions.mcp.errors import MCP_EXTRA_ERROR, MCPExtraRequiredError
from extensions.mcp.prompt import format_mcp_tools_prompt

if TYPE_CHECKING:
    from extensions.mcp.client import MCPClient

logger = logging.getLogger(__name__)

def _load_mcp_runtime() -> tuple[type[Any], type[Any], Any]:
    """Load the MCP SDK boundary only after the user explicitly enables MCP.

    client/adapter 会 import 官方 mcp 包。核心安装不含这个 extra，
    所以必须拖到「用户显式开启」之后，避免没装 SDK 时连主循环都起不来。
    """
    try:
        from extensions.mcp.adapter import register_mcp_tools
        from extensions.mcp.client import MCPClient, MCPClientConfig
    except ImportError as exc:
        raise MCPExtraRequiredError(MCP_EXTRA_ERROR) from exc
    return MCPClient, MCPClientConfig, register_mcp_tools


def _default_uv_env(project_root: str, env: dict[str, str] | None) -> dict[str, str]:
    merged = dict(env or {})
    root = Path(project_root)
    cache_dir = root / ".uv_cache"
    tool_dir = root / ".uv_tools"
    npm_cache = root / ".npm_cache"

    cache_dir.mkdir(parents=True, exist_ok=True)
    tool_dir.mkdir(parents=True, exist_ok=True)
    npm_cache.mkdir(parents=True, exist_ok=True)

    merged.setdefault("UV_CACHE_DIR", str(cache_dir))
    merged.setdefault("XDG_CACHE_HOME", str(cache_dir))
    merged.setdefault("UV_HOME", str(tool_dir))
    merged.setdefault("UV_TOOL_DIR", str(tool_dir))
    merged.setdefault("UV_TOOL_BIN_DIR", str(tool_dir / "bin"))

    merged.setdefault("NPM_CONFIG_CACHE", str(npm_cache))
    merged.setdefault("NPM_CONFIG_LOGLEVEL", "error")
    merged.setdefault("NPM_CONFIG_FUND", "false")
    merged.setdefault("NPM_CONFIG_AUDIT", "false")

    return merged


def _build_client_config(project_root: str, spec: dict[str, Any], client_config_class: type[Any]) -> Any:
    transport = spec.get("transport")
    url = spec.get("url") or spec.get("endpoint")
    command = spec.get("command")
    args = spec.get("args") or []
    env = spec.get("env") or {}

    if transport == "http" or url:
        if not url:
            raise ValueError("MCP server config requires url for http transport")
        return client_config_class(transport="http", url=url, env=env)

    if command in {"uvx", "uv"}:
        env = _default_uv_env(project_root, env)

    if not command:
        raise ValueError("MCP server config requires command for stdio transport")
    expanded_args = [os.path.expandvars(str(arg)) for arg in args]
    return client_config_class(transport="stdio", command=command, args=expanded_args, env=env)


def register_mcp_servers(tool_registry, project_root: str) -> tuple[list[MCPClient], list[dict[str, object | None]]]:
    """MCP 组装主入口：配置 → Client → 工具发现 → 注册到 ToolRegistry。

    整体流程：
    1. load_mcp_servers    读配置（JSON 文件 / 环境变量）拿到 server 清单
    2. connect_mode        决定连接策略（startup 立即连 / disabled 跳过）
    3. _build_client_config 为每个 server 构造 MCPClientConfig（stdio or http）
    4. MCPClient(config)   创建 client 对象，但尚未连接
    5. register_mcp_tools  连接 server、调 list_tools、把每个远端工具包成 Adapter 注入 Registry

    单个 server 失败只记 warning，不影响其他 server 和主循环启动。
    返回值：(clients 列表, 已注册工具的元信息列表)
    clients 由 host 持有，session 结束时调 client.close_sync() 释放连接。
    """
    # 懒加载 MCP SDK（核心安装不含 mcp 包，只有用户显式开启后才 import）
    MCPClient, MCPClientConfig, register_mcp_tools = _load_mcp_runtime()

    # load_mcp_servers：优先读环境变量 MCP_SERVERS（JSON 字符串），
    # 其次按顺序查找 mcp_servers.json / .mcp.json / mcp.json，
    # 兼容 {"mcpServers": {...}} 的 Claude 桌面端写法
    servers = load_mcp_servers(project_root)

    # connect_mode：读环境变量 MCP_CONNECT_MODE，默认 "startup"
    # "startup" → 启动时立即连接并发现工具
    # "disabled" → 配置文件存在但不连接（调试用）
    mode = connect_mode()
    if not servers or mode == "disabled":
        return [], []

    clients: list[MCPClient] = []
    registered_tools: list[dict[str, object | None]] = []

    for server_name, spec in servers.items():
        if not isinstance(spec, dict):
            continue

        # _build_client_config：把 JSON spec 转成 MCPClientConfig
        # 两种 transport：
        #   http  → spec 里有 url/endpoint，适合远程 HTTP server
        #   stdio → spec 里有 command（如 "uvx"/"node"），适合本地子进程
        # uvx/uv 命令额外注入缓存目录环境变量，防止子进程污染系统全局缓存
        config = _build_client_config(project_root, spec, MCPClientConfig)
        client = MCPClient(config)
        clients.append(client)

        if mode != "startup":
            continue

        try:
            # register_mcp_tools：连接 server → list_tools_sync → 为每个工具创建 MCPToolAdapter
            # MCPToolAdapter 把远端工具伪装成本地 Tool 对象：
            #   - get_parameters() 从 inputSchema 生成参数定义
            #   - run() 调 client.call_tool_sync() 发请求，把结果包成标准信封
            # 公开名 = sanitize(server_name:remote_name)，避免两个 server 都有 "search" 工具时冲突
            # 注册后和内置 Read/Bash 走同一套 Registry / Orchestrator / Executor 管道
            tools_meta = register_mcp_tools(tool_registry, client, namespace=server_name)
            registered_tools.extend(tools_meta)
        except Exception as exc:
            logger.warning("MCP tool registration failed for %s: %s", server_name, exc)
            continue

    return clients, registered_tools


__all__ = ["MCPExtraRequiredError", "format_mcp_tools_prompt", "register_mcp_servers"]
