# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**MyCodeAgent** is a Python-based code agent framework focused on:
- **Function Calling & Tool Protocols**: Standardized tool execution with structured response formats
- **Context Engineering**: Intelligent history compression, truncation, and context management
- **Subagent Mechanisms**: Delegating complex tasks to lightweight agents (Task system)
- **AgentTeams (MVP)**: Experimental feature for multi-agent collaboration and parallel task execution
- **Observability**: Comprehensive trace logging (JSONL + HTML), token statistics, and execution tracking

**Use Cases**: Learning LLM agent development, researching context engineering, experimenting with multi-agent systems, building extensible local agent playgrounds.

---

## Core Architecture

### Directory Structure

```
agents/              CodeAgent implementation (main agent class)
core/                Core runtime & context engineering
  ├─ agent.py       Abstract Agent base class
  ├─ llm.py         Unified LLM interface (OpenAI-compatible)
  ├─ config.py      Configuration management
  ├─ message.py     Message data model
  ├─ session_store.py Session persistence
  └─ context_engine/ Context engineering pipeline
      ├─ history_manager.py    (compression, retention, truncation)
      ├─ trace_logger.py       (JSONL + HTML logging, sanitization)
      ├─ observation_truncator.py (tool output truncation & spill)
      └─ summary_compressor.py (LLM-powered context summarization)

tools/               Tool system & registry
  ├─ base.py        Tool abstract class, ToolParameter, response protocol
  ├─ registry.py    ToolRegistry (unified tool execution, two registration modes)
  ├─ circuit_breaker.py (automatic tool disabling on repeated failures)
  ├─ builtin/       Built-in tools (LS, Glob, Grep, Read, Write, Edit, MultiEdit, etc.)
  └─ mcp/           MCP server integration

prompts/             System prompts & tool prompts
  ├─ agents_prompts/ Agent instructions
  └─ tools_prompts/  Tool descriptions & usage guidelines

skills/              User-defined skills (Skill.md format)
core/team_engine/    AgentTeams support (experimental)
docs/                Design documents & protocol specs
scripts/             CLI entry points (chat_test_agent.py)
tests/               Unit & integration tests
utils/               Utility functions

key files:
- mcp_servers.json   MCP tool configuration
- requirements.txt   Python dependencies
- .env.example       Environment variable template
```

### Key Components

#### 1. **LLM Interface** (`core/llm.py`)
- Unified OpenAI-compatible client supporting multiple providers (OpenAI, DeepSeek, Zhipu, SiliconFlow, Ollama, etc.)
- Prioritizes passed parameters, falls back to environment variables
- Supports streaming responses, retries, and custom timeouts
- Detects provider automatically or uses explicit configuration

#### 2. **Tool System** (`tools/`)
- **Tool Base Class** (`base.py`): Abstract Tool class with parameter definitions, response protocol, and validation
- **Registry** (`registry.py`): Two registration modes:
  - **Tool objects**: Full parameter definitions + structured validation (recommended)
  - **Functions**: Simple string-in, string-out functions (quick tools)
- **Response Protocol**: Standardized format (status, data, text, error, stats, context)
  - All tools must return JSON matching the protocol
  - Status: `success` | `partial` | `error`

#### 3. **Context Engineering** (`core/context_engine/`)
- **HistoryManager**: Rounds tracking, compression threshold monitoring, retention policy
- **SummaryCompressor**: LLM-powered context summarization when compression threshold is hit
- **ObservationTruncator**: Tool output truncation (head/tail/head_tail), spill to disk (`tool-output/`)
- **TraceLogger**: JSONL + HTML dual logging, automatic spill, retention cleanup (7 days default)

#### 4. **Built-in Tools** (`tools/builtin/`)
- **LS** (list_files): Safe directory browsing with pagination, hidden file control, symlink detection
- **Glob** (search_files_by_name): Pattern-based file search with dual circuit breakers (visit count + time limit)
- **Grep** (search_code): Regex content search, ripgrep priority, Python fallback, mtime sorting
- **Read** (read_file): Safe file reading with encoding detection, binary file warnings
- **Write** (write_file): File creation/overwrite with directory auto-creation
- **Edit** (edit_file): Single-point string replacement with context preservation
- **MultiEdit** (edit_file_multi): Batch replacements with validation
- **Bash**: Command execution with timeout, stderr/stdout capture, exit code tracking
- **TodoWrite**: Task list management (create, update status, mark complete)
- **Skill**: Load and execute user-defined skills from `skills/` directory
- **Task**: Subagent delegation (oneshot, persistent, or parallel modes)
- **AskUser** (send_message): Interactive user prompts

#### 5. **CodeAgent** (`agents/codeAgent.py`)
- Main agent implementation with ReAct loop (Reasoning → Action → Observation)
- Integrated tool execution via ToolRegistry
- History management with compression triggers
- Trace logging of all interactions
- Configurable response formatting (raw vs. formatted)

#### 6. **AgentTeams (Experimental)** (`core/team_engine/`)
- Feature flag: `ENABLE_AGENT_TEAMS=true` (default: false)
- Tools: `TeamCreate`, `SendMessage`, `TeamStatus`, `TeamDelete`, `TeamFanout`, `TeamCollect`
- Parallel task distribution across team members
- Message ACK states: pending → delivered → processed

---

## Development Workflow

### Quick Start

```bash
# Clone & install
git clone <repo>
cd MyCodeAgent
python -m venv venv
source venv/bin/activate  # or: .\venv\Scripts\activate on Windows
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your LLM API key & provider

# Run interactive CLI
python scripts/chat_test_agent.py
```

### Common Commands

| Command | Purpose |
|---------|---------|
| `python scripts/chat_test_agent.py` | Run interactive agent CLI |
| `python scripts/chat_test_agent.py --show-raw` | Show raw LLM responses (debug mode) |
| `python scripts/chat_test_agent.py --provider deepseek --model deepseek-chat --api-key YOUR_KEY` | Override LLM provider |
| `python -m pytest tests/ -v` | Run all tests |
| `python -m pytest tests/test_message.py -v` | Run specific test file |
| `python -m pytest tests/ --cov=.` | Run tests with coverage |

### Testing

- Unit tests in `tests/` directory
- Pytest configuration: standard pytest.ini (if present)
- To test a single tool: `pytest tests/test_<tool_name>.py -v`
- Coverage tracking with `--cov` flag

---

## Key Environment Variables

### LLM Configuration
- `LLM_PROVIDER`: Provider name (openai, deepseek, zhipu, siliconflow, etc.)
- `LLM_API_KEY`: API authentication key
- `LLM_BASE_URL`: Service endpoint (provider-specific)
- `LLM_MODEL_ID`: Model identifier
- `LLM_TIMEOUT`: Request timeout in seconds (default: 120)
- `LLM_MAX_RETRIES`: Retry count on transient failures (default: 2)
- `TEMPERATURE`: Sampling temperature (0.0-2.0, default: 0.7)
- `MAX_TOKENS`: Maximum output tokens (optional)

### Light Model (Subagents/Task)
- `LIGHT_LLM_PROVIDER`: Provider for lightweight model
- `LIGHT_LLM_API_KEY`: Light model API key
- `LIGHT_LLM_BASE_URL`: Light model endpoint
- `LIGHT_LLM_MODEL_ID`: Light model ID (e.g., deepseek-chat)

### Context Engineering
- `CONTEXT_WINDOW`: Token limit for context (default: 128000)
- `COMPRESSION_THRESHOLD`: Trigger compression at this usage ratio (default: 0.8, range: 0-1)
- `MIN_RETAIN_ROUNDS`: Minimum conversation rounds to keep (default: 10)
- `SUMMARY_TIMEOUT`: Timeout for LLM summarization (default: 120 seconds)

### Tool Output Truncation
- `TOOL_OUTPUT_TRUNCATE_DIRECTION`: Strategy (head | tail | head_tail, default: head_tail)
- `TOOL_OUTPUT_HEAD_TAIL_LINES`: Lines to keep at head/tail (default: 40)
- `TOOL_OUTPUT_MAX_LINES`: Maximum output lines (default: 2000)
- `TOOL_OUTPUT_MAX_BYTES`: Maximum output bytes (default: 51200)
- `TOOL_OUTPUT_DIR`: Spillover directory (default: tool-output/)
- `TOOL_OUTPUT_RETENTION_DAYS`: Cleanup retention period (default: 7)

### Skills & Subagents
- `SKILLS_REFRESH_ON_CALL`: Reload skills on each invocation (default: true)
- `SKILLS_PROMPT_CHAR_BUDGET`: Max characters for skill descriptions in prompts (default: 12000)
- `SUBAGENT_MAX_STEPS`: Maximum reasoning steps per Task subagent (default: 50)

### AgentTeams
- `ENABLE_AGENT_TEAMS`: Enable team features (default: false)
- `AGENT_TEAMS_STORE_DIR`: Team state storage directory (default: .teams)
- `AGENT_TASKS_STORE_DIR`: Task state storage directory (default: .tasks)
- `TEAMMATE_MODE`: Execution mode (auto | in-process | tmux, default: auto)
- `TEAM_DELEGATE_MODE`: Enable delegation patterns (default: false)
- `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`: Claude Code compatibility flag (same as ENABLE_AGENT_TEAMS)

### Trace & Debugging
- `TRACE_ENABLED`: Enable trace logging (default: true)
- `TRACE_DIR`: Trace output directory (default: memory/traces)
- `TRACE_SANITIZE`: Remove sensitive info from logs (default: true)
- `TRACE_HTML_INCLUDE_RAW_RESPONSE`: Include raw model responses in HTML logs (default: false)
- `DEBUG`: Enable debug mode (default: false)
- `LOG_LEVEL`: Logging verbosity (DEBUG | INFO | WARNING | ERROR, default: INFO)

### Tool Configuration
- `TAVILY_API_KEY`: Tavily Search API key (for MCP search tools)
- `CTX7_API_KEY`: Context7 MCP API key

---

## Tool Development Guidelines

### Unified Response Protocol

All tools **must** return JSON following the **Universal Tool Response Protocol** (see `docs/通用工具响应协议.md`):

```json
{
  "status": "success",
  "data": { /* core payload (object, never null) */ },
  "text": "Human-readable summary for LLM",
  "stats": {
    "time_ms": 123,
    /* tool-specific metrics */
  },
  "context": {
    "cwd": ".",
    "params_input": { /* original params */ },
    /* additional context */
  }
  /* error field only if status="error" */
}
```

**Status Rules**:
| Status | Condition |
|--------|-----------|
| `success` | Task fully completed, no truncation/fallback/error |
| `partial` | Result usable but discounted: truncation, fallback, dry-run, partial failure |
| `error` | No valid result: permission denied, invalid params, timeout with no output |

### Creating a New Tool

1. **Inherit from Tool base class** (`tools/base.py`):
   ```python
   from tools.base import Tool, ToolParameter, ToolStatus, ErrorCode

   class MyTool(Tool):
       def get_parameters(self) -> List[ToolParameter]:
           return [
               ToolParameter(name="input", type="string", required=True, description="..."),
           ]

       def run(self, parameters: Dict[str, Any]) -> str:
           # Implementation
           # Return JSON string following response protocol
           return self.create_success_response(
               data={"result": "..."},
               text="Summary",
               stats={"time_ms": 123},
               context={"cwd": self.get_cwd_rel(), "params_input": parameters}
           )
   ```

2. **Register tool in registry**:
   ```python
   from tools.registry import global_registry
   from tools.builtin.my_tool import MyTool

   tool = MyTool(name="MyTool", description="...", project_root=root, working_dir=cwd)
   global_registry.register_tool(tool)
   ```

3. **Add tool prompt** in `prompts/tools_prompts/my_tool_prompt.py`:
   - Clear usage examples
   - Common mistakes to avoid
   - Parameter guidance

4. **Add tests** in `tests/`:
   - Parameter validation
   - Response protocol compliance
   - Error handling
   - Edge cases

### Important Protocol Rules

- **No custom top-level fields**: Use only `status`, `data`, `text`, `stats`, `context`, `error`
- **Data is always an object**: Never `null`, even on error
- **Paths are consistent**: All relative paths should be relative to `project_root`
- **Sandbox safety**: Use `Path.relative_to(project_root)` to ensure paths stay within bounds
- **Timeout protection**: Set reasonable timeouts for external operations
- **Circuit breakers**: Use exponential backoff or skip failed tools temporarily

---

## Context Engineering Deep Dive

### Compression Workflow

1. **Tracking**: HistoryManager tracks rounds and token usage
2. **Threshold**: When `token_usage / context_window ≥ COMPRESSION_THRESHOLD`, compression is triggered
3. **Summarization**: SummaryCompressor creates a summary of older rounds (keeping `MIN_RETAIN_ROUNDS` recent)
4. **Replacement**: Old rounds are replaced with a single "summary" block
5. **Result**: More room for new interactions while retaining context

### Truncation Strategy

When tool output exceeds limits:
- **head**: Show first N lines
- **tail**: Show last N lines
- **head_tail**: Show first + last N lines each
- Excess content is written to `tool-output/<timestamp>_<tool>.txt`
- Reference included in truncation notice

### Trace Logging

- **JSONL**: Streaming JSON log (one event per line) in `memory/traces/<session>.jsonl`
- **HTML**: Human-readable report in `memory/traces/<session>.html`
- **Sanitization**: API keys, tokens, passwords removed by default
- **Retention**: Automatic cleanup after `TOOL_OUTPUT_RETENTION_DAYS` (7 days default)

---

## Common Development Tasks

### Running a Single Agent Query

```bash
python scripts/chat_test_agent.py
# Then type your message and press Enter
```

### Debugging Tool Failures

```bash
# Show raw LLM responses
python scripts/chat_test_agent.py --show-raw

# Check trace logs
cat memory/traces/<session>.jsonl | jq .

# View HTML report
open memory/traces/<session>.html
```

### Adding a New LLM Provider

1. Update `core/llm.py`:
   - Add provider to `SUPPORTED_PROVIDERS` literal
   - Add endpoint logic in `_resolve_config()`

2. Update `.env.example` with provider-specific variables

3. Test with `--provider <name>` flag

### Working with Skills

Create `skills/<skill-name>/SKILL.md`:
```markdown
---
name: my-skill
description: What this skill does
---

# My Skill

Instructions and context...

$ARGUMENTS
```

The `$ARGUMENTS` placeholder will be replaced with user arguments passed via the Skill tool.

### Enabling AgentTeams

```bash
export ENABLE_AGENT_TEAMS=true
python scripts/chat_test_agent.py
```

Then use: `TeamCreate`, `SendMessage`, `TeamStatus`, `TeamDelete` tools to manage teams.

---

## Code Style & Standards

**Python Version**: 3.8+

**Style Guide**:
- 4-space indentation (PEP 8)
- Type annotations for all functions
- Docstrings for public methods and classes
- Classes: PascalCase
- Functions/variables: snake_case
- Constants: UPPER_SNAKE_CASE

**Testing Standards**:
- Add unit tests for new tools
- Test both success and error paths
- Validate response protocol compliance
- Use pytest fixtures for common setup

**Linting** (if applicable):
- Follow PEP 8 via black or similar
- No unused imports
- Type checking with mypy (if enabled)

---

## Important Files & Concepts

| File/Dir | Purpose |
|----------|---------|
| `docs/通用工具响应协议.md` | Tool response protocol specification |
| `docs/上下文工程设计文档.md` | Context compression & truncation design |
| `docs/task(subagent)设计文档.md` | Subagent delegation system |
| `docs/TraceLogging设计文档.md` | Trace logging architecture |
| `prompts/agents_prompts/` | Agent system prompts |
| `prompts/tools_prompts/` | Tool descriptions & usage guidance |
| `core/config.py` | Configuration model (references all env vars) |
| `mcp_servers.json` | External MCP tool definitions |

---

## Known Limitations & Notes

- **AgentTeams**: Experimental feature; default disabled. Enable with caution in production.
- **Context Compression**: LLM-powered summarization may take 30-120 seconds depending on model and context size.
- **Tool Timeouts**: Most tools have 2-second default timeouts; adjust via environment if needed.
- **Sandbox**: All tools operate within `project_root`; absolute paths are converted to relative automatically.
- **MCP Integration**: Requires configured MCP servers in `mcp_servers.json`; not all tools will be available.

---

## References & Links

- **Main README**: [README.md](README.md) - Feature overview & quick start
- **Code Law**: [code_law.md](code_law.md) - Project principles & values
- **Development Handoff**: [docs/DEV_HANDOFF.md](docs/DEV_HANDOFF.md) - Detailed tool architecture & implementation guide
- **Tool Protocol**: [docs/通用工具响应协议.md](docs/通用工具响应协议.md) - Response format specification
- **Contribution Guide**: See README.md section "贡献指南"
- **Video Demo**: [Bilibili: MyCodeAgent Demonstration](https://www.bilibili.com/video/BV1vhkMBpEzq)

---

## Quick Reference: Tool Names (via Skill tool)

When calling the **Skill** tool in prompts, use these exact names:
- `read` - Read file contents
- `write` - Create/overwrite files
- `edit` - Single-point string replacement
- `bash` - Execute shell commands
- `glob` - Find files by pattern
- `grep` - Search code by regex
- `ls` - List directory contents
- `task` - Delegate to subagent
- `skill` - Load user-defined skill
- Custom skills in `skills/` with unique names

---

*Last updated: 2026-02-25*
