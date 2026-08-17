# Code Agent 系列 06：agent 的能力怎么用 Skills 动态扩展？

> 这是 MyCodeAgent 源码阅读系列的第六篇。上一篇我们完整走过了工具系统的注册→schema 生成→编排→执行→结果协议流程。本篇聚焦 **Skills 动态扩展机制**：一个不需要写 Python 代码、只需要放一个 Markdown 文件就能让 agent 获得新能力的系统。

---

## 从一个问题出发

内置工具（Bash、Read、Edit 等）在 agent 启动时就注册好了，能力是固定的。但如果想让 agent 执行一套特定的流程——比如「按照我们团队的规范做 code review」、「按照特定模板生成 commit message」——你有几个选择：

1. 写一个新的 Python Tool 类，注册进去
2. 把流程写进系统提示词里
3. **创建一个 SKILL.md 文件**

第三种方式就是 Skills 机制：用 Markdown 文件描述一套指令，agent 按需加载，不需要重启。本篇从 Skill 文件的创建开始，沿着 **定义→扫描→注入→调用→执行** 的生命周期完整走一遍。

---

## Step 1：定义一个 Skill（SKILL.md 格式）

Skills 的入口是项目根目录下的 `skills/` 文件夹。每个子目录放一个 `SKILL.md` 文件，格式是：

```markdown
---
name: code-review
description: 按照团队规范对指定文件做 code review
---

请对以下代码做 code review，关注：
1. 逻辑正确性
2. 边界条件处理
3. 命名规范

$ARGUMENTS
```

**格式规则：**
- 开头是 YAML frontmatter（`---` 包裹），必须有 `name` 和 `description` 两个字段
- `name` 只允许小写字母、数字、连字符（正则：`^[a-z0-9]+(?:-[a-z0-9]+)*$`）
- `---` 之后的内容是 Skill 正文（指令文本）
- `$ARGUMENTS` 是一个特殊占位符——模型调用这个 Skill 时传入的 `args` 参数会替换这个位置

这个文件本身就是完整的 Skill 定义，没有任何 Python 代码需要编写。

---

## Step 2：启动时扫描——SkillLoader 建立缓存

agent 启动时，`factory.py` 负责组装所有运行时组件，Skills 部分的逻辑在这里：

```python
# runtime/factory.py:27-33
host._skills_prompt = ""
if host.enable_skills and _project_has_skill_files(host.project_root):
    from extensions.skills import SkillLoader

    host._skill_loader = SkillLoader(host.project_root)
else:
    host._skill_loader = None
host._refresh_skills_prompt()
```

`_project_has_skill_files()` 做了一个快速检查——`skills/` 目录是否存在且有 SKILL.md 文件：

```python
# runtime/factory.py:103-105
def _project_has_skill_files(project_root: str) -> bool:
    skills_dir = Path(project_root) / "skills"
    return skills_dir.is_dir() and next(skills_dir.rglob("SKILL.md"), None) is not None
```

如果检查通过，`SkillLoader` 就被创建出来。它维护两个关键状态：

```python
# extensions/skills/loader.py:29-31
self._skills: Dict[str, SkillMeta] = {}        # 已解析的技能缓存（name → SkillMeta）
self._last_scan_mtime: float = 0.0              # 上次扫描时各文件的最大 mtime
self._last_scan_count: int = 0                  # 上次扫描时的文件数量
```

**`scan()` 扫描逻辑：**

```python
# extensions/skills/loader.py:33-61
def scan(self) -> List[SkillMeta]:
    files = self._iter_skill_files()   # rglob("SKILL.md") 遍历所有文件
    skills: Dict[str, SkillMeta] = {}
    max_mtime = 0.0

    for path in files:
        stat = path.stat()
        max_mtime = max(max_mtime, stat.st_mtime)   # 记录最新的 mtime
        parsed = self._parse_skill_file(path)
        if parsed:
            skills[parsed.name] = parsed

    self._skills = skills
    self._last_scan_mtime = max_mtime
    self._last_scan_count = len(files)
    return self.list_skills(refresh=False)
```

每个 SKILL.md 解析成一个 `SkillMeta` 对象：

```python
# extensions/skills/loader.py:14-20（SkillMeta dataclass）
@dataclass
class SkillMeta:
    name: str         # frontmatter 里的 name
    description: str  # frontmatter 里的 description（用于显示在 Skill tool prompt 里）
    path: str         # SKILL.md 的绝对路径
    base_dir: str     # 相对于 project_root 的目录（作为执行上下文）
    mtime: float      # 文件的 mtime（用于缓存失效检查）
```

### 增量刷新：`refresh_if_stale()`

Skills 支持热更新——不重启 agent，新增/修改 SKILL.md 后可以被自动感知。判断缓存是否过期的方法是比较文件状态：

```python
# extensions/skills/loader.py:63-71
def refresh_if_stale(self) -> List[SkillMeta]:
    if not self._skills:
        return self.scan()

    current_max_mtime, current_count = self._get_skills_state()
    # 任意文件的 mtime 变了，或者文件数量变了（新增/删除）→ 重新扫描
    if current_max_mtime != self._last_scan_mtime or current_count != self._last_scan_count:
        return self.scan()
    return self.list_skills(refresh=False)
```

**设计取舍**：这个缓存策略比较轻量，只看「最大 mtime」和「文件数量」，不逐文件 diff。好处是开销极小；代价是极少数情况下（两个文件同时改，其中一个 mtime 和原来相同）会漏刷——对 Skills 这种场景完全可以接受。

---

## Step 3：解析 frontmatter——`_parse_frontmatter()`

SkillLoader 不依赖 PyYAML，用了一个自己实现的简单解析器：

```python
# extensions/skills/loader.py:142-168
def _parse_frontmatter(content: str) -> Optional[Tuple[Dict[str, str], str]]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        return None

    frontmatter_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1:])   # --- 之后的部分是 Skill 正文

    frontmatter: Dict[str, str] = {}
    for line in frontmatter_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            return None
        key, value = stripped.split(":", 1)
        frontmatter[key.strip()] = value.strip().strip("\"'")

    return frontmatter, body
```

逻辑很直白：找到第一对 `---` 包裹的区域，逐行 `key: value` 解析，剩余内容作为 body 返回。不支持 YAML 的嵌套结构，只支持单层 key-value——对 Skill 场景已经够用了。

---

## Step 4：生成 Skill 列表文本注入系统提示词

回到 `factory.py`，`SkillLoader` 创建完成后立即调用 `_refresh_skills_prompt()`：

```python
# runtime/host.py:191-202
def _refresh_skills_prompt(self) -> None:
    if not self.enable_skills or self._skill_loader is None:
        self._skills_prompt = ""
        return
    refresh = self.config.skills_refresh_on_call
    if refresh:
        self._skill_loader.refresh_if_stale()
    elif not self._skills_prompt:
        self._skill_loader.scan()                   # 冷启动时强制扫描一次
    budget = int(os.getenv("SKILLS_PROMPT_CHAR_BUDGET", "12000"))
    from extensions.skills.prompt import format_skills_for_prompt
    self._skills_prompt = format_skills_for_prompt(
        self._skill_loader.list_skills(refresh=False), budget
    )
```

`format_skills_for_prompt()` 把 SkillMeta 列表转成一段文本，受 `char_budget` 约束：

```python
# extensions/skills/prompt.py:10-27
def format_skills_for_prompt(skills: Iterable[SkillMeta], char_budget: int) -> str:
    items = sorted(list(skills), key=lambda skill: skill.name)
    if not items:
        return "(none)"

    lines: list[str] = []
    used = 0
    for skill in items:
        line = f"- {skill.name}: {skill.description}"
        line_len = len(line) + 1
        if used + line_len > char_budget and lines:
            break   # 超出预算，截断
        lines.append(line)
        used += line_len

    return "\n".join(lines) if lines else "(none)"
```

生成的文本格式是这样的：

```
- code-review: 按照团队规范对指定文件做 code review
- gen-commit-msg: 生成符合 Conventional Commits 格式的提交信息
```

这段文本最终会填进哪里？答案是 Skill tool 的 prompt 里有一个 `{{available_skills}}` 插槽。

---

## Step 5：`{{available_skills}}` 插槽注入

`prompt_builder.py` 的 `_load_tool_prompts()` 方法负责把所有工具的 prompt 文件拼成系统提示词的 Tool Contracts 层，其中有一段关键逻辑：

```python
# runtime/prompt_builder.py:254-256
if self._skills_prompt and "{{available_skills}}" in prompt_value:
    prompt_value = prompt_value.replace("{{available_skills}}", self._skills_prompt)
prompts.append(prompt_value)
```

也就是说，`prompts/tools_prompts/skill_prompt.py` 里的 prompt 文本里包含 `{{available_skills}}`，在组装系统提示词时被替换成实际的 Skill 列表文本。

`skill_prompt.py` 里的定义是：

```
Available Skills
{{available_skills}}
```

替换后，模型在系统提示词里看到的是：

```
Available Skills
- code-review: 按照团队规范对指定文件做 code review
- gen-commit-msg: 生成符合 Conventional Commits 格式的提交信息
```

当 Skills 列表变化时（比如新增了一个 SKILL.md），`set_skills_prompt()` 会清空 system prompt 缓存，强制下一步重新组装：

```python
# runtime/prompt_builder.py:196-202
def set_skills_prompt(self, prompt: str) -> None:
    normalized = prompt or ""
    if normalized == self._skills_prompt:
        return
    self._skills_prompt = normalized
    self._cached_assembly = None   # 清空缓存，下次 get_prompt_assembly() 重新构建
```

**两层缓存**：系统提示词有 fingerprint 缓存（稳定层不变就复用），`_skills_prompt` 变化时触发 fingerprint 变化，从而让缓存失效。这样保证了每次 `get_prompt_assembly()` 拿到的都是最新的 Skills 列表，同时避免了每轮对话都重扫文件系统。

---

## Step 6：模型调用 Skill tool——SkillTool 执行

当模型在系统提示词里看到了可用 Skill 列表，决定调用某个 Skill 时，它发出一个 function call：

```json
{
  "name": "Skill",
  "arguments": {
    "name": "code-review",
    "args": "src/main.py"
  }
}
```

这个调用经过 ToolOrchestrator → ToolExecutor 链路，最终落到 `SkillTool.run()`。

`SkillTool` 在 `_register_builtin_tools()` 里被有条件地注册——只有 `_skill_loader is not None` 时才注册：

```python
# runtime/host.py:180-188
if self._skill_loader is not None:
    from tools.builtin.skill import SkillTool
    self.tool_registry.register_tool(
        SkillTool(
            project_root=self.project_root,
            skill_loader=self._skill_loader,
            refresh_on_call=self.config.skills_refresh_on_call,  # 是否每次调用都刷新缓存
        )
    )
```

`SkillTool.run()` 的执行步骤：

```python
# tools/builtin/skill.py:56-129（精简版）
def run(self, parameters: Dict[str, Any]) -> ToolResult:
    name = parameters.get("name")
    args = parameters.get("args") or ""

    # ① 查缓存，找不到则强制 refresh 一次
    skill_meta = self._skill_loader.get_skill(name.strip(), refresh=False)
    if not skill_meta:
        skill_meta = self._skill_loader.get_skill(name.strip(), refresh=True)
    if not skill_meta:
        return self.error_result(...)   # 找不到报 NOT_FOUND

    # ② 读取 SKILL.md 文件内容（实时读取，获得最新版本）
    raw_content = skill_path.read_text(encoding="utf-8")

    # ③ 解析 frontmatter，提取 body
    _frontmatter, body = _parse_frontmatter(raw_content)

    # ④ 把 args 填入 $ARGUMENTS 占位符
    expanded = _apply_arguments(body, args)

    # ⑤ 加上 base_dir 前缀，返回展开后的 Skill 内容
    content = f"Base directory for this skill: {base_dir}\n\n{expanded}".strip()
    return self.success_result(data={"name", "base_dir", "content": content}, ...)
```

**`$ARGUMENTS` 替换逻辑：**

```python
# tools/builtin/skill.py:132-138
def _apply_arguments(body: str, args: str) -> str:
    trimmed_args = args.strip()
    if "$ARGUMENTS" in body:
        return body.replace("$ARGUMENTS", trimmed_args)  # 有占位符 → 替换到指定位置
    if trimmed_args:
        return f"{body}\n\nARGUMENTS: {trimmed_args}"    # 无占位符 → 追加到末尾
    return body                                           # 没有 args → 原样返回
```

最终，`SkillTool.run()` 返回的是展开后的 Skill 内容文本（字符串），这段文本就是 Skill 的指令正文（替换了 `$ARGUMENTS` 之后的版本）。模型拿到这段文本后，按照其中的指令继续执行。

---

## 完整生命周期回顾

```
skills/code-review/SKILL.md    ← 你在这里写指令
        │
        ▼
[启动时] SkillLoader.scan()     ← 解析 frontmatter，建立 name→SkillMeta 缓存
        │
        ▼
[启动时] format_skills_for_prompt()  ← 生成 "- code-review: ..." 文本
        │
        ▼
[每轮对话] ContextBuilder._load_tool_prompts()
        → skill_prompt 里的 {{available_skills}} 被替换
        → 注入系统提示词的 Tool Contracts 层
        │
        ▼
[模型决策] LLM 看到可用 Skill 列表，发出 function call: Skill(name="code-review", args="src/main.py")
        │
        ▼
[执行] SkillTool.run()
        → get_skill() 查缓存
        → read_text() 读文件
        → _apply_arguments() 替换 $ARGUMENTS
        → 返回展开后的指令文本
        │
        ▼
[模型接收] 拿到 Skill 内容，按指令继续执行
```

---

## 设计亮点

**1. 零代码扩展**
新增能力只需要放一个 Markdown 文件，不需要写任何 Python 代码，不需要重启 agent。

**2. 按需加载**
Skill 内容（完整 SKILL.md body）只在模型主动调用时才读取文件，系统提示词里只放「名字+描述」摘要，避免把所有 Skill 正文全部塞进每轮对话的上下文。

**3. 热更新**
`refresh_if_stale()` 通过 mtime+count 检查，不重启就能感知文件变化。`SKILLS_REFRESH_ON_CALL=true` 时每次 Skill tool 调用都触发一次检查。

**4. 字符预算**
`SKILLS_PROMPT_CHAR_BUDGET`（默认 12000）限制了注入系统提示词的 Skill 描述总长度，防止 Skills 过多时撑爆上下文窗口。

**5. 条件注册**
`skills/` 目录不存在时，`SkillTool` 根本不会被注册进工具表，模型的 schema 里就没有这个工具，避免「工具存在但没有可用 Skill」的无效调用。

---

## 关键源码位置

| 文件 | 作用 |
|------|------|
| `extensions/skills/loader.py` | SkillLoader：扫描、缓存、刷新、解析 frontmatter |
| `extensions/skills/prompt.py` | `format_skills_for_prompt()`：把 SkillMeta 列表格式化成 prompt 文本 |
| `tools/builtin/skill.py` | SkillTool：处理模型的 Skill 调用，读文件，替换 $ARGUMENTS |
| `runtime/prompt_builder.py:254` | `{{available_skills}}` 插槽替换逻辑 |
| `runtime/factory.py:27-33` | 启动时创建 SkillLoader，触发首次扫描 |
| `runtime/host.py:173-202` | `_register_builtin_tools()` 条件注册 SkillTool，`_refresh_skills_prompt()` 更新 prompt |

---

*下一篇：外部工具怎么接进来？MCP 集成是怎么做的？*
