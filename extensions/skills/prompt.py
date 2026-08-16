"""Prompt formatting for project skills."""

from __future__ import annotations

from typing import Iterable

from extensions.skills.loader import SkillMeta


def format_skills_for_prompt(skills: Iterable[SkillMeta], char_budget: int) -> str:
    """把 SkillMeta 列表格式化为注入系统提示词的文本。

    输出格式（每行一条）：
      - code-review: 按照团队规范对指定文件做 code review
      - gen-commit-msg: 生成符合 Conventional Commits 格式的提交信息

    char_budget 控制总长度，防止 Skills 过多时撑爆上下文窗口。
    注意：只注入「名字+描述」摘要，完整 Skill 正文在模型调用 Skill tool 时才读取。
    """
    items = sorted(list(skills), key=lambda skill: skill.name)
    if not items:
        return "(none)"

    lines: list[str] = []
    used = 0
    for skill in items:
        line = f"- {skill.name}: {skill.description}"
        line_len = len(line) + 1  # +1 for newline
        if used + line_len > char_budget and lines:
            break  # 已有条目时超出预算 → 截断
        if used + line_len > char_budget and not lines:
            break  # 第一条就超预算 → 也截断（极端情况）
        lines.append(line)
        used += line_len

    return "\n".join(lines) if lines else "(none)"


__all__ = ["format_skills_for_prompt"]
