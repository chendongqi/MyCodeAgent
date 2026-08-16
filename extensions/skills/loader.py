"""Skill loader for project-local skills."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass
class SkillMeta:
    name: str
    description: str
    path: str
    base_dir: str
    mtime: float


class SkillLoader:
    """Scan and cache skills stored under project_root/skills.

    缓存策略：比较「所有 SKILL.md 的最大 mtime」+ 「文件数量」
    任意一个变化（文件增删、内容修改）→ 触发重新扫描。
    开销极小：只做 stat() 比对，不逐文件读取内容。
    """

    def __init__(self, project_root: str, skills_dir: str = "skills"):
        self._project_root = Path(project_root).resolve()
        self._skills_dir = (self._project_root / skills_dir).resolve()
        self._skills: Dict[str, SkillMeta] = {}     # 解析后的缓存：name → SkillMeta
        self._last_scan_mtime: float = 0.0           # 上次扫描时所有 SKILL.md 的最大 mtime
        self._last_scan_count: int = 0               # 上次扫描时的 SKILL.md 文件数量

    def scan(self) -> List[SkillMeta]:
        """Scan project-local skills and refresh the cache.

        遍历所有 SKILL.md，解析 frontmatter，更新 name→SkillMeta 缓存。
        同时记录最大 mtime 和文件数量，供 refresh_if_stale() 做增量检查。
        """
        files = self._iter_skill_files()   # rglob("SKILL.md") 的排序列表

        skills: Dict[str, SkillMeta] = {}
        max_mtime = 0.0
        count = 0

        for path in files:
            count += 1
            try:
                stat = path.stat()
                max_mtime = max(max_mtime, stat.st_mtime)   # 跟踪最新修改时间
            except OSError:
                continue

            parsed = self._parse_skill_file(path)
            if not parsed:
                continue

            meta = parsed
            if meta.name in skills:
                pass  # 同名 skill 后者覆盖前者（按路径排序，行为可预期）
            skills[meta.name] = meta

        self._skills = skills
        self._last_scan_mtime = max_mtime
        self._last_scan_count = count
        return self.list_skills(refresh=False)

    def refresh_if_stale(self) -> List[SkillMeta]:
        """Refresh cache if skill files changed.

        增量检查：只做 stat()，不读文件内容。
        mtime 变化（文件被修改）或 count 变化（文件增删）→ 触发全量 scan()。
        缓存未过期时直接返回内存中的列表，开销接近零。
        """
        if not self._skills:
            return self.scan()  # 首次调用：缓存为空，必须扫描

        current_max_mtime, current_count = self._get_skills_state()
        if current_max_mtime != self._last_scan_mtime or current_count != self._last_scan_count:
            return self.scan()  # 文件有变化 → 重新扫描
        return self.list_skills(refresh=False)  # 无变化 → 直接返回缓存

    def list_skills(self, refresh: bool = False) -> List[SkillMeta]:
        if refresh:
            self.refresh_if_stale()
        return sorted(self._skills.values(), key=lambda s: s.name)

    def get_skill(self, name: str, refresh: bool = False) -> Optional[SkillMeta]:
        if refresh:
            self.refresh_if_stale()
        return self._skills.get(name)

    def format_skills_for_prompt(self, char_budget: int) -> str:
        from extensions.skills.prompt import format_skills_for_prompt

        return format_skills_for_prompt(self.list_skills(refresh=False), char_budget)

    def _iter_skill_files(self) -> List[Path]:
        if not self._skills_dir.exists():
            return []
        return sorted(self._skills_dir.rglob("SKILL.md"))

    def _get_skills_state(self) -> Tuple[float, int]:
        max_mtime = 0.0
        count = 0
        for path in self._iter_skill_files():
            count += 1
            try:
                stat = path.stat()
                max_mtime = max(max_mtime, stat.st_mtime)
            except OSError:
                continue
        return max_mtime, count

    def _parse_skill_file(self, path: Path) -> Optional[SkillMeta]:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return None

        parsed = _parse_frontmatter(content)
        if not parsed:
            return None

        frontmatter, _body = parsed
        name = (frontmatter.get("name") or "").strip()
        description = (frontmatter.get("description") or "").strip()

        if not name or not description:
            return None
        if not _SKILL_NAME_PATTERN.match(name):
            return None

        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0

        try:
            base_dir = str(path.parent.relative_to(self._project_root)) or "."
        except ValueError:
            base_dir = str(path.parent)
        return SkillMeta(
            name=name,
            description=description,
            path=str(path),
            base_dir=base_dir,
            mtime=mtime,
        )


def _parse_frontmatter(content: str) -> Optional[Tuple[Dict[str, str], str]]:
    """解析 SKILL.md 的 YAML frontmatter。

    格式：
      ---
      name: code-review
      description: ...
      ---
      Skill 正文（body）...

    不依赖 PyYAML，只支持单层 key: value 结构。
    返回 (frontmatter_dict, body_str)，解析失败返回 None。
    """
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None  # 第一行必须是 ---

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i  # 找到结束 ---
            break

    if end_idx is None:
        return None  # 没有闭合的 ---，frontmatter 格式无效

    frontmatter_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1:])   # --- 之后的内容是 Skill 指令正文
    frontmatter: Dict[str, str] = {}

    for line in frontmatter_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            return None  # 无法解析的行 → 整个 frontmatter 视为无效
        key, value = stripped.split(":", 1)
        frontmatter[key.strip()] = value.strip().strip("\"'")  # 去掉引号包裹

    return frontmatter, body


__all__ = ["SkillLoader", "SkillMeta"]
