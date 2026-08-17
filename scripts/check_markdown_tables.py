#!/usr/bin/env python3
"""Fail when Markdown tables violate Kramdown block structure."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")
SEPARATOR_LIKE_CELL_RE = re.compile(r"^:?-+:?$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


@dataclass(frozen=True)
class Issue:
    path: Path
    line: int
    message: str


def is_table_row(line: str) -> bool:
    return TABLE_ROW_RE.fullmatch(line) is not None


def is_separator_row(line: str) -> bool:
    if not is_table_row(line):
        return False
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return len(cells) >= 2 and all(SEPARATOR_CELL_RE.fullmatch(cell) for cell in cells)


def is_separator_like_row(line: str) -> bool:
    if not is_table_row(line):
        return False
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return len(cells) >= 2 and all(SEPARATOR_LIKE_CELL_RE.fullmatch(cell) for cell in cells)


def find_issues(path: Path) -> list[Issue]:
    lines = path.read_text(encoding="utf-8").splitlines()
    issues: list[Issue] = []
    in_fence = False
    fence_marker = ""
    index = 0

    while index < len(lines) - 1:
        fence_match = FENCE_RE.match(lines[index])
        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker[0]
            elif marker[0] == fence_marker:
                in_fence = False
                fence_marker = ""
            index += 1
            continue
        if in_fence or not (is_table_row(lines[index]) and is_separator_like_row(lines[index + 1])):
            index += 1
            continue

        if not is_separator_row(lines[index + 1]):
            issues.append(Issue(path, index + 2, "table separator cells need at least 3 hyphens"))
            index += 2
            continue

        if index > 0 and lines[index - 1].strip():
            issues.append(Issue(path, index + 1, "missing blank line before table"))

        table_end = index + 2
        while table_end < len(lines) and is_table_row(lines[table_end]):
            table_end += 1
        if table_end == index + 2:
            issues.append(Issue(path, index + 1, "table requires at least one body row"))
        elif table_end < len(lines) and lines[table_end].lstrip().startswith("|"):
            issues.append(Issue(path, table_end + 1, "malformed table row; rows must end with |"))
        elif table_end < len(lines) and lines[table_end].strip():
            issues.append(Issue(path, table_end + 1, "missing blank line after table"))
        index = max(table_end, index + 1)

    return issues


def markdown_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(path.rglob("*.md"))
        elif path.suffix.lower() == ".md":
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("_tutorials"), Path("_posts")])
    args = parser.parse_args()

    files = list(markdown_files(args.paths))
    issues = [issue for path in files for issue in find_issues(path)]
    if issues:
        for issue in issues:
            print(f"{issue.path}:{issue.line}: {issue.message}")
        print(f"ERROR: {len(issues)} Markdown table structure issue(s) in {len(files)} file(s)")
        return 1

    print(f"Markdown tables OK: {len(files)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
