#!/usr/bin/env python3
"""Fail when Markdown code fences are malformed or left unclosed."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FENCE_RE = re.compile(r"^[ \t]*(?P<marker>`{3,}|~{3,})(?P<rest>.*)$")
MALFORMED_PREFIXES = ("``$$", "$$``", "$$`$$")


@dataclass(frozen=True)
class Issue:
    path: Path
    line: int
    message: str


def find_issues(path: Path) -> list[Issue]:
    lines = path.read_text(encoding="utf-8").splitlines()
    issues: list[Issue] = []
    opening_line = 0
    opening_character = ""
    opening_length = 0

    for line_number, line in enumerate(lines, start=1):
        if any(prefix in line for prefix in MALFORMED_PREFIXES):
            issues.append(
                Issue(
                    path,
                    line_number,
                    "malformed code fence; use ```language to open and ``` to close",
                )
            )

        match = FENCE_RE.match(line)
        if not match:
            continue

        marker = match.group("marker")
        rest = match.group("rest")
        if not opening_character:
            opening_line = line_number
            opening_character = marker[0]
            opening_length = len(marker)
        elif marker[0] == opening_character and len(marker) >= opening_length and not rest.strip():
            opening_line = 0
            opening_character = ""
            opening_length = 0

    if opening_character:
        issues.append(Issue(path, opening_line, "code fence is not closed"))

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
        print(f"ERROR: {len(issues)} Markdown fence issue(s) in {len(files)} file(s)")
        return 1

    print(f"Markdown fences OK: {len(files)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
