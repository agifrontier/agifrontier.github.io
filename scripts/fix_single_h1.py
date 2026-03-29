#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

FRONT_MATTER_RE = re.compile(r'^---\n.*?\n---\n?', re.DOTALL)
TITLE_RE = re.compile(r'^title:\s*["\']?(.*?)["\']?\s*$', re.IGNORECASE)
H1_RE = re.compile(r'^(#)\s+(.*)$')


def split_front_matter(text: str) -> tuple[str, str]:
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return "", text
    return match.group(0), text[match.end():]


def extract_title(front_matter: str) -> str | None:
    for line in front_matter.splitlines():
        m = TITLE_RE.match(line.strip())
        if m:
            return m.group(1).strip()
    return None


def normalize_heading_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip('"\'')
    return text.casefold()


def fix_markdown(text: str) -> tuple[str, dict[str, int | bool]]:
    front_matter, body = split_front_matter(text)
    title = extract_title(front_matter) or ""
    normalized_title = normalize_heading_text(title)

    lines = body.splitlines()
    new_lines: list[str] = []
    removed_title_h1 = False
    demoted_h1_count = 0
    seen_first_content = False

    for line in lines:
        stripped = line.strip()
        heading_match = H1_RE.match(line)

        if not seen_first_content and stripped == "":
            new_lines.append(line)
            continue

        if not seen_first_content:
            seen_first_content = True
            if heading_match:
                heading_text = normalize_heading_text(heading_match.group(2))
                if normalized_title and heading_text == normalized_title:
                    removed_title_h1 = True
                    continue

        if heading_match:
            new_lines.append(f"## {heading_match.group(2)}")
            demoted_h1_count += 1
        else:
            new_lines.append(line)

    new_body = "\n".join(new_lines)
    if body.endswith("\n"):
        new_body += "\n"

    return front_matter + new_body, {
        "removed_title_h1": removed_title_h1,
        "demoted_h1_count": demoted_h1_count,
        "changed": (front_matter + new_body) != text,
    }


def iter_markdown_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file() and path.suffix.lower() == '.md':
            yield path
        elif path.is_dir():
            for file in sorted(path.rglob('*.md')):
                yield file


def main() -> int:
    parser = argparse.ArgumentParser(description='Ensure markdown files produce only one logical H1.')
    parser.add_argument('paths', nargs='*', default=['_tutorials', '_posts'], help='Files or directories to process')
    parser.add_argument('--write', action='store_true', help='Write changes back to disk')
    args = parser.parse_args()

    changed_files = 0
    total_removed = 0
    total_demoted = 0

    for file_path in iter_markdown_files([Path(p) for p in args.paths]):
        original = file_path.read_text(encoding='utf-8')
        updated, stats = fix_markdown(original)
        if stats['changed']:
            changed_files += 1
            total_removed += int(stats['removed_title_h1'])
            total_demoted += int(stats['demoted_h1_count'])
            print(f"CHANGE {file_path} removed_title_h1={stats['removed_title_h1']} demoted_h1_count={stats['demoted_h1_count']}")
            if args.write:
                file_path.write_text(updated, encoding='utf-8')

    print(f"SUMMARY changed_files={changed_files} removed_title_h1={total_removed} demoted_h1_count={total_demoted} write={args.write}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
