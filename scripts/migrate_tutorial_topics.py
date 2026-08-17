#!/usr/bin/env python3
"""Migrate tutorial front matter to the canonical topic taxonomy."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from tutorial_topic_taxonomy import TOPIC_NAMES, migrate_legacy_topics


FIELD_RE = re.compile(r"^(?P<key>[a-zA-Z0-9_]+):\s*(?P<value>.*?)\s*$")
LIST_ITEM_RE = re.compile(r"^\s+-\s+(?P<value>.*?)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Jekyll repository root",
    )
    parser.add_argument("--write", action="store_true", help="Apply the migration")
    parser.add_argument("--report", type=Path, help="Write the full JSON audit report")
    return parser.parse_args()


def split_document(text: str) -> tuple[list[str], str]:
    if not text.startswith("---\n"):
        raise ValueError("missing front matter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("unterminated front matter")
    return text[4:end].splitlines(), text[end + 5 :]


def yaml_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        try:
            return json.loads(value) if value[0] == '"' else value[1:-1]
        except json.JSONDecodeError:
            return value[1:-1]
    return value


def scalar_field(lines: list[str], key: str) -> str:
    for line in lines:
        match = FIELD_RE.match(line)
        if match and match.group("key") == key:
            return yaml_scalar(match.group("value"))
    return ""


def list_field(lines: list[str], key: str) -> tuple[str, ...]:
    values: list[str] = []
    for index, line in enumerate(lines):
        match = FIELD_RE.match(line)
        if not match or match.group("key") != key:
            continue
        for child in lines[index + 1 :]:
            item = LIST_ITEM_RE.match(child)
            if item:
                values.append(yaml_scalar(item.group("value")))
                continue
            if child.startswith((" ", "\t")) or not child.strip():
                continue
            break
        return tuple(values)
    return ()


def replace_list_field(lines: list[str], key: str, values: tuple[str, ...]) -> list[str]:
    for index, line in enumerate(lines):
        match = FIELD_RE.match(line)
        if not match or match.group("key") != key:
            continue
        end = index + 1
        while end < len(lines) and (lines[end].startswith((" ", "\t")) or not lines[end].strip()):
            end += 1
        replacement = [f"{key}:", *[f"  - {json.dumps(value, ensure_ascii=False)}" for value in values]]
        return [*lines[:index], *replacement, *lines[end:]]
    raise ValueError(f"missing {key} field")


def transformed_document(path: Path, text: str) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    lines, body = split_document(text)
    existing_topics = list_field(lines, "topics")
    if not existing_topics:
        raise ValueError("missing topics")
    title = scalar_field(lines, "title")
    description = scalar_field(lines, "description")
    tags = list_field(lines, "tags")
    evidence = " ".join((path.stem, title, description, *tags)).casefold().replace("-", " ")
    migrated_topics = migrate_legacy_topics(existing_topics, evidence)
    migrated_lines = replace_list_field(lines, "topics", migrated_topics)
    migrated = "---\n" + "\n".join(migrated_lines) + "\n---\n" + body
    return migrated, existing_topics, migrated_topics


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    tutorials_dir = root / "_tutorials"
    changes: list[dict[str, object]] = []
    pending_writes: list[tuple[Path, str]] = []
    topic_counts: Counter[str] = Counter()
    errors: list[str] = []

    for path in sorted(tutorials_dir.glob("*.md")):
        try:
            current = path.read_text(encoding="utf-8")
            migrated, before, after = transformed_document(path, current)
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        topic_counts.update(after)
        if migrated == current:
            continue
        changes.append(
            {
                "path": path.relative_to(root).as_posix(),
                "before": list(before),
                "after": list(after),
            }
        )
        pending_writes.append((path, migrated))

    missing_topics = [topic for topic in TOPIC_NAMES if topic_counts[topic] == 0]
    report = {
        "tutorials": len(list(tutorials_dir.glob("*.md"))),
        "changed_files": len(changes),
        "topic_counts": {topic: topic_counts[topic] for topic in TOPIC_NAMES},
        "missing_topics": missing_topics,
        "errors": errors,
        "changes": changes,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({key: value for key, value in report.items() if key != "changes"}, ensure_ascii=False, indent=2))

    if errors or missing_topics:
        return 2
    if args.write:
        for path, migrated in pending_writes:
            path.write_text(migrated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
