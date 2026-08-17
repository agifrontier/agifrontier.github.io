#!/usr/bin/env python3
"""Generate intrinsic dimensions for local article images used by Jekyll."""

from __future__ import annotations

import json
import re
import struct
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


HTML_IMAGE_RE = re.compile(
    r"<img\b[^>]*?\bsrc\s*=\s*(['\"])(?P<src>.*?)\1",
    re.IGNORECASE | re.DOTALL,
)
MARKDOWN_IMAGE_RE = re.compile(
    r"!\[[^\n]*?\]\(\s*<?(?P<src>[^\s)>]+)>?(?:\s+['\"][^'\"]*['\"])?\s*\)",
    re.IGNORECASE,
)


def image_references(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    references: set[str] = set()
    for pattern in (HTML_IMAGE_RE, MARKDOWN_IMAGE_RE):
        references.update(match.group("src").strip() for match in pattern.finditer(text))
    return references


def resolve_local_image(root: Path, source_path: Path, reference: str) -> tuple[str, Path] | None:
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or reference.startswith("//") or "{{" in reference:
        return None
    web_path = unquote(parsed.path)
    if not web_path:
        return None
    candidate = root / web_path.lstrip("/") if web_path.startswith("/") else source_path.parent / web_path
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError(f"image path escapes repository: {source_path}: {reference}")
    return web_path, candidate


def jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    if not data.startswith(b"\xff\xd8"):
        return None
    offset = 2
    while offset + 9 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        offset += 2
        if marker in {0xD8, 0xD9}:
            continue
        if offset + 2 > len(data):
            break
        segment_length = struct.unpack(">H", data[offset : offset + 2])[0]
        if segment_length < 2 or offset + segment_length > len(data):
            break
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            height, width = struct.unpack(">HH", data[offset + 3 : offset + 7])
            return width, height
        offset += segment_length
    return None


def intrinsic_dimensions(path: Path) -> tuple[int, int] | None:
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data[:6] in {b"GIF87a", b"GIF89a"} and len(data) >= 10:
        return struct.unpack("<HH", data[6:10])
    jpeg_size = jpeg_dimensions(data)
    if jpeg_size:
        return jpeg_size

    result = subprocess.run(
        ["identify", "-format", "%w %h", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        width, height = result.stdout.strip().split()
        return int(width), int(height)
    except ValueError:
        return None


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    tutorial_sources = sorted((root / "_tutorials").glob("*"))
    post_sources = [
        path
        for path in sorted((root / "_posts").glob("*"))
        if re.match(r"^\d{4}-\d{2}-\d{2}-.+", path.name)
    ]
    sources = tutorial_sources + post_sources
    references: dict[str, Path] = {}
    errors: list[str] = []

    for source_path in sources:
        if not source_path.is_file() or source_path.suffix.lower() not in {".md", ".html"}:
            continue
        try:
            found_references = image_references(source_path)
        except (OSError, UnicodeError) as exc:
            errors.append(f"{source_path}: cannot read source: {exc}")
            continue
        for reference in found_references:
            try:
                resolved = resolve_local_image(root, source_path, reference)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if resolved is None:
                continue
            web_path, candidate = resolved
            if any(part.startswith("_") for part in Path(web_path).parts):
                errors.append(f"Jekyll-excluded image path: {source_path}: {web_path}")
                continue
            if not candidate.is_file():
                errors.append(f"missing local image: {source_path}: {web_path}")
                continue
            previous = references.setdefault(web_path, candidate)
            if previous != candidate:
                errors.append(f"ambiguous image path {web_path}: {previous} vs {candidate}")

    metadata: dict[str, dict[str, int]] = {}
    for web_path, image_path in sorted(references.items()):
        dimensions = intrinsic_dimensions(image_path)
        if dimensions is None:
            errors.append(f"cannot read image dimensions: {image_path}")
            continue
        width, height = dimensions
        metadata[web_path] = {"width": width, "height": height}

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"Image metadata generation failed: {len(errors)} error(s)", file=sys.stderr)
        return 1

    output = root / "_data" / "image_metadata.json"
    temporary = output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Image metadata OK: {len(metadata)} local image(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
