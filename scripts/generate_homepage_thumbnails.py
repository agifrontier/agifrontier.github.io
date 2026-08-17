#!/usr/bin/env python3
"""Generate lightweight homepage cover images for tutorial cards."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit


THUMBNAIL_SIZES = (320, 640)
THUMBNAIL_HEIGHTS = {320: 180, 640: 360}
BACKGROUND_COLOR = "#f4f6f8"
HTML_IMAGE_RE = re.compile(
    r"<img\b[^>]*?\bsrc\s*=\s*(['\"])(?P<src>.*?)\1",
    re.IGNORECASE | re.DOTALL,
)
MARKDOWN_IMAGE_RE = re.compile(
    r"!\[[^\n]*?\]\(\s*<?(?P<src>[^\s)>]+)>?(?:\s+['\"][^'\"]*['\"])?\s*\)",
    re.IGNORECASE,
)
FRONT_MATTER_VALUE_RE = re.compile(
    r"^(?P<key>thumbnail|slug):\s*(?P<value>.*?)\s*$", re.MULTILINE
)


@dataclass(frozen=True)
class TutorialImage:
    slug: str
    source_reference: str | None
    source_path: Path | None
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Jekyll repository root (defaults to the script's parent repository)",
    )
    return parser.parse_args()


def split_front_matter(text: str) -> tuple[str, str]:
    if not text.startswith("---"):
        return "", text
    lines = text.splitlines(keepends=True)
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "".join(lines[1:index]), "".join(lines[index + 1 :])
    return "", text


def unquote_yaml_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def front_matter_values(front_matter: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for match in FRONT_MATTER_VALUE_RE.finditer(front_matter):
        value = unquote_yaml_scalar(match.group("value"))
        if value:
            values[match.group("key")] = value
    return values


def first_body_image(body: str) -> str | None:
    matches: list[tuple[int, str]] = []
    for pattern in (HTML_IMAGE_RE, MARKDOWN_IMAGE_RE):
        match = pattern.search(body)
        if match:
            matches.append((match.start(), match.group("src").strip()))
    if not matches:
        return None
    return min(matches, key=lambda item: item[0])[1]


def resolve_local_image(
    root: Path, tutorial_path: Path, source_reference: str
) -> tuple[Path | None, str]:
    parsed = urlsplit(source_reference)
    if parsed.scheme or parsed.netloc or source_reference.startswith("//"):
        return None, "external_image"

    decoded_path = unquote(parsed.path)
    if not decoded_path:
        return None, "missing_file"

    if decoded_path.startswith("/"):
        candidate = root / decoded_path.lstrip("/")
    else:
        candidate = tutorial_path.parent / decoded_path

    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None, "path_outside_repository"

    if not candidate.is_file():
        return None, "missing_file"
    return candidate, "ready"


def inspect_tutorial(root: Path, tutorial_path: Path) -> TutorialImage:
    text = tutorial_path.read_text(encoding="utf-8")
    front_matter, body = split_front_matter(text)
    values = front_matter_values(front_matter)
    slug = values.get("slug", tutorial_path.stem)
    source_reference = values.get("thumbnail") or first_body_image(body)
    if not source_reference:
        return TutorialImage(slug, None, None, "no_image")

    source_path, status = resolve_local_image(root, tutorial_path, source_reference)
    return TutorialImage(slug, source_reference, source_path, status)


def safe_output_stem(slug: str) -> str:
    # Jekyll excludes files whose basename starts with an underscore. Strip all
    # special leading characters after transliterating unsupported characters.
    safe_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", slug).strip("._-") or "tutorial"
    digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:10]
    return f"{safe_slug[:120]}-{digest}"


def image_dimensions(image_path: Path) -> tuple[int, int] | None:
    result = subprocess.run(
        ["identify", "-format", "%w %h", str(image_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        width, height = result.stdout.strip().split()
        return int(width), int(height)
    except (TypeError, ValueError):
        return None


def generate_thumbnail(source: Path, destination: Path, width: int, height: int) -> bool:
    if (
        destination.is_file()
        and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns
        and image_dimensions(destination) == (width, height)
    ):
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.webp")
    command = [
        "convert",
        str(source),
        "-auto-orient",
        "-thumbnail",
        f"{width}x{height}>",
        "-background",
        BACKGROUND_COLOR,
        "-gravity",
        "center",
        "-extent",
        f"{width}x{height}",
        "-alpha",
        "remove",
        "-alpha",
        "off",
        "-strip",
        "-quality",
        "82",
        "-define",
        "webp:method=6",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        if image_dimensions(temporary) != (width, height):
            raise RuntimeError(f"unexpected output dimensions for {temporary}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return True


def relative_web_path(root: Path, path: Path) -> str:
    return "/" + path.relative_to(root).as_posix()


def write_mapping(path: Path, mapping: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    tutorials_dir = root / "_tutorials"
    output_dir = root / "assets" / "img" / "homepage-thumbnails"
    mapping_path = root / "_data" / "homepage_thumbnails.json"

    for executable in ("convert", "identify"):
        if shutil.which(executable) is None:
            print(f"ERROR: required ImageMagick command not found: {executable}", file=sys.stderr)
            return 2
    if not tutorials_dir.is_dir():
        print(f"ERROR: tutorials directory not found: {tutorials_dir}", file=sys.stderr)
        return 2

    tutorial_paths = sorted(tutorials_dir.glob("*.md"))
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, dict[str, object]] = {}
    expected_outputs: set[Path] = set()
    status_counts: dict[str, int] = {}
    generated_count = 0
    original_bytes = 0
    thumbnail_bytes = 0
    errors: list[str] = []

    for tutorial_path in tutorial_paths:
        try:
            tutorial = inspect_tutorial(root, tutorial_path)
        except (OSError, UnicodeError) as exc:
            errors.append(f"{tutorial_path.name}: cannot read tutorial: {exc}")
            continue

        status_counts[tutorial.status] = status_counts.get(tutorial.status, 0) + 1
        if tutorial.status != "ready" or tutorial.source_path is None:
            reference = tutorial.source_reference or "(none)"
            print(f"WARN [{tutorial.status}] {tutorial_path.name}: {reference}")
            continue
        if tutorial.slug in mapping:
            errors.append(f"{tutorial_path.name}: duplicate tutorial slug: {tutorial.slug}")
            continue

        output_stem = safe_output_stem(tutorial.slug)
        source_bytes = tutorial.source_path.stat().st_size
        original_bytes += source_bytes
        outputs: dict[int, Path] = {
            width: output_dir / f"{output_stem}-{width}.webp"
            for width in THUMBNAIL_SIZES
        }

        try:
            for width, destination in outputs.items():
                expected_outputs.add(destination.resolve())
                if generate_thumbnail(
                    tutorial.source_path,
                    destination,
                    width,
                    THUMBNAIL_HEIGHTS[width],
                ):
                    generated_count += 1
                thumbnail_bytes += destination.stat().st_size
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            errors.append(
                f"{tutorial_path.name}: thumbnail conversion failed for "
                f"{tutorial.source_reference}: {exc}"
            )
            for destination in outputs.values():
                destination.unlink(missing_ok=True)
                expected_outputs.discard(destination.resolve())
            continue

        mapping[tutorial.slug] = {
            "source": tutorial.source_reference,
            "small": relative_web_path(root, outputs[320]),
            "large": relative_web_path(root, outputs[640]),
            "width": 640,
            "height": 360,
        }

    removed_count = 0
    for stale_path in output_dir.glob("*.webp"):
        if stale_path.resolve() not in expected_outputs:
            stale_path.unlink()
            removed_count += 1

    write_mapping(mapping_path, mapping)

    print("Homepage thumbnail generation summary")
    print(f"  tutorials: {len(tutorial_paths)}")
    print(f"  mapped: {len(mapping)}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print(f"  files generated or refreshed: {generated_count}")
    print(f"  stale files removed: {removed_count}")
    print(f"  original cover bytes: {original_bytes}")
    print(f"  thumbnail bytes (320 + 640): {thumbnail_bytes}")
    if original_bytes:
        reduction = 100 * (1 - thumbnail_bytes / original_bytes)
        print(f"  byte reduction: {reduction:.1f}%")
    print(f"  mapping: {mapping_path}")

    if errors:
        print(f"ERROR: {len(errors)} tutorial(s) failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
