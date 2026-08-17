#!/usr/bin/env python3
"""Validate rendered Jekyll pages for SEO and content rendering regressions."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse


SITE_ORIGIN = "https://agifrontier.github.io"
USER_PREFIXES = ("tutorials/", "page/", "topics/", "blog/")
SKIP_TEXT_TAGS = {"pre", "code", "script", "style"}
VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
WEAK_ALT_TEXT = {
    "",
    "refer to caption",
    "refer tocaption",
    "[uncaptioned image]",
    "[无标题图片]",
    "插图",
    "img",
}
ALLOWED_DEVANAGARI_PAGES = {
    "tutorials/rethinking-cross-lingual-gaps-from-a-statistical-viewpoint/index.html"
}


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.title_parts: list[str] = []
        self.h1_parts: list[list[str]] = []
        self.heading_parts: list[list[str]] = []
        self.visible_parts: list[str] = []
        self.pre_parts: list[list[str]] = []
        self.descriptions: list[str] = []
        self.canonicals: list[str] = []
        self.images: list[dict[str, str]] = []
        self.references: list[tuple[str, str]] = []
        self.json_ld_parts: list[list[str]] = []
        self._current_h1: list[str] | None = None
        self._current_heading: list[str] | None = None
        self._current_pre: list[str] | None = None
        self._current_json_ld: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = {name.lower(): value or "" for name, value in attrs}
        if tag not in VOID_TAGS:
            self.stack.append(tag)
        if tag == "meta" and attributes.get("name", "").lower() == "description":
            self.descriptions.append(attributes.get("content", "").strip())
        elif tag == "link":
            rel_values = attributes.get("rel", "").lower().split()
            if "canonical" in rel_values:
                self.canonicals.append(attributes.get("href", "").strip())
            if "stylesheet" in rel_values and attributes.get("href"):
                self.references.append(("stylesheet", attributes["href"]))
        elif tag == "img":
            self.images.append(attributes)
            if attributes.get("src"):
                self.references.append(("image", attributes["src"]))
        elif tag == "a" and attributes.get("href"):
            self.references.append(("link", attributes["href"]))
        elif tag == "script":
            if attributes.get("src"):
                self.references.append(("script", attributes["src"]))
            if attributes.get("type", "").lower() == "application/ld+json":
                self._current_json_ld = []
        elif tag == "h1":
            self._current_h1 = []
        elif tag in {"h2", "h3", "h4", "h5", "h6"}:
            self._current_heading = []
        elif tag == "pre":
            self._current_pre = []

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "h1" and self._current_h1 is not None:
            self.h1_parts.append(self._current_h1)
            self._current_h1 = None
        elif tag in {"h2", "h3", "h4", "h5", "h6"} and self._current_heading is not None:
            self.heading_parts.append(self._current_heading)
            self._current_heading = None
        elif tag == "pre" and self._current_pre is not None:
            self.pre_parts.append(self._current_pre)
            self._current_pre = None
        elif tag == "script" and self._current_json_ld is not None:
            self.json_ld_parts.append(self._current_json_ld)
            self._current_json_ld = None
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        if self.stack and self.stack[-1] == "title":
            self.title_parts.append(data)
        if self._current_h1 is not None:
            self._current_h1.append(data)
        if self._current_heading is not None:
            self._current_heading.append(data)
        if self._current_pre is not None:
            self._current_pre.append(data)
        if self._current_json_ld is not None:
            self._current_json_ld.append(data)
        if not any(tag in SKIP_TEXT_TAGS for tag in self.stack):
            self.visible_parts.append(data)


def clean_text(parts: list[str]) -> str:
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def is_user_page(relative_path: str) -> bool:
    return relative_path in {"index.html", "404.html"} or relative_path.startswith(USER_PREFIXES)


def page_url(relative_path: str) -> str:
    if relative_path == "index.html":
        return "/"
    if relative_path.endswith("/index.html"):
        return "/" + relative_path[: -len("index.html")]
    return "/" + relative_path


def local_target(site: Path, page_relative_path: str, raw_url: str) -> Path | None:
    raw_url = raw_url.strip()
    if not raw_url or raw_url.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
        return None
    absolute = urljoin(SITE_ORIGIN + page_url(page_relative_path), raw_url)
    parsed = urlparse(absolute)
    if parsed.netloc and parsed.netloc != urlparse(SITE_ORIGIN).netloc:
        return None
    path = unquote(parsed.path)
    candidate = site / path.lstrip("/")
    if path.endswith("/"):
        candidate /= "index.html"
    elif not candidate.exists() and not candidate.suffix:
        candidate /= "index.html"
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", nargs="?", type=Path, default=Path("_site"))
    args = parser.parse_args()
    site = args.site.resolve()
    if not site.is_dir():
        print(f"ERROR: built site directory does not exist: {site}")
        return 2
    issues: list[str] = []
    titles: dict[str, list[str]] = defaultdict(list)
    descriptions: dict[str, list[str]] = defaultdict(list)
    checked_pages = 0

    for html_path in sorted(site.rglob("*.html")):
        relative_path = html_path.relative_to(site).as_posix()
        if not is_user_page(relative_path):
            continue
        checked_pages += 1
        page = PageParser()
        page.feed(html_path.read_text(encoding="utf-8", errors="replace"))
        title = clean_text(page.title_parts)
        h1_values = [clean_text(parts) for parts in page.h1_parts]
        description_values = [value for value in page.descriptions if value]

        if not title:
            issues.append(f"{relative_path}: missing title")
        else:
            titles[title].append(relative_path)
        if len(description_values) != 1:
            issues.append(f"{relative_path}: expected one non-empty description, got {len(description_values)}")
        else:
            descriptions[description_values[0]].append(relative_path)
        if len(h1_values) != 1 or not h1_values[0]:
            issues.append(f"{relative_path}: expected one non-empty H1, got {h1_values}")
        if len(page.canonicals) != 1:
            issues.append(f"{relative_path}: expected one canonical, got {page.canonicals}")

        for heading_parts in page.heading_parts:
            heading = clean_text(heading_parts)
            if re.match(r"^#{1,6}\s+", heading):
                issues.append(f"{relative_path}: literal Markdown marker in heading: {heading[:120]}")

        for pre_parts in page.pre_parts:
            pre_text = "".join(pre_parts)
            if "```" in pre_text and ("**" in pre_text or re.search(r"(?m)^\s*[*-]\s+", pre_text)):
                issues.append(f"{relative_path}: likely swallowed Markdown in code block")

        visible_text = "\n".join(page.visible_parts)
        if re.search(r"(?m)^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*\|", visible_text):
            issues.append(f"{relative_path}: raw Markdown table separator is visible")
        if re.search(r"\$\$[^\n]{0,500}`{1,3}", visible_text):
            issues.append(f"{relative_path}: broken math delimiter followed by backtick")
        if "```" in visible_text:
            issues.append(f"{relative_path}: raw Markdown fence is visible")

        suspicious_ranges = {
            "Thai": (0x0E00, 0x0E7F),
            "Cyrillic": (0x0400, 0x04FF),
            "Hangul": (0xAC00, 0xD7AF),
        }
        if relative_path not in ALLOWED_DEVANAGARI_PAGES:
            suspicious_ranges["Devanagari"] = (0x0900, 0x097F)
        for label, (lower, upper) in suspicious_ranges.items():
            if any(lower <= ord(character) <= upper for character in visible_text):
                issues.append(f"{relative_path}: unexpected {label} text")

        for image in page.images:
            source = image.get("src", "")
            target = local_target(site, relative_path, source)
            if target is None:
                continue
            alt = image.get("alt", "").strip()
            if alt.lower() in WEAK_ALT_TEXT:
                issues.append(f"{relative_path}: missing or weak image alt: {source}")
            if not image.get("width") or not image.get("height"):
                issues.append(f"{relative_path}: local image lacks width/height: {source}")

        for reference_kind, raw_url in page.references:
            target = local_target(site, relative_path, raw_url)
            if target is not None and not target.exists():
                issues.append(f"{relative_path}: missing local {reference_kind}: {raw_url}")

        if not page.json_ld_parts:
            issues.append(f"{relative_path}: missing JSON-LD")
        for json_parts in page.json_ld_parts:
            try:
                json.loads("".join(json_parts))
            except json.JSONDecodeError as exc:
                issues.append(f"{relative_path}: invalid JSON-LD: {exc}")

    for title, pages in titles.items():
        if len(pages) > 1:
            issues.append(f"duplicate title {title!r}: {pages}")
    for description, pages in descriptions.items():
        if len(pages) > 1:
            issues.append(f"duplicate description across {len(pages)} pages: {pages}")

    if checked_pages == 0:
        issues.append(f"no user pages found under built site directory: {site}")

    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        print(f"Built site validation failed: {len(issues)} issue(s) across {checked_pages} page(s)")
        return 1

    print(f"Built site OK: {checked_pages} user page(s), no rendering, SEO, link, or image issues")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
