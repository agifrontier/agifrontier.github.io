#!/usr/bin/env python3
"""Validate rendered Jekyll pages for SEO and content rendering regressions."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse


SITE_ORIGIN = "https://agifrontier.github.io"
USER_PREFIXES = ("tutorials/", "page/", "topics/", "guides/", "blog/")
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
        self.topic_published_dates: list[str] = []
        self.topic_article_links: list[str] = []
        self.topic_page_number: int | None = None
        self.topic_total_pages: int | None = None
        self.topic_total_items: int | None = None
        self.pagination_relations: list[tuple[str, str]] = []
        self.table_parts: list[list[str]] = []
        self.table_header_flags: list[bool] = []
        self._current_h1: list[str] | None = None
        self._current_heading: list[str] | None = None
        self._current_pre: list[str] | None = None
        self._current_json_ld: list[str] | None = None
        self._current_table: list[str] | None = None
        self._current_table_has_header = False
        self._in_topic_article = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = {name.lower(): value or "" for name, value in attrs}
        if tag not in VOID_TAGS:
            self.stack.append(tag)
        if tag == "table":
            self._current_table = []
            self._current_table_has_header = False
        elif tag in {"thead", "th"} and self._current_table is not None:
            self._current_table_has_header = True
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
            href = attributes["href"]
            self.references.append(("link", href))
            if self._in_topic_article:
                self.topic_article_links.append(href)
            for relation in attributes.get("rel", "").lower().split():
                if relation in {"prev", "next"}:
                    self.pagination_relations.append((relation, href))
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
        elif tag == "time" and "topic-page__published" in attributes.get("class", "").split():
            self.topic_published_dates.append(attributes.get("datetime", "").strip())
        elif tag == "article" and "topic-page__article" in attributes.get("class", "").split():
            self._in_topic_article = True
        elif tag == "section" and attributes.get("data-topic-page"):
            try:
                self.topic_page_number = int(attributes["data-topic-page"])
                self.topic_total_pages = int(attributes["data-topic-total-pages"])
                self.topic_total_items = int(attributes["data-topic-total-items"])
            except (KeyError, ValueError):
                self.topic_page_number = None
                self.topic_total_pages = None
                self.topic_total_items = None

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
        elif tag == "article" and self._in_topic_article:
            self._in_topic_article = False
        elif tag == "table" and self._current_table is not None:
            self.table_parts.append(self._current_table)
            self.table_header_flags.append(self._current_table_has_header)
            self._current_table = None
            self._current_table_has_header = False
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
        if self._current_table is not None:
            self._current_table.append(data)
        if not any(tag in SKIP_TEXT_TAGS for tag in self.stack):
            self.visible_parts.append(data)


def clean_text(parts: list[str]) -> str:
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def likely_kramdown_math_table(parts: list[str], has_header: bool) -> bool:
    """Identify headerless tables formed when Kramdown splits math on bare pipes."""
    return not has_header and "$" in "".join(parts)


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
    topic_pages: dict[str, list[tuple[int, str, PageParser, list[dt.datetime]]]] = defaultdict(list)

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

        for table_parts, has_header in zip(page.table_parts, page.table_header_flags):
            if likely_kramdown_math_table(table_parts, has_header):
                preview = clean_text(table_parts)[:120]
                issues.append(
                    f"{relative_path}: headerless table contains math delimiters; "
                    f"likely Kramdown pipe misparse: {preview}"
                )

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
            absolute_reference = urljoin(SITE_ORIGIN + page_url(relative_path), raw_url)
            parsed_reference = urlparse(absolute_reference)
            if parsed_reference.path.rstrip("/") == "/topics" and parsed_reference.fragment:
                issues.append(f"{relative_path}: topic link must use an independent page: {raw_url}")
            target = local_target(site, relative_path, raw_url)
            if target is not None and not target.exists():
                issues.append(f"{relative_path}: missing local {reference_kind}: {raw_url}")

        parsed_schemas: list[dict[str, object]] = []
        if not page.json_ld_parts:
            issues.append(f"{relative_path}: missing JSON-LD")
        for json_parts in page.json_ld_parts:
            try:
                schema = json.loads("".join(json_parts))
                if isinstance(schema, dict):
                    parsed_schemas.append(schema)
            except json.JSONDecodeError as exc:
                issues.append(f"{relative_path}: invalid JSON-LD: {exc}")

        topic_match = re.fullmatch(r"topics/([^/]+)(?:/page/(\d+))?/index\.html", relative_path)
        if topic_match:
            topic_slug = topic_match.group(1)
            topic_page_number = int(topic_match.group(2) or "1")
            parsed_dates: list[dt.datetime] = []
            if not page.topic_published_dates:
                issues.append(f"{relative_path}: missing topic article published dates")
            for value in page.topic_published_dates:
                try:
                    parsed_dates.append(dt.datetime.fromisoformat(value.replace("Z", "+00:00")))
                except ValueError:
                    issues.append(f"{relative_path}: invalid topic article date: {value!r}")
            if parsed_dates != sorted(parsed_dates, reverse=True):
                issues.append(f"{relative_path}: topic articles are not sorted by published date descending")

            schema_types = {schema.get("@type") for schema in parsed_schemas}
            if "CollectionPage" not in schema_types:
                issues.append(f"{relative_path}: missing CollectionPage JSON-LD")
            if "ItemList" not in schema_types:
                issues.append(f"{relative_path}: missing ItemList JSON-LD")

            if page.topic_page_number != topic_page_number:
                issues.append(
                    f"{relative_path}: data topic page {page.topic_page_number!r} != path page {topic_page_number}"
                )
            if page.topic_total_pages is None or page.topic_total_pages < 1:
                issues.append(f"{relative_path}: missing valid topic total pages")
            if page.topic_total_items is None or page.topic_total_items < len(page.topic_article_links):
                issues.append(f"{relative_path}: missing valid topic total items")
            if len(page.topic_article_links) != len(set(page.topic_article_links)):
                issues.append(f"{relative_path}: duplicate article link within topic page")
            topic_pages[topic_slug].append(
                (topic_page_number, relative_path, page, parsed_dates)
            )

    for title, pages in titles.items():
        if len(pages) > 1:
            issues.append(f"duplicate title {title!r}: {pages}")
    for description, pages in descriptions.items():
        if len(pages) > 1:
            issues.append(f"duplicate description across {len(pages)} pages: {pages}")

    sitemap_paths = sorted(site.glob("sitemap*.xml"))
    sitemap = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in sitemap_paths
    )
    if not sitemap:
        issues.append("sitemap.xml: missing or empty sitemap set")

    for topic_slug, records in sorted(topic_pages.items()):
        records.sort(key=lambda item: item[0])
        total_pages_values = {record[2].topic_total_pages for record in records}
        total_items_values = {record[2].topic_total_items for record in records}
        expected_total_pages = records[0][2].topic_total_pages
        expected_total_items = records[0][2].topic_total_items
        page_numbers = [record[0] for record in records]
        if len(total_pages_values) != 1 or expected_total_pages is None:
            issues.append(f"topics/{topic_slug}: inconsistent total page metadata: {total_pages_values}")
            continue
        if len(total_items_values) != 1 or expected_total_items is None:
            issues.append(f"topics/{topic_slug}: inconsistent total item metadata: {total_items_values}")
            continue
        expected_page_numbers = list(range(1, expected_total_pages + 1))
        if page_numbers != expected_page_numbers:
            issues.append(f"topics/{topic_slug}: page sequence {page_numbers} != {expected_page_numbers}")

        all_links = [link for record in records for link in record[2].topic_article_links]
        if len(all_links) != expected_total_items:
            issues.append(
                f"topics/{topic_slug}: rendered {len(all_links)} article links != total {expected_total_items}"
            )
        if len(all_links) != len(set(all_links)):
            issues.append(f"topics/{topic_slug}: duplicate article links across pagination")

        all_dates = [date for record in records for date in record[3]]
        if all_dates != sorted(all_dates, reverse=True):
            issues.append(f"topics/{topic_slug}: articles are not globally sorted by published date descending")

        for page_number, relative_path, page, _dates in records:
            expected_relations: set[tuple[str, str]] = set()
            if page_number > 1:
                previous_path = (
                    f"/topics/{topic_slug}/"
                    if page_number == 2
                    else f"/topics/{topic_slug}/page/{page_number - 1}/"
                )
                expected_relations.add(("prev", previous_path))
            if page_number < expected_total_pages:
                expected_relations.add(("next", f"/topics/{topic_slug}/page/{page_number + 1}/"))
            if set(page.pagination_relations) != expected_relations:
                issues.append(
                    f"{relative_path}: pagination relations {page.pagination_relations} != {sorted(expected_relations)}"
                )

            rendered_url = SITE_ORIGIN + page_url(relative_path)
            if rendered_url not in sitemap:
                issues.append(f"{relative_path}: missing from sitemap.xml")

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
