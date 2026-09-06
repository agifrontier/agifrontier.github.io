#!/usr/bin/env python3
"""Verify rendered SEO invariants for a Jekyll build."""

from __future__ import annotations

import argparse
import json
import datetime as dt
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote


class SeoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.in_title = False
        self.in_json_ld = False
        self.json_ld_parts: list[str] = []
        self.json_ld_blocks: list[str] = []
        self.meta: list[dict[str, str]] = []
        self.canonical: str | None = None
        self.h1_count = 0
        self.images_missing_alt = 0
        self.images_empty_alt = 0
        self.related_links = 0
        self.paper_sources: list[str] = []
        self.paper_dates: list[str] = []
        self.interpretation_dates: list[str] = []
        self.scripts: list[str] = []
        self.guide_articles: list[str] = []
        self.guide_sources: list[str] = []
        self.guide_backlinks: list[str] = []
        self.in_guide_backlinks = False

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {key: value or "" for key, value in attrs_list}
        classes = set(attrs.get("class", "").split())
        if tag == "title":
            self.in_title = True
        elif tag == "meta":
            self.meta.append(attrs)
        elif tag == "link" and "canonical" in attrs.get("rel", "").split():
            self.canonical = attrs.get("href")
        elif tag == "script" and attrs.get("type") == "application/ld+json":
            self.in_json_ld = True
            self.json_ld_parts = []
        elif tag == "h1":
            self.h1_count += 1
        elif tag == "img":
            if "alt" not in attrs:
                self.images_missing_alt += 1
            elif not attrs["alt"].strip():
                self.images_empty_alt += 1

        if tag == "a" and "related-tutorials__article" in classes:
            self.related_links += 1
        if tag == "a" and "paper-information__source" in classes:
            self.paper_sources.append(attrs.get("href", ""))
        if tag == "time" and "paper-information__published" in classes:
            self.paper_dates.append(attrs.get("datetime", ""))
        if tag == "time" and "paper-information__interpretation" in classes:
            self.interpretation_dates.append(attrs.get("datetime", ""))
        if tag == "script" and attrs.get("src"):
            self.scripts.append(attrs["src"])
        if tag == "nav" and "reading-guide-backlinks" in classes:
            self.in_guide_backlinks = True
        if tag == "a" and "reading-guide__article" in classes:
            self.guide_articles.append(attrs.get("href", ""))
        if tag == "a" and "reading-guide__source" in classes:
            self.guide_sources.append(attrs.get("href", ""))
        if tag == "a" and self.in_guide_backlinks:
            self.guide_backlinks.append(attrs.get("href", ""))

    def handle_endtag(self, tag: str) -> None:
        if tag == "nav":
            self.in_guide_backlinks = False
        if tag == "title":
            self.in_title = False
        elif tag == "script" and self.in_json_ld:
            self.in_json_ld = False
            self.json_ld_blocks.append("".join(self.json_ld_parts).strip())

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
        if self.in_json_ld:
            self.json_ld_parts.append(data)

    @property
    def title(self) -> str:
        return " ".join("".join(self.title_parts).split())

    def meta_content(self, *, name: str | None = None, prop: str | None = None) -> str | None:
        for item in self.meta:
            if name is not None and item.get("name") == name:
                return item.get("content")
            if prop is not None and item.get("property") == prop:
                return item.get("content")
        return None


def parse(path: Path) -> SeoParser:
    parser = SeoParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def add_error(errors: list[dict[str, str]], path: Path, message: str) -> None:
    errors.append({"path": str(path), "message": message})


def identity_issues(schema: object) -> list[str]:
    issues: list[str] = []
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "sameAs" and (not isinstance(value, list) or any(not isinstance(url, str) or not url.startswith("https://") for url in value)):
                issues.append("sameAs must contain nonempty HTTPS identities")
            issues.extend(identity_issues(value))
    elif isinstance(schema, list):
        for value in schema:
            issues.extend(identity_issues(value))
    elif isinstance(schema, str) and any(marker in schema for marker in ("alberteinstein.com", "qc6CJjYAAAAJ", "you@example.com", "example_pdf.pdf")):
        issues.append("template identity remains in JSON-LD")
    return issues


def guide_order_issues(page: SeoParser, schema: dict) -> list[str]:
    """Keep the visible curated order and search metadata in agreement."""
    issues = []
    articles = page.guide_articles
    if not articles or len(set(articles)) != len(articles):
        issues.append("guide needs unique visible article links")
    if len(page.guide_sources) != len(articles):
        issues.append("guide must link to each paper source")
    elements = schema.get("itemListElement", [])
    if schema.get("numberOfItems") != len(articles):
        issues.append("guide schema count differs from visible articles")
    if [item.get("position") for item in elements] != list(range(1, len(articles) + 1)):
        issues.append("guide schema positions are not consecutive")
    if [item.get("url") for item in elements] != [f"https://agifrontier.github.io{url}" for url in articles]:
        issues.append("guide schema order differs from visible articles")
    return issues


def main() -> int:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("build_dir", type=Path)
    argument_parser.add_argument("--report", type=Path)
    args = argument_parser.parse_args()
    build_dir = args.build_dir.resolve()

    tutorial_pages = sorted((build_dir / "tutorials").glob("*/index.html"))
    tutorial_pages = [path for path in tutorial_pages if path.parent.name != "index"]
    errors: list[dict[str, str]] = []
    descriptions: list[str] = []
    schema_types: Counter[str] = Counter()

    for path in tutorial_pages:
        page = parse(path)
        relative_url = "/" + str(path.relative_to(build_dir).parent) + "/"
        description = page.meta_content(name="description") or ""
        descriptions.append(description)
        if not 10 <= len(page.title) <= 60:
            add_error(errors, path, f"title length {len(page.title)}")
        if not 50 <= len(description) <= 160:
            add_error(errors, path, f"description length {len(description)}")
        expected_canonical = f"https://agifrontier.github.io{quote(relative_url, safe='/-._~')}"
        if page.canonical != expected_canonical:
            add_error(errors, path, f"canonical {page.canonical!r} != {expected_canonical!r}")
        for prop in ("og:title", "og:description", "og:url", "og:image"):
            if not page.meta_content(prop=prop):
                add_error(errors, path, f"missing {prop}")
        for name in ("twitter:card", "twitter:title", "twitter:description", "twitter:image"):
            if not page.meta_content(name=name):
                add_error(errors, path, f"missing {name}")
        if page.h1_count != 1:
            add_error(errors, path, f"h1 count {page.h1_count}")
        if page.images_missing_alt or page.images_empty_alt:
            add_error(
                errors,
                path,
                f"image alt missing={page.images_missing_alt} empty={page.images_empty_alt}",
            )
        if page.related_links != 4:
            add_error(errors, path, f"related links {page.related_links}")
        if len(page.paper_sources) != 1 or not page.paper_sources[0].startswith("https://"):
            add_error(errors, path, "missing or invalid clickable paper source")
        for label, dates in (("paper", page.paper_dates), ("interpretation", page.interpretation_dates)):
            if len(dates) != 1:
                add_error(errors, path, f"expected one {label} date")
            else:
                try:
                    dt.datetime.fromisoformat(dates[0])
                except ValueError:
                    add_error(errors, path, f"invalid {label} date: {dates[0]}")
        if not page.json_ld_blocks:
            add_error(errors, path, "missing JSON-LD")
        for block in page.json_ld_blocks:
            try:
                schema = json.loads(block)
            except json.JSONDecodeError as exc:
                add_error(errors, path, f"invalid JSON-LD: {exc}")
                continue
            schema_type = str(schema.get("@type", ""))
            for issue in identity_issues(schema):
                add_error(errors, path, issue)
            if schema.get("author", {}).get("@type") != "Organization":
                add_error(errors, path, "tutorial author must identify the editorial organization")
            if page.paper_sources and schema.get("isBasedOn") != page.paper_sources[0]:
                add_error(errors, path, "schema source differs from visible source")
            schema_types[schema_type] += 1
            if schema_type != "TechArticle":
                add_error(errors, path, f"schema type {schema_type!r}")

    duplicate_descriptions = [
        description for description, count in Counter(descriptions).items() if count > 1
    ]
    if duplicate_descriptions:
        errors.append({"path": "tutorials", "message": f"duplicate descriptions {len(duplicate_descriptions)}"})

    page_checks = {
        "homepage": (build_dir / "index.html", "WebSite"),
        "topics": (build_dir / "topics/index.html", "WebSite"),
        "guides": (build_dir / "guides/index.html", "CollectionPage"),
        "blog_post": (build_dir / "blog/2025/attention-is-all-you-need/index.html", "BlogPosting"),
    }
    for label, (path, expected_schema) in page_checks.items():
        if not path.is_file():
            add_error(errors, path, f"missing {label}")
            continue
        page = parse(path)
        if page.h1_count != 1:
            add_error(errors, path, f"h1 count {page.h1_count}")
        if not page.json_ld_blocks:
            add_error(errors, path, "missing JSON-LD")
            continue
        try:
            schema = json.loads(page.json_ld_blocks[0])
        except json.JSONDecodeError as exc:
            add_error(errors, path, f"invalid JSON-LD: {exc}")
            continue
        if schema.get("@type") != expected_schema:
            add_error(errors, path, f"schema type {schema.get('@type')!r}")
        for issue in identity_issues(schema):
            add_error(errors, path, issue)
        if label in {"homepage", "topics", "guides"}:
            unused = [src for src in page.scripts if any(marker in src for marker in ("mathjax", "masonry", "badge.dimensions.ai", "d1bxh8uas1mnw7"))]
            if unused:
                add_error(errors, path, f"unused listing scripts: {unused}")

    topic_pages = sorted(
        path
        for path in (build_dir / "topics").rglob("index.html")
        if path != build_dir / "topics/index.html"
    )
    guide_pages = sorted((build_dir / "guides").glob("*/index.html"))
    if not guide_pages:
        add_error(errors, build_dir / "guides", "missing curated guide pages")
    for path in topic_pages + guide_pages:
        page = parse(path)
        description = page.meta_content(name="description") or ""
        relative_url = "/" + str(path.relative_to(build_dir).parent) + "/"
        expected_canonical = f"https://agifrontier.github.io{quote(relative_url, safe='/-._~')}"
        if page.h1_count != 1:
            add_error(errors, path, f"h1 count {page.h1_count}")
        if not 10 <= len(page.title) <= 60:
            add_error(errors, path, f"title length {len(page.title)}")
        if not 50 <= len(description) <= 160:
            add_error(errors, path, f"description length {len(description)}")
        if page.canonical != expected_canonical:
            add_error(errors, path, f"canonical {page.canonical!r}")

        topic_schema_types: set[str] = set()
        for block in page.json_ld_blocks:
            try:
                schema = json.loads(block)
            except json.JSONDecodeError as exc:
                add_error(errors, path, f"invalid JSON-LD: {exc}")
                continue
            topic_schema_types.add(str(schema.get("@type", "")))
            if path in guide_pages and schema.get("@type") == "ItemList":
                for issue in guide_order_issues(page, schema):
                    add_error(errors, path, issue)
        if "CollectionPage" not in topic_schema_types:
            add_error(errors, path, "missing CollectionPage schema")
        if "ItemList" not in topic_schema_types:
            add_error(errors, path, "missing ItemList schema")
        if path in guide_pages:
            for url, source in zip(page.guide_articles, page.guide_sources):
                if not url.startswith("/tutorials/") or ".." in url:
                    add_error(errors, path, f"invalid guide article URL: {url}")
                    continue
                target = build_dir / url.strip("/") / "index.html"
                if not target.is_file():
                    add_error(errors, path, f"missing guide target: {url}")
                    continue
                article = parse(target)
                if relative_url not in article.guide_backlinks:
                    add_error(errors, target, f"missing guide backlink: {relative_url}")
                if article.paper_sources != [source]:
                    add_error(errors, path, f"guide paper source mismatch: {url}")

    report = {
        "tutorial_pages": len(tutorial_pages),
        "topic_pages": len(topic_pages),
        "guide_pages": len(guide_pages),
        "unique_descriptions": len(set(descriptions)),
        "schema_types": dict(schema_types),
        "errors": errors,
    }
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "errors": errors[:20]}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
