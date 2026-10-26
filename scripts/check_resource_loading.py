#!/usr/bin/env python3
"""Check lazy-search, icon subset and sitemap invariants in a Jekyll build."""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

ICON_CLASS = re.compile(r"(?:fa|ti|ai|si)-[a-z0-9-]+")


class ResourceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[dict[str, str | None]] = []
        self.styles: list[str] = []
        self.icons: set[str] = set()
        self.has_search = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "script":
            self.scripts.append(attributes)
        elif tag == "link" and "stylesheet" in (attributes.get("rel") or "").split():
            self.styles.append(attributes.get("href") or "")
        elif tag == "ninja-keys":
            self.has_search = True
        self.icons.update(item for item in (attributes.get("class") or "").split() if ICON_CLASS.fullmatch(item))


def check_page(page: ResourceParser, available_icons: set[str]) -> list[str]:
    issues = []
    for script in page.scripts:
        source = urlsplit(script.get("src") or "").path
        if "/assets/js/search/" in source or source.endswith(("/search-data.js", "/shortcut-key.js")):
            issues.append(f"Eager search resource: {source}")
    if page.has_search:
        loaders = [script for script in page.scripts if urlsplit(script.get("src") or "").path.endswith("/search-setup.js")]
        if len(loaders) != 1 or "defer" not in loaders[0]:
            issues.append("Search requires exactly one deferred loader")
        elif not all(loaders[0].get(name) for name in ["data-search-component", "data-search-index"]):
            issues.append("Search loader is missing component or index URL")
    for style in page.styles:
        parsed = urlsplit(style)
        if parsed.netloc == "fonts.googleapis.com" or parsed.path.endswith(("/academicons.min.css", "/scholar-icons.css")):
            issues.append(f"Unused global font stylesheet: {style}")
    issues.extend(f"Icon class missing from subset: {icon}" for icon in sorted(page.icons - available_icons))
    return issues


def pagination_urls(xml: str) -> list[str]:
    return [node.text or "" for node in ET.fromstring(xml).iter() if node.tag.endswith("}loc") and "/page/" in (node.text or "")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    css_path = args.build_root / "assets/css/main.css"
    css = css_path.read_text()
    available_icons = set(re.findall(r"\.((?:fa|ti|ai|si)-[a-z0-9-]+)", css))
    issues = []
    pages = sorted(args.build_root.rglob("*.html"))
    for file in pages:
        page = ResourceParser()
        page.feed(file.read_text())
        issues.extend({"file": str(file.relative_to(args.build_root)), "issue": issue} for issue in check_page(page, available_icons))
    for url in pagination_urls((args.build_root / "sitemap-pages.xml").read_text()):
        issues.append({"file": "sitemap-pages.xml", "issue": f"Pagination URL: {url}"})
    for icon in ["fa-clipboard", "fa-clipboard-check"]:
        if icon not in available_icons:
            issues.append({"file": "assets/css/main.css", "issue": f"Runtime clipboard icon missing: {icon}"})
    for font in ["site-tabler.woff", "site-fa-solid.woff"]:
        if not (args.build_root / "assets/fonts" / font).is_file():
            issues.append({"file": "assets/fonts", "issue": f"Subset font missing: {font}"})
    report = {"html_pages": len(pages), "main_css_bytes": css_path.stat().st_size, "issues": issues}
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({**report, "issues": issues[:20]}, ensure_ascii=False, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
