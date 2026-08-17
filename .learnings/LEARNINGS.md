## [LRN-20260817-001] markdown-table-spacing

**Logged**: 2026-08-17T11:10:33+08:00
**Priority**: high
**Status**: resolved
**Area**: tests

### Summary
Kramdown tables must be separated from a following caption or note by a blank line.

### Details
A table row followed immediately by caption, paragraph, or separator text causes Kramdown to render the entire table as a paragraph. Separator cells with fewer than three hyphens, malformed rows, and header-only tables also fail to render. The full repair covered 42 defects across 26 tutorials. Counting multiline `rg` output with `wc -l` is misleading because each match spans two output lines; count parsed matches or validate the final file set instead.

### Suggested Action
Run `scripts/check_markdown_tables.py` before every Jekyll build and keep the GitHub Pages workflow gate enabled.

### Metadata
- Source: investigation
- Related Files: scripts/check_markdown_tables.py, .github/workflows/jekyll.yml
- Tags: markdown, kramdown, tables, validation
