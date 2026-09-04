from __future__ import annotations

import unittest

from scripts.check_built_site import PageParser, likely_kramdown_math_table


class CheckBuiltSiteTest(unittest.TestCase):
    def _table(self, html: str) -> tuple[list[str], bool]:
        page = PageParser()
        page.feed(html)
        self.assertEqual(len(page.table_parts), 1)
        return page.table_parts[0], page.table_header_flags[0]

    def test_detects_headerless_table_created_from_math_pipes(self) -> None:
        parts, has_header = self._table(
            "<table><tbody><tr><td>集合大小为 $</td><td>D</td><td>$</td></tr></tbody></table>"
        )

        self.assertTrue(likely_kramdown_math_table(parts, has_header))

    def test_accepts_real_table_with_header(self) -> None:
        parts, has_header = self._table(
            "<table><thead><tr><th>公式</th></tr></thead>"
            "<tbody><tr><td>$O(1)$</td></tr></tbody></table>"
        )

        self.assertFalse(likely_kramdown_math_table(parts, has_header))

    def test_accepts_headerless_layout_table_without_math(self) -> None:
        parts, has_header = self._table(
            "<table><tbody><tr><td>普通文本</td></tr></tbody></table>"
        )

        self.assertFalse(likely_kramdown_math_table(parts, has_header))


if __name__ == "__main__":
    unittest.main()
