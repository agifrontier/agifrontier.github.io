import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_seo_build import SeoParser, guide_order_issues


class ReadingGuideSeoTests(unittest.TestCase):
    def setUp(self):
        self.page = SeoParser()
        self.page.feed('<a class="reading-guide__article" href="/tutorials/one/">One</a>'
                       '<a class="reading-guide__source" href="https://arxiv.org/abs/2608.11095">原文</a>')
        self.schema = {"numberOfItems": 1, "itemListElement": [
            {"position": 1, "url": "https://agifrontier.github.io/tutorials/one/"}]}

    def test_valid_order(self):
        self.assertEqual([], guide_order_issues(self.page, self.schema))

    def test_wrong_schema_order_is_rejected(self):
        self.schema["itemListElement"][0]["url"] = "https://agifrontier.github.io/tutorials/two/"
        self.assertIn("guide schema order differs from visible articles", guide_order_issues(self.page, self.schema))

    def test_missing_source_and_wrong_count_are_rejected(self):
        self.page.guide_sources = []
        self.schema["numberOfItems"] = 2
        self.assertEqual(2, len(guide_order_issues(self.page, self.schema)))

    def test_backlinks_are_not_confused_with_global_navigation(self):
        self.page.feed('<nav><a href="/guides/">专题</a></nav>'
                       '<nav class="reading-guide-backlinks"><a href="/guides/memory/">记忆</a></nav>'
                       '<a href="/guides/other/">其他</a>')
        self.assertEqual(["/guides/memory/"], self.page.guide_backlinks)


if __name__ == "__main__":
    unittest.main()
