import unittest

from scripts.verify_seo_build import SeoParser, identity_issues


class SeoPresentationTest(unittest.TestCase):
    def test_detects_nested_template_identity_and_null(self):
        self.assertTrue(identity_issues({'publisher': {'sameAs': [None]}}))
        self.assertTrue(identity_issues({'sameAs': ['https://www.alberteinstein.com/']}))
        self.assertEqual([], identity_issues({'publisher': {'sameAs': ['https://github.com/agifrontier']}}))

    def test_reads_visible_paper_information(self):
        parser = SeoParser()
        parser.feed('<a class="paper-information__source" href="https://arxiv.org/abs/2608.11095">原文</a>'
                    '<time class="paper-information__published" datetime="2026-08-12">日期</time>'
                    '<time class="paper-information__interpretation" datetime="2026-09-04T08:00:00+08:00">日期</time>')
        self.assertEqual(['https://arxiv.org/abs/2608.11095'], parser.paper_sources)
        self.assertEqual(['2026-08-12'], parser.paper_dates)
        self.assertEqual(1, len(parser.interpretation_dates))
