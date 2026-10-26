import unittest

from scripts.check_resource_loading import ResourceParser, check_page, pagination_urls


class ResourceLoadingTest(unittest.TestCase):
    def parse(self, html):
        page = ResourceParser()
        page.feed(html)
        return page

    def test_allows_lazy_urls_under_a_baseurl(self):
        page = self.parse('<ninja-keys></ninja-keys><script defer src="/preview/assets/js/search-setup.js?v=1" '
                          'data-search-component="/preview/assets/js/search/ninja-keys.min.js" '
                          'data-search-index="/preview/assets/js/search-data.js"></script>')
        self.assertEqual([], check_page(page, set()))

    def test_rejects_eager_component_and_index(self):
        page = self.parse('<script src="/assets/js/search-data.js"></script>'
                          '<script type="module" src="/assets/js/search/ninja-keys.min.js"></script>')
        self.assertEqual(2, len(check_page(page, set())))

    def test_search_disabled_does_not_require_loader(self):
        self.assertEqual([], check_page(self.parse('<p>正文</p>'), set()))

    def test_missing_lazy_url_is_not_accepted(self):
        page = self.parse('<ninja-keys></ninja-keys><script defer src="/assets/js/search-setup.js"></script>')
        self.assertIn("missing", check_page(page, set())[0])

    def test_new_icon_requires_updating_subset(self):
        page = self.parse('<i class="ti ti-search"></i><i class="fa-solid fa-calendar"></i>')
        self.assertEqual([], check_page(page, {"ti-search", "fa-solid", "fa-calendar"}))
        self.assertIn("fa-calendar", check_page(page, {"ti-search", "fa-solid"})[0])

    def test_pagination_exclusion_does_not_remove_topic_root(self):
        xml = '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://example.com/topics/ai-agent/</loc></url>' \
              '<url><loc>https://example.com/topics/ai-agent/page/2/</loc></url></urlset>'
        self.assertEqual(["https://example.com/topics/ai-agent/page/2/"], pagination_urls(xml))


if __name__ == "__main__":
    unittest.main()
