require 'minitest/autorun'
require 'tmpdir'
require 'jekyll'
require_relative '../../_plugins/reading-guides'

class ReadingGuidesTest < Minitest::Test
  def setup
    @root = Dir.mktmpdir('reading-guides')
    @site = Jekyll::Site.new(Jekyll.configuration('source' => @root, 'destination' => File.join(@root, '_site'),
                                                  'config' => [], 'plugins' => [], 'quiet' => true,
                                                  'collections' => { 'tutorials' => { 'output' => true } }))
    collection = @site.collections['tutorials']
    %w[one two untouched].each do |slug|
      document = Jekyll::Document.new(File.join(@root, '_tutorials', "#{slug}.md"), site: @site, collection: collection)
      document.data.merge!('title' => slug, 'paper_url' => 'https://arxiv.org/abs/2608.11095')
      collection.docs << document
    end
    @guide = { 'slug' => 'memory', 'title' => 'Agent记忆', 'seo_title' => 'Agent记忆论文阅读路线',
               'description' => '精选记忆论文', 'intro' => '从概念到工程',
               'choices' => [{ 'need' => '检索', 'route' => '阅读', 'check' => '成本' }],
               'items' => %w[one two].map { |slug| { 'slug' => slug, 'label' => slug, 'stage' => '入门',
                                                  'note' => '阅读理由', 'boundary' => '适用边界' } } }
    @site.data['reading_guides'] = [@guide]
    @generator = Jekyll::ReadingGuidesGenerator.new
  end

  def teardown
    FileUtils.remove_entry(@root)
  end

  def test_resolves_order_and_only_selected_backlinks
    @generator.generate(@site)
    assert_equal ['/guides/memory/'], @site.pages.map(&:url)
    assert_equal %w[one two], @site.pages.first.data['guide_items'].map { |item| item['tutorial'].data['title'] }
    documents = @site.collections['tutorials'].docs
    assert_equal [{ 'title' => 'Agent记忆', 'path' => '/guides/memory/' }], documents.first.data['reading_guides']
    assert_nil documents.last.data['reading_guides']
  end

  def test_repeated_generation_has_no_duplicate_pages_or_backlinks
    2.times { @generator.generate(@site) }
    assert_equal 1, @site.pages.length
    assert_equal 1, @site.collections['tutorials'].docs.first.data['reading_guides'].length
  end

  def test_removed_selection_clears_stale_backlink
    @generator.generate(@site)
    @guide['items'].shift
    @generator.generate(@site)
    assert_nil @site.collections['tutorials'].docs.first.data['reading_guides']
  end

  def test_invalid_article_does_not_partially_mutate_site
    @guide['items'].last['slug'] = 'not-found'
    assert_raises(Jekyll::Errors::FatalException) { @generator.generate(@site) }
    assert_empty @site.pages
    assert_nil @site.collections['tutorials'].docs.first.data['reading_guides']
  end

  def test_duplicate_article_is_rejected
    @guide['items'].last['slug'] = 'one'
    assert_raises(Jekyll::Errors::FatalException) { @generator.generate(@site) }
  end

  def test_invalid_guide_slugs_and_duplicate_guides_are_rejected
    ['../unsafe', 'https://bad.test', '', nil].each do |slug|
      @guide['slug'] = slug
      assert_raises(Jekyll::Errors::FatalException) { @generator.generate(@site) }
    end
    @guide['slug'] = 'memory'
    @site.data['reading_guides'] = [@guide, @guide]
    assert_raises(Jekyll::Errors::FatalException) { @generator.generate(@site) }
  end

  def test_empty_editorial_fields_and_missing_source_are_rejected
    @guide['items'].first['boundary'] = ' '
    assert_raises(Jekyll::Errors::FatalException) { @generator.generate(@site) }
    @guide['items'].first['boundary'] = '边界'
    @site.collections['tutorials'].docs.first.data.delete('paper_url')
    assert_raises(Jekyll::Errors::FatalException) { @generator.generate(@site) }
  end

  def test_existing_page_is_not_overwritten
    page = Jekyll::PageWithoutAFile.new(@site, @site.source, 'guides/memory', 'index.html')
    @site.pages << page
    assert_raises(Jekyll::Errors::FatalException) { @generator.generate(@site) }
    assert_equal [page], @site.pages
  end

  def test_one_article_can_belong_to_two_distinct_guides
    second = Marshal.load(Marshal.dump(@guide))
    second['slug'] = 'coding'
    @site.data['reading_guides'] << second
    @generator.generate(@site)
    assert_equal 2, @site.collections['tutorials'].docs.first.data['reading_guides'].length
  end

  def test_real_curated_data_has_three_guides_and_eighteen_unique_articles
    guides = YAML.safe_load_file(File.expand_path('../../_data/reading_guides.yml', __dir__))
    assert_equal 3, guides.length
    assert guides.all? { |guide| guide['items'].length == 6 }
    assert_equal 18, guides.flat_map { |guide| guide['items'].map { |item| item['slug'] } }.uniq.length
  end
end
