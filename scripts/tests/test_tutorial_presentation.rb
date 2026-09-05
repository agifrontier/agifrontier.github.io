require 'minitest/autorun'
require 'ostruct'
require 'jekyll'
require_relative '../../_plugins/tutorial-presentation'
require_relative '../../_plugins/article-image-filter'

class TutorialPresentationTest < Minitest::Test
  def prepare(data = {}, content = '', facts = {})
    document = OpenStruct.new(data: data, content: content)
    site = OpenStruct.new(collections: { 'tutorials' => OpenStruct.new(docs: [document]) },
                          data: { 'paper_metadata' => facts })
    Jekyll::TutorialPresentationGenerator.new.generate(site)
    document.data
  end

  def test_legacy_source_keeps_version_and_verified_date
    data = prepare({}, '> ArXiv URL：https://arxiv.org/abs/2608.11095v1',
                   '2608.11095' => { 'published' => '2026-08-12' })
    assert_equal 'https://arxiv.org/abs/2608.11095v1', data['paper_url']
    assert_equal '2026-08-12', data['paper_published']
  end

  def test_missing_paper_date_is_not_guessed
    data = prepare({ 'arxiv_id' => '2608.11095', 'published_at' => '2026-09-04' })
    assert_nil data['paper_published']
    assert_equal 'https://arxiv.org/abs/2608.11095', data['paper_url']
  end

  def test_non_arxiv_source_is_not_labeled_as_paper
    data = prepare({ 'source_url' => 'https://www.anthropic.com/engineering/context', 'paper_published' => '2025-09-29' })
    assert_equal 'https://www.anthropic.com/engineering/context', data['paper_url']
    assert data['source_is_article']
    assert_equal '2025-09-29', data['paper_published']
  end

  def test_openreview_paper_uses_its_real_source_and_date
    data = prepare({ 'arxiv_id' => 'openreview:3hXEPbG0dh' }, '',
                   'openreview:3hXEPbG0dh' => { 'published' => '2026-08-15' })
    assert_equal 'https://openreview.net/forum?id=3hXEPbG0dh', data['paper_url']
    assert_equal '2026-08-15', data['paper_published']
    refute data['source_is_article']
  end

  def test_rejects_untrusted_source_url
    assert_raises(Jekyll::Errors::FatalException) do
      prepare({ 'arxiv_id' => '2608.11095', 'paper_url' => 'javascript:alert(1)' })
    end
  end

  def test_source_must_match_declared_paper
    assert_raises(Jekyll::Errors::FatalException) do
      prepare({ 'arxiv_id' => '2608.11095', 'paper_url' => 'https://arxiv.org/abs/2608.00001' })
    end
  end

  def test_preserves_chinese_title_and_recovers_legacy_heading
    assert_equal '中文标题保持不变', prepare({ 'title' => '中文标题保持不变' }, '## 其他标题')['display_title']
    assert_equal '关于智能体记忆的系统综述', prepare({ 'title' => 'Memory Survey' }, '## 关于智能体记忆的系统综述')['display_title']
    assert_equal 'Memory Survey', prepare({ 'title' => 'Memory Survey' }, '## 引言')['display_title']
  end

  def test_detects_article_math
    assert prepare({}, '公式 $x + y$')['article_has_math']
    refute prepare({}, '纯文字文章')['article_has_math']
  end

  def test_information_bar_after_subtitle_and_idempotent
    renderer = Object.new.extend(Jekyll::TutorialPresentationFilter)
    html = '<h1>标题</h1><p class="paper-original-title">Original title</p><p>正文</p>'
    bar = '<div class="paper-information"><a href="https://arxiv.org/abs/2608.11095">论文原文</a></div>'
    result = renderer.with_paper_information(html, bar)
    fragment = Nokogiri::HTML::DocumentFragment.parse(result)
    assert_equal %w[h1 p div p], fragment.children.select(&:element?).map(&:name)
    assert_equal 1, Nokogiri::HTML::DocumentFragment.parse(renderer.with_paper_information(result, bar)).css('.paper-information').length
  end

  def test_later_images_lazy_with_zoom_and_dimensions
    renderer = Object.new.extend(Jekyll::ArticleImageFilter)
    metadata = { '/images/a.webp' => { 'width' => 1600, 'height' => 800 },
                 '/images/b.webp' => { 'width' => 1000, 'height' => 500 } }
    result = renderer.optimize_article_images('<img src="/images/a.webp"><img src="/images/b.webp">', metadata, '标题')
    images = Nokogiri::HTML::DocumentFragment.parse(result).css('img')
    assert_equal 'eager', images[0]['loading']
    assert_equal 'lazy', images[1]['loading']
    assert_equal '1600', images[0]['width']
    assert_equal '', images[1]['data-zoomable']
  end
end
