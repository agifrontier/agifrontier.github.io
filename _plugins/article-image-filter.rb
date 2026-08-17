require 'nokogiri'

module Jekyll
  module ArticleImageFilter
    WEAK_ALT_TEXT = [
      '',
      'refer to caption',
      'refer tocaption',
      '[uncaptioned image]',
      '[无标题图片]',
      '插图',
      'img'
    ].freeze

    def optimize_article_images(html, image_metadata = {}, page_title = nil)
      fragment = Nokogiri::HTML::DocumentFragment.parse(html.to_s)
      fallback_alt = "#{page_title} 论文图示".strip

      fragment.css('img').each do |image|
        alt = image['alt'].to_s.strip
        image['alt'] = fallback_alt if WEAK_ALT_TEXT.include?(alt.downcase) && !fallback_alt.empty?

        source = image['src'].to_s.split(/[?#]/, 2).first
        dimensions = image_metadata[source]
        next unless dimensions

        image['width'] ||= dimensions['width'].to_s
        image['height'] ||= dimensions['height'].to_s
        image['decoding'] ||= 'async'
      end

      fragment.to_html
    end
  end
end

Liquid::Template.register_filter(Jekyll::ArticleImageFilter)
