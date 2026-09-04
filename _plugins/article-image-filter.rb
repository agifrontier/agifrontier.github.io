require 'nokogiri'

module Jekyll
  module ArticleImageFilter
    LEGACY_GENERATED_STYLE = /\Awidth:(?:80|85|90)%;\s*max-width:(?:300|450|600|700)px;\s*margin:auto;\s*display:block;?\z/i
    WIDE_LANDSCAPE_STYLE = [
      'width:min(1000px, calc(100vw - 2rem))',
      'max-width:none',
      'height:auto',
      'margin:1.5rem auto',
      'position:relative',
      'left:50%',
      'transform:translateX(-50%)',
      'display:block'
    ].join('; ') + ';'
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

        width = dimensions['width'].to_i
        height = dimensions['height'].to_i
        image['width'] ||= width.to_s
        image['height'] ||= height.to_s
        image['decoding'] ||= 'async'

        legacy_style = image['style'].to_s.strip
        if width >= 900 && width > height && LEGACY_GENERATED_STYLE.match?(legacy_style)
          image['style'] = WIDE_LANDSCAPE_STYLE
        end
      end

      fragment.to_html
    end
  end
end

Liquid::Template.register_filter(Jekyll::ArticleImageFilter)
