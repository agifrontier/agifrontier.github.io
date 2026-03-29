require 'nokogiri'

module Jekyll
  module SingleH1Filter
    def ensure_single_h1(html, fallback_title = nil)
      fragment = Nokogiri::HTML::DocumentFragment.parse(html.to_s)
      headings = fragment.css('h1').to_a

      if headings.empty?
        prepend_fallback_heading(fragment, fallback_title)
        return fragment.to_html
      end

      headings.drop(1).each do |heading|
        heading.name = 'h2'
      end

      fragment.to_html
    end

    private

    def prepend_fallback_heading(fragment, fallback_title)
      return if fallback_title.to_s.strip.empty?

      heading = Nokogiri::XML::Node.new('h1', fragment.document)
      heading.content = fallback_title

      if fragment.children.empty?
        fragment.add_child(heading)
      else
        fragment.children.first.add_previous_sibling(heading)
      end
    end
  end
end

Liquid::Template.register_filter(Jekyll::SingleH1Filter)
