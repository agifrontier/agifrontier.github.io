require 'date'
require 'nokogiri'
require 'uri'

module Jekyll
  class TutorialPresentationGenerator < Generator
    ARXIV_ID = /(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?\/\d{7})(?:v\d+)?/.freeze
    ARXIV_URL = %r{https?://arxiv\.org/abs/(#{ARXIV_ID})}.freeze

    safe true
    priority :normal

    def generate(site)
      site.collections.fetch('tutorials').docs.each do |document|
        data = document.data
        declared = data['arxiv_id'].to_s
        openreview = declared.match(/\Aopenreview:([A-Za-z0-9_-]+)\z/)
        links = document.content.scan(ARXIV_URL).flatten
        identifier = links.find { |value| base_id(value) == base_id(declared) } || declared
        identifier = links.first if identifier.to_s.empty?
        identifier = nil unless identifier.to_s.match?(/\A#{ARXIV_ID}\z/)
        source = data['paper_url'] || ("https://arxiv.org/abs/#{identifier}" if identifier)
        source ||= "https://openreview.net/forum?id=#{openreview[1]}" if openreview
        if data['source_url'] && declared.empty?
          begin
            uri = URI.parse(data['source_url'])
            unless uri.is_a?(URI::HTTPS) && uri.host && !uri.userinfo
              raise URI::InvalidURIError
            end
          rescue URI::InvalidURIError
            raise Errors::FatalException, "invalid source_url: #{data['source_url']}"
          end
          source = data['source_url']
          data['source_is_article'] = true
        end
        if source
          match = source.match(/\A#{ARXIV_URL}\z/)
          openreview_source = openreview && source == "https://openreview.net/forum?id=#{openreview[1]}"
          unless data['source_is_article'] || openreview_source || (match && (declared.empty? || base_id(match[1]) == base_id(declared)))
            raise Errors::FatalException, "invalid or mismatched paper_url: #{source}"
          end
          data['paper_url'] = source.sub('http:', 'https:')
          key = openreview ? declared : (base_id(match[1]) if match)
          facts = key ? site.data.fetch('paper_metadata', {}).fetch(key, {}) : {}
          published = data['paper_published'] || facts['published']
          if published
            begin
              data['paper_published'] = Date.iso8601(published.to_s).iso8601
            rescue ArgumentError
              raise Errors::FatalException, "invalid paper_published for #{source}: #{published}"
            end
          end
        end
        title = data['seo_title'] || data['title']
        heading = document.content[/^\#{1,2}\s+([^\n]+)/, 1].to_s.strip
        if !title.to_s.match?(/\p{Han}/) && heading.match?(/\p{Han}/) && heading.length >= 6
          title = heading.gsub(/[*`]/, '')
        end
        data['display_title'] = title
        data['article_has_math'] = document.content.match?(/\$|\\[\[(]/)
      end
    end

    private

    def base_id(value)
      value.to_s.sub(/v\d+\z/, '')
    end
  end

  module TutorialPresentationFilter
    def with_paper_information(html, information)
      fragment = Nokogiri::HTML::DocumentFragment.parse(html.to_s)
      return fragment.to_html if fragment.at_css('.paper-information') || information.to_s.strip.empty?

      anchor = fragment.at_css('p.paper-original-title') || fragment.at_css('h1')
      anchor&.add_next_sibling(Nokogiri::HTML::DocumentFragment.parse(information.to_s))
      fragment.to_html
    end
  end
end

Liquid::Template.register_filter(Jekyll::TutorialPresentationFilter)
