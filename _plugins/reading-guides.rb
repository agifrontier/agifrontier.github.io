module Jekyll
  class ReadingGuidesGenerator < Generator
    safe true
    priority :lowest

    def generate(site)
      guides = site.data.fetch('reading_guides', [])
      fail!('reading_guides must be a list') unless guides.is_a?(Array)
      documents = site.collections.fetch('tutorials').docs
      by_slug = documents.to_h { |document| [document.basename_without_ext, document] }
      seen = []
      prepared = guides.map do |guide|
        fields!(guide, %w[slug title seo_title description intro], 'guide')
        slug = guide['slug']
        fail!("invalid or duplicate guide slug: #{slug}") unless slug.match?(/\A[a-z0-9]+(?:-[a-z0-9]+)*\z/) && !seen.include?(slug)
        seen << slug
        items = guide['items']
        fail!("#{slug}: items must be a nonempty list") unless items.is_a?(Array) && !items.empty?
        used = []
        resolved = items.map do |item|
          fields!(item, %w[slug label stage note boundary], slug)
          article_slug = item['slug']
          document = by_slug[article_slug]
          fail!("#{slug}: missing or duplicate tutorial: #{article_slug}") unless document && !used.include?(article_slug)
          fail!("#{slug}: missing paper source: #{article_slug}") if document.data['paper_url'].to_s.empty?
          used << article_slug
          item.merge('tutorial' => document)
        end
        choices = guide['choices']
        fail!("#{slug}: choices must be a nonempty list") unless choices.is_a?(Array) && !choices.empty?
        choices.each { |choice| fields!(choice, %w[need route check], slug) }
        guide.merge('guide_items' => resolved, 'guide_path' => "/guides/#{slug}/")
      end

      # Validate the entire selection before changing page or article state.
      prepared.each do |guide|
        collision = site.pages.any? do |page|
          page.url.sub(/index\.html\z/, '') == guide['guide_path'] && !page.data['generated_reading_guide']
        end
        fail!("existing page at #{guide['guide_path']}") if collision
      end
      site.pages.reject! { |page| page.data['generated_reading_guide'] }
      documents.each { |document| document.data.delete('reading_guides') }
      prepared.each do |guide|
        page = PageWithoutAFile.new(site, site.source, "guides/#{guide['slug']}", 'index.html')
        page.data = guide.merge('layout' => 'reading-guide', 'schema_type' => 'CollectionPage',
                                'generated_reading_guide' => true, 'sitemap' => true)
        site.pages << page
        guide['guide_items'].each do |item|
          links = (item['tutorial'].data['reading_guides'] ||= [])
          links << { 'title' => guide['title'], 'path' => guide['guide_path'] }
        end
      end
    end

    private

    def fields!(value, keys, context)
      unless value.is_a?(Hash) && keys.all? { |key| value[key].is_a?(String) && !value[key].strip.empty? }
        fail!("#{context}: required nonempty text fields: #{keys.join(', ')}")
      end
    end

    def fail!(message)
      raise Errors::FatalException, "reading-guides: #{message}"
    end
  end
end
