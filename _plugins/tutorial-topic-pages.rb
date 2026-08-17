require 'time'

module Jekyll
  class TutorialTopicPage < PageWithoutAFile
    def initialize(site, topic, tutorials, page_number, total_pages, first_position)
      slug = topic.fetch('slug')
      name = topic.fetch('name')
      directory = File.join('topics', slug)
      directory = File.join(directory, 'page', page_number.to_s) if page_number > 1
      super(site, site.source, directory, 'index.html')

      base_title = topic['seo_title'] || "#{name}论文解读与最新研究"
      base_description = topic['description'] || "聚合#{name}方向的AI论文解读、核心方法、实验结论与实践进展。"
      page_suffix = page_number > 1 ? " - 第#{page_number}页" : ''
      description_suffix = page_number > 1 ? " 当前为第#{page_number}页。" : ''
      self.data = {
        'layout' => 'topic',
        'title' => "#{name}论文解读",
        'seo_title' => "#{base_title}#{page_suffix}",
        'description' => "#{base_description}#{description_suffix}",
        'topic_name' => name,
        'topic_slug' => slug,
        'topic_tutorials' => tutorials,
        'topic_page_number' => page_number,
        'topic_total_pages' => total_pages,
        'topic_first_position' => first_position,
        'schema_type' => 'CollectionPage',
        'sitemap' => true
      }
    end
  end

  class TutorialTopicPagesGenerator < Generator
    DEFAULT_PER_PAGE = 24

    safe true
    priority :low

    def generate(site)
      topics = site.config.fetch('tutorial_topics', [])
      validate_topics!(topics)
      per_page = topic_per_page(site)
      topics.each { |topic| generate_topic_pages(site, topic, per_page) }
    end

    private

    def generate_topic_pages(site, topic, per_page)
      tutorials = site.collections.fetch('tutorials').docs.select do |tutorial|
        Array(tutorial.data['topics']).include?(topic.fetch('name'))
      end
      tutorials.sort_by! { |tutorial| [-published_timestamp(tutorial), tutorial.url.to_s] }

      total_pages = [(tutorials.length.to_f / per_page).ceil, 1].max
      total_pages.times do |page_index|
        page_number = page_index + 1
        page_tutorials = tutorials.slice(page_index * per_page, per_page) || []
        first_position = page_index * per_page + 1
        page = TutorialTopicPage.new(
          site,
          topic,
          page_tutorials,
          page_number,
          total_pages,
          first_position
        )
        page.data['topic_total_items'] = tutorials.length
        page.data['topic_per_page'] = per_page
        page.data['topic_previous_path'] = topic_page_path(topic.fetch('slug'), page_number - 1) if page_number > 1
        page.data['topic_next_path'] = topic_page_path(topic.fetch('slug'), page_number + 1) if page_number < total_pages
        page.data['topic_page_links'] = (1..total_pages).map do |number|
          { 'number' => number, 'path' => topic_page_path(topic.fetch('slug'), number) }
        end
        site.pages << page
      end
    end

    def published_timestamp(tutorial)
      value = tutorial.data['seo_published'] || tutorial.data['published_at'] || tutorial.data['date']
      return value.to_time.to_f if value.respond_to?(:to_time)

      Time.parse(value.to_s).to_f
    rescue ArgumentError
      0.0
    end

    def topic_page_path(slug, page_number)
      return "/topics/#{slug}/" if page_number == 1

      "/topics/#{slug}/page/#{page_number}/"
    end

    def topic_per_page(site)
      value = site.config.fetch('tutorial_topic_per_page', DEFAULT_PER_PAGE)
      return value if value.is_a?(Integer) && value.positive?

      raise Jekyll::Errors::FatalException, 'tutorial_topic_per_page must be a positive integer'
    end

    def validate_topics!(topics)
      slugs = topics.map { |topic| topic['slug'].to_s.strip }
      names = topics.map { |topic| topic['name'].to_s.strip }
      invalid = topics.select do |topic|
        topic['slug'].to_s !~ /\A[a-z0-9]+(?:-[a-z0-9]+)*\z/ || topic['name'].to_s.strip.empty?
      end

      return if invalid.empty? && slugs.uniq.length == slugs.length && names.uniq.length == names.length

      raise Jekyll::Errors::FatalException,
            'tutorial_topics must have unique names and lowercase kebab-case slugs'
    end
  end
end
