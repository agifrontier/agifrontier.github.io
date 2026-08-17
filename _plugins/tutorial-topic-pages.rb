module Jekyll
  class TutorialTopicPage < PageWithoutAFile
    def initialize(site, topic)
      slug = topic.fetch('slug')
      name = topic.fetch('name')
      super(site, site.source, File.join('topics', slug), 'index.html')

      self.data = {
        'layout' => 'topic',
        'title' => "#{name}论文解读",
        'seo_title' => topic['seo_title'] || "#{name}论文解读与最新研究",
        'description' => topic['description'] || "聚合#{name}方向的AI论文解读、核心方法、实验结论与实践进展。",
        'topic_name' => name,
        'topic_slug' => slug,
        'schema_type' => 'CollectionPage',
        'sitemap' => true
      }
    end
  end

  class TutorialTopicPagesGenerator < Generator
    safe true
    priority :low

    def generate(site)
      topics = site.config.fetch('tutorial_topics', [])
      validate_topics!(topics)
      topics.each { |topic| site.pages << TutorialTopicPage.new(site, topic) }
    end

    private

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
