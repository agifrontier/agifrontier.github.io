require 'open3'
require 'time'

module Jekyll
  class StableLastmodGenerator < Generator
    ISO_8601_PATTERN = /\A\d{4}-\d{2}-\d{2}T/.freeze

    safe true
    priority :low

    def generate(site)
      git_lastmod_index = build_git_lastmod_index(site)

      site_items(site).each do |item|
        lastmod = manual_lastmod(item)
        lastmod ||= git_lastmod(item, git_lastmod_index)
        lastmod ||= file_mtime(item, site.source)
        lastmod ||= explicit_date(item)

        item.data['seo_lastmod'] = lastmod if lastmod
      end
    end

    private

    def site_items(site)
      site.pages + site.collections.values.flat_map(&:docs)
    end

    def build_git_lastmod_index(site)
      relative_paths = site_items(site)
        .filter_map { |item| relative_path_for(item) if source_file?(item, site.source) }
        .uniq

      return {} if relative_paths.empty?

      stdout, status = Open3.capture2(
        'git',
        '-C',
        site.source,
        'log',
        '--format=%cI',
        '--name-only',
        '--',
        *relative_paths
      )

      unless status.success?
        Jekyll.logger.warn('stable-lastmod:', 'failed to read git history, falling back to file mtimes')
        return {}
      end

      parse_git_log(stdout)
    end

    def parse_git_log(output)
      current_timestamp = nil
      index = {}

      output.each_line do |line|
        value = line.strip
        next if value.empty?

        if value.match?(ISO_8601_PATTERN)
          current_timestamp = parse_time(value)
          next
        end

        index[value] ||= current_timestamp if current_timestamp
      end

      index
    end

    def manual_lastmod(item)
      parse_time(item.data['lastmod']) ||
        parse_time(item.data['last_modified_at']) ||
        parse_time(item.data['last_updated'])
    end

    def git_lastmod(item, git_lastmod_index)
      relative_path = relative_path_for(item)
      return unless relative_path

      git_lastmod_index[relative_path]
    end

    def file_mtime(item, site_source)
      source_path = source_path_for(item, site_source)
      return unless source_path && File.exist?(source_path)

      File.mtime(source_path)
    end

    def explicit_date(item)
      return unless item.data.key?('date')

      parse_time(item.data['date'])
    end

    def source_file?(item, site_source)
      source_path = source_path_for(item, site_source)
      source_path && File.exist?(source_path)
    end

    def relative_path_for(item)
      return item.relative_path if item.respond_to?(:relative_path) && item.relative_path
      return item.path if item.respond_to?(:path) && item.path

      nil
    end

    def source_path_for(item, site_source)
      return item.path if item.respond_to?(:path) && item.path && File.absolute_path(item.path) == item.path

      relative_path = relative_path_for(item)
      return unless relative_path

      File.join(site_source, relative_path)
    end

    def parse_time(value)
      return if value.nil? || value.to_s.strip.empty?
      return value if value.respond_to?(:iso8601)

      Time.parse(value.to_s)
    rescue ArgumentError
      nil
    end
  end
end
