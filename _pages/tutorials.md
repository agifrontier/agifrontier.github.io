---
layout: default
permalink: /
title: 首页
seo_title: AI前沿分享：AI论文解读与前沿技术指南
description: 聚合大模型、AI Agent、RAG、推理优化、多模态与具身智能论文解读，提炼核心方法、实验结果与工程实践启示。
nav: true
nav_order: 1
wide: true
pagination:
  enabled: true
  collection: tutorials
  permalink: /page/:num/
  per_page: 24
  sort_field: seo_published
  sort_reverse: true
  trail:
    before: 1 # The number of links before the current page
    after: 3 # The number of links after the current page
---

<div class="post homepage">
  <header class="homepage-intro">
    <div>
      <p class="homepage-intro__eyebrow">AGI FRONTIER</p>
      <h1>AI前沿分享{% if paginator.page and paginator.page > 1 %}<span>第{{ paginator.page }}页</span>{% endif %}</h1>
      <p class="homepage-intro__description">深度解读 AI 论文，追踪 Agent、推理、多模态与具身智能的最新进展。</p>
    </div>
    <a class="homepage-intro__link" href="{{ '/topics/' | relative_url }}">按主题浏览全部内容</a>
  </header>

  {% if page.pagination.enabled %}
    {% assign postlist = paginator.posts %}
  {% else %}
    {% assign postlist = site.tutorials | sort: "seo_published" | reverse %}
  {% endif %}

  {% if paginator.page == 1 or page.pagination.enabled == false %}
    {% assign hero_post = postlist[0] %}
    {% assign hero_date = hero_post.seo_published | default: hero_post.date %}
    {% assign hero_text_length = hero_post.content | strip_html | size %}
    {% assign hero_read_time = hero_text_length | divided_by: 500 | plus: 1 %}
    {% assign hero_image = hero_post.thumbnail %}
    {% if hero_image == blank %}
      {% assign hero_image_parts = hero_post.content | split: '<img src="' %}
      {% if hero_image_parts.size > 1 %}
        {% assign hero_image_src = hero_image_parts[1] | split: '"' %}
        {% assign hero_image = hero_image_src[0] %}
      {% endif %}
    {% endif %}

    <section class="homepage-lead" aria-label="最新论文解读">
      <article class="homepage-featured">
        <a class="homepage-featured__link" href="{{ hero_post.url | relative_url }}">
          <div class="homepage-featured__media">
            {% if hero_image != blank %}
              <img
                src="{% if hero_image contains '://' %}{{ hero_image }}{% else %}{{ hero_image | relative_url }}{% endif %}"
                alt="{{ hero_post.title | escape }}"
                loading="eager"
                fetchpriority="high"
                decoding="async"
              >
            {% else %}
              <img
                src="{{ '/assets/img/homepage-default-cover.svg' | relative_url }}"
                alt="{{ hero_post.title | escape }}"
                loading="eager"
                fetchpriority="high"
                decoding="async"
              >
            {% endif %}
          </div>
          <div class="homepage-featured__body">
            <p class="homepage-card__label">最新发布</p>
            <h2>{{ hero_post.title }}</h2>
            <p class="homepage-card__meta">{{ hero_date | date: "%Y年%m月%d日" }} · {{ hero_read_time }} 分钟阅读</p>
            <p class="homepage-featured__description">{{ hero_post.description | strip_html | truncate: 180 }}</p>
            {% if hero_post.topics and hero_post.topics.size > 0 %}
              <div class="homepage-card__topics">
                {% for topic in hero_post.topics limit: 2 %}<span>{{ topic }}</span>{% endfor %}
              </div>
            {% endif %}
          </div>
        </a>
      </article>

      <div class="homepage-lead__side">
        {% for post in postlist offset: 1 limit: 2 %}
          {% assign published_date = post.seo_published | default: post.date %}
          {% assign text_length = post.content | strip_html | size %}
          {% assign read_time = text_length | divided_by: 500 | plus: 1 %}
          {% assign card_image = post.thumbnail %}
          {% if card_image == blank %}
            {% assign card_image_parts = post.content | split: '<img src="' %}
            {% if card_image_parts.size > 1 %}
              {% assign card_image_src = card_image_parts[1] | split: '"' %}
              {% assign card_image = card_image_src[0] %}
            {% endif %}
          {% endif %}

          <article class="homepage-card homepage-card--lead">
            <a class="homepage-card__link" href="{{ post.url | relative_url }}">
              <div class="homepage-card__media">
                {% if card_image != blank %}
                  <img
                    src="{% if card_image contains '://' %}{{ card_image }}{% else %}{{ card_image | relative_url }}{% endif %}"
                    alt="{{ post.title | escape }}"
                    loading="eager"
                    decoding="async"
                  >
                {% else %}
                  <img
                    src="{{ '/assets/img/homepage-default-cover.svg' | relative_url }}"
                    alt="{{ post.title | escape }}"
                    loading="eager"
                    decoding="async"
                  >
                {% endif %}
              </div>
              <div class="homepage-card__body">
                {% if post.topics and post.topics.size > 0 %}
                  <p class="homepage-card__label">{{ post.topics[0] }}</p>
                {% endif %}
                <h2>{{ post.title }}</h2>
                <p class="homepage-card__description">{{ post.description | strip_html | truncate: 86 }}</p>
                <p class="homepage-card__meta">{{ published_date | date: "%Y年%m月%d日" }} · {{ read_time }} 分钟阅读</p>
              </div>
            </a>
          </article>
        {% endfor %}
      </div>
    </section>
  {% endif %}

  {% if site.tutorial_topics and site.tutorial_topics.size > 0 %}
    <nav class="homepage-topics" aria-label="AI主题导航">
      {% for topic in site.tutorial_topics %}
        <a href="{{ '/topics/' | relative_url }}#{{ topic.slug }}">{{ topic.name }}</a>
      {% endfor %}
    </nav>
  {% endif %}

  <section class="homepage-feed" aria-labelledby="homepage-feed-title">
    <div class="homepage-section-heading homepage-section-heading--feed">
      <div>
        <p class="homepage-card__label">PAPER INSIGHTS</p>
        <h2 id="homepage-feed-title">{% if paginator.page and paginator.page > 1 %}更多论文解读{% else %}最新解读{% endif %}</h2>
      </div>
      {% if paginator.page and paginator.page > 1 %}<a href="{{ '/' | relative_url }}">返回最新发布</a>{% endif %}
    </div>

    {% assign postlist_offset = 0 %}
    {% if paginator.page == 1 or page.pagination.enabled == false %}
      {% assign postlist_offset = 3 %}
    {% endif %}

    <div class="homepage-card-grid">
      {% for post in postlist offset: postlist_offset %}
        {% assign published_date = post.seo_published | default: post.date %}
        {% assign text_length = post.content | strip_html | size %}
        {% assign read_time = text_length | divided_by: 500 | plus: 1 %}
        {% assign card_image = post.thumbnail %}
        {% if card_image == blank %}
          {% assign card_image_parts = post.content | split: '<img src="' %}
          {% if card_image_parts.size > 1 %}
            {% assign card_image_src = card_image_parts[1] | split: '"' %}
            {% assign card_image = card_image_src[0] %}
          {% endif %}
        {% endif %}

        <article class="homepage-card">
          <a class="homepage-card__link" href="{{ post.url | relative_url }}">
            <div class="homepage-card__media">
              {% if card_image != blank %}
                <img
                  src="{% if card_image contains '://' %}{{ card_image }}{% else %}{{ card_image | relative_url }}{% endif %}"
                  alt="{{ post.title | escape }}"
                  loading="lazy"
                  decoding="async"
                >
              {% else %}
                <img
                  src="{{ '/assets/img/homepage-default-cover.svg' | relative_url }}"
                  alt="{{ post.title | escape }}"
                  loading="lazy"
                  decoding="async"
                >
              {% endif %}
            </div>
            <div class="homepage-card__body">
              <p class="homepage-card__meta">{{ published_date | date: "%Y年%m月%d日" }} · {{ read_time }} 分钟阅读</p>
              <h3>{{ post.title }}</h3>
              <p class="homepage-card__description">{{ post.description | strip_html | truncate: 112 }}</p>
              {% if post.topics and post.topics.size > 0 %}
                <div class="homepage-card__topics">
                  {% for topic in post.topics limit: 2 %}<span>{{ topic }}</span>{% endfor %}
                </div>
              {% endif %}
            </div>
          </a>
        </article>
      {% endfor %}
    </div>
  </section>

  {% if page.pagination.enabled %}
    {% include pagination.liquid %}
  {% endif %}
</div>
