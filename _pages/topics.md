---
layout: default
permalink: /topics/
title: AI 主题导航
seo_title: AI 主题导航：Agent、RAG、推理、多模态论文
description: 按 AI Agent、RAG、知识系统、推理、强化学习、多模态、具身智能等独立主题浏览 AI 前沿论文解读。
nav: true
nav_order: 2
---

<div class="post topic-directory">
  <header class="post-header">
    <h1 class="post-title">AI 主题导航</h1>
    <p>每个研究方向均拥有独立主题页，集中展示相关论文解读、核心方法与最新进展。</p>
  </header>

  <div class="topic-directory__grid">
    {% for topic in site.tutorial_topics %}
      {% assign topic_name = topic.name %}
      {% assign topic_tutorials = site.tutorials | where_exp: "tutorial", "tutorial.topics contains topic_name" %}
      {% if topic_tutorials.size > 0 %}
        {% if topic.legacy_slug %}<span class="topic-directory__legacy-anchor" id="{{ topic.legacy_slug }}" aria-hidden="true"></span>{% endif %}
        {% for legacy_slug in topic.legacy_slugs %}<span class="topic-directory__legacy-anchor" id="{{ legacy_slug }}" aria-hidden="true"></span>{% endfor %}
        <section class="topic-directory__section" id="{{ topic.slug }}">
          <h2><a href="{{ '/topics/' | append: topic.slug | append: '/' | relative_url }}">{{ topic.name }}</a></h2>
          <p>{{ topic.description }}</p>
          <a class="topic-directory__link" href="{{ '/topics/' | append: topic.slug | append: '/' | relative_url }}">
            查看 {{ topic_tutorials.size }} 篇论文解读 <span aria-hidden="true">→</span>
          </a>
        </section>
      {% endif %}
    {% endfor %}
  </div>
</div>
