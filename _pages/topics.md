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
    <p>按研究方向浏览论文解读，快速进入相关主题与延伸阅读。</p>
  </header>

  {% for topic in site.tutorial_topics %}
    {% assign topic_name = topic.name %}
    {% assign topic_tutorials = site.tutorials | where_exp: "tutorial", "tutorial.topics contains topic_name" | sort: "seo_lastmod" | reverse %}
    {% if topic_tutorials.size > 0 %}
      {% if topic.legacy_slug %}<span class="topic-directory__legacy-anchor" id="{{ topic.legacy_slug }}" aria-hidden="true"></span>{% endif %}
      <section class="topic-directory__section" id="{{ topic.slug }}">
        <h2>{{ topic.name }} <small>{{ topic_tutorials.size }} 篇</small></h2>
        <ul>
          {% for tutorial in topic_tutorials limit: 12 %}
            <li><a href="{{ tutorial.url | relative_url }}">{{ tutorial.title | replace: '$', '' | strip_html }}</a></li>
          {% endfor %}
        </ul>
        {% if topic_tutorials.size > 12 %}
          {% assign remaining_count = topic_tutorials.size | minus: 12 %}
          <details class="topic-directory__more">
            <summary>查看其余 {{ remaining_count }} 篇</summary>
            <ul>
              {% for tutorial in topic_tutorials offset: 12 %}
                <li><a href="{{ tutorial.url | relative_url }}">{{ tutorial.title | replace: '$', '' | strip_html }}</a></li>
              {% endfor %}
            </ul>
          </details>
        {% endif %}
      </section>
    {% endif %}
  {% endfor %}
</div>
