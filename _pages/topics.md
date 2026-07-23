---
layout: default
permalink: /topics/
title: AI 主题导航
seo_title: AI 主题导航：Agent、RAG、推理与多模态论文
description: 按 AI Agent、RAG 与知识系统、推理与强化学习、多模态视觉、具身智能、模型训练优化等主题浏览 AI 前沿论文解读。
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
      <section class="topic-directory__section" id="{{ topic.slug }}">
        <h2>{{ topic.name }} <small>{{ topic_tutorials.size }} 篇</small></h2>
        <ul>
          {% for tutorial in topic_tutorials limit: 12 %}
            <li><a href="{{ tutorial.url | relative_url }}">{{ tutorial.title | replace: '$', '' | strip_html }}</a></li>
          {% endfor %}
        </ul>
      </section>
    {% endif %}
  {% endfor %}
</div>
