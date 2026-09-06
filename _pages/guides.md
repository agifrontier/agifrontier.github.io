---
layout: default
permalink: /guides/
title: 精选专题
seo_title: AI论文精选专题：Agent记忆、Agent Harness、代码Agent
description: 按问题组织AI论文阅读路线，精选Agent记忆、Agent Harness和代码Agent研究，提供阅读顺序、比较维度、适用边界及论文原文入口。
schema_type: CollectionPage
nav: true
nav_order: 3
---

<div class="reading-guide-directory">
  <header class="post-header">
    <h1>精选专题</h1>
    <p>从一个具体问题出发，读懂方法之间的区别。</p>
  </header>
  <div class="reading-guide-directory__grid">
    {% for guide in site.data.reading_guides %}
      <section class="reading-guide-directory__card">
        <h2><a href="{{ '/guides/' | append: guide.slug | append: '/' | relative_url }}">{{ guide.title | escape }}</a></h2>
        <p>{{ guide.intro | escape }}</p>
        <a href="{{ '/guides/' | append: guide.slug | append: '/' | relative_url }}">查看阅读路线 <span aria-hidden="true">→</span></a>
      </section>
    {% endfor %}
  </div>
  <p><a href="{{ '/topics/' | relative_url }}">按研究方向浏览全部文章 →</a></p>
</div>
