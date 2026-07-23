---
layout: default
permalink: /
title: Tutorials
seo_title: AI前沿分享：AI论文解读与前沿技术指南
description: 聚合大模型、AI Agent、RAG、推理优化、多模态与具身智能论文解读，提炼核心方法、实验结果与工程实践启示。
nav: true
nav_order: 1
pagination:
  enabled: true
  collection: tutorials
  permalink: /page/:num/
  per_page: 50
  sort_field: seo_lastmod
  sort_reverse: true
  trail:
    before: 1 # The number of links before the current page
    after: 3 # The number of links after the current page
---

<div class="post">

  <div class="header-bar">
    <h1>Tutorials</h1>
    <h2>AI 论文解读、工程实践与前沿趋势</h2>
    <p><a href="{{ '/topics/' | relative_url }}">按主题浏览全部内容</a></p>
  </div>

{% if site.tutorial_topics and site.tutorial_topics.size > 0 %}

  <div class="tag-category-list">
    <ul class="p-0 m-0">
      {% for topic in site.tutorial_topics %}
        <li>
          <i class="fa-solid fa-hashtag fa-sm"></i>
          <a href="{{ '/topics/' | relative_url }}#{{ topic.slug }}">{{ topic.name }}</a>
        </li>
        {% unless forloop.last %}
          <p>&bull;</p>
        {% endunless %}
      {% endfor %}
    </ul>
  </div>
  {% endif %}

{% assign featured_tutorials = site.tutorials | where: "featured", "true" | sort: "seo_lastmod" | reverse %}
{% if featured_tutorials.size > 0 %}
<br>

<div class="container featured-posts">
{% assign is_even = featured_tutorials.size | modulo: 2 %}
<div class="row row-cols-{% if featured_tutorials.size <= 2 or is_even == 0 %}2{% else %}3{% endif %}">
{% for post in featured_tutorials %}
<div class="col mb-4">
<a href="{{ post.url | relative_url }}">
<div class="card hoverable">
<div class="row g-0">
<div class="col-md-12">
<div class="card-body">
<div class="float-right">
<i class="fa-solid fa-thumbtack fa-xs"></i>
</div>
<h3 class="card-title text-lowercase">{{ post.title }}</h3>
<p class="card-text">{{ post.description }}</p>

                    {% if post.external_source == blank %}
                      {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
                    {% else %}
                      {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
                    {% endif %}
                    {% assign display_date = post.seo_lastmod | default: post.date %}
                    {% assign year = display_date | date: "%Y" %}

                    <p class="post-meta">
                      {{ read_time }} min read &nbsp; &middot; &nbsp;
                      <span><i class="fa-solid fa-calendar fa-sm"></i> {{ year }}</span>
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </a>
        </div>
      {% endfor %}
      </div>
    </div>
    <hr>

{% endif %}

  <ul class="post-list">

    {% if page.pagination.enabled %}
      {% assign postlist = paginator.posts %}
    {% else %}
      {% assign postlist = site.tutorials | sort: "seo_lastmod" | reverse %}
    {% endif %}

    {% for post in postlist %}

    {% if post.external_source == blank %}
      {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
    {% else %}
      {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
    {% endif %}
    {% assign display_date = post.seo_lastmod | default: post.date %}
    {% assign year = display_date | date: "%Y" %}
    {% assign tags = post.tags | join: "" %}
    {% assign categories = post.categories | join: "" %}

    <li>

{% if post.thumbnail %}

<div class="row">
          <div class="col-sm-9">
{% endif %}
        <h3>
        {% if post.redirect == blank %}
          <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
        {% elsif post.redirect contains '://' %}
          <a class="post-title" href="{{ post.redirect }}" target="_blank">{{ post.title }}</a>
          <svg width="2rem" height="2rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
            <path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path>
          </svg>
        {% else %}
          <a class="post-title" href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
        {% endif %}
      </h3>
      <p>{{ post.description }}</p>
      <p class="post-meta">
        {{ read_time }} min read &nbsp; &middot; &nbsp;
        {{ display_date | date: '%B %d, %Y' }}
        {% if post.external_source %}
        &nbsp; &middot; &nbsp; {{ post.external_source }}
        {% endif %}
      </p>
      <p class="post-tags">
        <span><i class="fa-solid fa-calendar fa-sm"></i> {{ year }}</span>

          {% if tags != "" %}
          &nbsp; &middot; &nbsp;
            {% for tag in post.tags %}
            <span><i class="fa-solid fa-hashtag fa-sm"></i> {{ tag }}</span>
              {% unless forloop.last %}
                &nbsp;
              {% endunless %}
              {% endfor %}
          {% endif %}

          {% if categories != "" %}
          &nbsp; &middot; &nbsp;
            {% for category in post.categories %}
            <span><i class="fa-solid fa-tag fa-sm"></i> {{ category }}</span>
              {% unless forloop.last %}
                &nbsp;
              {% endunless %}
              {% endfor %}
          {% endif %}
    </p>

{% if post.thumbnail %}

</div>

  <div class="col-sm-3">
    <img class="card-img" src="{{ post.thumbnail | relative_url }}" style="object-fit: cover; height: 90%" alt="image">
  </div>
</div>
{% endif %}
    </li>

    {% endfor %}

  </ul>

{% if page.pagination.enabled %}
{% include pagination.liquid %}
{% endif %}

</div>
