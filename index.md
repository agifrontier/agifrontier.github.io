---
layout: default
title: AI前沿分享主页
---

# AI前沿分享

每日AI最新进展分享。

## 最新文章列表

<h2>文章列表</h2>

{% comment %} 先过滤确保所有项目都有 last_modified_at 属性，防止 nil 值破坏排序 {% endcomment %}
{% assign items_with_date = site.tutorials | where_exp: "item", "item.last_modified_at" %}
{% assign sorted_tutorials = items_with_date | sort: "last_modified_at" | reverse %}
<ul>
  {% for tutorial in sorted_tutorials %}
    <li>
      <h3>
        <a href="{{ tutorial.url | relative_url }}">{{ tutorial.title }}</a>
      </h3>
    </li>
  {% endfor %}
</ul>