#!/usr/bin/env python3
"""Generate the deterministic default Open Graph image."""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1200
HEIGHT = 630
FONT_PATH = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
OUTPUT = Path(__file__).resolve().parents[1] / "assets/img/og-default.png"


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size=size)


def main() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#F4F7F8")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, 26, HEIGHT), fill="#E76F51")
    draw.rectangle((26, 0, 52, HEIGHT), fill="#17252A")

    random.seed(42)
    center_x, center_y = 930, 315
    nodes: list[tuple[int, int, int, str]] = []
    colors = ["#2A9D8F", "#E76F51", "#E9C46A", "#264653"]
    for ring, count in ((92, 6), (178, 10), (252, 14)):
        for index in range(count):
            angle = (2 * math.pi * index / count) + random.uniform(-0.12, 0.12)
            x = int(center_x + math.cos(angle) * ring)
            y = int(center_y + math.sin(angle) * ring * 0.78)
            radius = random.choice((5, 6, 8))
            nodes.append((x, y, radius, colors[(index + count) % len(colors)]))

    for index, (x, y, _radius, _color) in enumerate(nodes):
        candidates = sorted(
            nodes[index + 1 :],
            key=lambda node: (node[0] - x) ** 2 + (node[1] - y) ** 2,
        )[:2]
        for x2, y2, _radius2, _color2 in candidates:
            if (x2 - x) ** 2 + (y2 - y) ** 2 < 44000:
                draw.line((x, y, x2, y2), fill="#B9C8CC", width=2)

    draw.ellipse((center_x - 64, center_y - 64, center_x + 64, center_y + 64), fill="#17252A")
    draw.text((center_x, center_y - 8), "AI", font=font(46), fill="#FFFFFF", anchor="mm")
    for x, y, radius, color in nodes:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    draw.text((108, 154), "AI前沿分享", font=font(72), fill="#17252A")
    draw.rectangle((108, 254, 194, 262), fill="#2A9D8F")
    draw.text((108, 300), "AI 论文解读 · Agent · RAG", font=font(32), fill="#264653")
    draw.text((108, 354), "推理优化 · 多模态 · 具身智能", font=font(32), fill="#264653")
    draw.text((108, 498), "agifrontier.github.io", font=font(23), fill="#5D7077")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG", optimize=True)


if __name__ == "__main__":
    main()
