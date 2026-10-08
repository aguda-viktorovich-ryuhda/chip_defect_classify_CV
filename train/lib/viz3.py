#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz3.py  —  图件工具 v3：支持中文（PIL + Noto Sans CJK）

v2 的致命缺陷：cv2.putText 不支持中文，所有中文标签渲染成 '???'，图件实际不可读。
v3 改用 PIL 渲染文字（字体：/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc），
其余（比例尺、网格、编号徽标、留白边框）沿用 v2 的设计。
"""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CJK_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
_FONTS = {}


def font(size=20, bold=False):
    key = (size, bold)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(CJK_B if bold else CJK, size)
    return _FONTS[key]


def text(img, s, xy, size=20, color=(255, 255, 255), bg=None, bold=False, pad=3):
    """在 BGR 数组上绘制中文/英文文本（就地修改）。xy 为文字左上角。"""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    f = font(size, bold)
    x, y = int(xy[0]), int(xy[1])
    if bg is not None:
        bbox = d.textbbox((x, y), s, font=f)
        d.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=bg)
    d.text((x, y), s, font=f, fill=(color[2], color[1], color[0]))
    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    np.copyto(img, out)
    return img


def frame(img, pad_l=20, pad_t=84, pad_b=104, pad_r=24, bg=(38, 38, 38)):
    h, w = img.shape[:2]
    canvas = np.full((h + pad_t + pad_b, w + pad_l + pad_r, 3), bg, np.uint8)
    canvas[pad_t:pad_t + h, pad_l:pad_l + w] = img
    return canvas, (pad_l, pad_t)


def axis_grid(canvas, off, size, step=100, color=(110, 110, 110), size_px=16):
    ox, oy = off
    w, h = size
    for x in range(0, w + 1, step):
        cv2.line(canvas, (ox + x, oy), (ox + x, oy + h - 1), color, 1)
        text(canvas, str(x), (ox + x - 12, oy - 20), size_px, (205, 205, 205))
    for y in range(0, h + 1, step):
        cv2.line(canvas, (ox, oy + y), (ox + w - 1, oy + y), color, 1)
        text(canvas, str(y), (ox + 4, oy - 20 if y == 0 else oy + y - 9), size_px,
             (205, 205, 205))


def scalebar(canvas, x, y, px, label=None, h=14, size=18):
    seg = max(1, px // 4)
    for i in range(4):
        x0 = x + i * seg
        x1 = x + px if i == 3 else x + (i + 1) * seg
        cv2.rectangle(canvas, (x0, y), (x1 - 1, y + h - 1),
                      (255, 255, 255) if i % 2 == 0 else (0, 0, 0), -1)
    cv2.rectangle(canvas, (x - 2, y - 2), (x + px + 1, y + h + 1), (255, 255, 255), 1)
    for t in (0, px):
        cv2.line(canvas, (x + t, y - 7), (x + t, y + h + 7), (0, 255, 255), 2)
    text(canvas, label if label else "{} px".format(px), (x, y - 26), size, (0, 255, 255))


def size_legend(canvas, x, y, radii=(5,), size=16):
    for i, r in enumerate(radii):
        cx = x + i * 150
        cv2.circle(canvas, (cx, y), r, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.circle(canvas, (cx, y), r, (0, 255, 0), 2, cv2.LINE_AA)
        text(canvas, "半径r={} 直径{}px".format(r, 2 * r), (cx - 40, y + 18), size,
             (170, 255, 170))
    return canvas


def title(canvas, s, sub=None, size=24, sub_size=18, x=12, y=10):
    text(canvas, s, (x, y), size, (0, 255, 255), bold=True)
    if sub:
        text(canvas, sub, (x, y + size + 8), sub_size, (120, 255, 120))
    return canvas


def tag(canvas, s, xy, color=(255, 0, 255), size=20, anchor_center=True):
    """带底色的标签，放在 xy 处"""
    x, y = int(xy[0]), int(xy[1])
    text(canvas, s, (x, y), size, (255, 255, 255), bg=tuple(int(c) for c in color))
    return canvas
