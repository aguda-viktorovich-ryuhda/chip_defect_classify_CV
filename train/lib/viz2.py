#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz2.py  —  科研级图件绘制工具 v2（推荐使用）

相对 v1 的修正（v1 的排版缺陷会误导使用者）：
  · 坐标轴刻度数字原来画在图内，被误读成"候选编号"。
    v2 先把图像放进一个带留白的外框（margin），网格线画在图内，
    刻度数字、比例尺、尺寸图例、标题全部画在图外留白区。
  · 编号改用"实心底+白字"徽标，形如 #12，不会与刻度数字混淆。
所有长度单位均为图像素（px）。JPEG 标称 300dpi 但不可尽信，故图上不直接标 mm。
"""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.6, fg=(255, 255, 255), bg=(0, 0, 0), th=2):
    cv2.putText(img, s, org, FONT, scale, bg, th + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, fg, th, cv2.LINE_AA)


def frame(img, pad_l=66, pad_t=72, pad_b=112, pad_r=30, bg=(38, 38, 38)):
    """把图放进带留白外框。返回 (canvas, (ox, oy))，ox/oy = 图像在 canvas 中的左上角。"""
    h, w = img.shape[:2]
    canvas = np.full((h + pad_t + pad_b, w + pad_l + pad_r, 3), bg, np.uint8)
    canvas[pad_t:pad_t + h, pad_l:pad_l + w] = img
    return canvas, (pad_l, pad_t)


def axis_grid(canvas, off, size, step=100, color=(110, 110, 110), scale=0.5):
    """网格画在图内；刻度数字画在图外留白区（x 在左侧、y 在上方）。"""
    ox, oy = off
    w, h = size
    for x in range(0, w + 1, step):
        cv2.line(canvas, (ox + x, oy), (ox + x, oy + h - 1), color, 1)
        _text(canvas, str(x), (ox + x - 14, oy - 14), scale, (205, 205, 205))
    for y in range(0, h + 1, step):
        cv2.line(canvas, (ox, oy + y), (ox + w - 1, oy + y), color, 1)
        _text(canvas, str(y), (ox - 58, oy + y + 5), scale, (205, 205, 205))


def scalebar(canvas, x, y, px, label=None, h=14, scale=0.6):
    """黑白相间四段比例尺 + 两端刻度 + 标签。"""
    seg = max(1, px // 4)
    for i in range(4):
        x0 = x + i * seg
        x1 = x + px if i == 3 else x + (i + 1) * seg
        cv2.rectangle(canvas, (x0, y), (x1 - 1, y + h - 1),
                      (255, 255, 255) if i % 2 == 0 else (0, 0, 0), -1)
    cv2.rectangle(canvas, (x - 2, y - 2), (x + px + 1, y + h + 1), (255, 255, 255), 1)
    for t in (0, px):
        cv2.line(canvas, (x + t, y - 7), (x + t, y + h + 7), (0, 255, 255), 2)
    _text(canvas, label if label else "{} px".format(px), (x, y - 24), scale, (0, 255, 255))


def size_legend(canvas, x, y, radii=(3, 5, 8), scale=0.5):
    """尺寸图例：参考半径的圆（黑描边 + 绿内圈），供肉眼比对比例尺。"""
    for i, r in enumerate(radii):
        cx = x + i * 130
        cv2.circle(canvas, (cx, y), r, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.circle(canvas, (cx, y), r, (0, 255, 0), 2, cv2.LINE_AA)
        _text(canvas, "r={} d={}".format(r, 2 * r), (cx - 34, y + 36), scale, (170, 255, 170))


def badge(canvas, idx, xy, color=(255, 0, 255), scale=0.55, prefix="#"):
    """编号徽标：实心底 + 白字，不与刻度数字混淆。"""
    s = "{}{}".format(prefix, idx)
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, 2)
    x, y = int(xy[0]), int(xy[1])
    cv2.rectangle(canvas, (x - 2, y - th - 4), (x + tw + 2, y + 4), color, -1)
    cv2.putText(canvas, s, (x, y), FONT, scale, (255, 255, 255), 2, cv2.LINE_AA)


def panel_title(canvas, title, sub=None, scale=0.75, x=12, y=30):
    _text(canvas, title, (x, y), scale, (0, 255, 255))
    if sub:
        _text(canvas, sub, (x, y + 30), scale * 0.8, (0, 255, 0))


def marker(canvas, xy, r, color=(255, 0, 255), th=2):
    cv2.circle(canvas, (int(xy[0]), int(xy[1])), int(r), color, th, cv2.LINE_AA)
    cv2.drawMarker(canvas, (int(xy[0]), int(xy[1])), (0, 255, 255),
                   cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
