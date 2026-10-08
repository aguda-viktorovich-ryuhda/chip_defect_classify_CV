#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_util.py  —  科研级图件绘制工具（标尺 / 网格 / 尺寸图例）
所有图件统一调用这里的函数，保证每张图都有：
  1. scale bar（比例尺）：黑白相间 4 段 + 刻度 + "N px" 标签，任何底色上都能看清
  2. 坐标网格 + 轴刻度数字
  3. size legend（尺寸图例）：画出 r=5px / r=15px 的参考圆，供肉眼比对
  4. 面板标题（含图名 / 放大倍率 / 源文件名 / 参数）
说明：JPEG 元数据标称 300dpi，但这是拍照件、物面尺度未知，
      故标尺一律用像素；若后续给出实物尺寸，可换算为 mm 标尺。
"""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.6, fg=(255, 255, 255), bg=(0, 0, 0), th=2):
    cv2.putText(img, s, org, FONT, scale, bg, th + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, fg, th, cv2.LINE_AA)


def scalebar(img, x, y, px, label=None, h=12, scale=0.6):
    """黑白相间比例尺。px 为图上长度（图像素），label 默认 "<px> px"。"""
    bar = img[y:y + h, x:x + px]
    if bar.shape[0] != h or bar.shape[1] != px:
        return
    seg = max(1, px // 4)
    for i in range(4):
        x0 = i * seg
        x1 = px if i == 3 else (i + 1) * seg
        bar[:, x0:x1] = (255, 255, 255) if i % 2 == 0 else (0, 0, 0)
    cv2.rectangle(img, (x - 2, y - 2), (x + px + 1, y + h + 1), (0, 0, 0), 1)
    cv2.rectangle(img, (x - 1, y - 1), (x + px, y + h), (255, 255, 255), 1)
    for t in (0, px):
        cv2.line(img, (x + t, y - 5), (x + t, y + h + 5), (0, 0, 0), 3)
        cv2.line(img, (x + t, y - 5), (x + t, y + h + 5), (255, 255, 255), 1)
    txt = label if label else "{} px".format(px)
    _text(img, txt, (x, y - 10), scale)


def size_legend(img, x, y, radii=(5, 15), scale=0.55):
    """尺寸图例：画出若干参考半径的圆，供肉眼比对标尺"""
    for i, r in enumerate(radii):
        cx = x + i * 110
        cy = y
        cv2.circle(img, (cx, cy), r, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r, (0, 255, 255), 1, cv2.LINE_AA)
        _text(img, "r={}px d={}px".format(r, 2 * r), (cx - r - 6, cy + r + 20), scale)


def grid(img, step=100, color=(170, 110, 0), axis_labels=True, scale=0.45):
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h - 1), color, 1)
        if axis_labels:
            _text(img, str(x), (x + 3, 14), scale)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w - 1, y), color, 1)
        if axis_labels:
            _text(img, str(y), (3, y + 14), scale)


def panel_title(img, title, sub=None, scale=0.7):
    _text(img, title, (8, 26), scale, (0, 255, 255))
    if sub:
        _text(img, sub, (8, 52), scale * 0.85, (0, 255, 0))


def footer(img, text, scale=0.6):
    h = img.shape[0]
    _text(img, text, (8, h - 12), scale, (0, 255, 0))


def border(img, pad=10, color=(40, 40, 40)):
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=color)
