# -*- coding: utf-8 -*-
"""patches.py — 候选点取块（自 recon/114_apply_cnn.py 第 44..55 行原样搬入 ✓ 48x48 ✓）

与训练/推理完全同一份取块代码 ✓ 否则分数对不上 ✗
"""
import cv2
import numpy as np

S = 48          # 取块边长（与训练/114 一致 ✓ 不许改 ✗）


def crop_at(g, px, py, s=S):
    r = s // 2
    ph, pw = g.shape
    y0, x0 = py - r, px - r
    top, left = max(0, -y0), max(0, -x0)
    bot, right = max(0, y0 + s - ph), max(0, x0 + s - pw)
    p = g[max(0, y0):min(ph, y0 + s), max(0, x0):min(pw, x0 + s)]
    if top or bot or left or right:
        p = cv2.copyMakeBorder(p, top, bot, left, right, cv2.BORDER_REPLICATE)
    return p
