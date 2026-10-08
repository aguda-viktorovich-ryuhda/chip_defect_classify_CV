# -*- coding: utf-8 -*-
"""mask.py — 现场生成板框掩膜（客户不提供任何掩膜 ✓）

来源：原样复用内置的 seg.module_mask（σ6 → Otsu → 闭运算25 → 开运算5 → 最大轮廓填充 ✓）
      —— 与 out/masks/ 里那些存档掩膜是**同一套代码** ✓ 所以现场算出来的与存档一致 ✓
      —— 再按 cfg 的 MASK_ERODE 走 seg.shape_mask ✓（焊点识别用腐蚀版 ✓ 与旧流程 MASK_ERODE=4 一致 ✓）

对外只暴露两个函数：
    read_gray(path)          读原图（灰度 ✓）
    board_mask(g, C)         现场生成板框掩膜 → (mask_bool, mids)
"""
import os
import sys

import cv2

_V = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "vendor")
if _V not in sys.path:
    sys.path.insert(0, _V)
import seg as _seg  # noqa: E402


def read_gray(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise IOError("读图失败（路径或格式 ✗）：" + path)
    return g


def board_mask(g, C=None):
    """现场生成板框掩膜。

    返回 (m4, mids)：
      m4   : bool 掩膜，已按 cfg.s1_candidates.MASK_ERODE 腐蚀（供焊点识别 ✓）
      mids : seg 的中间图 dict（可选保存，用于出中间图 ✓）
    掩膜来源若客户已有（cfg.io.mask_mode == "file"）则优先读客户给的目录 ✓（默认 auto = 现场算 ✓）
    """
    mode = (C or {}).get("io", {}).get("mask_mode", "auto")
    erode = int((C or {}).get("s1_candidates", {}).get("MASK_ERODE", 4))
    if mode == "file":
        d = (C or {}).get("io", {}).get("provided_mask_dir", "")
        if d and os.path.isdir(d):
            m = cv2.imread(os.path.join(d, "mask.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                return _seg.shape_mask(m > 0, erode=erode), {}
    m, _cnts, mids = _seg.module_mask(g)
    return _seg.shape_mask(m, erode=erode), mids
