#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seg.py  —  分阶段可调用的分割模块（中间数据）

设计（按使用者要求）：
  · 掩码作为独立中间数据保存到磁盘，任何阶段可直接读取，不必重算；
  · 提供 erode / dilate 参数，供不同任务取不同形态的掩码：
      - 识别焊点：先腐蚀掩码，把模块边缘的沟槽排除掉再分析
      - 识别焊座：用未腐蚀的掩码划定分析区域，保留焊座形状完整性
  · 分割预处理与检测预处理完全分离（σ6→Otsu→闭运算25→开运算5→findContours→填充）

数据布局：
  out/masks/<序号>_<组名>__mask.png    掩码（255=模块内）
  out/masks/manifest.json              索引：原图相对路径 → 掩码文件/覆盖率/参数
"""
import json
import os
import cv2
import numpy as np

DEFAULTS = dict(sigma=6.0, kclose=25, kopen=5, area_frac=0.10)


def module_mask(g, sigma=DEFAULTS["sigma"], kclose=DEFAULTS["kclose"],
                kopen=DEFAULTS["kopen"], area_frac=DEFAULTS["area_frac"]):
    """返回 (mask_bool, 轮廓列表, 中间图 dict)"""
    gb = cv2.GaussianBlur(g, (0, 0), sigma)
    t, b = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b2 = cv2.morphologyEx(b, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kclose, kclose)))
    b3 = cv2.morphologyEx(b2, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kopen, kopen)))
    cnts, _ = cv2.findContours(b3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    m = np.zeros(g.shape, np.uint8)
    used = 0
    if cnts:
        amax = cv2.contourArea(cnts[0])
        for c in cnts:
            if cv2.contourArea(c) >= area_frac * amax:
                cv2.drawContours(m, [c], -1, 255, -1)
                used += 1
    inter = dict(blur=gb, bina=b, close=b2, open=b3, otsu=t, nused=used)
    return (m > 0), cnts, inter


def disk(r):
    r = int(r)
    if r <= 0:
        r = 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def shape_mask(mask, erode=0, dilate=0):
    """按阶段需要取掩膜形态：erode 用于排除边缘沟槽；dilate 用于放宽区域。

       ⚠ 必须显式传 borderValue=0（图像外=背景）：
       cv2.erode/dilate 的默认形态学边界值是 DBL_MAX → 对 8U 饱和成 255，即"图像外=前景"。
       掩膜一旦贴到图像边缘（1_ 组六张贴上边、2_ 组四张贴右边，上排焊座被画面裁掉），
       腐蚀就完全吃不到那一边 —— 实测内圈掩膜（腐蚀 94px）在顶边保持原样，
       于是"色块=内圈"与"dtin=21 判外圈"自相矛盾。传 0 之后两边行为一致。"""
    m = (mask.astype(np.uint8) * 255)
    if erode > 0:
        m = cv2.erode(m, disk(erode), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if dilate > 0:
        m = cv2.dilate(m, disk(dilate), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return m > 0


def build_and_save(src_root, rels, out_dir, **kw):
    """为一批图计算掩码并存盘，返回 manifest"""
    os.makedirs(out_dir, exist_ok=True)
    p = dict(DEFAULTS)
    p.update(kw)
    mani_path = os.path.join(out_dir, "manifest.json")
    mani = {}
    if os.path.exists(mani_path):
        try:
            mani = json.load(open(mani_path, encoding="utf-8"))
        except Exception:
            mani = {}
    for i, rel in enumerate(rels, start=1):
        g = cv2.imread(os.path.join(src_root, rel), cv2.IMREAD_GRAYSCALE)
        mask, cnts, inter = module_mask(g, **p)
        tag = "{:02d}_{}".format(i, os.path.splitext(os.path.basename(rel))[0][:24])
        tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in tag)
        fn = tag + "__mask.png"
        cv2.imwrite(os.path.join(out_dir, fn), mask.astype(np.uint8) * 255)
        mani[rel] = dict(file=fn, coverage=float(mask.mean()), contours=inter["nused"],
                         otsu=float(inter["otsu"]), params=dict(p))
    json.dump(mani, open(mani_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return mani


def load_mask(out_dir, rel):
    mani = json.load(open(os.path.join(out_dir, "manifest.json"), encoding="utf-8"))
    return cv2.imread(os.path.join(out_dir, mani[rel]["file"]), cv2.IMREAD_GRAYSCALE) > 0
