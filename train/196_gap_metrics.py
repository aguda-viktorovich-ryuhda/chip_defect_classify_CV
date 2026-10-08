import os as _os, sys as _sys
# [2026-10-08 适配] 去掉写死路径：按脚本位置定位，全部可用环境变量覆盖
_ROOT = _os.environ.get("CHIP_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CHIP_DATA") or _os.path.join(_ROOT, "train", "data")
_LIB  = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "lib")
_SRC  = _os.environ.get("CHIP_SRC")  or ""
_PROJ = _os.environ.get("CHIP_PROJ") or ""
_LAB  = _os.environ.get("CHIP_LAB")  or ""
_ANN  = _os.environ.get("CHIP_ANN") or ""
for _p in (_ROOT, _LIB):
    if _p not in _sys.path: _sys.path.insert(0, _p)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""196_gap_metrics.py — 量化预测掩膜的【断线】与【两端是否落在焊点上】

使用者 2026-09-29 判断：假阳不是主要矛盾；**最大问题是断线**；
若假阳去不掉，可用"**预测出的线两端是否有焊点**"来判它是不是真飞线。

指标（对 GT 与 U-Net 预测各算一遍，可直接对比）：
  ① 连通域个数与面积/长轴（长轴用 minAreaRect；≥40px 才当作"线状片段"）
  ② 每一段线状片段：两端点（PCA 极值点）到最近【焊点候选】的距离
     · 两端都 ≤20px  → 判为 完整飞线 ✓
     · 只有一端      → 判为 断线（半截）✗
     · 都不在        → 判为 游离片段 ✗
  ③ 断线率 = （片段数 − 完整飞线数）/ 片段数
焊点候选：优先用 SharedData 里的标签集（weak_new/dataset.json），缺失才回退 /data。
"""
import json
import os
import sys

import cv2
import numpy as np

SRC = (_SRC + r"")
ROOT = (_ANN + r"")
UN = (_PROJ + r"/unet")
ANN = {"1_NG-1649": "1_NG/微信图片_20240711151649", "1_NG-1658": "1_NG/微信图片_20240711151658",
       "1_NG-1705": "1_NG/微信图片_20240711151705", "2_NG-3414": "2_NG/微信图片_20240711153414",
       "2_NG-3421": "2_NG/微信图片_20240711153421"}
ALL = {"1_NG-1649": "1_NG/微信图片_20240711151649.jpg", "1_NG-1658": "1_NG/微信图片_20240711151658.jpg",
       "1_NG-1705": "1_NG/微信图片_20240711151705.jpg", "1_OK-1507": "1_OK/微信图片_20240711151507.jpg",
       "1_OK-1528": "1_OK/微信图片_20240711151528.jpg", "1_OK-1535": "1_OK/微信图片_20240711151535.jpg",
       "2_NG-3414": "2_NG/微信图片_20240711153414.jpg", "2_NG-3421": "2_NG/微信图片_20240711153421.jpg",
       "2_OK-2": "2_OK/2.jpg", "2_OK-3": "2_OK/3.jpg"}


def pads_of(s):
    for p in ((_PROJ + r"/weak_new/dataset.json"),
              _os.path.join(H, "out/cnn/dataset.json")):
        if os.path.exists(p):
            ds = json.load(open(p, encoding="utf-8"))
            return np.array([[c["x"], c["y"]] for c in ds if c["sample"] == s], np.float32), p
    return np.zeros((0, 2), np.float32), "none"


def segs(mask, min_len=40):
    n, lab, st, ce = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        area = int(st[i, 4])
        if area < 60:
            continue
        ys, xs = np.where(lab == i)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        c = pts.mean(0)
        u, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
        d = (pts - c) @ vt[0]
        L = float(d.max() - d.min())
        if L < min_len:
            continue
        p1 = c + vt[0] * d.min()
        p2 = c + vt[0] * d.max()
        out.append({"area": area, "len": L, "p1": p1, "p2": p2})
    return out


def report(tag, mask, pads):
    ss = segs(mask)
    both = one = none = 0
    gaps = []
    for sg in ss:
        d1 = float(np.min(np.linalg.norm(pads - sg["p1"], axis=1))) if len(pads) else 1e9
        d2 = float(np.min(np.linalg.norm(pads - sg["p2"], axis=1))) if len(pads) else 1e9
        k = (d1 <= 20) + (d2 <= 20)
        if k == 2:
            both += 1
        elif k == 1:
            one += 1; gaps.append(round(sg["len"]))
        else:
            none += 1; gaps.append(round(sg["len"]))
    return ss, both, one, none, gaps


def main():
    print("{:<10} {:<8} {:>6} {:>8} {:>7} {:>7} {:>7} {:>9}".format(
        "板", "来源", "片段数", "中位长px", "两端有", "半截", "游离", "断线率"))
    for s in ALL:
        pads, src = pads_of(s)
        rows = []
        if s in ANN:
            raw = cv2.imread(os.path.join(ROOT, ANN[s] + "_m.png"), cv2.IMREAD_UNCHANGED)
            gt = (raw if raw.ndim == 2 else cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2GRAY)) > 127
            rows.append(("人工GT", gt))
        pm = cv2.imread(os.path.join(UN, "pred_masks", s + ".png"), cv2.IMREAD_GRAYSCALE)
        if pm is not None:
            rows.append(("U-Net", pm > 127))
        for tag, mk in rows:
            ss, both, one, none, gaps = report(tag, mk, pads)
            tot = max(1, len(ss))
            rate = 100.0 * (one + none) / tot
            med = float(np.median([x["len"] for x in ss])) if ss else 0.0
            print("{:<10} {:<8} {:>6} {:>8.0f} {:>7} {:>7} {:>7} {:>8.1f}%".format(
                s, tag, len(ss), med, both, one, none, rate))
    print("\n焊点候选来源：{}".format(pads_of("1_OK-1507")[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
