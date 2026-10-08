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
"""182_trainset_check_sheet.py — 训练集标签检查图（10 板 ✓ 供人工核对 ✓）

用途：阶段二（CNN 判定）出问题后，使用者要求"让我检查训练集" → 出一张能看标签的图 ✓
图上（每板一格 ✓ 1:1 原分辨率 ✓ 2×5 ✓ 板名与计数在图外 ✓ PNG 无损 ✓）：
  绿圈 = 标签 1（正常焊点）✓
  青圈 = 标签 2（断联焊点）✓
  灰点 = 标签 0（非焊点，不画圈以免糊 ✓ 半径 1px ✓）
  红圈 = 【今天改过标签的点】✓（= 现行 dataset.json 与 dataset.json.bak_beforelabel1005_1248 的差集 ✓）
  洋红圈 = 弱反差分支新增候选（并集的 1327–1329 号 ✓ 坐标取自数据集 ✓）
图外每格写：板名 + 总数 / 0 / 1 / 2 / 今日改动数 ✓
"""
import json
import os

import cv2
import numpy as np

H = (_PROJ or _ROOT)
SRC = (_SRC + r"")
PAN = (_PROJ + r"/panels")
BOARDS = ["1_NG-1649", "1_NG-1658", "1_NG-1705", "1_OK-1507", "1_OK-1528",
          "1_OK-1535", "2_NG-3414", "2_NG-3421", "2_OK-2", "2_OK-3"]
GRP = {"1_NG-1649": "1_NG", "1_NG-1658": "1_NG", "1_NG-1705": "1_NG", "1_OK-1507": "1_OK", "1_OK-1528": "1_OK",
       "1_OK-1535": "1_OK", "2_NG-3414": "2_NG", "2_NG-3421": "2_NG", "2_OK-2": "2_OK", "2_OK-3": "2_OK"}
WEAK = [("1_NG-1658", 810, 160), ("1_NG-1705", 674, 23), ("1_NG-1705", 808, 162)]


def find_img(grp, base):
    for f in sorted(os.listdir(os.path.join(SRC, grp))):
        if f.endswith(".jpg") and base in f:
            return cv2.imread(os.path.join(SRC, grp, f), cv2.IMREAD_GRAYSCALE)
    return None


def main():
    cur = json.load(open(os.path.join(H, "out/cnn/dataset.json"), encoding="utf-8"))
    bak = json.load(open(os.path.join(H, "out/cnn/dataset.json.bak_beforelabel1005_1248"), encoding="utf-8"))
    def key(c):
        return (c["sample"], int(c["x"]), int(c["y"]))
    old = {key(c): int(c["label"]) for c in bak}
    changed = {key(c) for c in cur if key(c) in old and int(c["label"]) != old[key(c)]}
    changed |= {key(c) for c in cur if key(c) not in old}
    print("今天标签改动的点：{} 个".format(len(changed)))
    for k in sorted(changed):
        print("   {} ({},{})  旧 {} → 新 {}".format(k[0], k[1], k[2], old.get(k, "无"), dict((key(c), int(c["label"])) for c in cur)[k]))
    cells = []
    for b in BOARDS:
        g = find_img(GRP[b], b.split("-")[-1])
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        cnt = {0: 0, 1: 0, 2: 0}
        pts = [c for c in cur if c["sample"] == b]
        for c in pts:
            x, y, L = int(c["x"]), int(c["y"]), int(c["label"])
            cnt[L] = cnt.get(L, 0) + 1
            if L == 0:
                cv2.circle(vis, (x, y), 1, (150, 150, 150), -1, cv2.LINE_AA)
            elif L == 1:
                cv2.circle(vis, (x, y), 7, (0, 200, 0), 2, cv2.LINE_AA)
            else:
                cv2.circle(vis, (x, y), 7, (255, 255, 0), 2, cv2.LINE_AA)
        for c in pts:
            k = key(c)
            if k in changed:
                cv2.circle(vis, (int(c["x"]), int(c["y"])), 14, (0, 0, 255), 2, cv2.LINE_AA)
        for (s, x, y) in WEAK:
            if s == b:
                cv2.circle(vis, (x, y), 20, (255, 0, 255), 2, cv2.LINE_AA)
        nch = sum(1 for k in changed if k[0] == b)
        cells.append((b, len(pts), cnt[0], cnt[1], cnt[2], nch, vis))
    Hh = max(c[6].shape[0] for c in cells)
    Ww = max(c[6].shape[1] for c in cells)
    pg = np.full((44 + 5 * (Hh + 46), 8 + 2 * (Ww + 8), 3), 18, np.uint8)
    cv2.putText(pg, "TRAINSET LABELS: green ring = label 1 (solder) | cyan ring = label 2 (disconnected) | gray dot = label 0 (non-solder) | RED ring = changed today | magenta = weak-contrast new cands",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 225, 60), 2, cv2.LINE_AA)
    for i, (b, n, n0, n1, n2, nch, vis) in enumerate(cells):
        r_, c_ = divmod(i, 2)
        oy, ox = 40 + r_ * (Hh + 46), 8 + c_ * (Ww + 8)
        cv2.putText(pg, "{}   n={}  0={} 1={} 2={}  changed-today={}".format(b, n, n0, n1, n2, nch),
                    (ox + 4, oy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
        pg[oy + 38:oy + 38 + vis.shape[0], ox + 4:ox + 4 + vis.shape[1]] = vis
    od = os.path.join(PAN, "state", "trainset")
    os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, "10板_训练集标签检查.png")
    cv2.imwrite(fn, pg)
    print("\n图 -> {}  {:.2f}MB  {}×{}".format(fn, os.path.getsize(fn) / 1048576, pg.shape[1], pg.shape[0]))
    print("逐板：n / 0 / 1 / 2 / 今日改动")
    for (b, n, n0, n1, n2, nch, _v) in cells:
        print("  {:<11} {:>4} {:>5} {:>4} {:>3} {:>4}".format(b, n, n0, n1, n2, nch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
