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
"""184_trainset_points_on_images.py — 统计训练集 + 把训练集全部点标注到原图上（10 板 ✓）

使用者（2026-10-05）："标签2在很久以前就和标签1合并，现在废止。你先统计训练集，
                     把训练集的点全部在原图上进行标注，我们检查缺哪个。"
数据源：重新分类后的目录（只读 ✓）SharedData/chip_wire_ng/weak_new_20261005/<样品>/{0,1,2}/
口径：**标签 2 并入 1**（废止 2 ✓）→ 图上只有两类：绿圈 = 焊点（1 ∪ 2）✓；灰点 = 非焊点（0）✓
排布：沿用既有交付口径 ✓ 10 板 2×5 ✓ 每格 1:1（不缩放不裁切 ✓）✓ 板名与计数在图外 ✓ PNG 无损 ✓
"""
import glob
import json
import os
import re

import cv2
import numpy as np

SRC = (_SRC + r"")
NEW = (_PROJ + r"/weak_new_20261005")
PAN = (_PROJ + r"/panels")
RE = re.compile(r"_(\d{1,4})_(\d{1,4})\.png$")
BOARDS = [("1_NG", "1649"), ("1_NG", "1658"), ("1_NG", "1705"), ("1_OK", "1507"), ("1_OK", "1528"),
          ("1_OK", "1535"), ("2_NG", "3414"), ("2_NG", "3421"), ("2_OK", "2"), ("2_OK", "3")]


def main():
    pts = {}
    stat = []
    for g, b in BOARDS:
        key = g + "/" + b
        per = {0: [], 1: []}
        for cls in ("0", "1", "2"):
            for f in glob.glob(os.path.join(NEW, g + "-" + b, cls, "*.png")):   # ← 只用一个精确目录名 ✓（原来两条 glob 会重复匹配 → 计数翻倍 ✗）
                mt = RE.search(os.path.basename(f))
                if not mt:
                    continue
                x, y = int(mt.group(1)), int(mt.group(2))
                per[0 if cls == "0" else 1].append((x, y))       # ← 2 并入 1 ✓
        pts[key] = per
        stat.append((key, len(per[0]), len(per[1])))
        print("  {:<11} 非焊点(0) {:>4} ｜ 焊点(1+2) {:>4} ｜ 合计 {:>4}".format(key, len(per[0]), len(per[1]),
                                                                                len(per[0]) + len(per[1])))
    print("  合计：非焊点 {} ｜ 焊点 {} ｜ 总 {}".format(sum(s[1] for s in stat), sum(s[2] for s in stat),
                                                      sum(s[1] + s[2] for s in stat)))
    cells = []
    for (g, b) in BOARDS:
        key = g + "/" + b
        img = None
        for f in sorted(os.listdir(os.path.join(SRC, g))):
            if f.endswith(".jpg") and b in f:
                img = cv2.imread(os.path.join(SRC, g, f), cv2.IMREAD_GRAYSCALE)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for (x, y) in pts[key][0]:
            cv2.circle(vis, (x, y), 1, (160, 160, 160), -1, cv2.LINE_AA)
        for (x, y) in pts[key][1]:
            cv2.circle(vis, (x, y), 7, (0, 220, 0), 2, cv2.LINE_AA)
        cells.append((key, len(pts[key][0]), len(pts[key][1]), vis))
    Hh = max(c[3].shape[0] for c in cells)
    Ww = max(c[3].shape[1] for c in cells)
    pg = np.full((46 + 5 * (Hh + 46), 8 + 2 * (Ww + 8), 3), 18, np.uint8)
    cv2.putText(pg, "TRAINSET POINTS on original: GREEN ring = solder point (label 1, label 2 merged into 1) | gray dot = non-solder (label 0) | check which points are MISSING",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 225, 60), 2, cv2.LINE_AA)
    for i, (key, n0, n1, vis) in enumerate(cells):
        r_, c_ = divmod(i, 2)
        oy, ox = 40 + r_ * (Hh + 46), 8 + c_ * (Ww + 8)
        cv2.putText(pg, "{}   0={}   1(+2)={}   total={}".format(key, n0, n1, n0 + n1), (ox + 4, oy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, (235, 235, 235), 2, cv2.LINE_AA)
        pg[oy + 38:oy + 38 + vis.shape[0], ox + 4:ox + 4 + vis.shape[1]] = vis
    od = os.path.join(PAN, "state", "trainset2")
    os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, "10板_训练集点_原图标注.png")
    cv2.imwrite(fn, pg)
    print("\n图 -> {}  {:.2f}MB  {}×{}".format(fn, os.path.getsize(fn) / 1048576, pg.shape[1], pg.shape[0]))
    json.dump({k: {"neg": v[0], "pos": v[1]} for k, v in pts.items()},
              open(_os.path.join(O, "out/cnn/trainset_points_new.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
