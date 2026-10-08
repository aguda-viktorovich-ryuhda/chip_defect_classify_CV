# -*- coding: utf-8 -*-
"""s259_stage - 自 recon/259_stage_figs.py 整份复制（仅移除绝对路径常量，其余原样）"""
import glob
import json
import os
import re
import sys
import time
import math
import itertools

import cv2
import numpy as np

_VENDOR = os.path.dirname(os.path.abspath(__file__))          # core/vendor/s5_steps
_COREV = os.path.dirname(_VENDOR)                           # core/vendor
for _p in (_COREV, _VENDOR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""259_stage_figs.py — 两阶段对照图 ✓：「直度判定前」/「直度判定后」各一张 10 板图

使用者（2026-10-02）："正常焊接的名单还是缺，不知道是之前统计焊点个数的时候就已经乱还是剪骨架的时候发生了污染。
                     你出两部图，一部在直度判定流程前，另一幅在后。"
数据来源：unet/pred_masks_all/（每条连通域掩膜 ✓）+ unet/reports/wire_reasons.jsonl（每条连通域的类别 ✓）
          + unet/reports/straightness_skeleton.json（哪条进了直度判定 ✓ 及其线段掩膜 pred_masks_wires/ ✓）
图 A（判定前）：每条连通域按【统计到的焊点数】着色 ✓ + 编号 ✓
             绿 = 2 个焊点（一内一外 ✓）｜ 橙 = 1 个焊点 ｜ 灰 = 0 个 ｜ 品红 = ≥3 个 ｜ 深蓝小点 = 焊点 ✓
图 B（判定后）：只保留【进了判定】的连通域（绿 ✓）+ 其判定用骨架（黄 ✓）+ 编号 = 直度编号 ✓
             【合格但没进判定】的连通域画成 青（cyan ✓）并保留原编号 ✗ —— 一眼看出名单在哪一步少了谁 ✓
排布：沿用既有交付口径 ✓ 10 板 2×5 ✓ 板名在图外 ✓ 1:1 无留白 ✓ PNG 无损 ✓
"""


# ---- 软件内默认路径（运行时由 core/s5_straightness.py 覆盖）----
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 软件根 chip_defect_classify/
SRC = os.path.join(HERE, "chip_samples__20260128")   # 输入（运行时由驱动器覆盖为 --input）
UN  = os.path.join(HERE, "outputs", "_masks")        # 掩膜目录（运行时覆盖）
PAN = os.path.join(HERE, "outputs")                  # 出图目录（运行时覆盖）
SOLD = os.path.join(HERE, "refs", "solder_table.json")
BOARD_FILE = os.path.join(HERE, "refs", "boards.json")
BOARDS = []
KEPT = ["TTA", "v13", "v18"]
ALL = {k: k for k in KEPT}
PY = sys.executable
ROOT = os.path.join(HERE, "refs", "annot")      # 板子清单驱动器（*_m.png 枚举；无标注时由 run_pipeline 生成占位）
ANN  = ROOT
MASKDIR = os.path.join(HERE, "outputs", "_masks")
OUT = PAN


# [搬运时移除的绝对路径常量] SRC = "<路径>"
# [搬运时移除的绝对路径常量] UN = "<项目根>/chip_wire_ng/unet"
# [搬运时移除的绝对路径常量] PAN = "<项目根>/chip_wire_ng/panels"
# [搬运时移除的绝对路径常量] SOLD = "<项目根>/chip_wire_ng/out/cnn/solder_final_20261001_v2.json"
BOARDS = ["1_NG/1649", "1_NG/1658", "1_NG/1705", "1_OK/1507", "1_OK/1528",
          "1_OK/1535", "2_NG/3414", "2_NG/3421", "2_OK/2", "2_OK/3"]


def find_src(grp, key):
    for f in sorted(os.listdir(os.path.join(SRC, grp))):
        if f.endswith(".jpg") and key.split("/")[-1] in f:
            return cv2.imread(os.path.join(SRC, grp, f), cv2.IMREAD_GRAYSCALE)
    return None


def skel_of(sub_u8):
    sk = cv2.ximgproc.thinning(np.ascontiguousarray(sub_u8))
    return (sk > 0).astype(np.uint8)


def main():
    rows = [json.loads(l) for l in open(os.path.join(UN, "reports/wire_reasons.jsonl"), encoding="utf-8")]
    rd = json.load(open(os.path.join(UN, "reports/straightness_skeleton.json"), encoding="utf-8"))
    pts = {}
    for e in json.load(open(SOLD, encoding="utf-8"))["rows"]:
        pts.setdefault(e["sample"], []).append((float(e["x"]), float(e["y"])))
    A, B, stat = [], [], {}
    for key in BOARDS:
        grp = key.split("/")[0]
        g = find_src(grp, key)
        if g is None:
            continue
        va, vb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        P = pts.get(key.replace("/", "-"), [])
        nA = nB = 0
        for r in [x for x in rows if x["board"] == key]:
            m = cv2.imread(os.path.join(UN, "pred_masks_all", r["file"]), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            sub = (m > 127)
            n = len(r["pts2"])
            col = (0, 200, 0) if n == 2 and len(set(r["rings"])) == 2 else \
                  ((0, 165, 255) if n == 1 else ((128, 128, 128) if n == 0 else (255, 0, 255)))
            va[sub] = (0.40 * va[sub] + 0.60 * np.array(col)).astype(np.uint8)
            ys, xs = np.where(sub)
            if len(ys) < 8:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            nA += 1
            cv2.rectangle(va, (cx + 4, cy - 22), (cx + 14 + 11 * len(str(nA)), cy - 3), (20, 20, 20), -1)
            cv2.putText(va, str(nA), (cx + 7, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        for (px, py) in P:
            cv2.circle(va, (int(round(px)), int(round(py))), 2, (120, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(vb, (int(round(px)), int(round(py))), 2, (120, 0, 0), -1, cv2.LINE_AA)
        for r in [x for x in rd if x["board"] == key]:      # 判定后的：只画进了判定的 ✓
            m = cv2.imread(os.path.join(UN, "pred_masks_wires", r["file"]), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            sub = (m > 127)
            vb[sub] = (0.42 * vb[sub] + 0.58 * np.array([0, 0, 255])).astype(np.uint8)
            ys, xs = np.where(sub)
            y0, y1 = max(0, ys.min() - 4), min(sub.shape[0], ys.max() + 5)
            x0, x1 = max(0, xs.min() - 4), min(sub.shape[1], xs.max() + 5)
            sk = skel_of((sub[y0:y1, x0:x1].astype(np.uint8) * 255))
            node = np.zeros(sub.shape, np.uint8)
            node[y0:y1, x0:x1] = sk
            for (px, py) in P:
                cv2.circle(node, (int(round(px)), int(round(py))), 10, 0, -1)
            n_s, lab_s, st_s, _c = cv2.connectedComponentsWithStats(node, 8)
            if n_s > 1:
                big = 1 + int(np.argmax(st_s[1:, 4]))
                node = (lab_s == big).astype(np.uint8)
            ys2, xs2 = np.where(node > 0)
            if len(ys2) < 8:
                continue
            vb[ys2, xs2] = (60, 255, 255)
            nB += 1
            cx, cy = int(xs2.mean()), int(ys2.mean())
            txt = r["file"].split("_")[-1].replace(".png", "").lstrip("0") or "0"
            cv2.rectangle(vb, (cx + 4, cy - 22), (cx + 15 + 11 * len(txt), cy - 3), (20, 20, 20), -1)
            cv2.putText(vb, txt, (cx + 7, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        # 合格但没进判定的（对比名单 ✓）
        rd_files = {r["file"] for r in rd if r["board"] == key}
        miss = 0
        for r in [x for x in rows if x["board"] == key and len(x["pts2"]) == 2 and None not in x["rings"]
                  and len(set(x["rings"])) == 2]:
            m = cv2.imread(os.path.join(UN, "pred_masks_all", r["file"]), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            sub = (m > 127)
            ys, xs = np.where(sub)
            if len(ys) < 8:
                continue
            y0, y1 = max(0, ys.min() - 4), min(sub.shape[0], ys.max() + 5)
            x0, x1 = max(0, xs.min() - 4), min(sub.shape[1], xs.max() + 5)
            sk = skel_of((sub[y0:y1, x0:x1].astype(np.uint8) * 255))
            node = np.zeros(sub.shape, np.uint8)
            node[y0:y1, x0:x1] = sk
            for (px, py) in P:
                cv2.circle(node, (int(round(px)), int(round(py))), 10, 0, -1)
            n_s, lab_s, st_s, _c = cv2.connectedComponentsWithStats(node, 8)
            if n_s > 1:
                big = 1 + int(np.argmax(st_s[1:, 4]))
                node = (lab_s == big).astype(np.uint8)
            if r.get("cid") in {x.get("cid") for x in rd if x["board"] == key}:
                continue                     # ← 按【稳定编号 cid】判断是否进了判定 ✓（原来按骨架像素数猜 ✗ 会误判 ✓）
            miss += 1
            vb[sub] = (0.40 * vb[sub] + 0.60 * np.array([255, 255, 0])).astype(np.uint8)   # 青 = 合格但没进判定 ✗
        stat[key] = (nA, nB, miss)
        A.append((key, va))
        B.append((key, vb))
    H = max(c[1].shape[0] for c in A)
    W = max(c[1].shape[1] for c in A)
    for tag, cells in [("A_直度判定前", A), ("B_直度判定后", B)]:
        pg = np.full((44 + 5 * (H + 46), 8 + 2 * (W + 8), 3), 18, np.uint8)
        cv2.putText(pg, {"A_直度判定前": "STAGE A (before straightness): comps colored by #solder pts  green=2(inner+outer) orange=1 gray=0 magenta>=3 | blue dots=solder pts",
                         "B_直度判定后": "STAGE B (after straightness): only judged comps kept  yellow=skeleton used, red=its mask, CYAN=qualified-but-not-judged | numbers = straightness ids"}[tag],
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 225, 60), 2, cv2.LINE_AA)
        for i, (key, vis) in enumerate(cells):
            r_, c_ = divmod(i, 2)
            oy, ox = 40 + r_ * (H + 46), 8 + c_ * (W + 8)
            a, b, ms = stat[key]
            cv2.putText(pg, "{}   comps {}   judged {}   qualified-not-judged {}".format(key, a, b, ms),
                        (ox + 4, oy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
            pg[oy + 38:oy + 38 + vis.shape[0], ox + 4:ox + 4 + vis.shape[1]] = vis
        od = os.path.join(PAN, "state", "stage"); os.makedirs(od, exist_ok=True)
        fn = os.path.join(od, "两阶段_{}.png".format(tag))
        cv2.imwrite(fn, pg)
        print("图 -> {}  {:.2f}MB  {}×{}".format(fn, os.path.getsize(fn) / 1048576, pg.shape[1], pg.shape[0]))
    print("\n逐板：连通域 / 进判定 / 合格但没进判定")
    for k, (a, b, m) in stat.items():
        print("  {:<12} {:>4} {:>4} {:>4}{}".format(k, a, b, m, "   ← 有丢 ✗" if m else ""))
    return 0


def _orig_main_guard():  # 保留原入口不动
    pass
if False:
    raise SystemExit(main())