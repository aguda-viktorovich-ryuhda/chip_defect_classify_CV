# -*- coding: utf-8 -*-
"""s256_panels - 自 recon/256_wire_panels.py 整份复制（仅移除绝对路径常量，其余原样）"""
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
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import core.s_paths as _SP2

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
"""256_wire_panels.py — 逐条飞线面板图（一条线一格 ✓ 放大 ✓ 只画这一条 ✓ 文字在图外 ✓）

使用者（2026-10-02）："不要看是不是NG或者OK，标签不是你来判定的。出入给我看是什么样的。"
做法：
  · 从 unet/reports/straightness_skeleton.json 取每条合格飞线的 6 个指标读数 ✓（不写 OK/NG 标签 ✗）
  · 按【抖动】从大到小取前 12 条 ✓（也带上弧弦比最大的若干条 ✓ 去重 ✓）
  · 每条一格：裁剪到该线段外接框 + 40px 边距 ✓，INTER_NEAREST 放大 3× ✓
    底色 = 原图（不压暗 ✓）；红 = 该条线段的掩膜 ✓；深蓝小点 = 焊点 ✓
    图外条写：板名 + 抖动/弧弦比/曲率P95/最小曲率半径/低频能量/弯曲能量/长度 + 50px 比例尺 ✓
产出：panels/state/wires/线段面板_前12条.png（PNG 无损 ✓）
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
BOARD_FILE = {
    # 示例占位（发布版）：键=板名，值=(组名, 原始文件名主干)。请按你的数据集填写。
    "GROUP/0001": ("GROUP", "raw_name_0001"),
    "GROUP/0002": ("GROUP", "raw_name_0002"),
}
Z, PAD = 3, 40


def main():
    rows = json.load(open(os.path.join(UN, "reports/straightness_skeleton.json"), encoding="utf-8"))
    rows = [r for r in rows if "file" in r]
    pts = {}
    for e in json.load(open(SOLD, encoding="utf-8"))["rows"]:
        pts.setdefault(e["sample"], []).append((float(e["x"]), float(e["y"])))
    pick = sorted(rows, key=lambda r: -r["jit_max"])[:8] + sorted(rows, key=lambda r: -r["arc_chord"])[:8]
    seen, sel = set(), []
    for r in pick:
        if r["file"] in seen: continue
        seen.add(r["file"]); sel.append(r)
    sel = sel[:12]
    print("入选 {} 条（按抖动前 8 + 弧弦比前 8 去重取 12 ✓）".format(len(sel)))
    cells = []
    for r in sel:
        grp, stem = BOARD_FILE[r["board"]]
        g = cv2.imread(os.path.join(SRC, grp, stem + ".jpg"), cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(os.path.join(UN, "pred_masks_wires", r["file"]), cv2.IMREAD_GRAYSCALE)
        m = m > 127
        ys, xs = np.where(m)
        y0, y1 = max(0, ys.min() - PAD), min(m.shape[0], ys.max() + PAD + 1)
        x0, x1 = max(0, xs.min() - PAD), min(m.shape[1], xs.max() + PAD + 1)
        vis = cv2.cvtColor(g[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
        sub = m[y0:y1, x0:x1]
        vis[sub] = (0.35 * vis[sub] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
        sk = cv2.ximgproc.thinning(np.ascontiguousarray((sub.astype(np.uint8) * 255)))
        # 剪枝 ✓（与 255 同一规则：剪掉从端头到分叉点、长度 ≤15px 的短支 ✓）
        sk = (sk > 0).astype(np.uint8)
        kk_nb = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], np.uint8)
        for _ in range(15):
            nb = cv2.filter2D(sk, -1, kk_nb, borderType=cv2.BORDER_CONSTANT)
            eys_, exs_ = np.where((sk > 0) & (nb == 11))
            again = False
            for (ex_, ey_) in zip(exs_.tolist(), eys_.tolist()):
                if sk[ey_, ex_] == 0:
                    continue
                br, cur, hit_j = [(ex_, ey_)], (ex_, ey_), False
                for _ in range(16):
                    if int(nb[cur[1], cur[0]]) >= 12:
                        hit_j = True
                        break
                    cand = [(cur[0] + dx, cur[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                            if (dx or dy) and sk[cur[1] + dy, cur[0] + dx] > 0 and (cur[0] + dx, cur[1] + dy) not in br]
                    if not cand:
                        break
                    cur = cand[0]
                    br.append(cur)
                if hit_j and len(br) <= 15:
                    for (bx, by) in br:
                        if int(nb[by, bx]) < 12:
                            sk[by, bx] = 0
                    again = True
            if not again:
                break
        # 骨架层剪掉"距焊点 R 以内"的部分 ✓ + 只留最长一段 ✓（与 255 同一规则 ✓）
        for (px2, py2) in pts.get(grp + "-" + r["board"].split("/")[-1], []):
            cv2.circle(sk, (int(round(px2)) - x0, int(round(py2)) - y0), 10, 0, -1)
        n_sk, lab_sk, st_sk, _c_sk = cv2.connectedComponentsWithStats(sk.astype(np.uint8), 8)
        if n_sk > 1:
            big_sk = 1 + int(np.argmax(st_sk[1:, 4]))
            sk = (lab_sk == big_sk).astype(np.uint8)
        vis[sk > 0] = (60, 255, 255)            # 黄 = 骨架（已剪枝 + 已剪掉端头 ✓）
        name = r["board"].split("/")[-1]
        for (px, py) in pts.get(grp + "-" + name, []):
            if x0 <= px < x1 and y0 <= py < y1:
                cv2.circle(vis, (int(round(px)) - x0, int(round(py)) - y0), 2, (120, 0, 0), -1, cv2.LINE_AA)
        big = cv2.resize(vis, ((x1 - x0) * Z, (y1 - y0) * Z), interpolation=cv2.INTER_NEAREST)
        cells.append((r, big))
    F = ImageFont.truetype(_SP2.font_path(), 20)
    FB = ImageFont.truetype(_SP2.font_path(), 24)
    cw = max(c[1].shape[1] for c in cells) + 16
    rh = max(c[1].shape[0] for c in cells) + 96
    cols, rows_n = 3, int(np.ceil(len(cells) / 3))
    pg = Image.new("RGB", (16 + cols * cw, 96 + rows_n * rh), (18, 18, 18))
    d = ImageDraw.Draw(pg)
    d.text((16, 10), "飞线线段面板（一条线一格 ✓ 只画这一条 ✓ 3× 最近邻放大 ✓；按抖动/弧弦比降序取 12 条 ✓）", font=FB, fill=(255, 225, 60))
    d.text((16, 44), "红 = 该条线段掩膜 ✓　黄 = 骨架（细化结果 ✓）　深蓝小点 = 焊点 ✓　底色 = 原图（不压暗 ✓）　文字与比例尺均在图外 ✓", font=F, fill=(180, 255, 180))
    for i, (r, big) in enumerate(cells):
        cr, cc = divmod(i, cols)
        ox, oy = 16 + cc * cw, 88 + cr * rh
        d.text((ox, oy), "{}  #{}".format(r["board"], r["file"].split("_")[-1].replace(".png", "")), font=FB, fill=(235, 235, 235))
        d.text((ox, oy + 26), "抖动 {:.2f}px ｜ 弧弦比 {:.4f} ｜ 曲率P95 {:.4f} ｜ 最小曲率半径 {:.1f}px".format(
            r["jit_max"], r["arc_chord"], r["kappa_p95"], r["R_min"]), font=F, fill=(200, 255, 200))
        d.text((ox, oy + 50), "低频能量 {:.3f} ｜ 弯曲能量 {:.3f} ｜ 线长 {}px".format(
            r["low_energy"], r["bend_energy"], r["len_px"]), font=F, fill=(200, 255, 200))
        pg.paste(Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB)), (ox, oy + 76))
    od = os.path.join(PAN, "state", "wires"); os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, "线段面板_前12条.png")
    pg.save(fn, "PNG", optimize=True)
    print("图 -> {}  {:.2f}MB  {}×{}".format(fn, os.path.getsize(fn) / 1048576, pg.size[0], pg.size[1]))
    return 0


def _orig_main_guard():  # 保留原入口不动
    pass
if False:
    raise SystemExit(main())
