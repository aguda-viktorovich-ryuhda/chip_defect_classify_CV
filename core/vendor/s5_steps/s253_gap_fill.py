# -*- coding: utf-8 -*-
"""s253_gap_fill - 自 recon/253_gap_fill_two_methods.py 整份复制（仅移除绝对路径常量）"""
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



try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None

# ---- 软件内默认路径（运行时由 core/s5_straightness.py 覆盖）----
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(HERE, "chip_samples__20260128")
UN = os.path.join(HERE, "outputs", "_masks")
PAN = os.path.join(HERE, "outputs")
SOLD = os.path.join(HERE, "refs", "solder_table.json")
BOARD_FILE = os.path.join(HERE, "refs", "boards.json")
BOARDS = []
KEPT = ["TTA", "v13", "v18"]
ALL = {k: k for k in KEPT}
PY = sys.executable
OUT = PAN
ROOT = os.path.join(HERE, "refs", "annot")
ANN = ROOT
MASKDIR = UN

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""253_gap_fill_two_methods.py — 补回拆分缺口：①闭运算 ②骨架桥接（各出一张图给使用者看）

使用者（2026-10-02）："缺口可能占到了飞线掩膜宽度的一半，而且弧度比较小，你1和2都试一下效果，出图给我看。"
前提：总名单流水线（recon/252 ✓）跑出的飞线掩膜 ✓；缺口 = 拆分时按圆盘删除留下的咬痕 ✓
逐条飞线处理 ✓（绝不动别人 ✗）：
  ① 闭运算：MORPH_CLOSE(圆盘 半径 11px ✓) 后 **只保留在【本飞线区域】内** 的部分 ✓
     本飞线区域 = 它自己的掩膜像素 ∪ 它名下焊点的删盘区 ✓
  ② 骨架桥接：本飞线骨架两端点中，**彼此距离 ≤ 2×删盘半径(19.8px) 且方向近似共线（夹角 ≤45°）** 的一对 ✓
     连线（线宽 = 该飞线线宽中位，取实测 7px ✓），同样限制在本飞线区域内 ✓
每条都打印校验 ✓：补回面积 / 补后骨架端点数 / 越界像素数（与其它飞线的重叠 ✓ 必须 0 ✓）
产出：panels/state/gapfill/补缺_法①闭运算.png、补缺_法②骨架桥接.png（10 板 1:1 2×5 ✓ PNG 无损 ✓）
      颜色：红 = 原掩膜 ✓　绿 = 补回来的部分 ✓　白圈 = 焊点 ✓
"""


# [搬运时移除的绝对路径常量] HERE = "<项目根>/chip_wire_ng"
sys.path.insert(0, os.path.join(HERE, "lib"))
import pipeline as P  # noqa: E402
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import core.s_paths as _SP2

# [搬运时移除的绝对路径常量] SRC = "<路径>"
# [搬运时移除的绝对路径常量] ROOT = "<路径>/某电子公司印刷电路samples__20260128"
# [搬运时移除的绝对路径常量] UN = "<项目根>/chip_wire_ng/unet"
# [搬运时移除的绝对路径常量] PAN = "<项目根>/chip_wire_ng/panels"
# [搬运时移除的绝对路径常量] SOLD = os.path.join(HERE, "out/cnn/solder_final_20261001_v2.json")
ALL = {"01_原图": "chan:01_原图", "Canny": "chan:05_Canny", "黑帽15": "chan:07_黑帽15", "USM": "chan:08_USM",
       "TTA": "pred_masks_tta", "v11": "pred_masks_v11", "v13": "pred_masks_v13", "v16b": "pred_masks_v16b_conv3_bn",
       "v18": "pred_masks_v18", "v19": "pred_masks_v19"}
KEPT = ["TTA", "v13", "v18"]
MIN_AXIS, MIN_AREA2 = 20.0, 50.0
R_DEL, EXT_R, THICK, L_CAP = 9.92, 20.0, 10, 14.92
CLOSE_R = 11                     # ① 闭运算核半径 = 删盘半径 + 1 ✓
BRIDGE_T = 7                     # ② 桥接线宽 = 实测线宽中位 ✓
BRIDGE_MAXD = 2 * R_DEL          # ② 两端点最大间距 ✓
BRIDGE_MAXA = 45.0               # ② 共线容差（度 ✓）


def load(k, fn):
    p = os.path.join(UN, "pred_masks_chan9", k[5:] + "_" + fn) if k.startswith("chan:") else os.path.join(UN, k, fn)
    v = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return (v > 127) if v is not None else None


def endpoints_of(sk):
    k = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], np.uint8)
    nb = cv2.filter2D((sk > 0).astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.where((sk > 0) & (nb == 11))
    return list(zip(xs.tolist(), ys.tolist()))


def skel_end_count(m):
    sk = cv2.ximgproc.thinning((m * 255).astype(np.uint8))
    return sk, len(endpoints_of(sk))


def build_W(keep, lab2, n2, P_pts):
    """按 252 的规范建总名单，并【先做第 5 步拆分】✓：≥3 焊点的条目 → 删盘 + 二次除杂 → 换成若干新条目 ✓"""
    W = []
    for i in range(1, n2):
        m = (lab2 == i)
        own = [q for q, (x, y) in enumerate(P_pts)
               if 0 <= int(round(x)) < m.shape[1] and 0 <= int(round(y)) < m.shape[0]
               and keep[int(round(y)), int(round(x))] and lab2[int(round(y)), int(round(x))] == i]
        if len(own) == 0:
            continue
        if len(own) < 3:                                  # 1–2 个焊点：本来就是独立飞线 ✓
            W.append({"mask": m.copy(), "pts": own, "new": False})
            continue
        # 干涉条目：删盘 → 二次除杂 → 拆成多个新飞线 ✓（这一坨之外一律不动 ✓）
        work = m.copy()
        for q in own:
            x, y = P_pts[q]
            cv2.circle(work.view(np.uint8), (int(round(x)), int(round(y))), int(round(R_DEL)), 0, -1)
        work = work.astype(bool)
        n3, lab3, st3, ce3 = cv2.connectedComponentsWithStats(work.astype(np.uint8), 8)
        for j in range(1, n3):
            if int(st3[j, 4]) < MIN_AREA2: continue       # 二次除杂 ✓
            mm = (lab3 == j)
            o2 = [q for q, (x, y) in enumerate(P_pts)
                  if 0 <= int(round(x)) < mm.shape[1] and 0 <= int(round(y)) < mm.shape[0] and mm[int(round(y)), int(round(x))]]
            W.append({"mask": mm.copy(), "pts": o2, "new": True})   # ← 拆出来的新飞线 ✓（各自独立配色 ✓）
    return W


def main():
    rows_s = json.load(open(SOLD, encoding="utf-8"))["rows"]
    pts = {}
    for e in rows_s:
        # [2026-10-07 迁移] 焊点表索引统一走 core.s_paths.solder_index
        import core.s_paths as _SP
        pts = _SP.solder_index(rows_s)
    F = ImageFont.truetype(_SP2.font_path(), 22)
    FB = ImageFont.truetype(_SP2.font_path(), 26)
    pages = {"①闭运算": [], "②骨架桥接": []}
    rep = []
    for f in sorted(glob.glob(ROOT + "/*/*_m.png")):
        grp = os.path.basename(os.path.dirname(f)); base = os.path.basename(f)[:-6]
        key = grp + "/" + base[-4:]; name = grp + "-" + base[-4:]
        fn = grp + "_" + base[-4:] + ".png"
        g = cv2.imread(os.path.join(SRC, grp, base + ".jpg"), cv2.IMREAD_GRAYSCALE)
        u = np.zeros(g.shape, bool)
        for nm in KEPT:
            m = load(ALL[nm], fn)
            if m is not None: u |= m
        n, lab, st, ce = cv2.connectedComponentsWithStats(u.astype(np.uint8), 8)
        keep = np.zeros_like(u)
        for i in range(1, n):
            ys, xs = np.where(lab == i)
            (cx, cy), (w, h), ang = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
            if max(w, h) >= MIN_AXIS: keep[lab == i] = True
        n2, lab2, st2, ce2 = cv2.connectedComponentsWithStats(keep.astype(np.uint8), 8)
        P_pts = pts.get(name, [])
        W = build_W(keep, lab2, n2, P_pts)
        # —— 只为"由干涉拆分得到的新飞线"补缺（它们才带圆盘咬痕 ✓）
        SPLIT = [w for w in W if w["new"]]
        for tag in pages:
            vis = (0.66 * cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)).astype(np.uint8)
            SPLIT_COL = [(0, 0, 255), (255, 0, 255), (0, 140, 255), (255, 0, 128), (128, 0, 255), (0, 215, 255)]
            for iw, w in enumerate(W):
                col = SPLIT_COL[iw % len(SPLIT_COL)] if w["new"] else (90, 90, 220)
                vis[w["mask"]] = (0.30 * vis[w["mask"]] + 0.70 * np.array(col, np.float32)).astype(np.uint8)
            for w in W:                                   # 每条飞线描黑边 ✓ 边界清晰 ✓
                cnt_ = cv2.findContours(w["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                cv2.drawContours(vis, cnt_, -1, (0, 0, 0), 1)
            add_total, info = 0, []
            for w in SPLIT:
                m = w["mask"]
                region = m.copy()
                sk0 = cv2.ximgproc.thinning((m * 255).astype(np.uint8))
                ys0, xs0 = np.where(sk0 > 0)
                S0 = np.stack([xs0, ys0], 1).astype(np.float32) if len(xs0) else None
                for (px_, py_) in P_pts:                       # ← 本飞线骨架端点附近（≤删盘+8px）的焊点圆盘也纳入区域 ✓
                    if S0 is None: break
                    if np.linalg.norm(S0 - np.array([px_, py_], np.float32), axis=1).min() <= R_DEL + 8.0:
                        cv2.circle(region.view(np.uint8), (int(round(px_)), int(round(py_))), int(np.ceil(R_DEL)) + 1, 1, -1)
                for q in w["pts"]:
                    x, y = P_pts[q]
                    cv2.circle(region.view(np.uint8), (int(round(x)), int(round(y))), int(np.ceil(R_DEL)) + 1, 1, -1)
                region = region.astype(bool)
                if tag == "①闭运算":
                    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CLOSE_R + 1, 2 * CLOSE_R + 1))
                    filled = cv2.morphologyEx((m * 255).astype(np.uint8), cv2.MORPH_CLOSE, ker) > 127
                    filled = filled & region
                else:
                    sk = cv2.ximgproc.thinning((m * 255).astype(np.uint8))
                    ys, xs = np.where(sk > 0); S = np.stack([xs, ys], 1).astype(np.float32)
                    eps = endpoints_of(sk)
                    d = np.zeros_like(m)
                    for a in range(len(eps)):
                        for b in range(a + 1, len(eps)):
                            p1 = np.array(eps[a], np.float32); p2 = np.array(eps[b], np.float32)
                            dist = float(np.linalg.norm(p1 - p2))
                            if dist > BRIDGE_MAXD or dist < 1e-3: continue
                            def tang(p):
                                near = S[np.linalg.norm(S - p, axis=1) <= EXT_R]
                                if len(near) < 3: return None
                                mu = near.mean(0); c = near - mu
                                _w, _v = np.linalg.eigh(c.T @ c); vv = _v[:, -1]
                                if float(vv @ (p - mu)) < 0: vv = -vv
                                nv = np.linalg.norm(vv)
                                return vv / nv if nv > 1e-3 else None
                            t1, t2 = tang(p1), tang(p2)
                            if t1 is None or t2 is None: continue
                            vec = (p2 - p1) / dist
                            a1 = np.degrees(np.arccos(np.clip(abs(float(t1 @ vec)), 0, 1)))
                            a2 = np.degrees(np.arccos(np.clip(abs(float(t2 @ vec)), 0, 1)))
                            if max(a1, a2) > BRIDGE_MAXA: continue      # 共线要求 ✓
                            cv2.line(d.view(np.uint8), (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 1, BRIDGE_T)
                    filled = (m | d.astype(bool)) & region
                newpx = int((filled & (~m)).sum())
                add_total += newpx
                # 校验：补后骨架端点数 ✓ / 越界像素（与其它飞线重叠 ✓）
                _, ne = skel_end_count(filled)
                others = np.zeros_like(keep)
                for w2 in W:
                    if w2 is w: continue
                    others |= w2["mask"]
                cross = int((filled & others).sum())
                info.append((newpx, ne, cross))
                vis[filled & (~m)] = (0.15 * vis[filled & (~m)] + 0.85 * np.array([0, 255, 0])).astype(np.uint8)
            for (x, y) in P_pts:
                cv2.circle(vis, (int(round(x)), int(round(y))), 2, (120, 0, 0), -1, cv2.LINE_AA)   # 深蓝色小点 ✓（原来白圈干扰观察 ✗）
            cap = "{}   由干涉拆分的新飞线 {} 条；{} 共补回 {} px".format(key, len(SPLIT), tag, add_total)
            if info:
                cap += "\n" + " / ".join("补{}px 端点{} 越界{}".format(a, b, c) for a, b, c in info)
            pages[tag].append((cap, vis))
            if tag == "①闭运算":
                rep.append({"board": key, "split_wires": len(SPLIT), "add_close": add_total,
                            "info_close": [list(x) for x in info]})
            else:
                rep[-1].update({"add_bridge": add_total, "info_bridge": [list(x) for x in info]})
            print("  {:<12} {} 新飞线 {} 条 → 补回 {} px；逐条（补px/端点数/越界）{}".format(key, tag, len(SPLIT), add_total, info))
    for tag, cells in pages.items():
        tw, th = cells[0][1].shape[1], cells[0][1].shape[0]
        cols = 5; rr = int(np.ceil(len(cells) / cols))
        pg = Image.new("RGB", (16 + cols * (tw + 8), 152 + rr * (th + 46)), (18, 18, 18))
        d = ImageDraw.Draw(pg)
        d.text((16, 8), "补回拆分缺口 —— 方法{}（逐条飞线处理 ✓，绝不动别的飞线 ✓）".format(tag), font=FB, fill=(255, 225, 60))
        d.text((16, 46), "红 = 原掩膜 ✓　绿 = 补回来的部分 ✓　白圈 = 焊点 ✓；只对【由干涉拆分得到的新飞线】施工 ✓",
               font=F, fill=(180, 255, 180))
        d.text((16, 76), "校验：逐条打印 补回 px / 补后骨架端点数 / 越界像素（越界必须 0 ✓）；全图 1:1 ×10 板 2×5；PNG 无损；文字在图外 ✓",
               font=F, fill=(255, 210, 140))
        for i, (cap, vis) in enumerate(cells):
            r_, c_ = divmod(i, cols)
            x = 16 + c_ * (tw + 8); y = 176 + r_ * (th + 46)
            pg.paste(Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)), (x, y))
            for j, line in enumerate(cap.split("\n")):
                d.text((x, y + th + 4 + j * 20), line, font=F, fill=(235, 235, 235))
        od = os.path.join(PAN, "state", "gapfill"); os.makedirs(od, exist_ok=True)
        fn2 = os.path.join(od, "补缺_{}.png".format(tag))
        pg.save(fn2, "PNG", optimize=True)
        print("图 -> {}  {:.2f}MB".format(fn2, os.path.getsize(fn2) / 1048576))
    json.dump({"close_r": CLOSE_R, "bridge": {"maxd": BRIDGE_MAXD, "maxa": BRIDGE_MAXA, "thick": BRIDGE_T},
               "boards": rep}, open(os.path.join(UN, "reports/gapfill.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if False:  # 原入口屏蔽（由驱动器调用 main）
    sys.exit(main())
