# -*- coding: utf-8 -*-
"""s257_numbered - 自 recon/257_all_boards_numbered.py 整份复制（仅移除绝对路径常量）"""

import glob
import json
import math
import os
import sys
import time

import cv2
import numpy as np
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VENDOR = os.path.dirname(os.path.abspath(__file__)); _COREV = os.path.dirname(_VENDOR)
for _p in (_COREV, _VENDOR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
SRC = os.path.join(HERE, "chip_samples__20260128")
UN = os.path.join(HERE, "outputs", "_masks")
PAN = os.path.join(HERE, "outputs")
OUT = PAN
SOLD = os.path.join(HERE, "refs", "solder_table.json")
BOARD_FILE = os.path.join(HERE, "refs", "boards.json")
BOARDS = []
KEPT = ["TTA", "v13", "v18"]
ALL = {k: k for k in KEPT}
PY = sys.executable
ROOT = os.path.join(HERE, "refs", "annot")
ANN = ROOT
MASKDIR = UN

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""257_all_boards_numbered.py — 10 板全跑 ✓ + 每条接受判定的线骨架【标号】✓（标签验证用 ✓）

使用者（2026-10-02）："你现在用这个方法把10张图的全跑一边看是什么样的。给每条接受判定的线骨架标号，
                     后续我们要进行标签验证。"
数据来源：recon/255 已存盘的逐条线段 ✓（unet/pred_masks_wires/ + unet/reports/straightness_skeleton.json ✓）
图上画什么：黄 = 该条线【接受判定用的骨架】（剪枝 + 剪掉距焊点 9.92px 以内 + 取最长一段 ✓）
            淡红 = 该条线的线段掩膜 ✓（半透明压在原图上 ✓ 不压暗原图 ✓）
            白底黑边数字 = 该条线的编号 ✓（= 文件名里的三位序号 ✓，可直接和 reports/straightness_skeleton.json 对号 ✓）
排布：沿用既有交付口径 ✓ 10 板 2×5 ✓ 每格 922×924（1:1 无留白 ✓）✓ 板名在图外 ✓ PNG 无损 ✓
"""
import json
import os

import cv2
import numpy as np

# [搬运时移除的绝对路径常量] SRC = "<路径>"
# [搬运时移除的绝对路径常量] UN = "<项目根>/chip_wire_ng/unet"
# [搬运时移除的绝对路径常量] PAN = "<项目根>/chip_wire_ng/panels"
BOARDS = [("1_NG", "微信图片_20240711151649", "1_NG/1649"), ("1_NG", "微信图片_20240711151658", "1_NG/1658"),
          ("1_NG", "微信图片_20240711151705", "1_NG/1705"), ("1_OK", "微信图片_20240711151507", "1_OK/1507"),
          ("1_OK", "微信图片_20240711151528", "1_OK/1528"), ("1_OK", "微信图片_20240711151535", "1_OK/1535"),
          ("2_NG", "微信图片_20240711153414", "2_NG/3414"), ("2_NG", "微信图片_20240711153421", "2_NG/3421"),
          ("2_OK", "微信图片_20240711150002", "2_OK/2"), ("2_OK", "微信图片_20240711150003", "2_OK/3")]
CELL = (922, 924)


def wire_skeleton(sub_u8):
    """与该条线【判定口径一致】的骨架：细化 → 剪枝 → 剪掉端头 → 只留最长一段 ✓"""
    sk = cv2.ximgproc.thinning(np.ascontiguousarray(sub_u8))
    sk = (sk > 0).astype(np.uint8)
    kk = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], np.uint8)
    for _ in range(15):
        nb = cv2.filter2D(sk, -1, kk, borderType=cv2.BORDER_CONSTANT)
        eys, exs = np.where((sk > 0) & (nb == 11))
        again = False
        for (ex, ey) in zip(exs.tolist(), eys.tolist()):
            if sk[ey, ex] == 0:
                continue
            br, cur, hit = [(ex, ey)], (ex, ey), False
            for _ in range(16):
                if int(nb[cur[1], cur[0]]) >= 12:
                    hit = True
                    break
                cand = [(cur[0] + dx, cur[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                        if (dx or dy) and sk[cur[1] + dy, cur[0] + dx] > 0 and (cur[0] + dx, cur[1] + dy) not in br]
                if not cand:
                    break
                cur = cand[0]
                br.append(cur)
            if hit and len(br) <= 15:
                for (bx, by) in br:
                    if int(nb[by, bx]) < 12:
                        sk[by, bx] = 0
                again = True
        if not again:
            break
    return sk


def main():
    rows = json.load(open(os.path.join(UN, "reports/straightness_skeleton.json"), encoding="utf-8"))
    pts = {}
    for e in json.load(open(SOLD, encoding="utf-8"))["rows"]:
        pts.setdefault(e["sample"], []).append((float(e["x"]), float(e["y"])))
    by_board = {}
    for r in rows:
        by_board.setdefault(r["board"], []).append(r)
    # ---- 不合格名单 ✓（使用者 2026-10-02 定：弯曲判据 + 断联 7 条 ✓）----
    OKB = {"1_OK/1507", "1_OK/1528", "1_OK/1535", "2_OK/2", "2_OK/3"}
    rsn = [json.loads(l) for l in open(os.path.join(UN, "reports/wire_reasons.jsonl"), encoding="utf-8")]
    okl = [r for r in rows if r["board"] in OKB]                      # rows = 直度读数 ✓
    TH1 = max(r["jit_max"] for r in okl) if okl else 7.8088      # 抖动阈值 = OK 板边界 ✓
    TH2 = max(r["arc_chord"] for r in okl) if okl else 1.0062    # 弧弦比阈值 = OK 板边界 ✓
    FAILSET = {(r["board"], int(r["file"].split("_")[-1].replace(".png", "")))
               for r in rows if r["jit_max"] > TH1 or r["arc_chord"] > TH2}
    BADCONN = [(r["board"], r["file"], r["area"]) for r in rsn if len(r["pts2"]) == 1]   # 断联 ✓
    print("不合格（弯曲）{} 条：{}".format(len(FAILSET), sorted("{}#{}".format(b.split("/")[-1], c) for b, c in FAILSET)))
    print("不合格（断联）{} 条：{}".format(len(BADCONN), ["{}#{}".format(b.split("/")[-1], f.split("_")[-1][:-4]) for b, f, a in BADCONN]))
    cells, ids = [], {}
    for _it in BOARDS:
        if isinstance(_it, str):          # 软件内注入的是【组名/尾4位】字符串
            key = _it; grp = _it.split('/')[0]
            stem = BOARD_FILE[_it][1] if (BOARD_FILE and isinstance(BOARD_FILE, dict) and _it in BOARD_FILE) else None
        else:
            grp, stem, key = _it
        g = None
        for _f in sorted(os.listdir(os.path.join(SRC, grp))):      # 按板号自动匹配原图 ✓（避免硬编码文件名出错 ✗）
            if _f.endswith(".jpg") and key.split("/")[-1] in _f:
                g = cv2.imread(os.path.join(SRC, grp, _f), cv2.IMREAD_GRAYSCALE)
                print("  {:<11} 原图 = {}".format(key, _f))
                break
        if g is None:
            print("  {:<11} 找不到原图 ✗（跳过 ✓）".format(key))
            cells.append((key, 0, np.zeros((CELL[1], CELL[0], 3), np.uint8)))
            continue
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        n_acc = 0
        for r in sorted(by_board.get(key, []), key=lambda r: r["file"]):
            m = cv2.imread(os.path.join(UN, "pred_masks_wires", r["file"]), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            sub = (m > 127)
            vis[sub] = (0.78 * vis[sub] + 0.22 * np.array([0, 0, 255])).astype(np.uint8)   # 掩膜压淡 ✓（原来 0.58 太浓 ✗ 与骨架混在一起 ✗）
            ys, xs = np.where(sub)
            y0, y1 = max(0, ys.min() - 4), min(sub.shape[0], ys.max() + 5)
            x0, x1 = max(0, xs.min() - 4), min(sub.shape[1], xs.max() + 5)
            sk = wire_skeleton((sub[y0:y1, x0:x1].astype(np.uint8) * 255))
            for (px, py) in pts.get(key.replace("/", "-"), []):
                if x0 <= px < x1 and y0 <= py < y1:
                    cv2.circle(sk, (int(round(px)) - x0, int(round(py)) - y0), 10, 0, -1)
            n_sk, lab_sk, st_sk, _c = cv2.connectedComponentsWithStats(sk.astype(np.uint8), 8)
            if n_sk > 1:
                big = 1 + int(np.argmax(st_sk[1:, 4]))
                sk = (lab_sk == big).astype(np.uint8)
            ys2, xs2 = np.where(sk > 0)
            if len(ys2) < 8:
                continue                       # 与 255 同一门槛 ✓（骨架被剪空/过短的线不参与判定 ✓）
            n_acc += 1
            num = r["file"].split("_")[-1].replace(".png", "")
            _fail = (key, int(num)) in FAILSET                                   # ← 不合格（弯曲 ✓）
            vis[y0 + ys2, x0 + xs2] = (0, 0, 255) if _fail else (0, 255, 0)       # 红 = 不合格骨架 ✓ 绿 = 合格骨架 ✓
            cx, cy = int(x0 + xs2.mean()), int(y0 + ys2.mean())
            ids.setdefault(key, []).append({"num": num, "file": r["file"], "不合格": bool(_fail), "抖动": r["jit_max"],
                                            "弧弦比": r["arc_chord"], "曲率P95": r["kappa_p95"],
                                            "最小曲率半径": r["R_min"], "低频能量": r["low_energy"],
                                            "弯曲能量": r["bend_energy"], "线长px": r["len_px"]})
            txt = num.lstrip("0") or "0"
            tw = 11 * len(txt) + 8
            cv2.rectangle(vis, (cx + 6, cy - 24), (cx + 6 + tw, cy - 4),
                          (0, 0, 170) if _fail else (20, 20, 20), -1)   # 不合格编号框 = 红底 ✓（不再写长串文字 ✓ 避免遮挡 ✗）
            cv2.putText(vis, txt, (cx + 10, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        for (_b, _f, _a) in BADCONN:                 # ← 不合格（断联 ✓ 只认到 1 个焊点的 7 条）用橙色标出 ✓
            if _b != key:
                continue
            _m = cv2.imread(os.path.join(UN, "pred_masks_all", _f), cv2.IMREAD_GRAYSCALE)
            if _m is None:
                continue
            _s = (_m > 127)
            vis[_s] = (0.35 * vis[_s] + 0.65 * np.array([0, 140, 255])).astype(np.uint8)
            _ys, _xs = np.where(_s)
            if len(_ys):
                cv2.putText(vis, "NG-open", (int(_xs.mean()) + 6, int(_ys.mean())), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 140, 255), 2, cv2.LINE_AA)
        cells.append((key, n_acc, vis))
        print("  {:<11} 标号骨架 {} 条".format(key, n_acc))
    H = max(c[2].shape[0] for c in cells); W = max(c[2].shape[1] for c in cells)   # 按原图实际尺寸 ✓
    TEXW = 360                                                                     # ← 每板右侧【图外文字栏】✓（不合格清单写在这里 ✓ 不压图 ✗）
    cw, ch = W + 8 + TEXW, H + 46
    pg = np.full((44 + 5 * ch, 8 + 2 * cw, 3), 18, np.uint8)
    cv2.putText(pg, "GREEN = OK wire skeleton | RED = NG-bend skeleton (red number box) | ORANGE = NG-open (1 solder pt) | faint red = its mask | numbers = ids; per-board NG lists in the right text column",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 225, 60), 2, cv2.LINE_AA)
    for i, (key, n_acc, vis) in enumerate(cells):
        r_, c_ = divmod(i, 2)
        oy, ox = 40 + r_ * ch, 8 + c_ * cw
        cv2.putText(pg, "{}   ({} skeletons)".format(key, n_acc), (ox + 4, oy + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (235, 235, 235), 2, cv2.LINE_AA)
        hh, ww = vis.shape[:2]
        pg[oy + 38:oy + 38 + hh, ox + 4:ox + 4 + ww] = vis          # 完整贴入 ✓ 不裁 ✗
        # ---- 图外文字栏：本板不合格清单 ✓（简洁 ✓ 不遮挡图 ✓）----
        _tx, _ty = ox + 12 + ww, oy + 66
        cv2.putText(pg, "-- NG bend (jit px / arc) --", (_tx, _ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2, cv2.LINE_AA)
        _ty += 30
        for _r in sorted([x for x in rows if x["board"] == key], key=lambda x: -x["jit_max"]):
            _cd = int(_r["file"].split("_")[-1].replace(".png", ""))
            if (key, _cd) not in FAILSET or _ty > oy + ch - 60:
                continue
            cv2.putText(pg, "#{:<4} {:.1f}px  arc {:.4f}".format(_cd, _r["jit_max"], _r["arc_chord"]),
                        (_tx, _ty), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            _ty += 26
        _ty += 14
        cv2.putText(pg, "-- NG open (1 solder pt) --", (_tx, _ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 140, 255), 2, cv2.LINE_AA)
        _ty += 30
        for (_b, _f, _a) in BADCONN:
            if _b != key or _ty > oy + ch - 20:
                continue
            cv2.putText(pg, "#{:<4} area {}px".format(int(_f.split("_")[-1][:-4]), _a),
                        (_tx, _ty), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 140, 255), 2, cv2.LINE_AA)
            _ty += 26
    od = os.path.join(PAN, "state", "wires"); os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, "10板_标号骨架.png")
    cv2.imwrite(fn, pg)
    json.dump(ids, open(os.path.join(UN, "reports/wire_ids.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    tot = sum(len(v) for v in ids.values())
    print("图 -> {}  {:.2f}MB  {}×{}  ✓ 标号骨架合计 {} 条（编号-读数对照 -> " + str(UN) + "/reports/wire_ids.json ✓）".format(
        fn, os.path.getsize(fn) / 1048576, pg.shape[1], pg.shape[0], tot))
    return 0


if False:  # 原入口屏蔽（由驱动器调用 main）
    raise SystemExit(main())