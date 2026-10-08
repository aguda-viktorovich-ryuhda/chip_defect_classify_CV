# -*- coding: utf-8 -*-
"""s_barepad.py - 裸焊盘（使用者 2026-10-06 口径，字面落地）

口径：候选 = 除杂丢掉的块（判定前有三成员并集、除杂后不在了）；
      其中【和焊点接触】的才算裸焊盘（接触 = 块内存在像素落在某焊点的圆盘内）。
独立于除杂结果落盘，供报告层标蓝计数；裸焊盘不计入板级不合格。
"""
import json
import os

import cv2
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import core.s_paths as _SP2


def _union(files, d):
    ms = []
    for f in files:
        _p = os.path.join(d, f)
        if not os.path.exists(_p):
            continue
        m = cv2.imread(_p, 0)
        if m is not None:
            ms.append(m > 0)
    return np.logical_or.reduce(ms) if ms else None


def build(run, solder_json, out_dir=None):
    run = os.path.abspath(run)
    rows = json.load(open(solder_json, encoding="utf-8"))["rows"]
    reas = os.path.join(run, "masks", "reports", "wire_reasons.jsonl")
    rws = [json.loads(l) for l in open(reas, encoding="utf-8")]
    out_dir = out_dir or os.path.join(run, "masks", "pred_masks_barepad")
    os.makedirs(out_dir, exist_ok=True)
    by_board = {}
    for r in rows:
        by_board.setdefault(r["sample"], []).append(r)
    res = []
    for b in sorted(by_board):
        grp, tail = b.split("/")
        key = grp + "_" + tail
        mem = []
        for k in ("TTA", "v13", "v18"):
            m = cv2.imread(os.path.join(run, "masks", k, key + ".png"), 0)
            if m is not None:
                mem.append(m > 0)
        if not mem:
            print("   成员掩膜缺 ✗ {}".format(b))
            continue
        pre = np.logical_or.reduce(mem)
        files = [x["file"] for x in rws if x["board"] == b]
        kept = _union(files, os.path.join(run, "masks", "pred_masks_all"))
        removed = pre & (~kept) if kept is not None else pre
        n, lab, st, _ = cv2.connectedComponentsWithStats(removed.astype(np.uint8), 8)
        n_bp = 0
        for i in range(1, n):
            a = int(st[i, 4])
            wire_u = _union(files, os.path.join(run, "masks", "pred_masks_wires"))
            if a < 50:
                continue
            blk = (lab == i)
            touch = 0
            for s in by_board[b]:
                x, y, r = float(s["x"]), float(s["y"]), float(s.get("r", 5.6))
                dsk = np.zeros_like(blk, np.uint8)
                cv2.circle(dsk, (int(round(x)), int(round(y))), int(np.ceil(r)), 1, -1)
                _yy, _xx = int(round(y)), int(round(x))
                _hit = (0 <= _yy < blk.shape[0] and 0 <= _xx < blk.shape[1] and bool(blk[_yy, _xx]))
                _nw = True if wire_u is None else (not bool((wire_u & blk).any()))
                if _hit and _nw:
                    touch += 1
                    _tx, _ty, _tr = x, y, r
                    touch += 1
                    _tx, _ty, _tr = x, y, r
            if touch:
                n_bp += 1
                mm = np.zeros_like(blk, np.uint8)
                mm[blk] = 255
                cv2.imwrite(os.path.join(out_dir, "{}_bp{:03d}.png".format(b.replace("/", "_"), n_bp)), mm)
                ys, xs = np.nonzero(blk)
                res.append({"board": b, "idx": n_bp, "area": a, "touch_pts": touch,
                            "x": float(_tx), "y": float(_ty), "r": float(_tr),
                            "cx": int(xs.mean()), "cy": int(ys.mean())})
            else:
                print("   剔除（未接触焊点）：{} 面积 {}".format(b, a))
        print("   裸焊盘 {} 个 ｜ {}".format(n_bp, b))
    json.dump(res, open(os.path.join(run, "masks", "reports", "bare_pad.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("   裸焊盘合计 =", len(res))
    return res

def crop_sheet(run, src, res, out_png=None):
    """把每个裸焊盘逐块裁图编号成一张总览图（流程内产物）。"""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    font = _SP2.font_path()
    if not res:
        return None
    run = os.path.abspath(run); src = os.path.abspath(src)
    out_png = out_png or os.path.join(run, "state", "report", "裸焊盘_{}个_逐块.png".format(len(res)))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    pan = []
    for i, r in enumerate(res, 1):
        b = r["board"]; grp, tail = b.split("/")
        fs = [x for x in sorted(os.listdir(os.path.join(src, grp))) if tail in x]
        if not fs:
            continue
        gm = cv2.imread(os.path.join(src, grp, fs[0]), 0)
        mp = os.path.join(run, "masks", "pred_masks_barepad", "{}_bp{:03d}.png".format(b.replace("/", "_"), r["idx"]))
        mm = cv2.imread(mp, 0) if os.path.exists(mp) else None
        x0, x1 = max(0, r["cx"] - 120), min(gm.shape[1], r["cx"] + 120)
        y0, y1 = max(0, r["cy"] - 120), min(gm.shape[0], r["cy"] + 120)
        vis = cv2.cvtColor(gm[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
        if mm is not None:
            vis[(mm[y0:y1, x0:x1] > 0)] = (255, 0, 180)
        cv2.circle(vis, (int(r["x"]) - x0, int(r["y"]) - y0), int(round(r["r"])), (0, 200, 0), 2)
        vis = cv2.resize(vis, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_NEAREST)
        cv2.line(vis, (10, vis.shape[0] - 16), (10 + int(100 * 2.2), vis.shape[0] - 16), (255, 0, 0), 3)
        vis = cv2.copyMakeBorder(vis, 46, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        im = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(im).text((6, 4), "#{}  {}  面积{}  100px".format(i, b, r["area"]),
                                font=ImageFont.truetype(font, 22), fill=(0, 0, 0))
        pan.append(np.array(im))
    if not pan:
        return None
    cw = max(p.shape[1] for p in pan); ch = max(p.shape[0] for p in pan)
    cols = 3; rn = (len(pan) + cols - 1) // cols
    canv = np.full((rn * (ch + 8) + 8, cols * (cw + 8) + 8, 3), 255, np.uint8)
    for i, p in enumerate(pan):
        rr, cc = divmod(i, cols)
        canv[8 + rr * (ch + 8):8 + rr * (ch + 8) + p.shape[0], 8 + cc * (cw + 8):8 + cc * (cw + 8) + p.shape[1]] = p
    Image.fromarray(canv).save(out_png)
    return out_png
