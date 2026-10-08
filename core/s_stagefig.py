# -*- coding: utf-8 -*-
"""s_stagefig.py - 流程内阶段图：飞线判定前 / 除杂后 / 最终飞线，三时点即时取数出图。
使用时必须在同一次运行内调用（用本次运行的现场产物，不得复用旧运行）。"""
import glob
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import core.s_paths as _SP2

FONT = _SP2.font_path()


def _stack(files, d):
    ms = []
    for f in files:
        _p = os.path.join(d, f)
        if not os.path.exists(_p):
            continue
        m = cv2.imread(_p, 0)
        if m is not None:
            ms.append(m > 0)
    return np.logical_or.reduce(ms) if ms else None


def build(run, src, boards, out_dir=None):
    run = os.path.abspath(run); src = os.path.abspath(src)
    reas = os.path.join(run, "masks", "reports", "wire_reasons.jsonl")
    rows = [json.loads(l) for l in open(reas, encoding="utf-8")]
    sold = json.load(open(os.path.join(run, "masks", "solder_table.json"), encoding="utf-8"))["rows"]
    out_dir = out_dir or os.path.join(run, "state", "stagefig")
    os.makedirs(out_dir, exist_ok=True)
    res = []
    for b in boards:
        grp, tail = b.split("/")
        key = grp + "_" + tail
        imgs = [f for f in sorted(os.listdir(os.path.join(src, grp))) if tail in f]
        if not imgs:
            continue
        g = cv2.imread(os.path.join(src, grp, imgs[0]), 0)
        files = [x["file"] for x in rows if x["board"] == b]
        mem = []
        for k in ("TTA", "v13", "v18"):
            m = cv2.imread(os.path.join(run, "masks", k, key + ".png"), 0)
            if m is not None:
                mem.append(m > 0)
        layers = [("1_判定前_三成员并集", np.logical_or.reduce(mem) if mem else None),
                  ("2_除杂后_判定保留块", _stack(files, os.path.join(run, "masks", "pred_masks_all"))),
                  ("3_最终飞线掩膜", _stack(files, os.path.join(run, "masks", "pred_masks_wires")))]
        cols = []
        for name, u in layers:
            vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            nb = 0
            nf = int(u.sum()) if u is not None else -1
            if u is not None:
                vis[u] = (255, 0, 180)
            for s in sold:
                if s["sample"] != b:
                    continue
                x, y, r = int(s["x"]), int(s["y"]), float(s.get("r", 5.6))
                cov = False
                if u is not None:
                    y0, y1 = max(0, int(y - r)), min(u.shape[0], int(y + r) + 1)
                    x0, x1 = max(0, int(x - r)), min(u.shape[1], int(x + r) + 1)
                    cov = bool(u[y0:y1, x0:x1].any())
                nb += (not cov)
                cv2.circle(vis, (x, y), int(round(r)), (0, 200, 0) if cov else (255, 0, 0), 2)
            im = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            ImageDraw.Draw(im).text((14, 8), "{} ｜ 掩膜 {}px ｜ 未覆盖焊点 {}".format(name, nf, nb),
                                    font=ImageFont.truetype(FONT, 26), fill=(0, 0, 0))
            cols.append(np.array(im))
            res.append({"board": b, "stage": name, "mask_px": nf, "uncovered": nb})
        ch = max(c.shape[0] for c in cols); cw = max(c.shape[1] for c in cols)
        canv = np.full((ch + 16, 3 * (cw + 8) + 8, 3), 255, np.uint8)
        for i, c in enumerate(cols):
            canv[8:8 + c.shape[0], 8 + i * (cw + 8):8 + i * (cw + 8) + c.shape[1]] = c
        Image.fromarray(canv).save(os.path.join(out_dir, "三段对照_{}.png".format(b.replace("/", "_"))))
    json.dump(res, open(os.path.join(out_dir, "stagefig.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return res
