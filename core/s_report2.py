# -*- coding: utf-8 -*-
"""s_report2.py - 产品视角报告（每张图一张结果图）

口径（使用者 2026-10-07 定）：
  · 报告主角是【每一个产品】：一行一个产品，给"合格 / 不合格"结论；
  · 缺陷只用于定位（图上圈出哪条飞线、什么类型），不做"某类共几次"的汇总宣传；
  · 阈值全部从 cfg 读，代码里不留数字（读取失败才退回内置默认并打日志）；
  · 缺数据不崩：没有结果的图照样出一张"无数据"的结果图。

产出：
  <run>/state/report/产品结果/<产品名>.png     每张图一张结果图
  <run>/state/report/产品结论.csv              机器可读的逐产品结论
  <run>/state/report/总览.png                  可选（默认不出，overview=True 时出）
"""
import csv
import json
import os
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import core.s_paths as _SP2

FONT = _SP2.font_path()
# 注意：OpenCV 是 BGR 顺序！这里按 BGR 定义，避免"名字和显示不符"
GREEN  = (0, 200, 0)      # 绿  合格飞线
RED    = (0, 0, 255)      # 红  弯曲不合格
ORANGE = (0, 140, 255)    # 橙  断联
YELLOW = (0, 220, 255)    # 黄 裸焊盘（BGR；与红/橙/绿明显区分，另加黑边）
GRAY   = (150, 150, 150)


def _font(sz):
    return ImageFont.truetype(FONT, sz)


def _thr(C):
    """阈值一律从 cfg 读；缺字段才用默认值，并把用到的值记下来。"""
    s5 = ((C or {}).get("s5_straightness") or {})
    jit = float(s5.get("jit_max", s5.get("抖动阈值", 7.8088)))
    arc = float(s5.get("arc_chord", s5.get("弧弦比阈值", 1.0062)))
    return jit, arc


def _safe_name(s):
    return re.sub(r"[^\w\u4e00-\u9fa5.\-]", "_", str(s)) or "product"


def _overlay(gray, layers, pts, size_hint):
    loaded = []
    for path, color in layers:
        if path and os.path.exists(path):
            m = cv2.imread(path, 0)
            if m is not None:
                loaded.append((m > 0, color))
    if gray is not None:
        h, w = gray.shape[:2]
    else:
        h = max([m.shape[0] for m, _ in loaded] + [int(size_hint[0])])
        w = max([m.shape[1] for m, _ in loaded] + [int(size_hint[1])])
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray is not None else np.full((h, w, 3), 255, np.uint8)
    for m, color in loaded:
        mh, mw = m.shape[:2]
        if (mh, mw) != (vis.shape[0], vis.shape[1]):
            mm = np.zeros(vis.shape[:2], bool)
            mm[:min(mh, vis.shape[0]), :min(mw, vis.shape[1])] = m[:min(mh, vis.shape[0]), :min(mw, vis.shape[1])]
            m = mm
        vis[m] = color
    for (x, y, r, color) in pts:
        rr = max(10, int(round(r * 2.4)))
        cv2.circle(vis, (int(round(x)), int(round(y))), rr + 3, (0, 0, 0), 2)   # 黑色外圈
        cv2.circle(vis, (int(round(x)), int(round(y))), rr, color, 3)           # 黄色内圈
    return vis


def build(run, src, C=None, out_dir=None, overview=False, grid=None):
    """为每个产品出一张结果图 + 一份 CSV。返回摘要 dict。"""
    run = os.path.abspath(run)
    src = os.path.abspath(src) if src else None
    jit_thr, arc_thr = _thr(C)
    rep = os.path.join(run, "masks", "reports")
    out_dir = out_dir or os.path.join(run, "state", "report", "产品结果")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 读现场产物（缺了就当作"无数据"，不抛异常）----
    rows, reasons, bare = [], [], []
    try:
        rows = json.load(open(os.path.join(rep, "straightness_skeleton.json"), encoding="utf-8"))
    except Exception:
        pass
    try:
        reasons = [json.loads(l) for l in open(os.path.join(rep, "wire_reasons.jsonl"), encoding="utf-8")]
    except Exception:
        pass
    try:
        bare = json.load(open(os.path.join(rep, "bare_pad.json"), encoding="utf-8"))
    except Exception:
        pass
    try:
        name_map = json.load(open(os.path.join(run, "inputs", "产品名对照.json"), encoding="utf-8"))
    except Exception:
        name_map = {}
    wires_dir = os.path.join(run, "masks", "pred_masks_wires")
    have_wire = set(os.listdir(wires_dir)) if os.path.isdir(wires_dir) else set()

    # ---- 按产品聚合 ----
    products = []
    for p in name_map:
        products.append(p)
    for r in rows:
        if r.get("board") not in products:
            products.append(r.get("board"))
    for x in reasons:
        if x.get("board") not in products:
            products.append(x.get("board"))
    products = [p for p in products if p]
    products.sort()

    def board_of(name):
        return name

    summary = []
    panels = []
    for p in products:
        bends = [(r["cid"], r["file"]) for r in rows
                 if r.get("board") == p and (r.get("jit_max", 0) > jit_thr or r.get("arc_chord", 0) > arc_thr)]
        disc = [x for x in reasons if x.get("board") == p and x.get("file") not in have_wire]
        bps = [b for b in bare if b.get("board") == p]
        _no_data = (not rows) or (len([r for r in rows if r.get("board") == p]) == 0)
        verdict = "无数据" if _no_data else ("合格" if (not bends and not disc) else "不合格")
        why = []
        if bends:
            why.append("弯曲 {} 条".format(len(bends)))
        if disc:
            why.append("断联 {} 条".format(len(disc)))
        if bps:
            why.append("裸焊盘 {} 处".format(len(bps)))
        if _no_data:
            why.append("流程未产出该产品的读数（见 run.log）")
        orig = (name_map.get(p) or {}).get("原始文件", "")

        gray = None
        if src:
            grp, _, tail = p.partition("/")
            tries = []
            if orig:
                tries.append(os.path.join(src, grp, orig))
            tries.append(os.path.join(src, grp, tail + ".jpg"))
            for t in tries:
                if os.path.exists(t):
                    gray = cv2.imread(t, 0)
                    if gray is not None:
                        break
            if gray is None:
                d = os.path.join(src, grp)
                if os.path.isdir(d):
                    for f in sorted(os.listdir(d)):
                        if tail and tail in os.path.splitext(f)[0]:
                            gray = cv2.imread(os.path.join(d, f), 0)
                            if gray is not None:
                                break
        all_dir = os.path.join(run, "masks", "pred_masks_all")
        layers = []
        for r in rows:
            if r.get("board") != p:
                continue
            f = r.get("file")
            if f in have_wire:
                layers.append((os.path.join(wires_dir, f), GREEN))
        for _, f in bends:
            src_p = os.path.join(wires_dir, f) if f in have_wire else os.path.join(all_dir, f)
            layers.append((src_p, RED))
        disc_files = [x.get("file") for x in disc]
        for f in disc_files:
            layers.append((os.path.join(all_dir, f), ORANGE))   # 断联：用判定阶段的掩膜画橙色
        pts = [(b["x"], b["y"], b.get("r", 5.6), YELLOW) for b in bps]
        size_hint = gray.shape if gray is not None else (900, 900)
        vis = _overlay(gray, layers, pts, size_hint)

        # ── 先量字再排版（2026-10-07 修：标题四行曾相互覆盖）──
        f_prod, f_verd, f_why, f_thr, f_leg = _font(30), _font(36), _font(26), _font(20), _font(22)
        lines = [("产品 {}（原图 {}）".format(p, orig or "—"), f_prod, (0, 0, 0)),
                 ("结论：{}".format(verdict), f_verd,
                  (0, 150, 0) if verdict == "合格" else ((200, 0, 0) if verdict == "不合格" else (120, 120, 120))),
                 ("原因：" + ("；".join(why) if why else "未发现不合格项"), f_why, (60, 60, 60)),
                 ("判据：抖动 > {} px 或 弧弦比 > {}（阈值取自 cfg）".format(jit_thr, arc_thr), f_thr, (110, 110, 110))]
        _tmp = Image.new("RGB", (16, 16), (255, 255, 255))
        _td = ImageDraw.Draw(_tmp)
        _sizes = []
        for (_t, _f, _c) in lines:
            _bb = _td.textbbox((0, 0), _t, font=_f)
            _sizes.append(_bb[3] - _bb[1] + 8)
        _pad_x, _pad_y, _leg_h = 16, 12, 44
        head = _pad_y + sum(_sizes) + _leg_h + _pad_y
        canvas = cv2.copyMakeBorder(vis, head, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        im = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(im)
        _y = _pad_y
        for (_t, _f, _c), _h in zip(lines, _sizes):
            d.text((_pad_x, _y), _t, font=_f, fill=_c)
            _y += _h
        # 图例：单独一行，按字宽依次排布（2026-10-07 收紧间距、缩短标签，避免右侧被截断）
        LX, LY, BW = _pad_x, _y + 6, 24
        _x = LX
        _gap = 16
        for (col, txt) in [(GREEN, "合格飞线"), (RED, "弯曲不合格"), (ORANGE, "断联"), (YELLOW, "裸焊盘（无飞线）")]:
            d.rectangle([_x, LY, _x + BW, LY + 16], fill=(col[2], col[1], col[0]), outline=(90, 90, 90))
            _tw = _td.textbbox((0, 0), txt, font=f_leg)[2]
            d.text((_x + BW + 6, LY - 5), txt, font=f_leg, fill=(40, 40, 40))
            _x += BW + 6 + _tw + _gap
        out_png = os.path.join(out_dir, _safe_name(p.replace("/", "_")) + ".png")
        im.save(out_png, "PNG")
        panels.append(np.array(im))
        summary.append({"产品": p, "原始文件": orig, "结论": verdict, "原因": "；".join(why) or "无",
                        "弯曲": len(bends), "断联": len(disc), "裸焊盘": len(bps), "结果图": out_png})

    csv_p = os.path.join(run, "state", "report", "产品结论.csv")
    with open(csv_p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["产品", "原始文件", "结论", "原因", "弯曲", "断联", "裸焊盘", "结果图"])
        w.writeheader()
        for s in summary:
            w.writerow(s)

    ov = None
    if overview and panels:
        cols = grid[0] if grid else min(5, max(1, len(panels)))
        cc = max(1, cols)
        rc = (len(panels) + cc - 1) // cc
        ch = max(x.shape[0] for x in panels); cw = max(x.shape[1] for x in panels)
        canv = np.full((rc * (ch + 8) + 8, cc * (cw + 8) + 8, 3), 255, np.uint8)
        for i, x in enumerate(panels):
            rr, c2 = divmod(i, cc)
            canv[8 + rr * (ch + 8):8 + rr * (ch + 8) + x.shape[0], 8 + c2 * (cw + 8):8 + c2 * (cw + 8) + x.shape[1]] = x
        ov = os.path.join(run, "state", "report", "总览.png")
        Image.fromarray(canv).save(ov, "PNG")

    n_bad = sum(1 for s in summary if s["结论"] != "合格")
    return {"产品数": len(summary), "不合格": n_bad, "合格": len(summary) - n_bad,
            "结果图目录": out_dir, "CSV": csv_p, "总览": ov,
            "阈值": {"抖动": jit_thr, "弧弦比": arc_thr}}
