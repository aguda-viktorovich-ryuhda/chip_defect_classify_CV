# -*- coding: utf-8 -*-
"""s_report.py - 客户向全局判定报告图（中文，一张 PNG）

口径（全部来自本次运行现场产物，无人工判定）：
  合格飞线   有线段掩膜 且 未越判据                              绿
  弯曲不合格 有线段掩膜 且 (抖动>7.8088px 或 弧弦比>1.0062)        红
  断联       无线段掩膜 且 甲判据判为非裸焊盘                     橙
  裸焊盘     无线段掩膜 且 (骨架长<SK 或 L/W<LW)（阈值取自 cfg）    蓝
  其它       非飞线连通域（不参与判定）                          灰
板级结论：弯曲=0 且 断联=0 => 合格（裸焊盘不计入不合格）
布局：10 板 2x5 | 原分辨率 1:1 | 板名与结论在图外 | 顶部判据 | 底部合计与图例
"""
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import core.s_paths as _SP2

FONT = _SP2.font_path()
JIT, ARC = 7.8088, 1.0062
GREEN, RED, ORANGE, BLUE = (0, 190, 0), (255, 0, 0), (0, 140, 255), (255, 90, 0)
MAGENTA = (255, 0, 255)
MEMBERS = ("TTA", "v13", "v18")


def _font(sz):
    return ImageFont.truetype(FONT, sz)


def _blob_of(run, board, area):
    grp, tail = (board.split("/") if "/" in board else board.rsplit("-", 1))
    ors = []
    for k in MEMBERS:
        m = cv2.imread(os.path.join(run, "masks", k, grp + "_" + tail + ".png"), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            ors.append(m > 0)
    if not ors:
        return None
    u = np.logical_or.reduce(ors).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(u, 8)
    if n <= 1:
        return None
    best, bd = None, None
    for i in range(1, n):
        d = abs(int(stats[i, cv2.CC_STAT_AREA]) - int(area or 0))
        if bd is None or d < bd:
            best, bd = i, d
    return (lab == best).astype(np.uint8)


def build(run, src, out_png, grid=(5, 2), C=None):
    rep = os.path.join(run, "masks", "reports")
    rows = json.load(open(os.path.join(rep, "straightness_skeleton.json"), encoding="utf-8"))
    reasons = [json.loads(l) for l in open(os.path.join(rep, "wire_reasons.jsonl"), encoding="utf-8")]
    wdir = os.path.join(run, "masks", "pred_masks_wires")
    boards = sorted({r["board"] for r in rows} | {x["board"] for x in reasons})

    rule = ((C or {}).get("s4_classify", {}) or {}).get("bare_pad_rule", {}) or {}
    SK, LW = float(rule.get("skeleton_px", 20.0)), float(rule.get("lw", 1.5))

    curved = {(r["board"], r["cid"]) for r in rows if r["jit_max"] > JIT or r["arc_chord"] > ARC}
    okwire = {(r["board"], r["cid"]) for r in rows} - curved
    by_file = {r["file"]: (r["board"], r["cid"]) for r in rows}

    # 按【数据】判类（不用"缺掩膜=断联"这个错误等式 ✗）
    #   pts2 长度 <=1  ⇒ 断联/裸焊盘（甲判据再分）
    #   长度==2 且 rings 内外不同 ⇒ 【名单异常】本应有掩膜却缺 ⇒ 属流水线问题，不计入断联
    #   其它（2 点同圈 / >=3 点粘连）⇒ 未参与（不计入不合格）
    disc, bare, anom, other = [], [], [], []
    for x in reasons:
        if os.path.exists(os.path.join(wdir, x["file"])):
            continue
        pts = x.get("pts2") or []
        rgs = x.get("rings") or []
        if len(pts) == 2 and len(rgs) == 2 and rgs[0] is not None and rgs[1] is not None and rgs[0] != rgs[1]:
            anom.append(x); continue
        if len(pts) > 2:
            other.append(x); continue
        blob = _blob_of(run, x["board"], x.get("area"))
        if blob is None:
            disc.append(x); continue
        cnt = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if not cnt:
            disc.append(x); continue
        c0 = max(cnt, key=cv2.contourArea)
        w0, h0 = cv2.minAreaRect(c0)[1]
        lw = max(w0, h0) / max(1.0, min(w0, h0))
        sk = float(cv2.arcLength(c0, False) / 2.0)
        x["_blob"] = blob
        (bare if (sk < SK or lw < LW) else disc).append(x)

    tiles = []
    for b in boards:
        grp, tail = (b.split("/") if "/" in b else b.rsplit("-", 1))
        cands = [f for f in sorted(os.listdir(os.path.join(src, grp))) if tail in f]
        assert cands, "找不到原图 " + b
        g = cv2.cvtColor(cv2.imread(os.path.join(src, grp, cands[0]), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        for f in sorted(os.listdir(wdir)):
            key = by_file.get(f)
            if not key or key[0] != b:
                continue
            m = cv2.imread(os.path.join(wdir, f), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                g[m > 0] = RED if key in curved else GREEN
        for x, col in ([(y, ORANGE) for y in disc if y["board"] == b]
                       + [(y, BLUE) for y in bare if y["board"] == b]
                       + [(y, MAGENTA) for y in anom if y["board"] == b]):
            bl = x.get("_blob")
            if bl is None:
                continue
            cnt = cv2.findContours(bl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            cv2.drawContours(g, cnt, -1, col, 3)
            g[bl > 0] = col
        tiles.append((b, g,
                      sum(1 for k in okwire if k[0] == b),
                      sum(1 for k in curved if k[0] == b),
                      sum(1 for y in disc if y["board"] == b),
                      sum(1 for y in bare if y["board"] == b),
                      sum(1 for y in anom if y["board"] == b)))

    cols, rws = grid
    th = max(t[1].shape[0] for t in tiles)
    tw = max(t[1].shape[1] for t in tiles)
    cap, head, foot, pad = 118, 150, 180, 20
    W = pad + cols * (tw + pad)
    H = head + rws * (th + cap + pad) + foot
    canvas = np.full((H, W, 3), 255, np.uint8)
    canvas[:head] = (243, 243, 243)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y0 = head + r * (th + cap + pad) + pad
        x0 = pad + c * (tw + pad)
        canvas[y0:y0 + t[1].shape[0], x0:x0 + t[1].shape[1]] = t[1]
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    im = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    # ── 裸焊盘：读流程产物 bare_pad.json（按点判定，独立于除杂）
    try:
            _bpj = json.load(open(os.path.join(run, "masks", "reports", "bare_pad.json"), encoding="utf-8"))
    except Exception:
        _bpj = []
    _bpc = {}
    for _bp in _bpj:
        _bpc[_bp["board"]] = _bpc.get(_bp["board"], 0) + 1
    f_big, f_mid, f_sm = _font(56), _font(40), _font(34)
    d.text((pad, 18), "飞线异常判定报告", font=f_big, fill=(20, 20, 20))
    d.text((pad, 92), "判据：抖动 > 7.8088 px 或 弧弦比 > 1.0062（越界即弯曲不合格）", font=f_sm, fill=(70, 70, 70))
    for i, (b, g, n_ok, n_cu, n_di, n_ba, n_an) in enumerate(tiles):
        r, c = divmod(i, cols)
        y0 = head + r * (th + cap + pad) + pad
        x0 = pad + c * (tw + pad)
        by = y0 + g.shape[0]
        d.text((x0, by + 6), "%s" % b, font=f_mid, fill=(0, 0, 0))
        cap_txt = "合格飞线 %d ｜ 弯曲 %d ｜ 断联 %d ｜ 裸焊盘 %d" % (n_ok, n_cu, n_di, _bpc.get(b, n_ba))
        if n_an:
            cap_tot += "" if False else ""
            cap_txt += " ｜ 名单异常 %d" % n_an
        d.text((x0, by + 62), cap_txt, font=f_sm, fill=(50, 50, 50))
        verdict = "合格" if (n_cu == 0 and n_di == 0) else "不合格"
        d.text((x0 + tw - 140, by + 28), verdict, font=f_mid,
               fill=(0, 140, 0) if verdict == "合格" else (200, 0, 0))
    tot_ok, tot_cu, tot_di, tot_ba = len(okwire), len(curved), len(disc), len(_bpj)
    tot_an = len(anom)
    fails = sum(1 for t in tiles if (t[3] or t[4]))
    d.text((pad, H - 136), "合计：板 %d ｜ 合格飞线 %d ｜ 弯曲 %d ｜ 断联 %d ｜ 裸焊盘 %d ｜ 名单异常 %d ｜ 不合格板 %d" % (
        len(tiles), tot_ok, tot_cu, tot_di, tot_ba, tot_an, fails), font=f_mid, fill=(0, 0, 0))
    d.text((pad, H - 82), "绿=合格飞线 ｜ 红=弯曲不合格 ｜ 橙=断联 ｜ 蓝=裸焊盘（不计入不合格）｜ 灰=非飞线连通域",
           font=f_sm, fill=(70, 70, 70))
    # ── 在报告图上标出裸焊盘（蓝圈，直径=焊点半径×4）
    _tidx = {t[0]: _i for _i, t in enumerate(tiles)}
    for _bp in _bpj:
        _i = _tidx.get(_bp["board"])
        if _i is None:
            continue
        _r, _c = divmod(_i, cols)
        _yy0 = head + _r * (th + cap + pad) + pad
        _xx0 = pad + _c * (tw + pad)
        _cx = _xx0 + float(_bp["x"])
        _cy = _yy0 + float(_bp["y"])
        _rr = float(_bp["r"]) * 2.0
        d.ellipse([_cx - _rr, _cy - _rr, _cx + _rr, _cy + _rr], outline=BLUE, width=4)
    im.save(out_png, "PNG")
    return {"png": out_png, "boards": len(tiles), "ok": tot_ok, "bend": tot_cu,
            "disconn": tot_di, "bare_pad": tot_ba, "anomaly": tot_an, "fail_boards": fails}
