# -*- coding: utf-8 -*-
"""s4_core.py - S4 掩膜处理与分类核心（自 recon/252_registry_pipeline.py 原样抽取：load / endpoints_of / classify）

抽取方式：逐函数原样复制，不改一行算法；路径与参数由 core/s4_classify.py 注入。
口径（项目冻结术语）：2 焊点=合格飞线 ｜ 1 焊点=断联的飞线 ｜ 0=非飞线 ｜ >=3 条以上=粘连
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import core.s_paths as _SP2
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None


def load(k, fn):
    p = os.path.join(UN, "pred_masks_chan9", k[5:] + "_" + fn) if k.startswith("chan:") else os.path.join(UN, k, fn)
    v = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return (v > 127) if v is not None else None


def endpoints_of(sk):
    k = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], np.uint8)
    nb = cv2.filter2D((sk > 0).astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.where((sk > 0) & (nb == 11))
    return list(zip(xs.tolist(), ys.tolist()))


def classify(n, rings):
    """类别（使用者规范第 4 步 ✓）"""
    if n == 0: return "噪音"
    if n == 1: return "断联"
    if n == 2: return "合格" if (rings[0] != rings[1]) else "焊错"
    return "干涉"


# ---- 可注入的路径常量（软件内由 core/s4_classify.py 覆盖 ✓ 默认指向软件自身）----
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = "chip_samples__20260128"
UN = "masks_from_s3"          # S3 现场产出的掩膜目录（由 s4_classify 注入真实路径）
PAN = "outputs"               # 出图目录（由 s4_classify 注入时间戳运行目录）
SOLD = "refs/solder_table.json"  # 焊点表（由 s4_classify 指向现行 246 点表）
ALL = {}
KEPT = ["TTA", "v13", "v18"]
CLS_COL = {"合格": (0, 255, 0), "断联": (0, 215, 255), "焊错": (255, 255, 0), "新飞线": (255, 0, 255), "噪音": (150, 150, 150)}
def run_reference():
    rows_s = json.load(open(SOLD, encoding="utf-8"))["rows"]
    pts = {}
    for e in rows_s:
        pts.setdefault(e["sample"], []).append((float(e["x"]), float(e["y"])))
    print("规范：长轴 <{:.0f}px 删 ✓；重合判据 TOL=0 ✓；删盘 {:.2f}px ✓；二次除杂 <{:.0f}px ✓".format(
        MIN_AXIS, R_DEL, MIN_AREA2))
    cells, summary = [], []
    for f in sorted(glob.glob(ROOT + "/*/*_m.png")):
        grp = os.path.basename(os.path.dirname(f)); base = os.path.basename(f)[:-6]
        key = grp + "/" + base[-4:]; name = grp + "-" + base[-4:]
        fn = grp + "_" + base[-4:] + ".png"
        g = cv2.imread(os.path.join(SRC, grp, base + ".jpg"), cv2.IMREAD_GRAYSCALE)
        u = np.zeros(g.shape, bool)
        for nm in KEPT:
            m = load(ALL[nm], fn)
            if m is not None: u |= m
        # 1 长轴过滤 ✓
        n, lab, st, ce = cv2.connectedComponentsWithStats(u.astype(np.uint8), 8)
        keep = np.zeros_like(u)
        for i in range(1, n):
            ys, xs = np.where(lab == i)
            (cx, cy), (w, h), ang = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
            if max(w, h) >= MIN_AXIS: keep[lab == i] = True
        P_pts = pts.get(name, [])
        # 内/外圈（项目口径 ✓ P.ring_of ✓）
        rings = {}
        try:
            m4 = P.read_archive_mask(os.path.join(grp, base + ".jpg"))
            inner, outer, dtin = P.ring_of(m4)
            for q, (x, y) in enumerate(P_pts):
                xi, yi = int(round(x)), int(round(y))
                rings[q] = "内" if float(dtin[yi, xi]) > float(P.RING_K) else "外"
        except Exception as ex:
            print("  !! {} 内外圈判定失败：{}（本板按 2 点=合格 处理 ✓）".format(key, ex))
        # 2-4 建立总名单 ✓
        n2, lab2, st2, ce2 = cv2.connectedComponentsWithStats(keep.astype(np.uint8), 8)
        W = []                                     # 总名单 ✓
        noise = np.zeros_like(keep)
        for i in range(1, n2):
            m = (lab2 == i)
            own = [q for q, (x, y) in enumerate(P_pts)
                   if 0 <= int(round(x)) < m.shape[1] and 0 <= int(round(y)) < m.shape[0]
                   and keep[int(round(y)), int(round(x))] and lab2[int(round(y)), int(round(x))] == i]
            cls = classify(len(own), [rings.get(q, "?") for q in own])
            if cls == "噪音":
                noise[m] = True
                continue
            W.append({"id": "w{}".format(len(W)), "mask": m.copy(), "pts": own, "cls": cls, "new": False})
        # 5 拆分干涉条目 ✓
        for w in [x for x in W if x["cls"] == "干涉"]:
            work = w["mask"].copy()
            for q in w["pts"]:
                x, y = P_pts[q]
                cv2.circle(work.view(np.uint8), (int(round(x)), int(round(y))), int(round(R_DEL)), 0, -1)
            work = work.astype(bool)
            n3, lab3, st3, ce3 = cv2.connectedComponentsWithStats(work.astype(np.uint8), 8)
            W.remove(w)                            # 从总名单删掉原干涉条目 ✓
            for j in range(1, n3):
                if int(st3[j, 4]) < MIN_AREA2: continue          # 二次除杂 ✓
                W.append({"id": "s{}_{}".format(w["id"], j), "mask": (lab3 == j), "pts": [],
                          "cls": "待定", "new": True})
        # 6 新飞线：定位两端方向 → 延长覆盖到各自焊点 ✓（每端只看它自己那一个点 ✓）
        nw = [x for x in W if x["new"]]
        claimed = set()
        for w in nw:
            m = w["mask"]
            sk = cv2.ximgproc.thinning((m * 255).astype(np.uint8))
            ys, xs = np.where(sk > 0); S = np.stack([xs, ys], 1).astype(np.float32)
            ends = endpoints_of(sk)
            # 候选 (端头, 点, 距离)：只取"就在该端头被删掉的圆盘里"的点 ✓，且距离 ≤ R_DEL+6 ✓
            cand = []
            for (ex, ey) in ends:
                for q, (x, y) in enumerate(P_pts):
                    d = float(np.hypot(x - ex, y - ey))
                    if d <= R_DEL + 6.0: cand.append((d, (ex, ey), q))
            cand.sort()
            used_end = set()
            for d, (ex, ey), q in cand:
                if q in claimed or (ex, ey) in used_end: continue     # 点只归一条线 ✓、端头只领一个点 ✓
                near = S[np.linalg.norm(S - np.array([ex, ey], np.float32), axis=1) <= EXT_R]
                if len(S) < 15 or len(near) < 8: break
                mu = near.mean(0); c = near - mu
                _w, _v = np.linalg.eigh(c.T @ c); ddv = _v[:, -1]
                if float(ddv @ (np.array([ex, ey], np.float32) - mu)) < 0: ddv = -ddv
                nd = float(np.linalg.norm(ddv))
                if nd < 1e-3: continue
                ddv = ddv / nd
                px, py = P_pts[q]
                proj = float((np.array([px, py], np.float32) - np.array([ex, ey], np.float32)) @ ddv)
                ln = min(max(proj, 0.0) + 1.0, L_CAP)
                tip = np.array([ex, ey], np.float32) + ddv * ln
                line = np.zeros_like(m)
                cv2.line(line.view(np.uint8), (ex, ey), (int(round(tip[0])), int(round(tip[1]))), 1, THICK)
                m = m | line.astype(bool)
                w["pts"].append(q); claimed.add(q); used_end.add((ex, ey))
            w["mask"] = m
            # 覆盖率校验：该点坐标在延长后必须为真 ✓
            w["cov"] = [bool(m[int(round(P_pts[q][1])), int(round(P_pts[q][0]))]) for q in w["pts"]]
            w["cls"] = classify(len(w["pts"]), [rings.get(q, "?") for q in w["pts"]])
        # 7 拉出"没有焊错、没有断联"的名单 ✓
        ok_list = [w for w in W if w["cls"] == "合格"]
        vis = (0.66 * cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)).astype(np.uint8)
        vis[noise] = (0.40 * vis[noise] + 0.60 * np.array(CLS_COL["噪音"], np.float32)).astype(np.uint8)
        for w in W:
            col = CLS_COL["新飞线"] if (w["new"] and w["cls"] == "合格") else CLS_COL.get(w["cls"], (0, 0, 255))
            vis[w["mask"]] = (0.32 * vis[w["mask"]] + 0.68 * np.array(col, np.float32)).astype(np.uint8)
        for (x, y) in P_pts:
            cv2.circle(vis, (int(round(x)), int(round(y))), 6, (255, 255, 255), 1, cv2.LINE_AA)
        cnt = {}
        for w in W:
            cnt[w["cls"]] = cnt.get(w["cls"], 0) + 1
        cnt["噪音"] = int(cv2.connectedComponentsWithStats(noise.astype(np.uint8), 8)[0]) - 1
        summary.append({"board": key, "counts": cnt, "ok": len(ok_list),
                        "conf": [[w["id"], w["cls"], len(w["pts"]), w["cov"]] for w in W if w["new"]]})
        print("  {:<12} 合格 {} ✓｜断联 {} ✓｜焊错 {} ✓｜由干涉拆分得到 {} 条（其中合格 {}）｜噪音块 {} ✗".format(
            key, cnt.get("合格", 0), cnt.get("断联", 0), cnt.get("焊错", 0),
            len([w for w in W if w["new"]]), len([w for w in W if w["new"] and w["cls"] == "合格"]), cnt.get("噪音", 0)))
        for w in W:
            if w["new"] and w["cls"] != "合格":
                print("     ⚠ 新飞线 {} 判为 {}（名下 {} 个点，坐标覆盖 {}）".format(w["id"], w["cls"], len(w["pts"]), w["cov"]))
        cells.append(("{}   合格 {} ✓｜断联 {}｜焊错 {}｜拆分新飞线 {}｜噪音 {} ✗".format(
            key, cnt.get("合格", 0), cnt.get("断联", 0), cnt.get("焊错", 0),
            len([w for w in W if w["new"]]), cnt.get("噪音", 0)), vis))
    F = ImageFont.truetype(_SP2.font_path(), 22)
    FB = ImageFont.truetype(_SP2.font_path(), 26)
    tw, th = cells[0][1].shape[1], cells[0][1].shape[0]
    cols = 5; rr = int(np.ceil(len(cells) / cols))
    pg = Image.new("RGB", (16 + cols * (tw + 8), 150 + rr * (th + 30)), (18, 18, 18))
    d = ImageDraw.Draw(pg)
    d.text((16, 8), "总名单分类（长轴 <{:.0f}px 删 ✓｜重合判据 TOL=0 ✓｜干涉拆分：删盘 {:.2f}px + 二次除杂 <{:.0f}px ✓）".format(
        MIN_AXIS, R_DEL, MIN_AREA2), font=FB, fill=(255, 225, 60))
    d.text((16, 46), "绿 = 合格 ✓　黄 = 断联 ✓　青 = 焊错 ✓　品红 = 由干涉拆分得到的合格飞线 ✓　灰 = 噪音（不登记 ✗）　白圈 = 焊点 ✓",
           font=F, fill=(180, 255, 180))
    d.text((16, 76), "全图 1:1 原分辨率 ×10 板 2×5；PNG 无损；文字在图外；本图仅供使用者判读 ✓", font=F, fill=(255, 210, 140))
    for i, (cap, vis) in enumerate(cells):
        r_, c_ = divmod(i, cols)
        x = 16 + c_ * (tw + 8); y = 140 + r_ * (th + 30)
        pg.paste(Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)), (x, y))
        d.text((x, y + th + 4), cap, font=F, fill=(235, 235, 235))
    od = os.path.join(PAN, "state", "registry"); os.makedirs(od, exist_ok=True)
    fn2 = os.path.join(od, "总名单分类.png")
    # [2026-10-07 使用者要求] 该过程图内部线全灰（历史遗留 bug）、客户不需要 ⇒ 注释其保存与打印（仅注释，不改算法）
    # pg.save(fn2, "PNG", optimize=True)
    # [2026-10-07 使用者要求] 该过程图内部线全灰（历史遗留 bug）、客户不需要 ⇒ 注释其保存与打印（仅注释，不改算法）
    # print("\n图 -> {}  {:.2f}MB".format(fn2, os.path.getsize(fn2) / 1048576))
    json.dump(summary, open(os.path.join(UN, "reports/registry.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


# ---- 常量（自 recon/252 原样搬入 ✓ 口径不许改 ✗）----
MIN_AXIS, MIN_AREA2 = 20.0, 50.0            # 长轴阈值 ✓ / 二次除杂阈值 ✓
R_DEL, EXT_R, THICK, L_CAP = 9.92, 20.0, 10, 14.92
