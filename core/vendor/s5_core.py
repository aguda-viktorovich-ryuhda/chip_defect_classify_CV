# -*- coding: utf-8 -*-
"""s5_core.py - S5（掩膜再处理 + 直度判定）核心，自 recon 253/255/256/257/259 原样抽取

逐函数原样复制；路径与结构常量由 core/s5_straightness.py 注入；口径与 recon/258 链一致。
"""
import glob
import json
import os
import re
import sys
import itertools
import math

import cv2
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import core.s_paths as _SP2


# ---- 可注入（驱动器覆盖）----
SRC="chip_samples__20260128"
UN="run_masks"
PAN="outputs"
SOLD="refs/solder_table.json"
BOARD_FILE="refs/boards.json"
BOARDS=[]
PY=sys.executable
ALL={}
KEPT=[]


# ---- 数值常量（仅搬括号配平的行 ✓ 结构类字面量一律不搬 ✗ 由驱动器注入 ✓）----
KEPT = ["TTA", "v13", "v18"]
MIN_AXIS, MIN_AREA2 = 20.0, 50.0
R_DEL, EXT_R, THICK, L_CAP = 9.92, 20.0, 10, 14.92
CLOSE_R = 11                     # ① 闭运算核半径 = 删盘半径 + 1 ✓
BRIDGE_T = 7                     # ② 桥接线宽 = 实测线宽中位 ✓
BRIDGE_MAXD = 2 * R_DEL          # ② 两端点最大间距 ✓
BRIDGE_MAXA = 45.0               # ② 共线容差（度 ✓）
KEPT = ["TTA", "v13", "v18"]
MIN_AXIS, MIN_AREA2 = 20.0, 50.0
R_DEL, EXT_R, THICK, L_CAP, CLOSE_R = 9.92, 20.0, 10, 14.92, 13
R_TRIM = 9.92
NG_BOARDS = set()  # 发布版置空：请按你的数据集填写板名
Z, PAD = 3, 40
CELL = (922, 924)


# ---- 来自 253_gap_fill_two_methods.py ----
def load(k, fn):
    p = os.path.join(UN, "pred_masks_chan9", k[5:] + "_" + fn) if k.startswith("chan:") else os.path.join(UN, k, fn)
    v = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return (v > 127) if v is not None else None


# ---- 来自 253_gap_fill_two_methods.py ----
def endpoints_of(sk):
    k = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], np.uint8)
    nb = cv2.filter2D((sk > 0).astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.where((sk > 0) & (nb == 11))
    return list(zip(xs.tolist(), ys.tolist()))


# ---- 来自 253_gap_fill_two_methods.py ----
def skel_end_count(m):
    sk = cv2.ximgproc.thinning((m * 255).astype(np.uint8))
    return sk, len(endpoints_of(sk))


# ---- 来自 253_gap_fill_two_methods.py ----
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


# ---- 来自 253_gap_fill_two_methods.py ----
def main():
    rows_s = json.load(open(SOLD, encoding="utf-8"))["rows"]
    pts = {}
    for e in rows_s:
        pts.setdefault(e["sample"], []).append((float(e["x"]), float(e["y"])))
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


# ---- 来自 255_straightness_skeleton.py ----
def load(k, fn):
    p = os.path.join(UN, "pred_masks_chan9", k[5:] + "_" + fn) if k.startswith("chan:") else os.path.join(UN, k, fn)
    v = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return (v > 127) if v is not None else None


# ---- 来自 255_straightness_skeleton.py ----
def thin_fast(u8):
    """细化加速 ✓：先裁到非零像素的外接框（+4px 边距）再 thin，然后贴回原尺寸 ✓
    （原来在整幅 922×924 上 thin：面积 85 万 vs 实际几千像素 ✗ → 慢 30–100 倍 ✗）"""
    ys, xs = np.where(u8 > 0)
    if len(ys) == 0:
        return np.zeros_like(u8)
    y0, y1 = max(0, int(ys.min()) - 4), min(u8.shape[0], int(ys.max()) + 5)
    x0, x1 = max(0, int(xs.min()) - 4), min(u8.shape[1], int(xs.max()) + 5)
    th = cv2.ximgproc.thinning(np.ascontiguousarray(u8[y0:y1, x0:x1]))
    out = np.zeros_like(u8)
    out[y0:y1, x0:x1] = th
    return out


# ---- 来自 255_straightness_skeleton.py ----
def order_path2(sk):
    """有序路径 = 骨架图上的【最长测地线】✓（两次 BFS 求直径路径 ✓）
    原 order_path 是"从端点贪心走"✗：分叉处走进短支就断 → 长骨架被判成"路径不足 8 点"而丢弃 ✗"""
    ys, xs = np.where(sk > 0)
    if len(ys) == 0:
        return np.zeros((0, 2), np.float32)
    pts = set(zip(xs.tolist(), ys.tolist()))
    nbr = {p: [(p[0] + dx, p[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
               if (dx or dy) and (p[0] + dx, p[1] + dy) in pts] for p in pts}
    def bfs(s):
        seen = {s: None}
        q = [s]
        last = s
        while q:
            cur = q.pop(0)
            last = cur
            for n in nbr[cur]:
                if n not in seen:
                    seen[n] = cur
                    q.append(n)
        return last, seen
    p0 = next(iter(pts))
    u, _ = bfs(p0)
    v, par = bfs(u)
    path = []
    cur = v
    while cur is not None:
        path.append(cur)
        cur = par[cur]
    return np.array(path[::-1], np.float32)


# ---- 来自 255_straightness_skeleton.py ----
def endpoints_of(sk):
    k = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], np.uint8)
    nb = cv2.filter2D((sk > 0).astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.where((sk > 0) & (nb == 11))
    return list(zip(xs.tolist(), ys.tolist()))


# ---- 来自 255_straightness_skeleton.py ----
def order_path(sk):
    """把骨架变成有序路径：从端点沿 8 邻域走到另一端 ✓（取最长的一条 ✓）"""
    ys, xs = np.where(sk > 0)
    if len(ys) == 0:
        return np.zeros((0, 2), np.float32)          # 骨架被剪空（极短线 ✓）→ 返回空 ✓，由 metrics 判为不可算 ✓
    pts = set(zip(xs.tolist(), ys.tolist()))
    eps = endpoints_of(sk)
    starts = eps if eps else [next(iter(pts))]
    best = []
    for s in starts:                     # ← 所有端点都试 ✓（原为 starts[:8] ✗：若这 8 个都在短分叉支上，长线会被判成"骨架太短"而丢弃 ✗）
        path, cur, prev = [s], s, None
        visited = {s}                      # ← 必须记住走过的点 ✓（否则分叉处来回弹跳 → path 无限增长 → 吃光内存 ✗）
        while True:
            nxts = [(cur[0] + dx, cur[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                    if (dx or dy) and (cur[0] + dx, cur[1] + dy) in pts and (cur[0] + dx, cur[1] + dy) != prev]
            nxts = [n for n in nxts if n not in visited]
            if not nxts or len(path) > 4 * len(pts): break
            prev, cur = cur, nxts[0]
            visited.add(cur)
            path.append(cur)
        if len(path) > len(best): best = path
    return np.array(best, np.float32)


# ---- 来自 255_straightness_skeleton.py ----
def metrics(path):
    if len(path) < 8: return None
    # 1D 平滑（±4 点移动平均 ✓ 抗二值量化的阶梯噪声 ✓）
    k = 9
    pad = np.vstack([path[0]] * (k // 2) + [path] + [path[-1]] * (k // 2))
    ker = np.ones(k, np.float32) / k
    sp = np.stack([np.convolve(pad[:, 0], ker, "valid"), np.convolve(pad[:, 1], ker, "valid")], 1)
    seg = np.linalg.norm(np.diff(sp, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    arc = float(s[-1])
    chord = float(np.linalg.norm(sp[-1] - sp[0])) + 1e-6
    v = (sp[-1] - sp[0]) / chord
    n = np.array([-v[1], v[0]], np.float32)
    dev = (sp - sp[0]) @ n
    # 曲率：以 7 点为跨度算转向角 ✓
    step = 7
    kap = []
    for i in range(step, len(sp) - step):
        a = sp[i] - sp[i - step]; b = sp[i + step] - sp[i]
        na, nb_ = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb_ < 1e-6: continue
        ang = float(np.arccos(np.clip(float(a @ b) / (na * nb_), -1, 1)))
        ds = 0.5 * (na + nb_)
        if ds > 1e-6: kap.append(ang / ds)
    kap = np.array(kap, np.float32)
    # 低频能量占比（横向偏离序列 FFT ✓）
    d = dev - dev.mean()
    f = np.abs(np.fft.rfft(d * np.hanning(len(d))))
    freqs = np.fft.rfftfreq(len(d))
    low = freqs <= 1.0 / max(2.0 * chord, 1e-6)
    e_low = float((f[low] ** 2).sum()) / float((f ** 2).sum() + 1e-9)
    return {"arc_chord": arc / chord,
            "jit_max": float(np.abs(dev).max()),
            "kappa_p95": float(np.percentile(kap, 95)) if len(kap) else 0.0,
            "R_min": float(1.0 / max(kap.max(), 1e-9)) if len(kap) else 1e9,
            "low_energy": e_low,
            "bend_energy": float((kap[:min(len(kap), max(0, len(seg) - step))] ** 2 *
                                  seg[step:step + min(len(kap), max(0, len(seg) - step))]).sum()),
            "len_px": int(arc)}


# ---- 来自 255_straightness_skeleton.py ----
def main():
    rows_s = json.load(open(SOLD, encoding="utf-8"))["rows"]
    pts = {}
    for e in rows_s:
        pts.setdefault(e["sample"], []).append((float(e["x"]), float(e["y"])))
    out = []
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
        rings = {}
        try:
            m4 = P.read_archive_mask(os.path.join(grp, base + ".jpg"))
            inner, outer, dtin = P.ring_of(m4)
            for q, (x, y) in enumerate(P_pts):
                rings[q] = "内" if float(dtin[int(round(y)), int(round(x))]) > float(P.RING_K) else "外"
        except Exception:
            rings = {}
        W = []
        for i in range(1, n2):
            m = (lab2 == i)
            own = [q for q, (x, y) in enumerate(P_pts)
                   if keep[int(round(y)), int(round(x))] and lab2[int(round(y)), int(round(x))] == i]
            if not own: continue
            if len(own) < 3:
                W.append({"mask": m.copy(), "pts": own, "new": False}); continue
            work = m.copy()
            for q in own:
                x, y = P_pts[q]
                cv2.circle(work.view(np.uint8), (int(round(x)), int(round(y))), int(round(R_DEL)), 0, -1)
            work = work.astype(bool)
            n3, lab3, st3, ce3 = cv2.connectedComponentsWithStats(work.astype(np.uint8), 8)
            for j in range(1, n3):
                if int(st3[j, 4]) < MIN_AREA2: continue
                mm = (lab3 == j)
                W.append({"mask": mm.copy(), "pts": [], "new": True})
        # 顺序 B：先延长 → 再闭运算（核 13px ✓）
        claimed = set()
        for w in W:
            if not w["new"]: continue
            m = w["mask"]; sk = thin_fast((m * 255).astype(np.uint8))
            ys, xs = np.where(sk > 0); S = np.stack([xs, ys], 1).astype(np.float32)
            cand = []
            for (ex, ey) in endpoints_of(sk):
                best_q, best_d = None, 1e9                    # ← 逐条飞线独立认点 ✓：只看"离我这个端头最近的焊点" ✓
                for q, (x, y) in enumerate(P_pts):            #    不再全板先到先得 ✗（那样后处理的线会丢点 → 正常长线被误判断联 ✗）
                    d = float(np.hypot(x - ex, y - ey))
                    if d < best_d:
                        best_q, best_d = q, d
                if best_q is not None and best_d <= R_DEL + 6.0:
                    cand.append((best_d, (ex, ey), best_q))
            cand.sort(); used = set()
            for d, (ex, ey), q in cand:
                if (ex, ey) in used: continue
                near = S[np.linalg.norm(S - np.array([ex, ey], np.float32), axis=1) <= EXT_R]
                if len(S) < 15 or len(near) < 8: continue
                mu = near.mean(0); c = near - mu
                _w, _v = np.linalg.eigh(c.T @ c); dd = _v[:, -1]
                if float(dd @ (np.array([ex, ey], np.float32) - mu)) < 0: dd = -dd
                nv = float(np.linalg.norm(dd))
                if nv < 1e-3: continue
                dd = dd / nv
                px, py = P_pts[q]
                proj = float((np.array([px, py], np.float32) - np.array([ex, ey], np.float32)) @ dd)
                tip = np.array([ex, ey], np.float32) + dd * min(max(proj, 0.0) + 1.0, L_CAP)
                line = np.zeros_like(m)
                cv2.line(line.view(np.uint8), (ex, ey), (int(round(tip[0])), int(round(tip[1]))), 1, THICK)
                m = m | line.astype(bool)
                w["pts"].append(q); claimed.add(q); used.add((ex, ey))
            reg = m.copy()
            sk2 = thin_fast((m * 255).astype(np.uint8))
            ys2, xs2 = np.where(sk2 > 0)
            if len(xs2):
                S2 = np.stack([xs2, ys2], 1).astype(np.float32)
                for (px, py) in P_pts:
                    if np.linalg.norm(S2 - np.array([px, py], np.float32), axis=1).min() <= R_DEL + 8.0:
                        cv2.circle(reg.view(np.uint8), (int(round(px)), int(round(py))), int(np.ceil(R_DEL)) + 1, 1, -1)
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CLOSE_R + 1, 2 * CLOSE_R + 1))
            w["mask"] = (cv2.morphologyEx((m * 255).astype(np.uint8), cv2.MORPH_CLOSE, ker) > 127) & reg.astype(bool)
        # 复核归属与分类（在修补后的掩膜上 ✓）→ 只取合格（2 点、一内一外 ✓）
        qual = []
        import json as _json
        with open(os.path.join(UN, "reports/wire_reasons.jsonl"), "a", encoding="utf-8") as _fh:
            for _i, w in enumerate(W, 1):   # ← 每条连通域都记录 ✓ + 掩膜存盘 ✓（用于"直度判定前/后"两阶段对照图 ✗）
                w["cid"] = _i               # ← 【稳定编号】✓：直度判定前后都用它 ✓（原来用读数序号 ✗ 名单传递会错位 ✗）
                _own = [q for q, (x, y) in enumerate(P_pts) if w["mask"][int(round(y)), int(round(x))]]
                _dir = os.path.join(UN, "pred_masks_all"); os.makedirs(_dir, exist_ok=True)
                _f = "{}_{:03d}.png".format(base, _i)
                cv2.imwrite(os.path.join(_dir, _f), (w["mask"] * 255).astype("uint8"))
                _fh.write(_json.dumps({"board": key, "idx": _i, "file": _f, "area": int(w["mask"].sum()),
                                       "pts2": [int(q) for q in _own],
                                       "rings": [rings.get(int(q)) for q in _own],
                                       "from_split": bool(w["new"])}, ensure_ascii=False) + "\n")
        for w in W:
            m = w["mask"]
            own = [q for q, (x, y) in enumerate(P_pts)
                   if m[int(round(y)), int(round(x))]]
            w["pts2"] = own
            if len(own) == 2 and rings and rings.get(own[0]) != rings.get(own[1]):
                qual.append(w)
        # 剪端（半径 = 平均 + 5 = 9.92px ✓，只在本条自己的像素上删 ✓）
        res = []
        for w in qual:
            m = w["mask"].copy()          # ← 撤掉掩膜层的"直线切端" ✓（在焊盘喇叭口处方向不可靠 ✗ 会把线切短 ✗）；端头只在骨架层剪 ✓
            big = None
            n4, lab4, st4, ce4 = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
            for j in range(1, n4):
                if big is None or st4[j, 4] > st4[big, 4]: big = j
            if big is None: continue
            m = (lab4 == big)
            sk = thin_fast((m * 255).astype(np.uint8))
            # ← 骨架剪枝 ✓：反复剪掉"从端头走到分叉点、长度 ≤15px"的短支 ✓
            #    （端头分叉就是这么来的 ✗：掩膜在焊盘处变宽/切口不齐 → 细化必给 Y 形分叉 ✗）
            sk = (sk > 0).astype(np.uint8)
            kk_nb = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], np.uint8)
            for _ in range(15):
                nb = cv2.filter2D(sk, -1, kk_nb, borderType=cv2.BORDER_CONSTANT)
                eys_, exs_ = np.where((sk > 0) & (nb == 11))
                again = False
                for (ex_, ey_) in zip(exs_.tolist(), eys_.tolist()):
                    if sk[ey_, ex_] == 0:
                        continue
                    br, cur = [(ex_, ey_)], (ex_, ey_)
                    hit_j = False
                    for _ in range(16):
                        if int(nb[cur[1], cur[0]]) >= 12:      # 已到分叉点 → 停 ✓
                            hit_j = True
                            break
                        cand = [(cur[0] + dx, cur[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                if (dx or dy) and 0 <= cur[0] + dx < sk.shape[1] and 0 <= cur[1] + dy < sk.shape[0]
                                and sk[cur[1] + dy, cur[0] + dx] > 0 and (cur[0] + dx, cur[1] + dy) not in br]
                        if not cand:
                            break
                        cur = cand[0]
                        br.append(cur)
                    if hit_j and len(br) <= 15:                # 短支 → 删掉（分叉点本身保留 ✓）
                        for (bx, by) in br:
                            if int(nb[by, bx]) < 12:
                                sk[by, bx] = 0
                        again = True
                if not again:
                    break
            # ← 骨架层直接剪掉"距焊点 R 以内"的部分 ✓（使用者 2026-10-02 指定 ✓）
            #    判直度只用【线身】，端头附近的弯折/分叉在骨架层一次剪干净 ✓ → 再取最长的一段 ✓
            # 剪端半径【自适应】✓：两端焊点间距小的短线不能按 9.92px 剪 ✗（否则骨架被剪空 → 正常飞线漏判 ✗）
            if len(w["pts2"]) == 2:
                _d = float(np.hypot(P_pts[w["pts2"][0]][0] - P_pts[w["pts2"][1]][0],
                                    P_pts[w["pts2"][0]][1] - P_pts[w["pts2"][1]][1]))
                _r = max(3.0, min(float(os.environ.get("CUT_R", "9.92")), 0.30 * _d))   # 剪端半径可调 ✓（9.92 剪不完焊盘喇叭口 ✗）
            else:
                _r = float(R_TRIM)
            for q in w["pts2"]:
                x_p, y_p = P_pts[q]
                cv2.circle(sk, (int(round(x_p)), int(round(y_p))), int(round(_r)), 0, -1)
            n_sk, lab_sk, st_sk, _c_sk = cv2.connectedComponentsWithStats(sk.astype(np.uint8), 8)
            if n_sk > 1:
                big_sk = 1 + int(np.argmax(st_sk[1:, 4]))
                sk = (lab_sk == big_sk).astype(np.uint8)
            path = order_path2(sk)                      # ← 最长测地线 ✓（原 order_path 贪心走法在分叉处会断 ✗）
            mt = metrics(path)
            if mt is None:                       # ← 丢弃【必须留痕】✗（原来静默 continue → 名单前后对不上号 ✗）
                with open(os.path.join(UN, "reports/wire_dropped.jsonl"), "a", encoding="utf-8") as _fd:
                    _fd.write(_json.dumps({"board": key, "cid": w.get("cid"), "area": int(w["mask"].sum()),
                                           "skel_px": int((sk > 0).sum()),
                                           "reason": "剪端后骨架不足 8 点 → 不参与直度"},
                                          ensure_ascii=False) + "\n")
                print("     丢弃：{} cid={} 面积 {}px 骨架 {}px（剪端后不足 8 点 ✗）".format(
                    key, w.get("cid"), int(w["mask"].sum()), int((sk > 0).sum())))
                continue
            od_w = os.path.join(UN, "pred_masks_wires"); os.makedirs(od_w, exist_ok=True)
            wf = "{}_{:03d}.png".format(base, int(w.get("cid", len(res))))     # ← 用【稳定编号】存盘 ✓
            cv2.imwrite(os.path.join(od_w, wf), (m * 255).astype(np.uint8))
            mt["board"] = key                        # ← 不写 OK/NG 标签 ✓（标签不由我判定 ✗）
            mt["cid"] = int(w.get("cid", len(res)))
            mt["file"] = wf
            res.append(mt)
        out += res
        print("  {:<12} 合格飞线 {} 条 → 可算直度 {} 条".format(key, len(qual), len(res)))
    out.sort(key=lambda r: -r["jit_max"])
    print("\n逐条读数（按抖动降序 ✓；不写 OK/NG 标签 ✗ —— 标签不由助手判定 ✓）")
    for r in out:
        print("  {:<12} #{}  抖动 {:>6.2f}px ｜ 弧弦比 {:.4f} ｜ 曲率P95 {:.4f} ｜ 最小曲率半径 {:>7.1f}px ｜ 低频能量 {:.3f} ｜ 弯曲能量 {:.4f} ｜ 线长 {}px".format(
            r["board"], r["file"].split("_")[-1].replace(".png", ""), r["jit_max"], r["arc_chord"],
            r["kappa_p95"], r["R_min"], r["low_energy"], r["bend_energy"], r["len_px"]))
    json.dump(out, open(os.path.join(UN, "reports/straightness_skeleton.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n共 {} 条合格飞线的线段读数 ✓；线段掩膜 -> unet/pred_masks_wires/ ✓".format(len(out)))
    return 0


# ---- 来自 256_wire_panels.py ----
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


# ---- 来自 257_all_boards_numbered.py ----
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


# ---- 来自 257_all_boards_numbered.py ----
def main():
    rows = json.load(open(os.path.join(UN, "reports/straightness_skeleton.json"), encoding="utf-8"))
    pts = {}
    for e in json.load(open("<项目根>/chip_wire_ng/out/cnn/solder_final_20261001_v2.json",
                            encoding="utf-8"))["rows"]:
        pts.setdefault(e["sample"], []).append((float(e["x"]), float(e["y"])))
    by_board = {}
    for r in rows:
        by_board.setdefault(r["board"], []).append(r)
    # ---- 不合格名单 ✓（使用者 2026-10-02 定：弯曲判据 + 断联 7 条 ✓）----
    OKB = set()  # 发布版置空：请按你的数据集填写板名
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
    for grp, stem, key in BOARDS:
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
    print("图 -> {}  {:.2f}MB  {}×{}  ✓ 标号骨架合计 {} 条（编号-读数对照 -> unet/reports/wire_ids.json ✓）".format(
        fn, os.path.getsize(fn) / 1048576, pg.shape[1], pg.shape[0], tot))
    return 0


# ---- 来自 259_stage_figs.py ----
def find_src(grp, key):
    for f in sorted(os.listdir(os.path.join(SRC, grp))):
        if f.endswith(".jpg") and key.split("/")[-1] in f:
            return cv2.imread(os.path.join(SRC, grp, f), cv2.IMREAD_GRAYSCALE)
    return None


# ---- 来自 259_stage_figs.py ----
def skel_of(sub_u8):
    sk = cv2.ximgproc.thinning(np.ascontiguousarray(sub_u8))
    return (sk > 0).astype(np.uint8)


# ---- 来自 259_stage_figs.py ----
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
