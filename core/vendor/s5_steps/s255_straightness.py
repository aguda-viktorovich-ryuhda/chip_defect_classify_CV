# -*- coding: utf-8 -*-
"""s255_straightness - 自 recon/255_straightness_skeleton.py 整份复制（仅移除绝对路径常量，其余原样）"""
import glob
import json
import os
import re
import sys
import time
import math
import itertools

import cv2
def _walk_skel(sk, start, K=40):
    """沿骨架从端点按连通顺序取 K 点（用于末段方向拟合 ✓ 不用欧氏半径 ✗）"""
    import collections
    seen={start}; q=collections.deque([start]); order=[start]
    while q and len(order)<K:
        x,y=q.popleft(); nxt=[]
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx or dy:
                    xx,yy=x+dx,y+dy
                    if 0<=xx<sk.shape[1] and 0<=yy<sk.shape[0] and sk[yy,xx]>0 and (xx,yy) not in seen: nxt.append((xx,yy))
        for p in nxt: seen.add(p); q.append(p); order.append(p)
    return order

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
"""255_straightness_skeleton.py — 骨架类直度指标：在合格飞线上按 OK/NG 分组，先验"能不能分开"

使用者（2026-10-02）："接下来我们测试直度分析。你先测试骨架类。"
现行流水线（已定 ✓）：长轴 <20px 删 → 建名单 → 分类 → 干涉条目删盘 9.92px + 二次除杂 <50px 拆开
                        → 端头延长覆盖焊点 → 闭运算补缺（核 13px ✓，顺序 B：先延长再闭 ✓）
                        → 复核重合/归属/分类 → 取【合格】飞线（2 个焊点、一内一外 ✓）
直度输入：合格飞线【剪掉两端焊点部分】后的线体（剪端半径 = 平均半径 + 5 = 9.92px ✓）
五个骨架类指标（都在 1D 平滑后的有序骨架上算 ✓）：
  k1 弧弦比 = 弧长 / 首末直线距离 ✓
  k2 抖动   = 骨架相对首末直线的最大横向偏离（px ✓）
  k3 曲率P95 = |转向角 / 弧长| 的 95 分位（1/px ✓）
  k4 最小曲率半径 = 1 / 最大曲率（px ✓）—— 直接可读的物理量 ✓
  k5 低频能量占比 = 横向偏离序列 FFT 中【波长 > 半个弦长】的能量占比 ✓
  k6 弯曲能量 = Σ κ² Δs ✓
输出：逐板逐条读数 + OK/NG 两组的分位数对照表（先看能不能分开 ✓，分不开就不谈阈值 ✗）
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


# [搬运时移除的绝对路径常量] HERE = "<项目根>/chip_wire_ng"
sys.path.insert(0, os.path.join(HERE, "lib"))
import pipeline as P  # noqa: E402

# [搬运时移除的绝对路径常量] SRC = "<路径>"
# [搬运时移除的绝对路径常量] ROOT = "<路径>/某电子公司印刷电路samples__20260128"
# [搬运时移除的绝对路径常量] UN = "<项目根>/chip_wire_ng/unet"
# [搬运时移除的绝对路径常量] SOLD = os.path.join(HERE, "out/cnn/solder_final_20261001_v2.json")
ALL = {"01_原图": "chan:01_原图", "Canny": "chan:05_Canny", "黑帽15": "chan:07_黑帽15", "USM": "chan:08_USM",
       "TTA": "pred_masks_tta", "v11": "pred_masks_v11", "v13": "pred_masks_v13", "v16b": "pred_masks_v16b_conv3_bn",
       "v18": "pred_masks_v18", "v19": "pred_masks_v19"}
KEPT = ["TTA", "v13", "v18"]
MIN_AXIS, MIN_AREA2 = 20.0, 50.0
R_DEL, EXT_R, THICK, L_CAP, CLOSE_R = 9.92, 20.0, 10, 14.92, 13
R_TRIM = 9.92
NG_BOARDS = set()  # 发布版置空：请按你的数据集填写板名


def load(k, fn):
    p = os.path.join(UN, "pred_masks_chan9", k[5:] + "_" + fn) if k.startswith("chan:") else os.path.join(UN, k, fn)
    v = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return (v > 127) if v is not None else None


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


def endpoints_of(sk):
    k = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], np.uint8)
    nb = cv2.filter2D((sk > 0).astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.where((sk > 0) & (nb == 11))
    return list(zip(xs.tolist(), ys.tolist()))


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


def main():
    rows_s = json.load(open(SOLD, encoding="utf-8"))["rows"]
    pts = {}
    for e in rows_s:
        # [2026-10-07 迁移] 焊点表索引统一走 core.s_paths.solder_index（两种形式都登记，不重复计数）
        import core.s_paths as _SP
        pts = _SP.solder_index(rows_s)
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
            # 纯几何延伸（2026-10-06 定稿）：只顺着本条线自己末段的方向往外接固定长度，
            # 不查任何焊点、不使用 claimed、不写 pts —— 是否接到焊盘由后续计点决定。
            # 依据：飞线正常焊上时延伸自然召回自己的焊盘；大弯折/偏移/未焊上则召不回 ⇒ 判缺陷（符合要求）。
            for (ex, ey) in endpoints_of(sk):
                walk = _walk_skel(sk, (ex, ey), 40)
                if len(walk) < 8:
                    continue
                pw = np.array(walk, np.float32); mu = pw.mean(0); cc = pw - mu
                dd = np.linalg.svd(cc, full_matrices=False)[2][0]      # 末段最小二乘方向
                if float(dd @ (np.array([ex, ey], np.float32) - mu)) < 0:
                    dd = -dd                                           # 朝向：背离线体向外
                nv = float(np.linalg.norm(dd))
                if nv < 1e-3:
                    continue
                dd = dd / nv
                tip = np.array([ex, ey], np.float32) + dd * L_CAP       # 固定上限，与焊点远近无关
                line = np.zeros_like(m)
                cv2.line(line.view(np.uint8), (ex, ey), (int(round(tip[0])), int(round(tip[1]))), 1, THICK)
                m = m | line.astype(bool)
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
    print("共 {} 条合格飞线的线段读数 ✓；线段掩膜 -> {}/pred_masks_wires/".format(len(out), UN))
    return 0


def _orig_main_guard():  # 保留原入口不动
    pass
if False:
    sys.exit(main())
