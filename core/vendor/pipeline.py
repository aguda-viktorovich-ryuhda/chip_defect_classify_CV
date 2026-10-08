#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py  —  焊点识别的权威流程（单一定义处，所有参数集中在此）

定案（2026-09-22，由使用者确认）：
  · 范围掩码 = 存档模块掩码 腐蚀 k=4  → MASK_ERODE = 4
        · 掩码外的像素不参与任何计算（外圈沟槽在结构上被排除）
  · 半径窗口放宽为 2.5~8.0（原 4.5~7.5 是按 OK-1 标定的，把 NG-1 整组切掉）

七个步骤（每步都可单独 dump 中间图核对）：
  S1 掩码       read_archive_mask → shape_mask(erode=MASK_ERODE)     范围限定
  S2 预滤波     中值 k5（可插拔：none/median/gaussian/bilateral）      扫参表唯一满分点
  S3 局部反差   D = 闭运算(椭圆41) − 预滤波图；M = D > (D 的 96 分位)   逐图自适应（约选中 4%）
  S4 抹细线     O = 开运算(M, 圆盘半径2)                              飞线宽4~5.6px被抹掉，焊点留下
  S5 距离变换   DT = distanceTransform(M)；只在 O 内取峰（11×11 局部极大）  内切半径
  S6 三重过滤   半径窗口 2.5~8.0 → k4 掩膜内 → SEP 非极大值抑制          得到候选
  S7 焊点判据   mf = 环带(1.4r~2.4r)均值 − 点心(0.6r)均值；**分圈阈值 内127/外102**
  S8 误检剔除   A 分界带 |dtin−RING_K|≤40 / B 20px 局部最优 / C half_min>100
  S9 交点剔除   外圈：点是否落在两条线段的交点（Canny+概率霍夫）→ 焊点不会是两线交点
  —— 另有分圈：ring_of() 用【补边版】距离变换，RING_K=94 为内外圈分界。

⚠ 当前 CNN 路线（`out/cnn/REPORT.md`）：S1–S6 不动，S7 可由 CNN 概率替代（外圈阈值 102 定标得来），
   S8/S9 保留作兜底；CNN 逐点仲裁在 CPU 上 88ms/图，比 S9 的 Canny+霍夫（362ms/图）更便宜。
"""
import os
import sys
import json
import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "lib"))
import seg   # noqa: E402

SRC = "chip_samples__20260128"   # 交付软件内不使用此常量（s1 走 core/mask.py 现场读图）；仅为兼容旧脚本保留，指向软件自身的输入目录
MASKDIR = os.path.join(HERE, "out", "masks")

# ---- 全部参数（改这里）----
MASK_ERODE = 4          # 定案：焊点识别的掩码腐蚀半径
GAUSS_PRE = 5           # S2 预平滑窗口
KCLOSE = 41             # S2 闭运算核直径（局部背景估计）
D_PCT = 96              # S3 分位阈值（扫参表唯一满分点；脚本显式传 d_pct 时以传入为准）
RE = 2.0                # S4 开运算半径
DT_MIN = 1.5            # S5 峰的最小内切半径
RMIN, RMAX = 2.5, 8.0   # S6 半径窗口（定案）
SEP = 9.0               # S6 非极大值抑制间距
MF_T = 126.0            # S7 mf 阈值（已知对 2_ 组偏严，待修）
RING_IN, RING_OUT = 1.4, 2.4   # S7 环带半径倍数
INNER_K = 0.6                  # S7 点心半径倍数

# ---- 弱反差候选分支（与 S1–S6 并列的补充来源；标定脚本 recon/171_weak_branch.py）----
#   3 个断联焊点 D=119/140/122 < 所在图 tau=158/147/147 → S3 二值化即漏、S5 无峰 →
#   它们压根不是候选。但它们的 mf（环带均值 − 点心均值）= 118/127.7/120.4，与真焊点同区间。
#   ⇒ 另设一条【以 mf 为判据】的候选分支：只补点，不改 S1–S6 的任何参数 → 原候选坐标逐点不变。
#   阈值标定（10 板扫表）：t=90 是能召回全部 3 个目标点的最紧设置
#   （目标点 mf = 105.8/96.0/97.1；t=105 起 ①③ 丢失）。
#   并集去重半径 = 10px（实测：5px 时新增 16 个，其中 13 个是"旧候选已判非焊点、只因换判据
#   峰位偏 5–8px"的同一暗斑；10px 时新增正好 3 个 = 3 个目标点，且 10~25px 区间结果不变）。
#   使用者 2026-09-28 裁定：采用 t=90 + 去重 10px ⇒ 并集 1326 → 1329。
WEAK_MF_T = 90.0
WEAK_DEDUP = 10.0
WEAK_WIN_BIG, WEAK_WIN_SMALL = 25, 7      # 环带/点心均值方窗边长（奇数）


def mfmap(g):
    """mf 图：大窗均值 − 小窗均值 ≈ 环带均值 − 点心均值。越大越像焊点（暗心被亮焊座包住）。"""
    a = g.astype(np.float32)
    n1, n2 = WEAK_WIN_SMALL ** 2, WEAK_WIN_BIG ** 2
    s1 = cv2.boxFilter(a, -1, (WEAK_WIN_SMALL, WEAK_WIN_SMALL), normalize=False)
    s2 = cv2.boxFilter(a, -1, (WEAK_WIN_BIG, WEAK_WIN_BIG), normalize=False)
    return (s2 - s1) / float(n2 - n1) - s1 / float(n1)


def weak_candidates(g, mask, t=None):
    """弱反差分支候选：mf 图 > t → 开运算 → DT 峰 → 半径窗口(RMIN/RMAX) + NMS(SEP)，只留 mask 内。"""
    t = WEAK_MF_T if t is None else t
    M = (mfmap(g) > t).astype(np.uint8) * 255
    M = cv2.morphologyEx(M, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    DT = cv2.distanceTransform((M > 0).astype(np.uint8), cv2.DIST_L2, 5)
    dil = cv2.dilate(DT, np.ones((3, 3), np.uint8))
    ys, xs = np.where((DT >= dil - 1e-6) & (DT > DT_MIN) & (DT >= RMIN) & (DT <= RMAX))
    pts = sorted([(float(x), float(y), float(DT[y, x])) for y, x in zip(ys, xs)],
                 key=lambda q: -q[2])
    keep = []
    for x, y, r in pts:
        if mask[int(round(y)), int(round(x))] == 0:
            continue
        if all((x - a) ** 2 + (y - b) ** 2 > SEP * SEP for a, b, _ in keep):
            keep.append((x, y, r))
    return keep


def candidates_union(g, mask, d_pct=None, weak_t=None, dedup=None):
    """原 S1–S6 候选 ∪ 弱反差分支候选（dedup 内去重，原候选优先）。
    返回 (base, extra)：base 逐点等于 candidates(stages(...)) 的结果（硬约束，勿改路径）。"""
    _, kept = candidates(stages(g, mask, prefilter=pre_median,
                                d_pct=(D_PCT if d_pct is None else d_pct))["peaks"], mask)
    base = [(float(x), float(y), float(r)) for (x, y, r) in kept]
    dedup = WEAK_DEDUP if dedup is None else dedup
    extra = [w for w in weak_candidates(g, mask, t=weak_t)
             if min(((w[0] - b[0]) ** 2 + (w[1] - b[1]) ** 2) ** 0.5 for b in base) > dedup]
    return base, extra

# ---- 分圈（内圈 / 外圈）参数：由已确认焊点到掩膜内边界的距离分布定标 ----
#   OK-1：外圈 16 个在 21~34px、内圈 18 个在 155~173px，中间空档 121px
#   NG-2：外圈  4 个在 36~38px、内圈  5 个在 236~299px，中间空档 198px
#   取空档中点 → 分界线两侧各留 ~60px 余量（k 在 40~150 全区间归属不变）。
#   外圈零假点（假点全在边界或内圈），故外圈阈值可单独放松；内圈保持严格。
RING_K = 94
RING_TI, RING_TO = 127.0, 102.0

# ---- 误检剔除（S8）：由 NG-2-样例板G 上 5 个误检 + NG-1-样例板A 上 3 个误检的标定而来 ----
#   A 分界带剔除：两圈之间的空档（走廊）内不该有焊点 → 距 RING_K ±BAND 的候选直接丢弃。
#     两侧约束（实测）：下边界必须 > 38（NG-2 外圈真焊点在 dtin 36~38）、
#     上边界必须 ≥ 118（NG-1-样例板A 三个误检在 dtin 116~118）。
#     取 BAND=40（带子 54~134）→ 两侧各留 16px 余量。
#     BAND=20 时那三个点擦边溜过（116~118 > 114）；BAND=60 起开始吃真焊点（NG-2 掉到 4/9）。
#   B 局部最优响应：20px 内只留 mf 最高的点（原 9px NMS 太松，同一结构可出多个响应）
#   C 环带最暗侧 half_min：真焊点四周被亮焊座包住、环带没有"暗的一侧"；
#     沟槽/飞线交点必有一侧是暗的。pct=96+A+B 后 假点 hm=79/85、真点 hm=108~201，取 100 居中。
BAND = 40.0
NMS2 = 20.0
HALF_MIN_T = 100.0


def half_min(g, px, py, r, nbins=12, darkest=4):
    """环带(1.4r~2.4r)按角度分 nbins 档，返回最暗 darkest 档的平均亮度。
       物理含义：真焊点四周是亮焊座 → 各方向都亮；沟槽/飞线交点必有一侧暗。"""
    ri, ro = max(1, int(round(r * 1.4))), max(2, int(round(r * 2.4)))
    if py - ro < 0 or px - ro < 0 or py + ro >= g.shape[0] or px + ro >= g.shape[1]:
        return None
    yy, xx = np.ogrid[-ro:ro + 1, -ro:ro + 1]
    dist = np.hypot(yy, xx)
    ring = (dist >= ri) & (dist <= ro)
    patch = g[py - ro:py + ro + 1, px - ro:px + ro + 1].astype(np.float32)
    rv = patch[ring]
    ang = np.arctan2(yy, xx)[ring]
    bins = np.linspace(-np.pi, np.pi, nbins + 1)
    prof = [float(rv[(ang >= bins[i]) & (ang < bins[i + 1])].mean())
            for i in range(nbins) if ((ang >= bins[i]) & (ang < bins[i + 1])).any()]
    if not prof:
        return None
    return float(np.sort(np.array(prof))[:darkest].mean())


def ring_of(mask, erode=RING_K):
    """按使用者方案分圈：内圈 = 掩膜再腐蚀 RING_K；外圈 = 掩膜 − 内圈。
       返回 (inner, outer, dtin)，dtin = 到掩膜内边界的距离。

       dtin 必须用【补边后的】距离变换：cv2.distanceTransform 只把图像内部的 0 当背景，
       若掩膜贴到图像边缘（1_ 组六张都贴上边、2_ 组四张都贴右边——上排焊座被画面裁掉了），
       边缘附近像素"朝外"找不到背景，距离会绕去量到别处（实测 (349,20) 算出 275px），
       于是外圈焊点被误判进内圈，且分界带剔除无法生效。
       补一圈 0 之后："图像外"也算边界 → (349,20) 的 dtin 由 275 修正为 21。"""
    inner = seg.shape_mask(mask, erode=erode)
    outer = mask & (~inner)
    u = (mask.astype(np.uint8) * 255)
    p = cv2.copyMakeBorder(u, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dtin = cv2.distanceTransform(p, cv2.DIST_L2, 5)[1:-1, 1:-1]
    return inner, outer, dtin


def detect_rings(g, mask, prefilter=None, d_pct=None):
    """分圈检出 + S8 误检剔除（A 分界带剔除 / B 局部最优响应 / C 环带最暗侧）。
       返回的点带 ring 与 half_min 字段。"""
    inner, outer, dtin = ring_of(mask)
    st = stages(g, mask, prefilter=prefilter, d_pct=d_pct)
    _, kept = candidates(st["peaks"], mask)
    dots = with_mf(g, kept)
    # A：分界带内的候选直接丢弃（两圈之间的空档里没有焊点）
    stage1 = []
    for d in dots:
        dd = float(dtin[int(d["y"]), int(d["x"])])
        if abs(dd - RING_K) <= BAND:
            continue
        if d["mf"] > (RING_TI if dd >= RING_K else RING_TO):
            d["ring"] = "inner" if dd >= RING_K else "outer"
            d["dtin"] = dd
            stage1.append(d)
    # B：局部最优响应（间距 NMS2，保留 mf 最高者）
    stage2 = []
    for d in sorted(stage1, key=lambda v: -v["mf"]):
        if any(np.hypot(d["x"] - e["x"], d["y"] - e["y"]) <= NMS2 for e in stage2):
            continue
        stage2.append(d)
    # C：环带最暗侧下限
    out = []
    for d in stage2:
        hm = half_min(g, int(d["x"]), int(d["y"]), float(d["r"]))
        d["half_min"] = hm
        if hm is not None and hm > HALF_MIN_T:
            out.append(d)
    # D（S9）：外圈交点剔除 —— 落在两条线交点上的不是焊点（使用者指认的飞线×沟槽交点）
    segs = crossing_lines(g)
    fin = []
    for d in out:
        if d["ring"] == "outer" and outer_crossing(g, int(d["x"]), int(d["y"]), segs):
            continue
        fin.append(d)
    return fin


def crossing_lines(g):
    """S9 用的线段表（Canny + 概率霍夫）。每张图算一次，别逐点重算。"""
    gb = cv2.medianBlur(g, 5)
    edges = cv2.Canny(gb, 40, 110)
    r = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=18, minLineLength=25, maxLineGap=6)
    return [] if r is None else [tuple(map(float, s[0])) for s in r]


def outer_crossing(g, px, py, segs=None):
    """S9：判该点是否落在"两条线段的交点"上（使用者思路：焊点不会是两线交点）。

    实现要点（两次踩坑）：
      ① 不能用 pct=96 暗结构的骨架：沟槽是中灰（100~200），根本不在暗结构里，
         骨架中只有飞线 → 飞线×沟槽交点测不出来（(582,43) 就是这样漏掉的）。
      ② 不能只要求"点附近有 ≥2 个不同方向的线段"：内圈焊座密集、边缘到处都是，
         误报 42/248（含 mf 160+ 的强焊点）。必须求两条线段的**交点本身**落在点附近。
    参数：交点距点 ≤5px、两条线距点 ≤12px、方向差 >20°、线段长 ≥25px。
    实测（这些参数下）：已确认真焊点误判 2/34 与 0/9、使用者标的 8 个真焊点 0/8；
    使用者指认的飞线×沟槽交点 (582,43) 判出（距交点 1.7px）。
    ⚠ 内圈仍有少数误报警（焊座密集），故本规则**只在外圈生效**。
    """
    if segs is None:
        segs = crossing_lines(g)
    loc = []
    for s in segs:
        x1, y1, x2, y2 = [float(v) for v in s]
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        if L2 < 1e-9:
            continue
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
        cx, cy = x1 + t * dx, y1 + t * dy
        d = float(np.hypot(px - cx, py - cy))
        if d <= 12.0:
            loc.append(((x1, y1, x2, y2), abs(np.degrees(np.arctan2(dy, dx))) % 180.0))
    for i in range(len(loc)):
        for j in range(i + 1, len(loc)):
            s1, a1 = loc[i]
            s2, a2 = loc[j]
            if min(abs(a1 - a2) % 180, 180 - abs(a1 - a2) % 180) <= 20.0:
                continue
            x1, y1, x2, y2 = s1
            x3, y3, x4, y4 = s2
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(den) < 1e-9:
                continue
            aa = x1 * y2 - y1 * x2
            bb = x3 * y4 - y3 * x4
            ix = (aa * (x3 - x4) - (x1 - x2) * bb) / den
            iy = (aa * (y3 - y4) - (y1 - y2) * bb) / den
            if float(np.hypot(ix - px, iy - py)) <= 5.0:
                return True
    return False


def read_archive_mask(rel, maskdir=MASKDIR, erode=MASK_ERODE):
    return seg.shape_mask(seg.load_mask(maskdir, rel), erode=erode)


def pre_gaussian(g):
    """S2 默认预滤波：高斯 5×5（sigma 自动 ≈1.1）"""
    return cv2.GaussianBlur(g, (GAUSS_PRE, GAUSS_PRE), 0)


def pre_median(g):
    return cv2.medianBlur(g, GAUSS_PRE)


def pre_bilateral(g):
    return cv2.bilateralFilter(g, 7, 50, 7)


def pre_none(g):
    return g.copy()


PREFILTERS = dict(none=pre_none, median=pre_median, gaussian=pre_gaussian,
                  bilateral=pre_bilateral)


def stages(g, mask, prefilter=None, d_pct=None):
    """逐步骤计算，返回所有中间量（供出图/核对）
       prefilter: 可插拔的 S2 预滤波（不传=高斯）；决定 S3 起的一切下游结果
       d_pct:     临时覆盖 S3 的二值化分位（不传=用 D_PCT 常量）"""
    gb = (prefilter or pre_gaussian)(g)                                       # S2a
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KCLOSE, KCLOSE))
    close = cv2.morphologyEx(gb, cv2.MORPH_CLOSE, k)                          # S2b
    D = np.clip(close.astype(np.int16) - gb.astype(np.int16), 0, 255).astype(np.float32)
    tau = float(np.percentile(D, D_PCT if d_pct is None else d_pct))          # S3a
    M = ((D > tau).astype(np.uint8)) * 255                                    # S3b
    d2 = 2 * int(np.ceil(RE)) + 1
    O = cv2.morphologyEx(M, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (d2, d2)))                                         # S4
    DT = cv2.distanceTransform(M, cv2.DIST_L2, 5)                             # S5a
    DTo = np.where(O > 0, DT, 0).astype(np.float32)                           # S5b
    mx = cv2.dilate(DTo, np.ones((11, 11), np.float32))
    ys, xs = np.nonzero((DTo >= mx - 1e-6) & (DTo > DT_MIN))                  # S5c
    order = np.argsort(-DTo[ys, xs])
    peaks = [(int(xs[i]), int(ys[i]), float(DTo[ys[i], xs[i]])) for i in order]
    return dict(gb=gb, close=close, D=D, tau=tau, M=M, O=O, DT=DT, DTo=DTo, peaks=peaks)


def candidates(peaks, mask):
    """S6 三重过滤：半径窗口 → 掩码内 → 非极大值抑制"""
    raw, kept = [], []
    for x, y, r in peaks:
        if not (RMIN <= r <= RMAX):
            continue
        if any((x - a) ** 2 + (y - b) ** 2 <= SEP * SEP for a, b, _ in raw):
            continue
        raw.append((x, y, r))
        if not mask[y, x]:
            continue
        kept.append((x, y, r))
    return raw, kept


def with_mf(g, pts):
    """S7 焊点判据：黑心 + 亮环"""
    out = []
    for (x, y, r) in pts:
        mm = np.zeros(g.shape, np.uint8)
        cv2.circle(mm, (x, y), max(1, int(round(r * INNER_K))), 255, -1)
        inner = float(g[mm > 0].mean())
        rr = np.zeros(g.shape, np.uint8)
        cv2.circle(rr, (x, y), int(round(r * RING_OUT)), 255, -1)
        cv2.circle(rr, (x, y), int(round(r * RING_IN)), 0, -1)
        ring = float(g[rr > 0].mean()) if (rr > 0).any() else 0.0
        out.append(dict(x=float(x), y=float(y), r=float(r), mf=ring - inner,
                        inner=inner, ring=ring))
    return out


def detect(g, mask, return_trace=False, prefilter=None, d_pct=None):
    """端到端：原图 + 掩码 → 焊点列表（mf>MF_T）"""
    st = stages(g, mask, prefilter=prefilter, d_pct=d_pct)
    raw, kept = candidates(st["peaks"], mask)
    dots = with_mf(g, kept)
    hits = [d for d in dots if d["mf"] > MF_T]
    if return_trace:
        return hits, dots, raw, st
    return hits


def main():
    import glob
    rels = [os.path.relpath(f, SRC) for f in sorted(glob.glob(os.path.join(SRC, "*", "*")))]
    print("参数: MASK_ERODE={} KCLOSE={} D_PCT={} RE={} RMIN~RMAX={}~{} SEP={} MF_T={}".format(
        MASK_ERODE, KCLOSE, D_PCT, RE, RMIN, RMAX, SEP, MF_T))
    print("\n{:34} {:>8} {:>8} {:>8} {:>10}".format(
        "image", "掩膜%", "候选(前)", "掩膜内", "焊点数"))
    total = 0
    for rel in rels:
        g = cv2.imread(os.path.join(SRC, rel), cv2.IMREAD_GRAYSCALE)
        mask = read_archive_mask(rel)
        hits, dots, raw, st = detect(g, mask, return_trace=True)
        total += len(hits)
        print("{:34} {:>7.1f}% {:>8} {:>8} {:>10}".format(
            rel[:34], 100 * mask.mean(), len(raw), len(dots), len(hits)))
    print("{:34} {:>8} {:>8} {:>8} {:>10}".format("合计", "", "", "", total))


if __name__ == "__main__":
    sys.exit(main())
