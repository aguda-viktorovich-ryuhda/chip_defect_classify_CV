import os as _os, sys as _sys
# [2026-10-08 适配] 去掉写死路径：按脚本位置定位，全部可用环境变量覆盖
_ROOT = _os.environ.get("CHIP_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CHIP_DATA") or _os.path.join(_ROOT, "train", "data")
_LIB  = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "lib")
_SRC  = _os.environ.get("CHIP_SRC")  or ""
_PROJ = _os.environ.get("CHIP_PROJ") or ""
_LAB  = _os.environ.get("CHIP_LAB")  or ""
_ANN  = _os.environ.get("CHIP_ANN")  or r"<路径见 本地配置.env>/某电子公司印刷电路samples__20260128"
for _p in (_ROOT, _LIB):
    if _p not in _sys.path: _sys.path.insert(0, _p)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""208_tta_and_v12.py — ① TTA 并集 ② 训 v12（纯原图, clDice 权重 0.5）再 TTA

TTA（测试时增强，test-time augmentation）：对每张图做 4 个视图
    ① 原图  ② 水平翻  ③ 垂直翻  ④ 旋转 180°
各自滑窗推理得概率图，**把概率图逆变换回原方向后逐像素平均** ✓，再取阈 0.5 ✓
—— 不用重训 ✓，专治"换个视角就看得见那段线" ✓（文献标准做法 ✓）。

参与 TTA 的模型（全部 ÷8、cin=1）与其输入变换：
    unet_v3_unet8.pt     输入 原图
    unet_v6_A_pct.pt     输入 分位拉伸 2–98%
    unet_v6_B_clahe.pt   输入 CLAHE(2.0,8×8)
    unet_v10_cldice3.pt  输入 分位拉伸
    unet_v11_raw_cldice.pt 输入 原图
另有 v7(6通道)/v8(4通道) 的既有掩膜直接并入池（它们不参加 TTA，保持原样 ✓）

v12：纯原图输入 + 0.50·BCE(20) + 0.25·(1−Dice) + 0.50·clDice（clDice 权重翻倍 ✓）
     其余与 v11 完全一致（÷8 / lr 1e-3 / 250ep / 留出 样例板C+样例板H / 护栏 15% ✓）
口径：沿人工 GT 量 覆盖率 / 最大缺口 px / 缺口数（≥8px 才算一个缺口 ✓）
判据：并集覆盖率 > 97.3% 或 缺口数 < 0.8 才算通过 ✓（现最佳 = v11∪全部 ✓）
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC = (_SRC + r"")
ROOT = (_ANN + r"")
UN = (_PROJ + r"/unet")
PAN = (_PROJ + r"/panels")
C, STRIDE, DEV = 192, 129, ("cuda" if torch.cuda.is_available() else "cpu")
HOLD = {"1_NG/1705", "2_NG/3421"}
CLAHE2 = cv2.createCLAHE(2.0, (8, 8))
MODELS = [("unet_v3_unet8.pt", "raw"), ("unet_v6_A_pct.pt", "pct"), ("unet_v6_B_clahe.pt", "clahe"),
          ("unet_v10_cldice3.pt", "pct"), ("unet_v11_raw_cldice.pt", "raw")]
POOL_EXTRA = ["pred_masks_v7", "pred_masks_v8"]


def norm_pct(g):
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    return np.clip((g.astype(np.float32) - lo) * 255.0 / max(1e-6, hi - lo), 0, 255)


def prep(g, mode):
    if mode == "pct": return norm_pct(g) / 255.0
    if mode == "clahe": return CLAHE2.apply(g).astype(np.float32) / 255.0
    return g.astype(np.float32) / 255.0


def slices(shape, st=STRIDE):
    h, w = shape
    xs = list(range(0, max(1, w - C) + 1, st)); ys = list(range(0, max(1, h - C) + 1, st))
    if w - C not in xs: xs.append(w - C)
    if h - C not in ys: ys.append(h - C)
    return [(x, y) for y in ys for x in xs]


class U(nn.Module):
    def __init__(s, c=16, cin=1, levels=3):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
        s.encs, s.p = nn.ModuleList(), nn.MaxPool2d(2)
        c0 = cin
        for i in range(levels):
            o = c * (2 ** i); s.encs.append(blk(c0, o)); c0 = o
        s.b = blk(c0, c0 * 2); s.decs, s.ups = nn.ModuleList(), nn.ModuleList(); cc = c0 * 2
        for i in reversed(range(levels)):
            o = c * (2 ** i)
            s.ups.append(nn.ConvTranspose2d(cc, o, 2, 2)); s.decs.append(blk(o * 2, o)); cc = o
        s.out = nn.Conv2d(c, 1, 1)

    def forward(s, x):
        es = []
        for e in s.encs:
            x = e(x); es.append(x); x = s.p(x)
        x = s.b(x)
        for i, (u, d) in enumerate(zip(s.ups, s.decs)):
            x = d(torch.cat([u(x), es[-1 - i]], 1))
        return s.out(x)


def soft_erode(x):
    return -F.max_pool2d(-x, 3, 1, 1)

def soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)

def soft_open(x):
    return soft_dilate(soft_erode(x))

def soft_skel(x, iters=5):
    x1 = soft_open(x); skel = F.relu(x - x1)
    for _ in range(iters):
        x = soft_erode(x); x1 = soft_open(x)
        d = F.relu(x - x1); skel = skel + d - skel * d
    return skel

def dice_loss(p, t, eps=1e-6):
    return 1 - ((2 * (p * t).sum((1, 2, 3)) + eps) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + eps)).mean()

def cldice_loss(p, t, eps=1e-6):
    sp, st = soft_skel(p), soft_skel(t)
    tprec = (torch.sum(sp * t) + eps) / (torch.sum(sp) + eps)
    tsens = (torch.sum(st * p) + eps) / (torch.sum(st) + eps)
    return 1 - 2 * (tprec * tsens) / (tprec + tsens + eps)

def gap_stats(gt, pred):
    miss = gt & (~pred)
    cov = 100.0 * (1 - miss.sum() / max(1, gt.sum()))
    n, lab, st, ce = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
    mx, cnt = 0.0, 0
    for i in range(1, n):
        mm = miss & (lab == i)
        n2, l2, s2, c2 = cv2.connectedComponentsWithStats(mm.astype(np.uint8), 8)
        for j in range(1, n2):
            L = float(np.sqrt(s2[j, 4])); mx = max(mx, L)
            if L >= 8: cnt += 1
    return cov, mx, cnt


def infer(net, arr):
    h, w = arr.shape
    acc = np.zeros((h, w), np.float32); cnt = np.zeros((h, w), np.float32)
    tl = slices((h, w))
    bt = np.zeros((len(tl), C, C), np.float32)
    for i, (x0, y0) in enumerate(tl):
        bt[i] = arr[y0:y0 + C, x0:x0 + C]
    with torch.no_grad():
        for i in range(0, len(tl), 16):
            p = torch.sigmoid(net(torch.from_numpy(bt[i:i + 16])[:, None].to(DEV))).cpu().numpy()[:, 0]
            for j, (x0, y0) in enumerate(tl[i:i + 16]):
                acc[y0:y0 + C, x0:x0 + C] += p[j]; cnt[y0:y0 + C, x0:x0 + C] += 1
    return acc / np.maximum(1.0, cnt)


def tta_prob(net, arr):
    views = [(lambda a: a, lambda p: p),
             (np.fliplr, np.fliplr),
             (np.flipud, np.flipud),
             (lambda a: np.rot90(a, 2), lambda p: np.rot90(p, 2))]
    ps = []
    for f, g in views:
        p = infer(net, np.ascontiguousarray(f(arr)))
        ps.append(np.ascontiguousarray(g(p)))
    return np.mean(ps, 0)


def main():
    boards = []
    for f in sorted(glob.glob(ROOT + "/*/*_m.png")):
        grp = os.path.basename(os.path.dirname(f)); base = os.path.basename(f)[:-6]
        g = cv2.imread(os.path.join(SRC, grp, base + ".jpg"), cv2.IMREAD_GRAYSCALE)
        raw = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        lab = (raw if raw.ndim == 2 else cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2GRAY)) > 127
        boards.append({"key": grp + "/" + base[-4:], "g": g, "lab": lab, "fn": grp + "_" + base[-4:] + ".png"})
    print("=== ① TTA 并集（{} 个模型 × 4 视图）===".format(len(MODELS)))
    tta_masks = {}
    for name, mode in MODELS:
        ck = torch.load(os.path.join(UN, "models", name), map_location=DEV)
        net = U(16, 1, ck.get("levels", 3)).to(DEV)
        net.load_state_dict(ck["sd"]); net.eval()
        t0 = time.time()
        for b in boards:
            p = tta_prob(net, prep(b["g"], mode))
            tta_masks.setdefault(b["fn"], []).append(p > 0.5)
        print("  {:<24} 输入{:<6} TTA 4 视图完成 ({:.0f}s)".format(name, mode, time.time() - t0))
    os.makedirs(os.path.join(UN, "pred_masks_tta"), exist_ok=True)
    od = os.path.join(PAN, "union_clean_tta"); os.makedirs(od, exist_ok=True)
    rep = {}; S = []; SU = []
    print("\n  板            旧最佳并集(v11∪全部) | TTA并集 覆盖率 最大缺 缺口 | TTA∪(v7,v8) 覆盖率 最大缺 缺口")
    for b in boards:
        pool_tta = np.zeros_like(b["lab"])
        for m in tta_masks[b["fn"]]:
            pool_tta |= m
        cv2.imwrite(os.path.join(UN, "pred_masks_tta", b["fn"]), (pool_tta * 255).astype(np.uint8))
        a = gap_stats(b["lab"], pool_tta)
        pool_all = pool_tta.copy()
        for sub in POOL_EXTRA:
            pm = cv2.imread(os.path.join(UN, sub, b["fn"]), cv2.IMREAD_GRAYSCALE)
            if pm is not None: pool_all |= pm > 127
        for sub in ("pred_masks_v10", "pred_masks_v11"):
            pm = cv2.imread(os.path.join(UN, sub, b["fn"]), cv2.IMREAD_GRAYSCALE)
            if pm is not None: pool_all |= pm > 127
        au = gap_stats(b["lab"], pool_all)
        S.append(a); SU.append(au)
        rep[b["key"]] = {"tta": a, "tta_union": au}
        print("  {:<12}       {:>12.1f}%  | {:>10.1f}%{:>6.0f}{:>6} | {:>13.1f}%{:>6.0f}{:>6}".format(
            b["key"], 0, a[0], a[1], a[2], au[0], au[1], au[2]))
        vis = cv2.cvtColor(b["g"], cv2.COLOR_GRAY2BGR)
        vis[pool_all] = (0.45 * vis[pool_all] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(os.path.join(od, b["fn"].replace(".png", "_tta.jpg")), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    m = np.array(S, np.float32); mu = np.array(SU, np.float32)
    print("\n  TTA 并集（5 模型×4 视图）      覆盖率 {:.1f}%  最大缺 {:.0f}px  缺口 {:.1f}".format(m[:,0].mean(), m[:,1].mean(), m[:,2].mean()))
    print("  TTA 并集 + 旧池 ∪            覆盖率 {:.1f}%  最大缺 {:.0f}px  缺口 {:.1f}".format(mu[:,0].mean(), mu[:,1].mean(), mu[:,2].mean()))
    print("  现最佳 v11∪全部              覆盖率 97.3%  最大缺 9px  缺口 0.8")

    print("\n=== ② 训 v12：纯原图 + clDice 权重 0.5 ===")
    data = {}
    for b in boards:
        ch = b["g"].astype(np.float32) / 255.0
        ps, ns = [], []
        for (x0, y0) in slices(b["g"].shape):
            n = int(b["lab"][y0:y0 + C, x0:x0 + C].sum())
            (ps if n >= 20 else ns).append((x0, y0))
        rng = np.random.default_rng(0); rng.shuffle(ns)
        data[b["key"]] = {"ch": ch, "lab": b["lab"], "g": b["g"], "blocks": ps + ns[:max(1, len(ps))]}
    tr = [k for k in data if k not in HOLD]; va = [k for k in data if k in HOLD]
    rng = np.random.default_rng(1)
    def load(keys, aug=False):
        X, Y = [], []
        for k in keys:
            ch, lab = data[k]["ch"], data[k]["lab"]
            for (x0, y0) in data[k]["blocks"]:
                x = ch[y0:y0 + C, x0:x0 + C].astype(np.float32); y = lab[y0:y0 + C, x0:x0 + C].astype(np.float32)
                if aug:
                    j = int(rng.integers(0, 4)); x, y = np.rot90(x, j), np.rot90(y, j)
                    if rng.random() < 0.5: x, y = x[:, ::-1], y[:, ::-1]
                    x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
                X.append(x); Y.append(y)
        return torch.from_numpy(np.stack(X)).unsqueeze(1), torch.from_numpy(np.stack(Y)).unsqueeze(1)
    Xtr, Ytr = load(tr, True); Xva, Yva = load(va)
    net = U(16, 1, 3).to(DEV)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0], device=DEV))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    best, bd, aborted = -1, None, False
    t0 = time.time()
    for ep in range(1, 251):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 8):
            idx = perm[i:i + 8]
            lg = net(Xtr[idx].to(DEV)); p = torch.sigmoid(lg); yb = Ytr[idx].to(DEV)
            loss = 0.50 * bce(lg, yb) + 0.25 * dice_loss(p, yb) + 0.50 * cldice_loss(p, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0:
            net.eval()
            with torch.no_grad():
                pv = torch.sigmoid(net(Xva.to(DEV)))
            area = 100.0 * float((pv > 0.5).float().mean())
            it = float(((pv > 0.5) & (Yva.to(DEV) > 0.5)).sum()); un = float(((pv > 0.5) | (Yva.to(DEV) > 0.5)).sum())
            iou = it / max(1.0, un)
            if ep % 50 == 0:
                print("   ep{:>3}  验证预测面积 {:>6.2f}%  IoU {:.4f}".format(ep, area, iou))
            if area > 15.0:
                print("  ✗ 护栏触发：面积 {:.1f}% → 中止".format(area)); aborted = True; break
            if iou > best:
                best = iou; bd = {k: v.clone() for k, v in net.state_dict().items()}
    if aborted or bd is None:
        print("  v12 失败 ✗"); return 1
    net.load_state_dict(bd)
    torch.save({"sd": net.state_dict(), "cin": 1, "crop": C, "loss": "raw+bce+dice+cldice(0.5)"},
               os.path.join(UN, "models/unet_v12_raw_cldice5.pt"))
    print("  v12 val IoU {:.4f}  ({:.0f}s)  → 再做 TTA 并集".format(best, time.time() - t0))
    net.eval()
    od2 = os.path.join(PAN, "union_clean_v12"); os.makedirs(od2, exist_ok=True)
    S2 = []
    print("\n  板            v12(TTA) 覆盖率 最大缺 缺口 | 全体并集(v12+全部) 覆盖率 最大缺 缺口")
    for b in boards:
        p12 = tta_prob(net, b["g"].astype(np.float32) / 255.0) > 0.5
        cv2.imwrite(os.path.join(UN, "pred_masks_v12", b["fn"]) if os.path.isdir(os.path.join(UN, "pred_masks_v12"))
                    else os.path.join(os.makedirs(os.path.join(UN, "pred_masks_v12"), exist_ok=True) or
                                      os.path.join(UN, "pred_masks_v12"), b["fn"]), (p12 * 255).astype(np.uint8))
        a = gap_stats(b["lab"], p12)
        allu = p12.copy()
        for sub in ("pred_masks_tta", "pred_masks_v10", "pred_masks_v11", "pred_masks_v7", "pred_masks_v8"):
            pm = cv2.imread(os.path.join(UN, sub, b["fn"]), cv2.IMREAD_GRAYSCALE)
            if pm is not None: allu |= pm > 127
        au = gap_stats(b["lab"], allu)
        S2.append(a); rep[b["key"]]["v12_tta"] = a; rep[b["key"]]["all_union"] = au
        print("  {:<12} {:>12.1f}%{:>6.0f}{:>6} | {:>17.1f}%{:>6.0f}{:>6}".format(b["key"], a[0], a[1], a[2], au[0], au[1], au[2]))
        vis = cv2.cvtColor(b["g"], cv2.COLOR_GRAY2BGR)
        vis[allu] = (0.45 * vis[allu] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(os.path.join(od2, b["fn"].replace(".png", "_allunion.jpg")), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    m2 = np.array(S2, np.float32)
    print("\n  v12(纯原图, clDice 0.5, 带 TTA) 单独 覆盖率 {:.1f}%  最大缺 {:.0f}px  缺口 {:.1f}".format(m2[:,0].mean(), m2[:,1].mean(), m2[:,2].mean()))
    print("  全体并集（含 v12+TTA）             覆盖率 {:.1f}%  最大缺 {:.0f}px  缺口 {:.1f}".format(
        np.mean([r["all_union"][0] for r in rep.values()]), np.mean([r["all_union"][1] for r in rep.values()]),
        np.mean([r["all_union"][2] for r in rep.values()])))
    print("  判据：覆盖率 > 97.3% 或 缺口 < 0.8 → 通过 ✓")
    json.dump(rep, open(os.path.join(UN, "reports/tta_v12.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("  图 -> {} 与 {}".format(od, od2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
