import os as _os, sys as _sys
# [2026-10-08 适配] 去掉写死路径：按脚本位置定位，全部可用环境变量覆盖
_ROOT = _os.environ.get("CHIP_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CHIP_DATA") or _os.path.join(_ROOT, "train", "data")
_LIB  = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "lib")
_SRC  = _os.environ.get("CHIP_SRC")  or ""
_PROJ = _os.environ.get("CHIP_PROJ") or ""
_LAB  = _os.environ.get("CHIP_LAB")  or ""
_ANN  = _os.environ.get("CHIP_ANN") or ""
for _p in (_ROOT, _LIB):
    if _p not in _sys.path: _sys.path.insert(0, _p)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""203_train_unet_multichan.py — 路线 3：多种前处理当【通道】同时喂给模型

使用者 2026-09-29："第三条路线，继续增加前处理的种类，包括但不限于适应性二值化、轮廓提取等，
把这些图当作不同通道同时喂给模型。"

诚实前提：多通道试过两次都崩 ✗（v4 全判正 / v5 几乎不判），但那两次同时改了 lr 与 pos_weight ✗
→ 结论不可用。这次**只改输入通道这一处**，超参与最佳配置（v6_A）完全一致：
    ÷8 三级下采样、U(16, cin=6, 3)、pos_weight 20、lr 1e-3、250 epoch、按板留出 样例板C+样例板H。

六通道（全部按训练集统计做逐通道标准化 ✓，尺度拉齐）：
    c0 灰度分位拉伸(2–98%)      c1 CLAHE
    c2 自适应二值化（blockSize=51, C=10, 反相，线为亮）  c3 Canny(30,90) 轮廓
    c4 局部反差（GaussianBlur σ=8 − gray）              c5 黑帽（椭圆 15）
评判口径（这次用对的 ✓）：沿人工 GT 量 覆盖率 / 最大缺口 px / 缺口(≥8px)个数；
并与现有最佳【A∪B】对照，再看【v7 多通道 ∪ A∪B】能否再涨。
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

SRC = (_SRC + r"")
ROOT = (_ANN + r"")
UN = (_PROJ + r"/unet")
PAN = (_PROJ + r"/panels")
C, STRIDE, EP, DEV = 192, 129, 250, ("cuda" if torch.cuda.is_available() else "cpu")
HOLD = {"1_NG/1705", "2_NG/3421"}
POS_W, LR, CIN = 20.0, 1e-3, 6
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
BH = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


def channels(g):
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    c0 = np.clip((g.astype(np.float32) - lo) * 255.0 / max(1e-6, hi - lo), 0, 255)
    c1 = CLAHE.apply(g).astype(np.float32)
    c2 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10).astype(np.float32)
    c3 = cv2.Canny(g, 30, 90).astype(np.float32)
    c4 = cv2.GaussianBlur(g, (0, 0), 8).astype(np.float32) - g.astype(np.float32)
    c5 = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, BH).astype(np.float32)
    return np.stack([c0, c1, c2, c3, c4, c5], -1)


def slices(shape, st):
    h, w = shape
    xs = list(range(0, max(1, w - C) + 1, st)); ys = list(range(0, max(1, h - C) + 1, st))
    if w - C not in xs: xs.append(w - C)
    if h - C not in ys: ys.append(h - C)
    return [(x, y) for y in ys for x in xs]


class U(nn.Module):
    def __init__(s, c=16, cin=6, levels=3):
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


def dice_loss(p, t, eps=1e-6):
    p = torch.sigmoid(p)
    return 1 - ((2 * (p * t).sum((1, 2, 3)) + eps) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + eps)).mean()


def gap_stats(gt, pred):
    miss = gt & (~pred)
    cov = 100.0 * (1 - miss.sum() / max(1, gt.sum()))
    n, lab, st, ce = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
    mx, cnt = 0.0, 0
    for i in range(1, n):
        mm = miss & (lab == i)
        n2, l2, s2, c2 = cv2.connectedComponentsWithStats(mm.astype(np.uint8), 8)
        for j in range(1, n2):
            L = float(np.sqrt(s2[j, 4]))
            mx = max(mx, L)
            if L >= 8: cnt += 1
    return cov, mx, cnt


def main():
    bs = []
    for f in sorted(glob.glob(ROOT + "/*/*_m.png")):
        grp = os.path.basename(os.path.dirname(f)); base = os.path.basename(f)[:-6]
        bs.append({"key": grp + "/" + base[-4:], "src": os.path.join(SRC, grp, base + ".jpg"), "m": f})
    data = {}
    for b in bs:
        g = cv2.imread(b["src"], cv2.IMREAD_GRAYSCALE)
        raw = cv2.imread(b["m"], cv2.IMREAD_UNCHANGED)
        lab = (raw if raw.ndim == 2 else cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2GRAY)) > 127
        ch = channels(g)
        ps, ns = [], []
        for (x0, y0) in slices(g.shape, STRIDE):
            n = int(lab[y0:y0 + C, x0:x0 + C].sum())
            (ps if n >= 20 else ns).append((x0, y0))
        rng = np.random.default_rng(0); rng.shuffle(ns)
        data[b["key"]] = {"ch": ch, "lab": lab, "g": g, "blocks": ps + ns[:max(1, len(ps))]}
    tr = [k for k in data if k not in HOLD]; va = [k for k in data if k in HOLD]
    # 逐通道标准化（只用训练板）
    acc = []
    for k in tr:
        for (x0, y0) in data[k]["blocks"][:60]:
            acc.append(data[k]["ch"][y0:y0 + C, x0:x0 + C].reshape(-1, CIN))
    A = np.concatenate(acc, 0)
    # 校验：通道 c2（自适应二值化）在线内/环带的点亮率，确认它确实带信息
    print("=== 六通道在线内 vs 环带的信号强度（标准化前）===")
    for k in sorted(data):
        lab = data[k]["lab"]; ch = data[k]["ch"]
        dil = cv2.dilate(lab.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
        ring = dil & (~lab)
        msg = []
        for i, nm in enumerate(("灰", "CLAHE", "自适应二值", "Canny", "局部反差", "黑帽")):
            a = float(ch[:, :, i][lab].mean()); b2 = float(ch[:, :, i][ring].mean())
            msg.append("{} {:.0f}/{:,.0f}".format(nm, a, b2) if i not in (0, 1) else "{} {:.0f}/{:.0f}".format(nm, a, b2))
        print("  {:<11} {}".format(k, "  ".join(msg)))
    mu, sd = A.mean(0), A.std(0) + 1e-6
    print("\n  逐通道统计 均值 {}  标准差 {}".format(np.round(mu, 1), np.round(sd, 1)))
    for k in data:
        data[k]["ch"] = (data[k]["ch"] - mu) / sd
    rng = np.random.default_rng(1)
    def load(keys, aug=False):
        X, Y = [], []
        for k in keys:
            ch, lab = data[k]["ch"], data[k]["lab"]
            for (x0, y0) in data[k]["blocks"]:
                a = ch[y0:y0 + C, x0:x0 + C].astype(np.float32)
                m = lab[y0:y0 + C, x0:x0 + C].astype(np.float32)
                if aug:
                    j = int(rng.integers(0, 4)); a, m = np.rot90(a, j, (0, 1)), np.rot90(m, j)
                    if rng.random() < 0.5: a, m = a[:, ::-1], m[:, ::-1]
                    a, m = np.ascontiguousarray(a), np.ascontiguousarray(m)
                X.append(a); Y.append(m)
        return torch.from_numpy(np.stack(X)).transpose(1, 3), torch.from_numpy(np.stack(Y)).unsqueeze(1)
    Xtr, Ytr = load(tr, True); Xva, Yva = load(va)
    print("  训练 {} 块（{} 板）/ 验证 {} 块（{}）→ {}   唯一改动 = 输入 6 通道".format(
        len(Xtr), len(tr), len(Xva), "/".join(va), DEV))
    net = U(16, CIN, 3).to(DEV)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_W], device=DEV))
    opt = torch.optim.Adam(net.parameters(), LR)
    best, bd = -1, None
    t0 = time.time()
    for ep in range(1, EP + 1):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 8):
            idx = perm[i:i + 8]
            o = net(Xtr[idx].to(DEV))
            loss = 0.5 * bce(o, Ytr[idx].to(DEV)) + 0.5 * dice_loss(o, Ytr[idx].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == EP:
            net.eval()
            with torch.no_grad():
                pv = torch.sigmoid(net(Xva.to(DEV)))
            it = float(((pv > 0.5) & (Yva.to(DEV) > 0.5)).sum()); un = float(((pv > 0.5) | (Yva.to(DEV) > 0.5)).sum())
            if it / max(1.0, un) > best:
                best = it / max(1.0, un); bd = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(bd)
    torch.save({"sd": net.state_dict(), "cin": CIN, "crop": C, "mu": mu, "sd": sd},
               os.path.join(UN, "models/unet_v7_multichan.pt"))
    print("  val IoU（留出板）{:.4f}   {:.1f}s   参数 {:.1f}k".format(
        best, time.time() - t0, sum(p.numel() for p in net.parameters()) / 1000))
    net.eval()
    os.makedirs(os.path.join(UN, "pred_masks_v7"), exist_ok=True)
    od = os.path.join(PAN, "union_clean_v7"); os.makedirs(od, exist_ok=True)
    print("\n  板            覆盖率 最大缺 缺口数 | v7∪(A∪B) 覆盖率 最大缺 缺口数")
    rep = {}
    S7, S7U = [], []
    for k in sorted(data):
        g = data[k]["g"]; ch = data[k]["ch"]; h, w = g.shape
        acc2 = np.zeros((h, w), np.float32); cnt = np.zeros((h, w), np.float32)
        tl = slices((h, w), STRIDE)
        bt = np.zeros((len(tl), C, C, CIN), np.float32)
        for i, (x0, y0) in enumerate(tl):
            bt[i] = ch[y0:y0 + C, x0:x0 + C]
        with torch.no_grad():
            for i in range(0, len(tl), 16):
                p = torch.sigmoid(net(torch.from_numpy(bt[i:i + 16]).transpose(1, 3).to(DEV))).cpu().numpy()[:, 0]
                for j, (x0, y0) in enumerate(tl[i:i + 16]):
                    acc2[y0:y0 + C, x0:x0 + C] += p[j]; cnt[y0:y0 + C, x0:x0 + C] += 1
        pred = (acc2 / np.maximum(1.0, cnt)) > 0.5
        fn = k.replace("/", "_") + ".png"
        cv2.imwrite(os.path.join(UN, "pred_masks_v7", fn), (pred * 255).astype(np.uint8))
        pa = cv2.imread(os.path.join(UN, "pred_masks_v6", "A_pct", fn), cv2.IMREAD_GRAYSCALE) > 127
        pb = cv2.imread(os.path.join(UN, "pred_masks_v6", "B_clahe", fn), cv2.IMREAD_GRAYSCALE) > 127
        a7 = gap_stats(data[k]["lab"], pred)
        a7u = gap_stats(data[k]["lab"], pred | pa | pb)
        S7.append(a7); S7U.append(a7u)
        rep[k] = {"v7": a7, "v7_union_AB": a7u, "pred_pct": 100 * float(pred.mean())}
        print("  {:<12}  {:>6.1f}% {:>6.0f} {:>6} |         {:>6.1f}% {:>6.0f} {:>6}".format(
            k, a7[0], a7[1], a7[2], a7u[0], a7u[1], a7u[2]))
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        mk = pred | pa | pb
        vis[mk] = (0.45 * vis[mk] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(os.path.join(od, fn.replace(".png", "_v7union.jpg")), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    m7 = np.array(S7, np.float32); m7u = np.array(S7U, np.float32)
    print("\n=== 对照（平均；口径 = 沿人工 GT 量缺口）===")
    print("  A∪B 基线             覆盖率 93.6%  最大缺 14px  缺口数 2.2")
    print("  v7 多通道            覆盖率 {:.1f}%  最大缺 {:.0f}px  缺口数 {:.1f}".format(m7[:,0].mean(), m7[:,1].mean(), m7[:,2].mean()))
    print("  v7 ∪ (A∪B)           覆盖率 {:.1f}%  最大缺 {:.0f}px  缺口数 {:.1f}".format(m7u[:,0].mean(), m7u[:,1].mean(), m7u[:,2].mean()))
    json.dump(rep, open(os.path.join(UN, "reports/unet_v7_multichan.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("  图 -> {}（红 = v7∪A∪B；无任何圈/文字）".format(od))
    return 0


if __name__ == "__main__":
    sys.exit(main())
