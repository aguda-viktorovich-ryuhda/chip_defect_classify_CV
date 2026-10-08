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
"""207_train_unet_raw_cldice.py — 原图（不处理）+ BCE+Dice+clDice

使用者提问："只用原图跑的模型改损失函数是什么效果？"
对照设计（只改损失这一处 ✓）：
    基准 = v3：**原图灰度（不做任何拉伸）** + 0.5·BCE(pos_weight=20) + 0.5·Dice
             → 覆盖率 90.6% / 最大缺 16px / 缺口 3.8（10 板口径 ✓）
    本次 = v11：**原图灰度（同样不处理）** + 0.50·BCE(20) + 0.25·(1−Dice) + 0.25·clDice
其余全部一致：÷8 U-Net(16-32-64)、lr 1e-3、250 epoch、按板留出 NG-1-样例板C + NG-2-样例板H ✓
（注：v6_A 用的是【分位拉伸】灰度 ✗，v10 也是分位拉伸 ✗ —— 本脚本才是"纯原图"✓）
护栏：验证预测面积 > 15% 即中止（防塌 ✗）
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
C, STRIDE, EP, DEV = 192, 129, 250, ("cuda" if torch.cuda.is_available() else "cpu")
HOLD = {"1_NG/1705", "2_NG/3421"}
W_BCE, W_DICE, W_CLDICE, LR, ITERS, AREA_ABORT = 0.50, 0.25, 0.25, 1e-3, 5, 15.0


def slices(shape, st):
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

def soft_skel(x, iters=ITERS):
    x1 = soft_open(x); skel = F.relu(x - x1)
    for _ in range(iters):
        x = soft_erode(x); x1 = soft_open(x)
        delta = F.relu(x - x1); skel = skel + delta - skel * delta
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


def main():
    bs = []
    for f in sorted(glob.glob(ROOT + "/*/*_m.png")):
        grp = os.path.basename(os.path.dirname(f)); base = os.path.basename(f)[:-6]
        bs.append({"key": grp + "/" + base[-4:], "src": os.path.join(SRC, grp, base + ".jpg"), "m": f})
    data = {}
    for b in bs:
        g = cv2.imread(b["src"], cv2.IMREAD_GRAYSCALE)          # ← 原图，不做任何处理
        raw = cv2.imread(b["m"], cv2.IMREAD_UNCHANGED)
        lab = (raw if raw.ndim == 2 else cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2GRAY)) > 127
        ch = g.astype(np.float32) / 255.0
        ps, ns = [], []
        for (x0, y0) in slices(g.shape, STRIDE):
            n = int(lab[y0:y0 + C, x0:x0 + C].sum())
            (ps if n >= 20 else ns).append((x0, y0))
        rng = np.random.default_rng(0); rng.shuffle(ns)
        data[b["key"]] = {"ch": ch, "lab": lab, "g": g, "blocks": ps + ns[:max(1, len(ps))]}
    tr = [k for k in data if k not in HOLD]; va = [k for k in data if k in HOLD]
    rng = np.random.default_rng(1)
    def load(keys, aug=False):
        X, Y = [], []
        for k in keys:
            ch, lab = data[k]["ch"], data[k]["lab"]
            for (x0, y0) in data[k]["blocks"]:
                a = ch[y0:y0 + C, x0:x0 + C].astype(np.float32)
                m = lab[y0:y0 + C, x0:x0 + C].astype(np.float32)
                if aug:
                    j = int(rng.integers(0, 4)); a, m = np.rot90(a, j), np.rot90(m, j)
                    if rng.random() < 0.5: a, m = a[:, ::-1], m[:, ::-1]
                    a, m = np.ascontiguousarray(a), np.ascontiguousarray(m)
                X.append(a); Y.append(m)
        return torch.from_numpy(np.stack(X)).unsqueeze(1), torch.from_numpy(np.stack(Y)).unsqueeze(1)
    Xtr, Ytr = load(tr, True); Xva, Yva = load(va)
    print("  【纯原图】输入（不做拉伸/不做任何前处理）✓  损失 = {:.2f}·BCE(20) + {:.2f}·(1−Dice) + {:.2f}·clDice".format(
        W_BCE, W_DICE, W_CLDICE))
    print("  验证集标签面积 {:.2f}%（健康预测应同量级 ✓）".format(100.0 * float(Yva.mean())))
    print("  训练 {} 块（{} 板）/ 验证 {} 块（{}）→ {}".format(len(Xtr), len(tr), len(Xva), "/".join(va), DEV))
    net = U(16, 1, 3).to(DEV)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0], device=DEV))
    opt = torch.optim.Adam(net.parameters(), LR)
    best, bd, aborted = -1, None, False
    t0 = time.time()
    for ep in range(1, EP + 1):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 8):
            idx = perm[i:i + 8]
            logits = net(Xtr[idx].to(DEV)); p = torch.sigmoid(logits); yb = Ytr[idx].to(DEV)
            loss = W_BCE * bce(logits, yb) + W_DICE * dice_loss(p, yb) + W_CLDICE * cldice_loss(p, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == EP:
            net.eval()
            with torch.no_grad():
                pv = torch.sigmoid(net(Xva.to(DEV)))
            area = 100.0 * float((pv > 0.5).float().mean())
            it = float(((pv > 0.5) & (Yva.to(DEV) > 0.5)).sum()); un = float(((pv > 0.5) | (Yva.to(DEV) > 0.5)).sum())
            iou = it / max(1.0, un)
            if ep % 50 == 0 or ep == EP:
                print("   ep{:>3}  验证预测面积 {:>6.2f}%  IoU {:.4f}".format(ep, area, iou))
            if area > AREA_ABORT:
                print("  ✗ 护栏触发：预测面积 {:.1f}% → 判塌陷，中止".format(area)); aborted = True; break
            if iou > best:
                best = iou; bd = {k: v.clone() for k, v in net.state_dict().items()}
    if aborted or bd is None:
        print("  结果：失败 ✗"); return 1
    net.load_state_dict(bd)
    torch.save({"sd": net.state_dict(), "cin": 1, "crop": C, "loss": "raw+bce+dice+cldice"},
               os.path.join(UN, "models/unet_v11_raw_cldice.pt"))
    print("  val IoU（留出板）{:.4f}   {:.1f}s".format(best, time.time() - t0))
    net.eval()
    os.makedirs(os.path.join(UN, "pred_masks_v11"), exist_ok=True)
    od = os.path.join(PAN, "union_clean_v11"); os.makedirs(od, exist_ok=True)
    print("\n  板            预测%  标签%  IoU  | v11覆盖率 最大缺 缺口 | v11∪全部 覆盖率 最大缺 缺口")
    rep = {}; S = []; SU = []
    for k in sorted(data):
        g = data[k]["g"]; ch = data[k]["ch"]; h, w = g.shape
        acc2 = np.zeros((h, w), np.float32); cnt = np.zeros((h, w), np.float32)
        tl = slices((h, w), STRIDE)
        bt = np.zeros((len(tl), C, C), np.float32)
        for i, (x0, y0) in enumerate(tl):
            bt[i] = ch[y0:y0 + C, x0:x0 + C]
        with torch.no_grad():
            for i in range(0, len(tl), 16):
                p = torch.sigmoid(net(torch.from_numpy(bt[i:i + 16])[:, None].to(DEV))).cpu().numpy()[:, 0]
                for j, (x0, y0) in enumerate(tl[i:i + 16]):
                    acc2[y0:y0 + C, x0:x0 + C] += p[j]; cnt[y0:y0 + C, x0:x0 + C] += 1
        pred = (acc2 / np.maximum(1.0, cnt)) > 0.5
        fn = k.replace("/", "_") + ".png"
        cv2.imwrite(os.path.join(UN, "pred_masks_v11", fn), (pred * 255).astype(np.uint8))
        pool = np.zeros_like(pred)
        for sub in ("pred_masks_v6/A_pct", "pred_masks_v6/B_clahe", "pred_masks_v7", "pred_masks_v8", "pred_masks_v10"):
            pm = cv2.imread(os.path.join(UN, sub, fn), cv2.IMREAD_GRAYSCALE)
            if pm is not None:
                pool |= pm > 127
        union = pred | pool
        a = gap_stats(data[k]["lab"], pred); au = gap_stats(data[k]["lab"], union)
        S.append(a); SU.append(au)
        iou = float((pred & data[k]["lab"]).sum()) / max(1.0, float((pred | data[k]["lab"]).sum()))
        rep[k] = {"pred_pct": 100 * float(pred.mean()), "gt_pct": 100 * float(data[k]["lab"].mean()), "iou": iou,
                  "v11": a, "v11_union_all": au}
        print("  {:<12} {:>6.2f} {:>6.2f} {:>5.3f} |{:>9.1f}%{:>6.0f}{:>6} |{:>11.1f}%{:>6.0f}{:>6}".format(
            k, rep[k]["pred_pct"], rep[k]["gt_pct"], iou, a[0], a[1], a[2], au[0], au[1], au[2]))
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        vis[union] = (0.45 * vis[union] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(os.path.join(od, fn.replace(".png", "_v11union.jpg")), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    m = np.array(S, np.float32); mu = np.array(SU, np.float32)
    print("\n=== 只改损失的对照（纯原图输入）===")
    print("  v3  原图 + BCE+Dice         覆盖率 90.6%  最大缺 16px  缺口 3.8   ← 基准")
    print("  v11 原图 + BCE+Dice+clDice  覆盖率 {:.1f}%  最大缺 {:.0f}px  缺口 {:.1f}   ← 本次".format(m[:,0].mean(), m[:,1].mean(), m[:,2].mean()))
    print("  v11 ∪ 全部                  覆盖率 {:.1f}%  最大缺 {:.0f}px  缺口 {:.1f}   ← 并集".format(mu[:,0].mean(), mu[:,1].mean(), mu[:,2].mean()))
    print("  参考：v10（分位拉伸 + 新损失）单独 88.4%/17px/3.9；并集 96.3%/10px/1.1")
    json.dump(rep, open(os.path.join(UN, "reports/unet_v11_raw_cldice.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("  图 -> {}".format(od))
    return 0


if __name__ == "__main__":
    sys.exit(main())
