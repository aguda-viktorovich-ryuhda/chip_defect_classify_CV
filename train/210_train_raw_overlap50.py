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
"""210_train_raw_overlap50.py — 纯原图 + 重叠率 33%→50%（唯一变量），对比 v11

使用者要求（2026-09-29）："改变裁剪的重叠率，增加到 50% 来获取样本量和多样性。用原图重新训练"
比较基准 v11（纯原图、重叠 33%、stride 129）：331 训练块 / 单独覆盖率 92.0% / 最大缺 14px / 缺口 2.8
本次 v13：同样纯原图、**重叠 50%（stride 96）** → 块数约 1.6 倍 ✓
其余全部一致：÷8 U-Net(16-32-64)、0.50·BCE(20)+0.25·(1−Dice)+0.25·clDice、lr 1e-3、250 ep、
              按板留出 NG-1-样例板C + NG-2-样例板H、护栏（验证预测面积 >15% 中止 ✗）
口径：沿人工 GT 量 覆盖率 / 最大缺口 px / 缺口数；并报与现最佳池的并集效果 + 干净拼接图。
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
C, STRIDE, EP, DEV = 192, 96, 250, ("cuda" if torch.cuda.is_available() else "cpu")
HOLD = {"1_NG/1705", "2_NG/3421"}
POOL = ["pred_masks_tta", "pred_masks_v10", "pred_masks_v11", "pred_masks_v12", "pred_masks_v7",
        "pred_masks_v8", "pred_masks_chan9/01_原图", "pred_masks_chan9/06_局部反差s8", "pred_masks_chan9/08_USM"]


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


def soft_erode(x): return -F.max_pool2d(-x, 3, 1, 1)

def soft_dilate(x): return F.max_pool2d(x, 3, 1, 1)

def soft_open(x): return soft_dilate(soft_erode(x))

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


def gap(gt, pred):
    miss = gt & (~pred); cov = 100.0 * (1 - miss.sum() / max(1, gt.sum()))
    n, lab, st, ce = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8); mx = 0.0; cnt = 0
    for i in range(1, n):
        mm = miss & (lab == i)
        n2, l2, s2, c2 = cv2.connectedComponentsWithStats(mm.astype(np.uint8), 8)
        for j in range(1, n2):
            L = float(np.sqrt(s2[j, 4])); mx = max(mx, L)
            if L >= 8: cnt += 1
    return cov, mx, cnt


def main():
    boards = []
    for f in sorted(glob.glob(ROOT + "/*/*_m.png")):
        grp = os.path.basename(os.path.dirname(f)); base = os.path.basename(f)[:-6]
        g = cv2.imread(os.path.join(SRC, grp, base + ".jpg"), cv2.IMREAD_GRAYSCALE)
        raw = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        boards.append({"key": grp + "/" + base[-4:], "fn": grp + "_" + base[-4:] + ".png", "g": g,
                       "lab": (raw if raw.ndim == 2 else cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2GRAY)) > 127})
    data = {}
    for b in boards:
        ch = b["g"].astype(np.float32) / 255.0
        ps, ns = [], []
        for (x0, y0) in slices(b["g"].shape, STRIDE):
            n = int(b["lab"][y0:y0 + C, x0:x0 + C].sum())
            (ps if n >= 20 else ns).append((x0, y0))
        rng = np.random.default_rng(0); rng.shuffle(ns)
        data[b["key"]] = {"ch": ch, "lab": b["lab"], "g": b["g"], "blocks": ps + ns[:max(1, len(ps))]}
    tr = [k for k in data if k not in HOLD]; va = [k for k in data if k in HOLD]
    npos = sum(len([x for x in data[k]["blocks"] if int(data[k]["lab"][x[1]:x[1] + C, x[0]:x[0] + C].sum()) >= 20]) for k in tr)
    ntot = sum(len(data[k]["blocks"]) for k in tr)
    print("  重叠 50%（stride {}）→ 训练块 {}（v11 的 33% 重叠为 331）".format(STRIDE, ntot))
    print("  训练 {} 板 / 验证 {} 板（{}）→ {}".format(len(tr), len(va), "/".join(va), DEV))
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
    for ep in range(1, EP + 1):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 8):
            idx = perm[i:i + 8]
            lg = net(Xtr[idx].to(DEV)); p = torch.sigmoid(lg); yb = Ytr[idx].to(DEV)
            loss = 0.50 * bce(lg, yb) + 0.25 * dice_loss(p, yb) + 0.25 * cldice_loss(p, yb)
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
        print("  结果：失败 ✗"); return 1
    net.load_state_dict(bd); net.eval()
    torch.save({"sd": net.state_dict(), "cin": 1, "crop": C, "stride": STRIDE, "overlap": 0.5},
               os.path.join(UN, "models/unet_v13_raw_overlap50.pt"))
    print("  val IoU（留出板）{:.4f}  ({:.0f}s)".format(best, time.time() - t0))
    os.makedirs(os.path.join(UN, "pred_masks_v13"), exist_ok=True)
    od = os.path.join(PAN, "union_clean_v13"); os.makedirs(od, exist_ok=True)
    print("\n  板            预测%  标签%  | v13覆盖率 最大缺 缺口 | v13∪池 覆盖率 最大缺 缺口 | 全池 覆盖率 最大缺 缺口")
    S = []; SU = []; SA = []
    for b in boards:
        g = b["g"]; ch, h, w = data[b["key"]]["ch"], b["g"].shape[0], b["g"].shape[1]
        acc = np.zeros((h, w), np.float32); cnt = np.zeros((h, w), np.float32)
        tl = slices((h, w), STRIDE)
        bt = np.zeros((len(tl), C, C), np.float32)
        for i, (x0, y0) in enumerate(tl):
            bt[i] = ch[y0:y0 + C, x0:x0 + C]
        with torch.no_grad():
            for i in range(0, len(tl), 16):
                p = torch.sigmoid(net(torch.from_numpy(bt[i:i + 16])[:, None].to(DEV))).cpu().numpy()[:, 0]
                for j, (x0, y0) in enumerate(tl[i:i + 16]):
                    acc[y0:y0 + C, x0:x0 + C] += p[j]; cnt[y0:y0 + C, x0:x0 + C] += 1
        pred = (acc / np.maximum(1.0, cnt)) > 0.5
        cv2.imwrite(os.path.join(UN, "pred_masks_v13", b["fn"]), (pred * 255).astype(np.uint8))
        pool = np.zeros_like(pred)
        for sub in POOL:
            pm = cv2.imread(os.path.join(UN, sub, b["fn"]), cv2.IMREAD_GRAYSCALE)
            if pm is not None: pool |= pm > 127
        up = pred | pool
        a = gap(b["lab"], pred); ap = gap(b["lab"], up); aa = gap(b["lab"], pool)
        S.append(a); SU.append(ap); SA.append(aa)
        print("  {:<12} {:>6.2f} {:>6.2f}  |{:>9.1f}%{:>6.0f}{:>4} |{:>11.1f}%{:>6.0f}{:>4} |{:>9.1f}%{:>6.0f}{:>4}".format(
            b["key"], 100 * float(pred.mean()), 100 * float(b["lab"].mean()), a[0], a[1], a[2], ap[0], ap[1], ap[2], aa[0], aa[1], aa[2]))
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        vis[up] = (0.45 * vis[up] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(os.path.join(od, b["fn"].replace(".png", "_v13union.jpg")), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    m = np.array(S, np.float32); mp = np.array(SU, np.float32); ma = np.array(SA, np.float32)
    print("\n=== 对比（沿人工 GT 量缺口；平均）===")
    print("  v11（33% 重叠，331 块）单独     覆盖率 92.0%  最大缺 14px  缺口 2.8")
    print("  v13（50% 重叠，{} 块）单独  覆盖率 {:.1f}%  最大缺 {:.0f}px  缺口 {:.1f}".format(ntot, m[:,0].mean(), m[:,1].mean(), m[:,2].mean()))
    print("  仅现最佳池（无 v13）           覆盖率 {:.1f}%  最大缺 {:.0f}px  缺口 {:.1f}".format(ma[:,0].mean(), ma[:,1].mean(), ma[:,2].mean()))
    print("  v13 ∪ 池                        覆盖率 {:.1f}%  最大缺 {:.0f}px  缺口 {:.1f}".format(mp[:,0].mean(), mp[:,1].mean(), mp[:,2].mean()))
    print("  对照：合上 8 通道并集后的历史最佳 = 98.9% / 6px / 0.2")
    json.dump({"v13": S, "v13_union_pool": SU, "pool_only": SA, "train_blocks": ntot, "stride": STRIDE},
              open(os.path.join(UN, "reports/unet_v13_overlap50.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("  图 -> {}".format(od))
    return 0


if __name__ == "__main__":
    sys.exit(main())
