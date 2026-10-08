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
"""212_train_premlp3.py — U-Net 前接【3 层 nn.Linear + ReLU】的矩阵乘法 MLP（v15）

使用者要求（2026-09-29）："我的想法并不是做单个 y=ax+b 的线性变换……而是把图片直接当成矩阵，
用 nn.Linear 进行矩阵乘法，叠 3 层，带 relu。修改这个网络。"

实现（唯一改动 = 入口模块 ✓）：
    x (1×192×192) → flatten(36864)
      → Linear(36864,512) → ReLU → Linear(512,512) → ReLU → Linear(512,36864)
      → reshape 回 (1×192×192) → + 未处理原图（残差 ✓）→ 进入 U-Net
    最后一层 zero-init ⇒ 起点 = 恒等（等价"纯原图"✓），让网络自己决定要学什么变换 ✓
参数量：36864×512 + 512×512 + 512×36864 ≈ 37.8M（U-Net 本体 0.48M）
其余与 v13/v14 完全一致：纯原图、重叠 50%（stride 96，555 块）、÷8 U-Net(16-32-64)、
    0.50·BCE(20)+0.25·(1−Dice)+0.25·clDice、lr 1e-3、250 ep、留出 NG-1-样例板C+NG-2-样例板H、护栏 15%
对照：v13 单独 93.4%/11px/2.2；v14（1×1 线性）单独 92.7%/11px/1.8；干净池 98.9%/5px/0.1
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
HID = 512
CHAN8 = ["01_原图", "02_分位拉伸", "03_CLAHE", "05_Canny", "06_局部反差s8", "07_黑帽15", "08_USM", "09_gamma0.6"]
SINGLE = ["pred_masks_tta", "pred_masks_v10", "pred_masks_v11", "pred_masks_v12", "pred_masks_v13"]


def slices(shape, st):
    h, w = shape
    xs = list(range(0, max(1, w - C) + 1, st)); ys = list(range(0, max(1, h - C) + 1, st))
    if w - C not in xs: xs.append(w - C)
    if h - C not in ys: ys.append(h - C)
    return [(x, y) for y in ys for x in xs]


class UPreMLP(nn.Module):
    def __init__(s, c=16, cin=1, levels=3, hid=HID, crop=C):
        super().__init__()
        n = crop * crop
        s.mlp = nn.Sequential(nn.Linear(n, hid), nn.ReLU(True), nn.Linear(hid, hid), nn.ReLU(True), nn.Linear(hid, n))
        nn.init.zeros_(s.mlp[-1].weight); nn.init.zeros_(s.mlp[-1].bias)     # 起点 = 恒等
        s.crop = crop
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
        b = x.shape[0]
        h = s.mlp(x.reshape(b, -1)).reshape(b, 1, s.crop, s.crop)     # 3 层矩阵乘法 + ReLU
        x = h + x                                                     # 残差：未处理原图直通
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
    ntot = sum(len(data[k]["blocks"]) for k in tr)
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
    net = UPreMLP(16, 1, 3).to(DEV)
    npar = sum(p.numel() for p in net.parameters())
    nmlp = sum(p.numel() for p in net.mlp.parameters())
    print("  训练块 {}（同 v13/v14 ✓）；总参数 {:.1f}M（其中前置 MLP {:.1f}M，U-Net 本体 {:.2f}M）".format(
        ntot, npar / 1e6, nmlp / 1e6, (npar - nmlp) / 1e6))
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
            if ep % 25 == 0 or ep == EP:
                print("   ep{:>3}  验证预测面积 {:>6.2f}%  IoU {:.4f}  累计 {:.0f}s".format(ep, area, iou, time.time() - t0))
            if area > 15.0:
                print("  ✗ 护栏触发：面积 {:.1f}% → 中止".format(area)); aborted = True; break
            if iou > best:
                best = iou; bd = {k: v.clone() for k, v in net.state_dict().items()}
    if aborted or bd is None:
        print("  结果：失败 ✗"); return 1
    net.load_state_dict(bd); net.eval()
    torch.save({"sd": net.state_dict(), "cin": 1, "crop": C, "stride": STRIDE, "pre": "mlp3x512+residual"},
               os.path.join(UN, "models/unet_v15_premlp3.pt"))
    print("  val IoU {:.4f}  （{:.0f}s；前置 MLP 3 层 36864→{}→{}→36864 ✓）".format(best, time.time() - t0, HID, HID))
    os.makedirs(os.path.join(UN, "pred_masks_v15"), exist_ok=True)
    od = os.path.join(PAN, "union_clean_v15"); os.makedirs(od, exist_ok=True)
    print("\n  板            预测%  标签%  | v15覆盖率 最大缺 缺口 | 干净池 覆盖率 最大缺 缺口 | v15∪干净池 覆盖率 最大缺 缺口")
    S = []; SP = []; SU = []
    for b in boards:
        g = b["g"]; ch = data[b["key"]]["ch"]; h, w2 = g.shape
        acc = np.zeros((h, w2), np.float32); cnt = np.zeros((h, w2), np.float32)
        tl = slices((h, w2), STRIDE)
        bt = np.zeros((len(tl), C, C), np.float32)
        for i, (x0, y0) in enumerate(tl):
            bt[i] = ch[y0:y0 + C, x0:x0 + C]
        with torch.no_grad():
            for i in range(0, len(tl), 16):
                p = torch.sigmoid(net(torch.from_numpy(bt[i:i + 16])[:, None].to(DEV))).cpu().numpy()[:, 0]
                for j, (x0, y0) in enumerate(tl[i:i + 16]):
                    acc[y0:y0 + C, x0:x0 + C] += p[j]; cnt[y0:y0 + C, x0:x0 + C] += 1
        pred = (acc / np.maximum(1.0, cnt)) > 0.5
        cv2.imwrite(os.path.join(UN, "pred_masks_v15", b["fn"]), (pred * 255).astype(np.uint8))
        pool = np.zeros_like(pred)
        for c in CHAN8:
            p = os.path.join(UN, "pred_masks_chan9", c + "_" + b["fn"])
            if os.path.exists(p): pool |= cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127
        for sub in SINGLE:
            p = os.path.join(UN, sub, b["fn"])
            if os.path.exists(p): pool |= cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127
        a = gap(b["lab"], pred); ap = gap(b["lab"], pool); au = gap(b["lab"], pred | pool)
        S.append(a); SP.append(ap); SU.append(au)
        print("  {:<12} {:>6.2f} {:>6.2f}  |{:>9.1f}%{:>6.0f}{:>4} |{:>12.1f}%{:>6.0f}{:>4} |{:>12.1f}%{:>6.0f}{:>4}".format(
            b["key"], 100 * float(pred.mean()), 100 * float(b["lab"].mean()), a[0], a[1], a[2], ap[0], ap[1], ap[2], au[0], au[1], au[2]))
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        vis[pred | pool] = (0.45 * vis[pred | pool] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(os.path.join(od, b["fn"].replace(".png", "_v15union.jpg")), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    m = np.array(S, np.float32); mp = np.array(SP, np.float32); mu = np.array(SU, np.float32)
    print("\n=== 对照（平均；沿人工 GT 量缺口）===")
    print("  v13（无前置）单独          覆盖率 93.4%  最大缺 11px  缺口 2.2")
    print("  v14（1×1 线性 + 残差）单独  覆盖率 92.7%  最大缺 11px  缺口 1.8")
    print("  v15（3 层 MLP + ReLU + 残差）单独 覆盖率 {:.1f}%  最大缺 {:.0f}px  缺口 {:.1f}".format(m[:,0].mean(), m[:,1].mean(), m[:,2].mean()))
    print("  干净池                      覆盖率 {:.1f}%  最大缺 {:.0f}px  缺口 {:.1f}".format(mp[:,0].mean(), mp[:,1].mean(), mp[:,2].mean()))
    print("  v15 ∪ 干净池                覆盖率 {:.1f}%  最大缺 {:.0f}px  缺口 {:.1f}".format(mu[:,0].mean(), mu[:,1].mean(), mu[:,2].mean()))
    json.dump({"v15": S, "pool": SP, "union": SU, "params": npar, "mlp_params": nmlp},
              open(os.path.join(UN, "reports/unet_v15_premlp3.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("  图 -> {}".format(od))
    return 0


if __name__ == "__main__":
    sys.exit(main())
