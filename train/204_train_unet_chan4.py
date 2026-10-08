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
"""204_train_unet_chan4.py — 按排查结果选的 4 通道（含【原图】）

依据 203 之后的逐像素 AUC 排查（原图 0.845 为基准 ✓）：
    真增强：局部反差 σ8 (AUC 0.917 ✓)
    疑似增强：USM σ3 a1.5 (0.886 ✓)
    中性补充：CLAHE(4,16×16) (0.856 ✓)
    必须直接进：原图（未处理 ✓）
    剔除：黑帽 15 (带斑 21.8% ✗)、Canny (噪音 0.745 ✗)、自适应二值化 b11 (0.787 ✗)

唯一改动 = 输入通道集（4 通道）✓，其余与 v6_A 完全一致：÷8、U(16,cin=4,3)、
pos_weight 20、lr 1e-3、250 epoch、按板留出 NG-1-样例板C + NG-2-样例板H。

健康自检（关键 ✓）：预测面积占比应落在 2.5–3.4%（1_ 系）/ 0.8–1.2%（2_ 系）
    —— 若像 v7 那样到 12–18% ✗，说明又学成"斑"了 ✗，直接判失败。
口径：沿人工 GT 量 覆盖率 / 最大缺口 px / 缺口数；并报与现有最佳并集的合并效果。
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
POS_W, LR, CIN = 20.0, 1e-3, 4
CLAHE4 = cv2.createCLAHE(4.0, (16, 16))


def channels(g):
    c0 = g.astype(np.float32)                                                    # 原图（未处理）
    c1 = cv2.GaussianBlur(g, (0, 0), 8).astype(np.float32) - g.astype(np.float32)  # 局部反差 σ8
    b = cv2.GaussianBlur(g, (0, 0), 3).astype(np.float32)
    c2 = np.clip(g.astype(np.float32) + 1.5 * (g.astype(np.float32) - b), 0, 255)  # USM σ3 a1.5
    c3 = CLAHE4.apply(g).astype(np.float32)                                       # CLAHE(4,16×16)
    return np.stack([c0, c1, c2, c3], -1)


def slices(shape, st):
    h, w = shape
    xs = list(range(0, max(1, w - C) + 1, st)); ys = list(range(0, max(1, h - C) + 1, st))
    if w - C not in xs: xs.append(w - C)
    if h - C not in ys: ys.append(h - C)
    return [(x, y) for y in ys for x in xs]


class U(nn.Module):
    def __init__(s, c=16, cin=4, levels=3):
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
    acc = []
    for k in tr:
        for (x0, y0) in data[k]["blocks"][:60]:
            acc.append(data[k]["ch"][y0:y0 + C, x0:x0 + C].reshape(-1, CIN))
    A = np.concatenate(acc, 0); mu, sd = A.mean(0), A.std(0) + 1e-6
    print("  四通道：0原图  1局部反差σ8  2USM  3CLAHE(4,16)；逐通道均值 {} 标准差 {}".format(np.round(mu, 1), np.round(sd, 1)))
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
    print("  训练 {} 块（{} 板）/ 验证 {} 块（{}）→ {}".format(len(Xtr), len(tr), len(Xva), "/".join(va), DEV))
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
               os.path.join(UN, "models/unet_v8_chan4.pt"))
    print("  val IoU（留出板）{:.4f}   {:.1f}s   参数 {:.1f}k".format(
        best, time.time() - t0, sum(p.numel() for p in net.parameters()) / 1000))
    net.eval()
    os.makedirs(os.path.join(UN, "pred_masks_v8"), exist_ok=True)
    od = os.path.join(PAN, "union_clean_v8"); os.makedirs(od, exist_ok=True)
    print("\n  板            预测%  标签%  |  v8覆盖率 最大缺 缺口 | v8∪(A∪B∪v7) 覆盖率 最大缺 缺口")
    rep = {}; S8 = []; SU = []
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
        cv2.imwrite(os.path.join(UN, "pred_masks_v8", fn), (pred * 255).astype(np.uint8))
        pa = cv2.imread(os.path.join(UN, "pred_masks_v6", "A_pct", fn), cv2.IMREAD_GRAYSCALE) > 127
        pb = cv2.imread(os.path.join(UN, "pred_masks_v6", "B_clahe", fn), cv2.IMREAD_GRAYSCALE) > 127
        p7 = cv2.imread(os.path.join(UN, "pred_masks_v7", fn), cv2.IMREAD_GRAYSCALE) > 127
        union = pred | pa | pb | p7
        a8 = gap_stats(data[k]["lab"], pred); au = gap_stats(data[k]["lab"], union)
        S8.append(a8); SU.append(au)
        rep[k] = {"pred_pct": 100 * float(pred.mean()), "gt_pct": 100 * float(data[k]["lab"].mean()),
                  "v8": a8, "v8_union_all": au}
        print("  {:<12} {:>6.2f} {:>6.2f}  |{:>9.1f}%{:>6.0f}{:>6} |{:>14.1f}%{:>6.0f}{:>6}".format(
            k, rep[k]["pred_pct"], rep[k]["gt_pct"], a8[0], a8[1], a8[2], au[0], au[1], au[2]))
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        vis[union] = (0.45 * vis[union] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(os.path.join(od, fn.replace(".png", "_v8union.jpg")), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    m8 = np.array(S8, np.float32); mu8 = np.array(SU, np.float32)
    print("\n=== 对照（平均；沿人工 GT 量缺口）===")
    print("  原图单通道 v6_A      覆盖率 90.6%  最大缺 16px  缺口 3.8")
    print("  A∪B                 覆盖率 93.6%  最大缺 14px  缺口 2.2")
    print("  A∪B∪v7（现最佳）      覆盖率 94.9%  最大缺 11px  缺口 1.9")
    print("  v8 四通道(含原图)     覆盖率 {:.1f}%  最大缺 {:.0f}px  缺口 {:.1f}   ← 单独".format(m8[:,0].mean(), m8[:,1].mean(), m8[:,2].mean()))
    print("  v8 ∪ 全部            覆盖率 {:.1f}%  最大缺 {:.0f}px  缺口 {:.1f}   ← 并集".format(mu8[:,0].mean(), mu8[:,1].mean(), mu8[:,2].mean()))
    json.dump(rep, open(os.path.join(UN, "reports/unet_v8_chan4.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("  图 -> {}（红 = v8∪全部；无圈无字）".format(od))
    return 0


if __name__ == "__main__":
    sys.exit(main())
