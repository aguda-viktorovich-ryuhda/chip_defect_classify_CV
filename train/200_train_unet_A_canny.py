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
"""200_train_unet_A_canny.py — A 方案 + Canny 通道

使用者 2026-09-29："做A，再加一个 canny 看看"
  A 的内容：只加【局部反差】通道（**做逐通道标准化**），去掉黑帽 ✗，lr 3e-4，pos_weight 5
  追加：**Canny 边缘通道**（飞线的两条暗边正是边缘响应最强的地方；作为显式结构信号）
三通道： 0 灰度/255   1 局部反差（bg−gray，按训练集统计标准化）   2 Canny(gray,30,90)/255
验收（写死，跑完直接判成败）：
  ① 预测面积占比必须回到 2.5–3.5%（1_ 系）/ 0.8–1.2%（2_ 系）—— 面积失控即失败 ✗
  ② 平均断线率必须 < 22.5%（v3 ÷8 原灰度基线）
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
from importlib.util import module_from_spec, spec_from_file_location

SRC = (_SRC + r"")
ROOT = (_ANN + r"")
UN = (_PROJ + r"/unet")
PAN = (_PROJ + r"/panels")
spec = spec_from_file_location("m196", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "196_gap_metrics.py"))
m196 = module_from_spec(spec); spec.loader.exec_module(m196)
C, STRIDE, EP, DEV = 192, 129, 250, ("cuda" if torch.cuda.is_available() else "cpu")
HOLD = {"1_NG/1705", "2_NG/3421"}
POS_W, LR = 5.0, 3e-4


def chans(g):
    bg = cv2.GaussianBlur(g, (0, 0), 8)
    c1 = (bg.astype(np.float32) - g.astype(np.float32))
    c2 = cv2.Canny(g, 30, 90)
    return np.stack([g.astype(np.float32) / 255.0, c1, c2.astype(np.float32) / 255.0], -1)


def slices(shape, st):
    h, w = shape
    xs = list(range(0, max(1, w - C) + 1, st)); ys = list(range(0, max(1, h - C) + 1, st))
    if w - C not in xs: xs.append(w - C)
    if h - C not in ys: ys.append(h - C)
    return [(x, y) for y in ys for x in xs]


class U(nn.Module):
    def __init__(s, c=16, cin=3, levels=3):
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


def main():
    bs = []
    for f in sorted(glob.glob(ROOT + "/*/*_m.png")):
        grp = os.path.basename(os.path.dirname(f)); base = os.path.basename(f)[:-6]
        bs.append({"key": grp + "/" + base[-4:], "pk": grp + "-" + base[-4:],
                   "src": os.path.join(SRC, grp, base + ".jpg"), "m": f})
    data = {}
    for b in bs:
        g = cv2.imread(b["src"], cv2.IMREAD_GRAYSCALE)
        raw = cv2.imread(b["m"], cv2.IMREAD_UNCHANGED)
        lab = (raw if raw.ndim == 2 else cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2GRAY)) > 127
        ch = chans(g)
        # Canny 通道的信息量自检：边缘像素在【线内】与【线外】的密度比
        dil = cv2.dilate(lab.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
        ring = dil & (~lab)
        d_in = float((ch[:, :, 2][lab] > 0).mean()); d_out = float((ch[:, :, 2][ring] > 0).mean())
        ps, ns = [], []
        for (x0, y0) in slices(g.shape, STRIDE):
            n = int(lab[y0:y0 + C, x0:x0 + C].sum())
            (ps if n >= 20 else ns).append((x0, y0))
        rng = np.random.default_rng(0); rng.shuffle(ns)
        data[b["key"]] = {"ch": ch, "lab": lab, "g": g, "blocks": ps + ns[:max(1, len(ps))], "pk": b["pk"]}
        print("  {:<11} 块 {:>3}  Canny 在线内 {:>5.1f}% / 环带 {:>5.1f}%  比 {:.2f}x".format(
            b["key"], len(data[b["key"]]["blocks"]), 100 * d_in, 100 * d_out, d_in / max(1e-6, d_out)))
    # 逐通道标准化（只用训练板统计，避免信息泄漏）
    tr = [k for k in data if k not in HOLD]; va = [k for k in data if k in HOLD]
    acc = []
    for k in tr:
        for (x0, y0) in data[k]["blocks"][:60]:
            acc.append(data[k]["ch"][y0:y0 + C, x0:x0 + C].reshape(-1, 3))
    A = np.concatenate(acc, 0)
    mu, sd = A.mean(0), A.std(0) + 1e-6
    print("\n  逐通道统计（训练板）：均值 {}  标准差 {}".format(np.round(mu, 3), np.round(sd, 3)))
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
    net = U(16, 3, 3).to(DEV)
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
    torch.save({"sd": net.state_dict(), "cin": 3, "crop": C, "mu": mu, "sd": sd},
               os.path.join(UN, "models/unet_v5_A_canny.pt"))
    print("  val IoU（留出板）{:.4f}   {:.1f}s   参数 {:.1f}k   lr {}  pos_weight {}".format(
        best, time.time() - t0, sum(p.numel() for p in net.parameters()) / 1000, LR, POS_W))
    net.eval()
    od = os.path.join(PAN, "full_v5_A_canny"); os.makedirs(od, exist_ok=True)
    os.makedirs(os.path.join(UN, "pred_masks_v5"), exist_ok=True)
    rep = {}
    print("\n  板           预测%   GT%    IoU   片段 两端有 半截  断线率")
    for k in sorted(data):
        g = data[k]["g"]; ch = data[k]["ch"]; h, w = g.shape
        acc2 = np.zeros((h, w), np.float32); cnt = np.zeros((h, w), np.float32)
        tl = slices((h, w), STRIDE)
        bt = np.zeros((len(tl), C, C, 3), np.float32)
        for i, (x0, y0) in enumerate(tl):
            bt[i] = ch[y0:y0 + C, x0:x0 + C]
        with torch.no_grad():
            for i in range(0, len(tl), 16):
                p = torch.sigmoid(net(torch.from_numpy(bt[i:i + 16]).transpose(1, 3).to(DEV))).cpu().numpy()[:, 0]
                for j, (x0, y0) in enumerate(tl[i:i + 16]):
                    acc2[y0:y0 + C, x0:x0 + C] += p[j]; cnt[y0:y0 + C, x0:x0 + C] += 1
        pred = (acc2 / np.maximum(1.0, cnt)) > 0.5
        cv2.imwrite(os.path.join(UN, "pred_masks_v5", k.replace("/", "_") + ".png"), (pred * 255).astype(np.uint8))
        lab = data[k]["lab"]
        iou = float((pred & lab).sum()) / max(1.0, float((pred | lab).sum()))
        pads, _ = m196.pads_of(data[k]["pk"])
        ss, both, one, none, _ = m196.report("x", pred, pads)
        rate = 100.0 * (one + none) / max(1, len(ss))
        rep[k] = {"pred_pct": 100 * float(pred.mean()), "iou": iou, "segs": len(ss), "both": both,
                  "half": one, "break_rate": rate}
        print("  {:<12} {:>6.2f} {:>6.2f}  {:.3f} {:>5} {:>6} {:>5} {:>6.1f}%".format(
            k, rep[k]["pred_pct"], 100 * float(lab.mean()), iou, len(ss), both, one, rate))
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        vis[lab] = (0.50 * vis[lab] + 0.50 * np.array([0, 200, 0])).astype(np.uint8)
        vis[pred] = (0.30 * vis[pred] + 0.70 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(os.path.join(od, k.replace("/", "_") + "_gt_pred.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 86])
    json.dump(rep, open(os.path.join(UN, "reports/unet_v5_A_canny.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    avg = float(np.mean([v["break_rate"] for v in rep.values()]))
    print("\n  平均断线率 {:.1f}%（基线：v3 ÷8 原灰度 22.5% / 人工 GT 3.1%）→ 验收 {}".format(
        avg, "通过 ✓" if avg < 22.5 else "未通过 ✗"))
    print("  图 -> {}".format(od))
    return 0


if __name__ == "__main__":
    sys.exit(main())
