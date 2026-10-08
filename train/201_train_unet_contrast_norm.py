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
"""201_train_unet_contrast_norm.py — 每张图做对比度均一化（两个臂）

使用者 2026-09-29 提问："有没有尝试对每张图做对比度均一化？"
之前只做过【数据集级】逐通道标准化（200）与【局部反差】通道（199，失败 ✗），
**没有做过按图自身的对比度均一化**。本脚本补上，且**只改输入这一处**（其余同最佳配置 v3）：
  臂 A：按图 2%–98% 分位线性拉伸到 0–255（单通道）
  臂 B：CLAHE（clipLimit=2.0, tile=8×8，单通道）
对照基线：v3（原灰度）平均断线率 22.5%、val IoU 0.683
验收（写死）：平均断线率 < 22.5% 且 val IoU > 0.683 才算通过 ✓；否则判决失败 ✗
标签：你画的 _m（转灰度 >127，**不加宽**，一次只改一处）
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
POS_W, LR = 20.0, 1e-3
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def norm_pct(g):
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    return np.clip((g.astype(np.float32) - lo) * 255.0 / max(1e-6, hi - lo), 0, 255)


def norm_clahe(g):
    return CLAHE.apply(g).astype(np.float32)


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


def dice_loss(p, t, eps=1e-6):
    p = torch.sigmoid(p)
    return 1 - ((2 * (p * t).sum((1, 2, 3)) + eps) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + eps)).mean()


def main():
    bs = []
    for f in sorted(glob.glob(ROOT + "/*/*_m.png")):
        grp = os.path.basename(os.path.dirname(f)); base = os.path.basename(f)[:-6]
        bs.append({"key": grp + "/" + base[-4:], "pk": grp + "-" + base[-4:],
                   "src": os.path.join(SRC, grp, base + ".jpg"), "m": f})
    raw = {}
    for b in bs:
        g = cv2.imread(b["src"], cv2.IMREAD_GRAYSCALE)
        lab = cv2.imread(b["m"], cv2.IMREAD_UNCHANGED)
        lab = (lab if lab.ndim == 2 else cv2.cvtColor(lab[:, :, :3], cv2.COLOR_BGR2GRAY)) > 127
        ps, ns = [], []
        for (x0, y0) in slices(g.shape, STRIDE):
            n = int(lab[y0:y0 + C, x0:x0 + C].sum())
            (ps if n >= 20 else ns).append((x0, y0))
        rng = np.random.default_rng(0); rng.shuffle(ns)
        raw[b["key"]] = {"g": g, "lab": lab, "blocks": ps + ns[:max(1, len(ps))], "pk": b["pk"]}
    print("=== 对比度均一化的效果（线内 vs 环带，灰阶落差 / 图内动态范围）===")
    for k in sorted(raw):
        g, lab = raw[k]["g"], raw[k]["lab"]
        dil = cv2.dilate(lab.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
        ring = dil & (~lab)
        for tag, fn in (("原图", None), ("分位拉伸", norm_pct), ("CLAHE", norm_clahe)):
            x = g.astype(np.float32) if fn is None else fn(g)
            print("  {:<11} {:<7} 落差 {:>6.1f}  动态范围(P2–P98) {:>6.1f}".format(
                k if tag == "原图" else "", tag, float(x[ring].mean() - x[lab].mean()),
                float(np.percentile(x, 98) - np.percentile(x, 2))))
        print()
    tr = [k for k in raw if k not in HOLD]; va = [k for k in raw if k in HOLD]
    for tag, fn in (("A_pct", norm_pct), ("B_clahe", norm_clahe)):
        data = {}
        for k in raw:
            data[k] = {"ch": (raw[k]["g"].astype(np.float32) if fn is None else fn(raw[k]["g"])) / 255.0,
                       "lab": raw[k]["lab"], "blocks": raw[k]["blocks"], "pk": raw[k]["pk"], "g": raw[k]["g"]}
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
        net = U(16, 1, 3).to(DEV)
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
                    best = it / max(1.0, un); bd = {k2: v.clone() for k2, v in net.state_dict().items()}
        net.load_state_dict(bd)
        torch.save({"sd": net.state_dict(), "cin": 1, "crop": C, "norm": tag},
                   os.path.join(UN, "models/unet_v6_%s.pt" % tag))
        net.eval()
        od = os.path.join(PAN, "full_v6_" + tag); os.makedirs(od, exist_ok=True)
        os.makedirs(os.path.join(UN, "pred_masks_v6", tag), exist_ok=True)
        rep = {}
        print("[{}] val IoU {:.4f}  ({:.1f}s)".format(tag, best, time.time() - t0))
        print("  板           预测%   GT%    IoU  片段 两端有 断线率")
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
            cv2.imwrite(os.path.join(UN, "pred_masks_v6", tag, k.replace("/", "_") + ".png"), (pred * 255).astype(np.uint8))
            lab = data[k]["lab"]
            iou = float((pred & lab).sum()) / max(1.0, float((pred | lab).sum()))
            pads, _ = m196.pads_of(data[k]["pk"])
            ss, both, one, none, _ = m196.report("x", pred, pads)
            rate = 100.0 * (one + none) / max(1, len(ss))
            rep[k] = {"pred_pct": 100 * float(pred.mean()), "iou": iou, "segs": len(ss), "break_rate": rate}
            print("  {:<12} {:>6.2f} {:>6.2f}  {:.3f} {:>5} {:>6} {:>6.1f}%".format(
                k, rep[k]["pred_pct"], 100 * float(lab.mean()), iou, len(ss), both, rate))
            vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            vis[lab] = (0.50 * vis[lab] + 0.50 * np.array([0, 200, 0])).astype(np.uint8)
            vis[pred] = (0.30 * vis[pred] + 0.70 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.imwrite(os.path.join(od, k.replace("/", "_") + "_gt_pred.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 86])
        avg = float(np.mean([v["break_rate"] for v in rep.values()]))
        rep["_summary"] = {"val_iou": best, "avg_break_rate": avg,
                           "pass": bool(avg < 22.5 and best > 0.683)}
        json.dump(rep, open(os.path.join(UN, "reports/unet_v6_%s.json" % tag), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("  平均断线率 {:.1f}%（基线 22.5%）→ 验收 {}\n".format(
            avg, "通过 ✓" if (avg < 22.5 and best > 0.683) else "未通过 ✗"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
