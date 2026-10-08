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
"""198_retrain_with_all_labels.py — 用【全部已画标签的板】重训，并出拼接回原图的结果图

相对 193 的变化：
  · 自动发现标签：glob ROOT/*/*_m.png（不再写死板名）✓
  · 两个结构臂一起训（分开"样本量"与"下采样层级"两个假设）：
      unet8 = 3 级下采样（现状，÷8）
      unet4 = 2 级下采样（÷4，细线在瓶颈层还有约 4px，治断线）
  · 按板留出：优先沿用上一轮的验证板（样例板C / 样例板H，若这次仍有标签）以保持可比；否则取最后两张
  · 产：全图拼接叠加图（原分辨率，每板每臂一张）+ 图版 + 指标 JSON
指标复用 196（IoU vs 人工 GT / 断线率 / 各段两端是否有焊点）。
"""
import glob
import json
import os
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location

import cv2
import numpy as np
import torch
import torch.nn as nn

SRC = (_SRC + r"")
ROOT = (_ANN + r"")
UN = (_PROJ + r"/unet")
PAN = (_PROJ + r"/panels")
spec = spec_from_file_location("m196", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "196_gap_metrics.py"))
m196 = module_from_spec(spec); spec.loader.exec_module(m196)
C, STRIDE, EP = 192, 129, 250
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HOLD = ["1_NG-1705", "2_NG-3421"]


def boards():
    out = []
    for f in sorted(glob.glob(ROOT + "/*/*_m.png")):
        grp = os.path.basename(os.path.dirname(f))
        base = os.path.basename(f)[:-len("_m.png")]
        src = os.path.join(SRC, grp, base + ".jpg")
        if not os.path.exists(src):
            print("  ! 找不到原图，跳过：{}".format(src)); continue
        out.append({"key": grp + "/" + base[-4:], "grp": grp, "base": base, "src": src, "m": f})
    return out


def slices(shape, st):
    h, w = shape
    xs = list(range(0, max(1, w - C) + 1, st)); ys = list(range(0, max(1, h - C) + 1, st))
    if w - C not in xs: xs.append(w - C)
    if h - C not in ys: ys.append(h - C)
    return [(x, y) for y in ys for x in xs]


class U(nn.Module):
    """levels=3 → ÷8（现状）；levels=2 → ÷4"""
    def __init__(s, c=16, levels=3):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
        s.levels, s.encs, s.p = levels, nn.ModuleList(), nn.MaxPool2d(2)
        cin = 1
        for i in range(levels):
            o = c * (2 ** i); s.encs.append(blk(cin, o)); cin = o
        s.b = blk(cin, cin * 2)
        s.decs, s.ups = nn.ModuleList(), nn.ModuleList()
        cch = cin * 2
        for i in reversed(range(levels)):
            o = c * (2 ** i)
            s.ups.append(nn.ConvTranspose2d(cch, o, 2, 2)); s.decs.append(blk(o * 2, o)); cch = o
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
    bs = boards()
    print("=== 发现 {} 张有标签的板 ===".format(len(bs)))
    data = {}
    tr, va = [], []
    for b in bs:
        g = cv2.imread(b["src"], cv2.IMREAD_GRAYSCALE)
        raw = cv2.imread(b["m"], cv2.IMREAD_UNCHANGED)
        lab = (raw if raw.ndim == 2 else cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2GRAY)) > 127
        assert g is not None and g.shape[:2] == lab.shape[:2], b["src"]
        ps, ns = [], []
        for (x0, y0) in slices(g.shape, STRIDE):
            n = int(lab[y0:y0 + C, x0:x0 + C].sum())
            (ps if n >= 20 else ns).append((x0, y0))
        rng = np.random.default_rng(0); rng.shuffle(ns)
        keep = ps + ns[:max(1, len(ps))]
        is_va = b["key"] in HOLD
        (va if is_va else tr).append(b["key"])
        data[b["key"]] = {"g": g, "lab": lab, "blocks": keep, "npos": len(ps), "gt_pct": 100 * lab.mean()}
        print("  {:<12} 块 {:>3}（正 {:>3} / 负 {:>3}） GT占比 {:.3f}%  {}".format(
            b["key"], len(keep), len(ps), len(keep) - len(ps), 100 * lab.mean(), "验证" if is_va else "训练"))
    if not va:
        va = [tr.pop()]; va = list(va)
    print("  训练板 {} 张 / 验证板 {} 张".format(len(tr), len(va)))

    rng = np.random.default_rng(1)
    def load(keys, aug=False):
        X, Y = [], []
        for k in keys:
            g, lab = data[k]["g"], data[k]["lab"]
            for (x0, y0) in data[k]["blocks"]:
                a = g[y0:y0 + C, x0:x0 + C].astype(np.float32) / 255.0
                m = lab[y0:y0 + C, x0:x0 + C].astype(np.float32)
                if aug:
                    j = int(rng.integers(0, 4)); a, m = np.rot90(a, j), np.rot90(m, j)
                    if rng.random() < 0.5: a, m = a[:, ::-1], m[:, ::-1]
                    a, m = np.ascontiguousarray(a), np.ascontiguousarray(m)
                X.append(a); Y.append(m)
        return (torch.from_numpy(np.stack(X)).unsqueeze(1), torch.from_numpy(np.stack(Y)).unsqueeze(1))
    Xtr, Ytr = load(tr, True); Xva, Yva = load(va)
    print("  训练 {} 块 / 验证 {} 块".format(len(Xtr), len(Xva)))

    results = {}
    for tag, levels in (("unet8", 3), ("unet4", 2)):
        net = U(16, levels).to(DEV)
        npar = sum(p.numel() for p in net.parameters())
        pw = torch.tensor([min(20.0, max(1.0, (1 - float(Ytr.mean())) / max(1e-6, float(Ytr.mean()))))], device=DEV)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        best, bd = -1, None
        t0 = time.time()
        for ep in range(1, EP + 1):
            net.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), 8):
                idx = perm[i:i + 8]
                out = net(Xtr[idx].to(DEV))
                loss = 0.5 * bce(out, Ytr[idx].to(DEV)) + 0.5 * dice_loss(out, Ytr[idx].to(DEV))
                opt.zero_grad(); loss.backward(); opt.step()
            if ep % 10 == 0 or ep == EP:
                net.eval()
                with torch.no_grad():
                    pv = torch.sigmoid(net(Xva.to(DEV)))
                it = float(((pv > 0.5) & (Yva.to(DEV) > 0.5)).sum())
                un = float(((pv > 0.5) | (Yva.to(DEV) > 0.5)).sum())
                if it / max(1.0, un) > best:
                    best = it / max(1.0, un)
                    bd = {k: v.clone() for k, v in net.state_dict().items()}
        net.load_state_dict(bd)
        torch.save({"sd": net.state_dict(), "levels": levels, "crop": C}, os.path.join(UN, "models/unet_v3_%s.pt" % tag))
        print("  [{}] {} 参数 {:.1f}k  val IoU {:.4f}  ({:.1f}s)".format(tag, "÷8" if levels == 3 else "÷4", npar / 1000, best, time.time() - t0))
        # 全图推理 + 指标
        net.eval()
        per = {}
        for k in data:
            g = data[k]["g"]; h, w = g.shape
            acc = np.zeros((h, w), np.float32); cnt = np.zeros((h, w), np.float32)
            tl = slices((h, w), STRIDE)
            bt = np.zeros((len(tl), 1, C, C), np.float32)
            for i, (x0, y0) in enumerate(tl):
                bt[i, 0] = g[y0:y0 + C, x0:x0 + C].astype(np.float32) / 255.0
            with torch.no_grad():
                for i in range(0, len(tl), 16):
                    p = torch.sigmoid(net(torch.from_numpy(bt[i:i + 16]).to(DEV))).cpu().numpy()[:, 0]
                    for j, (x0, y0) in enumerate(tl[i:i + 16]):
                        acc[y0:y0 + C, x0:x0 + C] += p[j]; cnt[y0:y0 + C, x0:x0 + C] += 1
            pred = (acc / np.maximum(1.0, cnt)) > 0.5
            os.makedirs(os.path.join(UN, "pred_masks_v3", tag), exist_ok=True)
            cv2.imwrite(os.path.join(UN, "pred_masks_v3", tag, k.replace("/", "_") + ".png"), (pred * 255).astype(np.uint8))
            gt = data[k]["lab"]
            inter = float((pred & gt).sum()); union = float((pred | gt).sum())
            pk = k.split("/")[0] + "-" + k.split("/")[1]      # 焊点表里的样品名形如 1_NG-1649
            pads, _ = m196.pads_of(pk)
            ss, both, one, none, gaps = m196.report(tag, pred, pads)
            per[k] = {"pred_pct": 100 * float(pred.mean()), "gt_pct": data[k]["gt_pct"],
                      "iou": inter / max(1.0, union), "segs": len(ss), "both": both, "half": one, "free": none,
                      "break_rate": 100.0 * (one + none) / max(1, len(ss))}
            # 原分辨率叠加图（预测=红）
            vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            vis[pred] = (0.35 * vis[pred] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
            od = os.path.join(PAN, "full_v3_" + tag); os.makedirs(od, exist_ok=True)
            cv2.imwrite(os.path.join(od, k.replace("/", "_") + ".jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        results[tag] = {"levels": levels, "params": npar, "val_iou": best, "per_board": per}
        print("      逐板：")
        for k, v in per.items():
            print("        {:<12} 预测{:>6.2f}% GT{:>6.2f}% IoU {:.3f}  片段{:>3} 两端有{:>3} 断线率 {:>5.1f}%".format(
                k, v["pred_pct"], v["gt_pct"], v["iou"], v["segs"], v["both"], v["break_rate"]))
    os.makedirs(os.path.join(UN, "reports"), exist_ok=True)
    json.dump(results, open(os.path.join(UN, "reports/unet_v3.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 图版：每板一行（unet8 上 / unet4 下），0.33x
    from PIL import Image, ImageDraw, ImageFont
    F = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 15)
    FB = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 20)
    keys = sorted(data.keys())
    sc = 0.33; cw, ch = int(922 * sc), int(924 * sc)
    page = Image.new("RGB", (14 + len(keys) * (cw + 6), 96 + 2 * (ch + 22)), (18, 18, 18))
    d = ImageDraw.Draw(page)
    d.text((14, 8), "重训结果（上=÷8 三级下采样，下=÷4 两级下采样；红=预测；0.33x）", font=FB, fill=(255, 225, 60))
    d.text((14, 36), "训练板 {} 张 / 验证板 {}；白条=原图 50px".format(len(tr), "/".join(va)), font=F, fill=(180, 255, 180))
    for i, k in enumerate(keys):
        g = data[k]["g"]; gt = data[k]["lab"]
        x = 14 + i * (cw + 6)
        for r, tag in enumerate(("unet8", "unet4")):
            pm = cv2.imread(os.path.join(UN, "pred_masks_v3", tag, k.replace("/", "_") + ".png"), cv2.IMREAD_GRAYSCALE) > 127
            vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            vis[gt] = (0.45 * vis[gt] + 0.55 * np.array([0, 200, 0])).astype(np.uint8)
            vis[pm] = (0.35 * vis[pm] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
            y = 60 + r * (ch + 22)
            page.paste(Image.fromarray(cv2.cvtColor(cv2.resize(vis, (cw, ch), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)), (x, y))
            d.rectangle([x + 4, y + ch - 11, x + 4 + int(50 * sc), y + ch - 7], fill=(255, 255, 255))
        d.text((x, 60 + 2 * (ch + 22)), "{}\n断线率 {:.0f}%/{:.0f}%".format(
            k, results["unet8"]["per_board"][k]["break_rate"], results["unet4"]["per_board"][k]["break_rate"]),
            font=F, fill=(235, 235, 235))
    p = os.path.join(PAN, "198_retrain_alllabels.jpg")
    page.save(p, "JPEG", quality=76)
    print("\n图 -> {}  {:.2f}MB".format(p, os.path.getsize(p) / 1048576))
    print("报告 -> {}/reports/unet_v3.json".format(UN))
    return 0


if __name__ == "__main__":
    sys.exit(main())
