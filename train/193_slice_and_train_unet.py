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
"""193_slice_and_train_unet.py — 192x192 切块(重叠33%) + 小 U-Net 训练（一次跑完）

数据（全部在 SharedData，训练产物不回 /data ✗）：
  原图 只读 <原始样品图目录>（由 CHIP_SRC 指定）
  标签 <路径见 本地配置.env>/某电子公司印刷电路samples__20260128/*/*_m.png
       统一口径：转灰度 → >127 取标注（5 板：样例板A/样例板B/样例板C/样例板G/样例板H）
切块：192×192、步长 129（33% 重叠）、补边对齐；
     正样本=含标注像素≥20；负样本=含 0 标注像素（等量随机抽，含沟槽/黑框等困难负例）
划分：训练=样例板A/样例板B/样例板G；验证=样例板C/样例板H（**按板留出**，不混板）
模型：小 U-Net（1→16→32→64 三级下采样，每层 Conv3+BN+ReLU ×2，上采样用最近邻+Conv）
     输入归一化，末端 sigmoid；损失 0.5*BCE(pos_weight) + 0.5*Dice
产出：SharedData/chip_wire_ng/unet/{blocks_train,labels_train,models,reports}
"""
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

SRC = (_SRC + r"")
ROOT = (_ANN + r"")
OUT = (_PROJ + r"/unet")
BOARDS = ["1_NG/微信图片_20240711151649", "1_NG/微信图片_20240711151658", "1_NG/微信图片_20240711151705",
          "2_NG/微信图片_20240711153414", "2_NG/微信图片_20240711153421"]
SHORT = {"微信图片_20240711151649": "1_NG-1649", "微信图片_20240711151658": "1_NG-1658",
         "微信图片_20240711151705": "1_NG-1705", "微信图片_20240711153414": "2_NG-3414",
         "微信图片_20240711153421": "2_NG-3421"}
TRAIN_B = ["1_NG-1649", "1_NG-1658", "2_NG-3414"]
VAL_B = ["1_NG-1705", "2_NG-3421"]
C, STRIDE = 192, 129
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def slices(shape, st):
    h, w = shape
    xs = list(range(0, max(1, w - C) + 1, st))
    ys = list(range(0, max(1, h - C) + 1, st))
    if w - C not in xs:
        xs.append(w - C)
    if h - C not in ys:
        ys.append(h - C)
    return [(x, y) for y in ys for x in xs]


def build():
    rng = np.random.default_rng(0)
    data = {}
    for b in BOARDS:
        s = SHORT[b.split("/")[-1]]
        g = cv2.imread(os.path.join(SRC, b + ".jpg"), cv2.IMREAD_GRAYSCALE)
        raw = cv2.imread(os.path.join(ROOT, b + "_m.png"), cv2.IMREAD_UNCHANGED)
        assert g is not None and raw is not None, b
        lab = (raw if raw.ndim == 2 else cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2GRAY)) > 127
        if os.environ.get("LABEL_MODE", "raw") == "wide":     # 197 生成的加宽版（焊点处未加宽、未粘连）
            wp = os.path.join(OUT, "labels_widened", s + ".png")
            if os.path.exists(wp):
                lab = cv2.imread(wp, cv2.IMREAD_GRAYSCALE) > 127
        ps, ns = [], []
        for (x0, y0) in slices(g.shape, STRIDE):
            if g[y0:y0 + C, x0:x0 + C].shape != (C, C):
                continue
            n = int(lab[y0:y0 + C, x0:x0 + C].sum())
            (ps if n >= 20 else ns).append((x0, y0, n))
        rng.shuffle(ns)
        keep = ps + ns[:max(1, int(len(ps) * 1.0))]
        data[s] = {"img": g, "lab": lab, "blocks": keep, "n_pos": len(ps), "n_neg": len(keep) - len(ps)}
        print("  {:<10} 块 {}（正 {} / 负 {}）".format(s, len(keep), len(ps), len(keep) - len(ps)))
    return data


def load(data, boards, train=True, rng=None):
    X, Y = [], []
    for s in boards:
        g, lab = data[s]["img"], data[s]["lab"]
        for (x0, y0, _) in data[s]["blocks"]:
            a = g[y0:y0 + C, x0:x0 + C].astype(np.float32) / 255.0
            m = lab[y0:y0 + C, x0:x0 + C].astype(np.float32)
            if train and rng is not None:
                k = int(rng.integers(0, 4))
                a, m = np.rot90(a, k), np.rot90(m, k)
                if rng.random() < 0.5:
                    a, m = a[:, ::-1], m[:, ::-1]
                a, m = np.ascontiguousarray(a), np.ascontiguousarray(m)
            X.append(a); Y.append(m)
    return (torch.from_numpy(np.stack(X)).unsqueeze(1), torch.from_numpy(np.stack(Y)).unsqueeze(1))


class UNet(nn.Module):
    def __init__(s, c=16):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        s.e1, s.e2, s.e3 = blk(1, c), blk(c, c * 2), blk(c * 2, c * 4)
        s.p = nn.MaxPool2d(2)
        s.b = blk(c * 4, c * 8)
        s.u3, s.d3 = nn.ConvTranspose2d(c * 8, c * 4, 2, 2), blk(c * 8, c * 4)
        s.u2, s.d2 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2), blk(c * 4, c * 2)
        s.u1, s.d1 = nn.ConvTranspose2d(c * 2, c, 2, 2), blk(c * 2, c)
        s.out = nn.Conv2d(c, 1, 1)

    def forward(s, x):
        e1 = s.e1(x)
        e2 = s.e2(s.p(e1))
        e3 = s.e3(s.p(e2))
        b = s.b(s.p(e3))
        d3 = s.d3(torch.cat([s.u3(b), e3], 1))
        d2 = s.d2(torch.cat([s.u2(d3), e2], 1))
        d1 = s.d1(torch.cat([s.u1(d2), e1], 1))
        return s.out(d1)


def dice_loss(p, t, eps=1e-6):
    p = torch.sigmoid(p)
    num = 2 * (p * t).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + eps
    return 1 - (num / den).mean()


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== 切块（{}×{}，步长 {} = 33% 重叠，补边对齐）===".format(C, C, STRIDE))
    data = build()
    tr_boards = [k for k in data if k in TRAIN_B]
    va_boards = [k for k in data if k in VAL_B]
    rng = np.random.default_rng(1)
    Xtr, Ytr = load(data, tr_boards, True, rng)
    Xva, Yva = load(data, va_boards, False)
    print("  训练 {} 块（{} 板）  验证 {} 块（{} 板）  设备 {}".format(
        len(Xtr), len(tr_boards), len(Xva), len(va_boards), DEV))
    print("  标签正例占比：训练 {:.2f}%  验证 {:.2f}%".format(100 * Ytr.mean(), 100 * Yva.mean()))

    net = UNet().to(DEV)
    n_par = sum(p.numel() for p in net.parameters())
    pos_w = torch.tensor([min(20.0, max(1.0, (1.0 - float(Ytr.mean())) / max(1e-6, float(Ytr.mean()))))], device=DEV)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    EP, BS = int(os.environ.get("EP", "40")), 8
    print("  模型 {} 参数 {:.1f}k；损失 0.5*BCE(pos_weight={:.1f}) + 0.5*Dice".format(
        "UNet-16", n_par / 1000, float(pos_w)))
    best, hist = -1.0, []
    t0 = time.time()
    for ep in range(1, EP + 1):
        net.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), BS):
            idx = perm[i:i + BS]
            xb, yb = Xtr[idx].to(DEV), Ytr[idx].to(DEV)
            out = net(xb)
            loss = 0.5 * bce(out, yb) + 0.5 * dice_loss(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xva.to(DEV)))
        inter = float(((pv > 0.5) & (Yva.to(DEV) > 0.5)).sum())
        union = float(((pv > 0.5) | (Yva.to(DEV) > 0.5)).sum())
        iou = inter / max(1.0, union)
        dice = 2 * inter / max(1.0, float((pv > 0.5).sum() + (Yva.to(DEV) > 0.5).sum()))
        hist.append((ep, tot / len(Xtr), iou, dice))
        if dice > best:
            best = dice
            torch.save({"sd": net.state_dict(), "iou": iou, "dice": dice, "ep": ep, "c": C},
                       os.path.join(OUT, "models/unet_v1.pt"))
        if ep % 5 == 0 or ep == 1:
            print("   ep{:>3}  loss {:.4f}  val IoU {:.4f}  Dice {:.4f}".format(ep, tot / len(Xtr), iou, dice))
    print("  最佳 val Dice {:.4f}（{:.1f}s，{}）".format(best, time.time() - t0, DEV))

    # 逐板 IoU
    net.load_state_dict(torch.load(os.path.join(OUT, "models/unet_v1.pt"))["sd"])
    net.eval()
    per = {}
    for s in va_boards:
        Xb, Yb = load(data, [s], False)
        with torch.no_grad():
            pv = torch.sigmoid(net(Xb.to(DEV))) > 0.5
        Yb = Yb.to(DEV) > 0.5
        per[s] = float(((pv & Yb).sum()) / max(1.0, float(((pv | Yb).sum()))))
    print("  逐板 val IoU：" + "  ".join("{} {:.3f}".format(k, v) for k, v in per.items()))
    json.dump({"hist": hist, "best_dice": best, "per_board_iou": per, "params": n_par,
               "train_blocks": len(Xtr), "val_blocks": len(Xva),
               "train_boards": tr_boards, "val_boards": va_boards, "crop": C, "stride": STRIDE},
              open(os.path.join(OUT, "reports/unet_v1.json"), "w", encoding="utf-8"), ensure_ascii=False)

    # 出图：验证板各 2 例（原图 / GT / 预测）
    from PIL import Image, ImageDraw, ImageFont
    F = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 16)
    FB = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 21)
    rows = []
    for s in va_boards:
        g, lab = data[s]["img"], data[s]["lab"]
        pos = [b for b in data[s]["blocks"] if b[2] >= 20][:2]
        for (x0, y0, _) in pos:
            a = g[y0:y0 + C, x0:x0 + C].astype(np.float32) / 255.0
            with torch.no_grad():
                pv = (torch.sigmoid(net(torch.from_numpy(a)[None, None].to(DEV)))[0, 0].cpu().numpy() > 0.5)
            rows.append((s, x0, y0, g[y0:y0 + C, x0:x0 + C], lab[y0:y0 + C, x0:x0 + C], pv))
    ph = 3 * 172
    page = Image.new("RGB", (14 + 4 * (C + 10), ph + 96), (18, 18, 18))
    d = ImageDraw.Draw(page)
    d.text((14, 8), "U-Net v1 验证结果（每行：原图 / 人工标签 / 模型预测 / 叠加）", font=FB, fill=(255, 225, 60))
    d.text((14, 36), "验证板 {}（按板留出）；最佳 val Dice {:.3f}".format("/".join(va_boards), best),
           font=F, fill=(180, 255, 180))
    for r, (s, x0, y0, gi, gt, pv) in enumerate(rows[:3]):
        y = 62 + r * 172
        ov = cv2.cvtColor(gi, cv2.COLOR_GRAY2BGR)
        ov[pv] = (0.45 * ov[pv] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        for k, im in enumerate((gi, (gt * 255).astype(np.uint8), (pv * 255).astype(np.uint8), ov)):
            if k < 3:
                page.paste(Image.fromarray(im).convert("RGB").resize((C, C)), (14 + k * (C + 10), y))
            else:
                page.paste(Image.fromarray(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)).resize((C, C)), (14 + k * (C + 10), y))
        d.text((14 + 4 * (C + 10) - 190, y + 4), "{} ({},{})".format(s, x0, y0), font=F, fill=(0, 0, 0))
    p = os.path.join((_PROJ + r"/panels/193_unet_v1_val.jpg"))
    page.save(p, "JPEG", quality=82)
    print("  图 -> {}  {:.2f}MB".format(p, os.path.getsize(p) / 1048576))
    return 0


if __name__ == "__main__":
    sys.exit(main())
