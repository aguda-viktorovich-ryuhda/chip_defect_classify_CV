# -*- coding: utf-8 -*-
"""unet_v18.py - v18 成员（自 recon/215_conv3_sig_resnorm.py 第 46..86 行【按缩进整类】抽取）

输入 = 原图/255（源第 132 行 ch ✓）；z-score 标准化 + 残差标准化 + sigmoid 全在该网络前端 front 内 ✓
切块/拼回：192×192 / 步长 96 / 按覆盖次数逐像元平均 ✓（与 recon/208 infer 同构 ✓）
禁止改结构 - 结构一致才能加载既有权重 ✓
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


C = 192
STRIDE = 96
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class Net(nn.Module):
    def __init__(s, c=16, cin=1, levels=3):
        super().__init__()
        s.front = nn.Sequential(                                  # ← sigmoid 激活 + BN（BN 在激活前 ✓）
            nn.Conv2d(cin, 16, 3, padding=1), nn.BatchNorm2d(16), nn.Sigmoid(),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.Sigmoid(),
            nn.Conv2d(16, cin, 3, padding=1), nn.BatchNorm2d(cin))
        nn.init.zeros_(s.front[-2].weight); nn.init.zeros_(s.front[-2].bias)
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

    def norm(s, x):
        m = x.mean(dim=(1, 2, 3), keepdim=True); sd = x.std(dim=(1, 2, 3), keepdim=True) + 1e-6
        return (x - m) / sd

    def forward(s, x, probe=None):
        xn = s.norm(x)                                   # ← ① 先标准化
        h = s.front(xn)
        if probe is not None:
            probe["h"] = float(h.abs().mean()); probe["xn"] = float(xn.abs().mean())
            probe["x_raw"] = float(x.abs().mean()); probe["h_max"] = float(h.abs().max())
        x = h + xn                                       # ← ② 残差支也标准化（两支同尺度 ✓）
        es = []
        for e in s.encs:
            x = e(x); es.append(x); x = s.p(x)
        x = s.b(x)
        for i, (u, d) in enumerate(zip(s.ups, s.decs)):
            x = d(torch.cat([u(x), es[-1 - i]], 1))
        return s.out(x)


def infer_v18(net, ch, thr=0.5):
    h, w = ch.shape
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
    return (acc / np.maximum(1.0, cnt)) > thr
