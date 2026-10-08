# -*- coding: utf-8 -*-
"""unet_v13.py - v13 成员（自 210_train_raw_overlap50.py 第 42..67 行原样抽取）

输入 = 原图/255（ch ✓ 源第 113 行）；v13 无额外前端 ✓
切块/拼回统一口径：192×192 / 步长 96（50% 重叠）/ 按覆盖次数逐像元平均 ✓（与 recon/208 的 infer 同构 ✓）
禁止改网络结构 - 结构一致才能直接加载既有权重 ✓
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


def infer_v13(net, ch, thr=0.5):
    dec = net
    if hasattr(net, "eval"):
        pass
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
