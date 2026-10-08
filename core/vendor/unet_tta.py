# -*- coding: utf-8 -*-
"""unet_tta.py - TTA 成员推理（自 recon/208_tta_and_v12.py 第 47..157 行原样搬入）

含 norm_pct / prep / slices / infer / tta_prob（9 个单通道模型 x 4 视图）
C=192 STRIDE=96 与本模块里的常量一致 - 不许改，改了与存盘 pred_masks_tta 对不上。
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


C = 192
STRIDE = 96
CLAHE2 = cv2.createCLAHE(2.0, (8, 8))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 208 的 DEV（CUDA 可用则 GPU ✓）   # 208 第 41 行原样 ✓（clahe 模式用 ✓）


def norm_pct(g):
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    return np.clip((g.astype(np.float32) - lo) * 255.0 / max(1e-6, hi - lo), 0, 255)


def prep(g, mode):
    if mode == "pct": return norm_pct(g) / 255.0
    if mode == "clahe": return CLAHE2.apply(g).astype(np.float32) / 255.0
    return g.astype(np.float32) / 255.0


def slices(shape, st=STRIDE):
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


def infer(net, arr):
    h, w = arr.shape
    acc = np.zeros((h, w), np.float32); cnt = np.zeros((h, w), np.float32)
    tl = slices((h, w))
    bt = np.zeros((len(tl), C, C), np.float32)
    for i, (x0, y0) in enumerate(tl):
        bt[i] = arr[y0:y0 + C, x0:x0 + C]
    with torch.no_grad():
        for i in range(0, len(tl), 16):
            p = torch.sigmoid(net(torch.from_numpy(bt[i:i + 16])[:, None].to(DEV))).cpu().numpy()[:, 0]
            for j, (x0, y0) in enumerate(tl[i:i + 16]):
                acc[y0:y0 + C, x0:x0 + C] += p[j]; cnt[y0:y0 + C, x0:x0 + C] += 1
    return acc / np.maximum(1.0, cnt)


def tta_prob(net, arr):
    views = [(lambda a: a, lambda p: p),
             (np.fliplr, np.fliplr),
             (np.flipud, np.flipud),
             (lambda a: np.rot90(a, 2), lambda p: np.rot90(p, 2))]
    ps = []
    for f, g in views:
        p = infer(net, np.ascontiguousarray(f(arr)))
        ps.append(np.ascontiguousarray(g(p)))
    return np.mean(ps, 0)
