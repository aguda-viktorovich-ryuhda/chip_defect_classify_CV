# -*- coding: utf-8 -*-
"""s2_cnn.py — 第②步：CNN 筛选焊点（mean(p) >= thr ✓）

口径（与定格版 V-20261005-003 一致 ✓）：arch=res2_deep, act=relu, 25 个权重（5 种子 × 5 折 ✓）
  新板没有"折归属"（原流程按样品分折 ✗）→ 这里对 25 个模型**全体取均值** ✓（更稳 ✓ 也无需知道折 ✓）
  S7/S8/S9 规则段全部停用 ✓ 判定只由 CNN 给出 ✓
"""
import glob
import os

import numpy as np
import torch

from core import cfg as _cfg
from core import patches as _patches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _device(C):
    d = C["s2_cnn"].get("device", "auto")
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def score_board(g, rows, C, log=None):
    """对 rows（s1 的候选 ✓）逐点打分。返回 (probs(25,N) 数组, 均值(N,), 接受 bool(N,))"""
    from core.vendor import cnn_model as CM
    dev = _device(C)
    ws = os.path.join(ROOT, C["s2_cnn"]["weights_dir"])
    # 聚合口径（cfg 显式 ✓ 不静默切换 ✗）：
    #   fold5  = 复用确认过的口径：该板所属折的 5 个种子模型求均值 ✓（已知板 ✓ 复现 246 ✓）
    #   all25  = 5 种子 × 5 折全体均值（客户新板无折归属时的兜底 ✓）
    agg = C["s2_cnn"].get("aggregate", "fold5")
    smp = rows[0].get("sample") if rows else None
    fmap = C["s2_cnn"].get("fold_map", {})
    if agg == "fold5" and smp in fmap:
        ks = [int(fmap[smp])]
    elif agg == "fold5":
        agg = C["s2_cnn"].get("aggregate_new_board", "all25")   # 新板 → 显式兜底 ✓
        ks = list(C["s2_cnn"]["folds"])
    else:
        ks = list(C["s2_cnn"]["folds"])
    files = []
    for s in C["s2_cnn"]["seeds"]:
        for k in ks:
            f = os.path.join(ws, "model_s{}_f{}.pt".format(s, k))
            if os.path.exists(f):
                files.append(f)
    # ← 归一化必须与训练/114 逐字一致 ✓（recon/114_apply_cnn.py 第 106-109 行原样照搬 ✓）
    #    口径 = 整幅图的 z-score：(块 - 图均值) / 图标准差 ✓（不是 /255 ✗ 我先前猜错过一次 ✓）
    mu, sd = float(g.mean()), float(g.std()) + 1e-6
    X = np.zeros((len(rows), 1, _patches.S, _patches.S), np.float32)
    for i, r in enumerate(rows):
        X[i, 0] = (_patches.crop_at(g, int(r["x"]), int(r["y"])).astype(np.float32) - mu) / sd
    ps = []
    with torch.no_grad():
        for f in files:
            net = CM.build_and_load(f, C["s2_cnn"]["arch"], C["s2_cnn"]["act"], dev)
            out = []
            for i in range(0, len(X), C["s2_cnn"].get("batch", 512)):
                xb = torch.from_numpy(X[i:i + C["s2_cnn"]["batch"]]).to(dev)
                # ← 不许再加 sigmoid ✗：SmallCNN.forward 末尾已有 sigmoid ✓
                #    （114 就是直接 net(xt) 取值 ✓；多套一层会把概率压到 0.5 附近 ✗ 已踩过一次 ✓）
                out.append(net(xb).cpu().numpy().ravel())
            ps.append(np.concatenate(out))
    P = np.array(ps)
    mean = P.mean(axis=0)
    if log:
        log("    用了 {} 个权重（设备 {} ✓）".format(len(files), dev))
    return P, mean, mean >= float(C["s2_cnn"]["thr"])
