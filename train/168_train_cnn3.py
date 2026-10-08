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
"""168_train_cnn3.py — 焊点三分类 CNN（0 非焊点 / 1 正常焊点 / 2 断联焊点）

与 111_train_cnn.py 的关系：**完全复用**它的模型（Stage / SmallCNN）、数据加载（load_dataset，
含逐图归一化与 16TB 广播 bug 的修复）、增强（jitter）、分折（folds_by_sample）与内存日志。
本脚本只做三件事：
  1 把输出头从 1 维改成 3 维（logits + CrossEntropyLoss，而不是 sigmoid + BCE）；
  2 标签改用 out/labels/cnn3_labels.json（类 -1 = 尚未人工裁定 → **排除，不参与训练**）；
  3 指标改为三分类混淆矩阵 + 每类召回/精度，并单独报告**类2（断联焊点）**——这是这套东西存在的理由。

标签来源与规则（见 out/labels/cnn3_labels.json 的 meta）：
  类0 = 标注库 label 0；类1 = 已配对 或 人工判“有飞线”；类2 = 人工判“无飞线”（断联）；
  类-1 = 未配对且未裁定 → 排除。

用法：
  # 连通性自检（2 epoch、1 种子、只跑 2 折）—— 只是验证管道，不是结果
  python 168_train_cnn3.py --smoke
  # 正式：5 折 × 3 种子
  python 168_train_cnn3.py --epochs 40 --seeds 0 1 2
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 复用 111 的全部积木（导入不会执行它的 main：有 __main__ 保护）----
_spec = importlib.util.spec_from_file_location("t111", os.path.join(HERE, "recon", "111_train_cnn.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

NAMES = ("非焊点", "正常焊点", "断联焊点")


class CNN3(T.SmallCNN):
    """SmallCNN 的三分类版：输出头 3 维；前向返回 logits（不接 sigmoid）。"""

    def __init__(self, act="relu", arch="res2_deep", n_out=3):
        super().__init__(act=act, arch=arch)
        self.f2 = nn.Linear(32, n_out)

    def forward(self, x):
        x = self.in_bn(x)
        for st in self.stages:
            x = self.pool(st(x))
        x = self.gap(x).flatten(1)
        x = torch.sigmoid(self.f1(x))          # 保持使用者习惯：线性层后接 sigmoid
        return self.f2(x)                      # logits → CrossEntropyLoss


def load_three_class():
    """返回 ds, xs, y(3类, int64), idx(在 ds 中的下标), groups, samples

    标签直接取自 dataset.json 的 label 字段（0 非焊点 / 1 正常焊点 / 2 断联焊点）。
    该字段由 175_sync_labels_from_folder.py 按使用者的目录归属写入，
    快照 out/labels/labels_20260928_final.json。label<0 视为未裁定 → 排除。
    """
    ds, xs, _ys2, _stat = T.load_dataset()
    keep, y = [], []
    for i, r in enumerate(ds):
        c = int(r["label"])
        if c < 0:
            continue                            # 未裁定 → 不参与训练
        keep.append(i); y.append(c)
    keep = np.array(keep)
    print("  三分类标签来源：dataset.json 的 label 字段（快照 out/labels/labels_20260928_final.json）")
    print("  参与训练 {} 条；分布 0:{} 1:{} 2:{}".format(
        len(y), int((np.array(y) == 0).sum()), int((np.array(y) == 1).sum()),
        int((np.array(y) == 2).sum())))
    return ds, xs[keep], np.array(y, np.int64), keep


def confusion(y, pred, k=3):
    cm = np.zeros((k, k), np.int64)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return cm


def per_class(cm):
    out = []
    for i in range(len(cm)):
        tp = cm[i, i]; fn = cm[i, :].sum() - tp; fp = cm[:, i].sum() - tp
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        pre = tp / (tp + fp) if (tp + fp) else float("nan")
        out.append((int(tp), int(fn), int(fp), rec, pre))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--arch", default="res2_deep")
    ap.add_argument("--act", default="relu")
    ap.add_argument("--max-folds", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true", help="连通性自检：2 epoch / 1 种子 / 2 折")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.seeds, args.max_folds = 2, [0], 2

    dev = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available())
                       else "cpu") if args.device != "cpu" else torch.device("cpu")
    ds, xs, y, keep = load_three_class()
    samples = [ds[i]["sample"] for i in keep]
    groups = {r["sample"]: r["group"] for r in ds}   # folds_by_sample 要的是 {样品: 组} 字典
    cnt = np.bincount(y, minlength=3)
    print("=" * 74)
    print("三分类数据集：{} 条（排除类-1 {} 条）".format(len(y), len(ds) - len(y)))
    print("  类0 非焊点 {} ｜ 类1 正常焊点 {} ｜ 类2 断联焊点 {}".format(*cnt))
    print("  类别权重（N/(3*n_c)）: {}".format(np.round(len(y) / (3.0 * np.maximum(cnt, 1)), 3)))
    print("  设备 {}".format(dev))
    print("=" * 74)

    uniq = sorted(set(samples))          # folds_by_sample 要“样品名去重列表”
    folds = T.folds_by_sample(uniq, groups)
    w = torch.tensor(len(y) / (3.0 * np.maximum(cnt, 1)), dtype=torch.float32).to(dev)
    X = torch.tensor(xs).unsqueeze(1)          # (N,1,48,48)
    Y = torch.tensor(y)

    all_rows = []
    for seed in args.seeds:
        used = list(range(len(folds)))[:args.max_folds] if args.max_folds else list(range(len(folds)))
        fold_acc = []
        for fi in used:
            held = set(map(str, folds[fi]))
            va = np.array([i for i, s in enumerate(samples) if s in held])
            tr = np.array([i for i, s in enumerate(samples) if s not in held])
            torch.manual_seed(seed * 100 + fi); np.random.seed(seed * 100 + fi)
            net = CNN3(act=args.act, arch=args.arch).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=args.lr)
            lossf = nn.CrossEntropyLoss(weight=w)
            ntr = len(tr)
            for ep in range(args.epochs):
                net.train()
                perm = np.random.permutation(tr)
                for b0 in range(0, ntr, args.batch):
                    idx = perm[b0:b0 + args.batch]
                    xb = T.jitter(X[idx].clone()).to(dev)
                    yb = Y[idx].to(dev)
                    opt.zero_grad()
                    loss = lossf(net(xb), yb)
                    loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                pred = net(X[va].to(dev)).argmax(1).cpu().numpy()
            cm = confusion(y[va], pred)
            fold_acc.append((fi + 1, sorted(held), cm))
        # 汇总该种子所有折的混淆矩阵
        cm_all = sum(f[2] for f in fold_acc)
        rows = per_class(cm_all)
        print("\n--- seed {} 汇总（{} 折，共 {} 条留出）---".format(seed, len(fold_acc), int(cm_all.sum())))
        print("  混淆矩阵（行=真值，列=预测）：")
        print("               " + "".join("{:>10}".format(n) for n in NAMES))
        for i, n in enumerate(NAMES):
            print("  真 {:>8} ".format(n) + "".join("{:>10}".format(int(v)) for v in cm_all[i]))
        for i, n in enumerate(NAMES):
            tp, fn, fp, rec, pre = rows[i]
            print("  {:<8} 召回 {:.3f}（{}/{}）  精度 {:.3f}  误报 {}  漏检 {}".format(
                n, rec, tp, tp + fn, pre, fp, fn))
        all_rows.append((seed, cm_all.tolist()))
        acc = float((cm_all.diagonal().sum()) / max(cm_all.sum(), 1))
        print("  总体准确率 {:.4f}".format(acc))

    out = {"tag": args.tag, "arch": args.arch, "act": args.act, "epochs": args.epochs,
           "seeds": args.seeds, "smoke": args.smoke, "counts": cnt.tolist(),
           "results": [{"seed": s, "cm": c} for s, c in all_rows],
           "note": "类2 样本极少（当前 2 条）→ 指标仅供参考，见 out/labels/cnn3_labels.json 的 meta"}
    fp = os.path.join(HERE, "out", "cnn", "train3{}.json".format("_" + args.tag if args.tag else ""))
    json.dump(out, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n结果已写：{}".format(fp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
