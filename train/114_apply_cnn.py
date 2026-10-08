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
"""
114_apply_cnn.py  —  CNN 仲裁器：对一张图的**全部候选点**逐个判分（使用者要求"所有的点由 CNN 判断"）

流程：S1–S6 产生候选（与现行管线完全一致）→ 每个候选裁 48×48 → 逐图归一化
      → **多种子集成**打分（每个种子用"该图被留出"的那一折模型，保证与评估口径一致）
      → 阈值判 0/1 → 输出逐图内外圈检出数、与规则链的差异、CPU/GPU 耗时。

用法：
  python 114_apply_cnn.py --tag gpu5 --device cuda   # 用 GPU 打分并计时
  python 114_apply_cnn.py --tag gpu5 --device cpu    # 用 CPU 打分并计时（部署成本参考）
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "lib"))
import pipeline as P   # noqa: E402

import importlib.util as _iu
_spec = _iu.spec_from_file_location("t", os.path.join(HERE, "recon", "111_train_cnn.py"))
T = _iu.module_from_spec(_spec)
sys.modules["t"] = T
_spec.loader.exec_module(T)

S = 48


def short_name(rel):
    grp = os.path.dirname(rel).replace("/", "")
    base = os.path.splitext(os.path.basename(rel))[0].split("_")[-1][-4:]
    return "{}-{}".format(grp, base)


def crop_at(g, px, py, s=S):
    r = s // 2
    ph, pw = g.shape
    y0, x0 = py - r, px - r
    top, left = max(0, -y0), max(0, -x0)
    bot, right = max(0, y0 + s - ph), max(0, x0 + s - pw)
    p = g[max(0, y0):min(ph, y0 + s), max(0, x0):min(pw, x0 + s)]
    if top or bot or left or right:
        p = cv2.copyMakeBorder(p, top, bot, left, right, cv2.BORDER_REPLICATE)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gpu5")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--model-dir", default=os.path.join(HERE, "out", "cnn"))
    args = ap.parse_args()

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else ("cpu" if args.device == "auto" else args.device))
    # 折划分（与训练脚本一致，保证"该图被留出"的那一折模型可用）
    ds = json.load(open(os.path.join(HERE, "out", "cnn", "dataset.json"), encoding="utf-8"))
    samples = sorted(set(r["sample"] for r in ds))
    groups = {r["sample"]: r["group"] for r in ds}
    folds = T.folds_by_sample(samples, groups)
    fold_of = {s: i for i, f in enumerate(folds) for s in f}
    seeds = sorted(int(os.path.basename(p).split("_s")[1].split("_")[0])
                   for p in glob.glob(os.path.join(args.model_dir, "model_s*_f1.pt")))
    print("设备={} 阈值={} 种子={} 折数={}".format(dev.type, args.thr, seeds, len(folds)))

    # 预载每个种子对应的模型（按图取"留出它"的那一折）
    nets = {}
    for seed in seeds:
        for k in range(len(folds)):
            p = os.path.join(args.model_dir, "model_s{}_f{}.pt".format(seed, k + 1))
            if not os.path.exists(p):
                print("!! 缺模型 {}".format(p)); continue
            net = T.SmallCNN(act="relu", arch="res2_deep").to(dev)   # 2026-09-28 修：现行权重是 res2_deep（4 阶段），原来默认 plain1 会加载失败
            net.load_state_dict(torch.load(p, map_location=dev))
            net.eval()
            nets[(seed, k)] = net

    rels = [os.path.relpath(f, P.SRC) for f in sorted(glob.glob(os.path.join(P.SRC, "*", "*")))]
    rows, allt = [], {0.0: 0.0}
    for rel in rels:
        smp = short_name(rel)
        g = cv2.imread(os.path.join(P.SRC, rel), cv2.IMREAD_GRAYSCALE)
        m4 = P.read_archive_mask(rel)
        _, _, dtin = P.ring_of(m4)
        st = P.stages(g, m4, prefilter=P.PREFILTERS["median"], d_pct=96)
        _, kept = P.candidates(st["peaks"], m4)
        # ← 与训练集同口径 ✓：候选 = 基础候选 ∪ 弱反差分支（candidates_union）✓
        #    （原来只取基础 1326 ✗ → 3 个弱反差外圈点拿不到 CNN 分数 ✗ = 使用者说的"3 个点又缺了" ✓）
        _u = P.candidates_union(g, m4)
        _extra = _u[-1] if isinstance(_u, tuple) else _u
        kept = list(kept) + [p for p in _extra
                             if not any(abs(p[0] - q[0]) + abs(p[1] - q[1]) <= 10 for q in kept)]
        mu, sd = float(g.mean()), float(g.std()) + 1e-6
        X = np.zeros((len(kept), 1, S, S), np.float32)
        for i, (x, y, r) in enumerate(kept):
            X[i, 0] = (crop_at(g, int(x), int(y)).astype(np.float32) - mu) / sd
        k = fold_of[smp]
        t0 = time.time()
        probs = []
        with torch.no_grad():
            xt = torch.tensor(X).to(dev)
            for seed in seeds:
                net = nets.get((seed, k))
                if net is None:
                    continue
                probs.append(net(xt).cpu().numpy())
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        p = np.mean(probs, axis=0)
        hits = P.detect_rings(g, m4, prefilter=P.PREFILTERS["median"], d_pct=96)
        # 规则链判定 vs CNN 判定
        cnt = {0: 0, 1: 0}
        for i, (x, y, r) in enumerate(kept):
            dd = float(dtin[int(y), int(x)])
            if p[i] >= args.thr:
                cnt[0 if dd <= P.RING_K else 1] += 1
        ri = sum(1 for h in hits if h["ring"] == "inner")
        ro = len(hits) - ri
        rows.append(dict(image=smp, n_cand=len(kept), cnn_inner=cnt[0], cnn_outer=cnt[1],
                         cnn_total=cnt[0] + cnt[1], rule_inner=ri, rule_outer=ro,
                         rule_total=len(hits), sec=dt))
        print("{:12} 候选{:>4} | CNN: 内{:>3}/外{:>3} 共{:>3} | 规则: 内{:>3}/外{:>3} 共{:>3} | {:>7.2f}s".format(
            smp, len(kept), cnt[0], cnt[1], cnt[0] + cnt[1], ri, ro, len(hits), dt))

    tot = sum(r["sec"] for r in rows)
    ncand = sum(r["n_cand"] for r in rows)
    print("\n合计：{} 张图 / {} 个候选点，全部由 CNN 判分".format(len(rows), ncand))
    print("总耗时 {:.2f}s（设备 {}）→ 平均 {:.1f} ms/点，单张图 {:.0f} ms".format(
        tot, dev.type, 1000 * tot / max(1, ncand), 1000 * tot / max(1, len(rows))))
    if dev.type == "cuda":
        print("显存峰值 {:.1f} MB".format(torch.cuda.max_memory_allocated() / 1048576))
    out = os.path.join(HERE, "out", "cnn", "arbiter_{}_{}.json".format(args.tag, dev.type))
    json.dump(dict(tag=args.tag, device=dev.type, thr=args.thr, rows=rows),
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→ {}".format(out))


if __name__ == "__main__":
    sys.exit(main())
