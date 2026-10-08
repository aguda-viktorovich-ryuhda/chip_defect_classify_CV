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
121_cnn_overlays.py  —  出 CNN 判定叠加图（供使用者目视核对漏检/误报）

口径：S1–S6 出候选 → CNN（res2_deep，5 种子集成，阈值默认 0.6）逐个判分
  圈 = CNN 接受（绿=内圈 / 橙=外圈，线宽 2）
  小灰点 = CNN 拒绝的候选点（线宽 1，便于发现"被误杀的真焊点"）
  右下角给出比例尺（黑白四段 50px）与 r=5px 尺寸参考圆；左上角写图名与接受/拒绝数。
交付：一图一文件（坐标即原图坐标）+ 查看器（内嵌版 + 全分辨率面板版），
      并带聊天/收集器两个回报按钮，供使用者点出漏检/误报。
"""
import glob
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "lib"))
import pipeline as P   # noqa: E402
import viz3 as V       # noqa: E402

import importlib.util as _iu
_spec = _iu.spec_from_file_location("t", os.path.join(HERE, "recon", "111_train_cnn.py"))
T = _iu.module_from_spec(_spec)
sys.modules["t"] = T
_spec.loader.exec_module(T)

PY = "" + (_os.environ.get("CHIP_PY") or _sys.executable) + ""
S, THR = 48, 0.6
OUT = os.path.join(HERE, "out", "cnn", "overlays")


def short_name(rel):
    grp = os.path.dirname(rel).replace("/", "")
    base = os.path.splitext(os.path.basename(rel))[0].split("_")[-1][-4:]
    return "{}-{}".format(grp, base)


def scalebar(vis, x, y, px=50, h=6):
    for i in range(4):
        col = (255, 255, 255) if i % 2 == 0 else (0, 0, 0)
        x0 = x + i * (px // 4)
        cv2.rectangle(vis, (x0, y), (x0 + px // 4, y + h), col, -1)
    cv2.rectangle(vis, (x, y), (x + px, y + h), (255, 255, 255), 1)
    V.text(vis, "{}px".format(px), (x, y + h + 2), 16, (0, 255, 255), bg=(0, 0, 0))


def main():
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else THR
    os.makedirs(OUT, exist_ok=True)
    ds = json.load(open(os.path.join(HERE, "out", "cnn", "dataset.json"), encoding="utf-8"))
    samples = sorted(set(r["sample"] for r in ds))
    groups = {r["sample"]: r["group"] for r in ds}
    folds = T.folds_by_sample(samples, groups)
    fold_of = {s: i for i, f in enumerate(folds) for s in f}
    truth = {(r["image"], r["x"], r["y"]): r["label"] for r in ds}
    nets = {}
    for seed in range(5):
        for k in range(len(folds)):
            p = os.path.join(HERE, "out", "cnn", "model_s{}_f{}.pt".format(seed, k + 1))
            if os.path.exists(p):
                net = T.SmallCNN(act="relu", arch="res2_deep")
                net.load_state_dict(torch.load(p, map_location="cpu")); net.eval()
                nets[(seed, k)] = net

    rels = [os.path.relpath(f, P.SRC) for f in sorted(glob.glob(os.path.join(P.SRC, "*", "*")))]
    pages, meta_json = [], {}
    tot_acc = tot_rej = 0
    for rel in rels:
        smp = short_name(rel)
        g = cv2.imread(os.path.join(P.SRC, rel), cv2.IMREAD_GRAYSCALE)
        m4 = P.read_archive_mask(rel)
        _, _, dtin = P.ring_of(m4)
        st = P.stages(g, m4, prefilter=P.PREFILTERS["median"], d_pct=96)
        raw, kept = P.candidates(st["peaks"], m4)
        # ← 与训练集/114 同口径 ✓：候选 = 基础 ∪ 弱反差分支（candidates_union）✓
        _u = P.candidates_union(g, m4)
        _extra = _u[-1] if isinstance(_u, tuple) else _u
        kept = list(kept) + [p for p in _extra
                             if not any(abs(p[0] - q[0]) + abs(p[1] - q[1]) <= 10 for q in kept)]
        mu, sd = float(g.mean()), float(g.std()) + 1e-6
        k = fold_of[smp]; r = S // 2
        X = np.zeros((len(kept), 1, S, S), np.float32)
        for i, (x, y, rr) in enumerate(kept):
            x, y = int(x), int(y)
            p = g[max(0, y - r):y + r, max(0, x - r):x + r]
            tt, ll = max(0, r - y), max(0, r - x)
            bb, rr2 = max(0, y + r - g.shape[0]), max(0, x + r - g.shape[1])
            if tt or bb or ll or rr2:
                p = cv2.copyMakeBorder(p, tt, bb, ll, rr2, cv2.BORDER_REPLICATE)
            X[i, 0] = (p.astype(np.float32) - mu) / sd
        with torch.no_grad():
            xt = torch.tensor(X)
            prob = np.mean([nets[(s2, k)](xt).numpy() for s2 in range(5) if (s2, k) in nets], axis=0)
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        acc = rej = 0
        for i, (x, y, rr) in enumerate(kept):
            px, py = int(x), int(y)
            ring_in = float(dtin[py, px]) > P.RING_K
            if prob[i] >= thr:
                col = (0, 255, 0) if ring_in else (60, 165, 255)
                cv2.circle(vis, (px, py), 15, col, 2, cv2.LINE_AA)
                acc += 1
            else:
                cv2.circle(vis, (px, py), 2, (170, 170, 170), -1, cv2.LINE_AA)
                rej += 1
        tot_acc += acc; tot_rej += rej
        # 图外留白：顶部信息条 + 中段原图（图内只有圈/灰点）+ 底部图例条，避免文字压住焊点
        H, W = vis.shape[:2]
        HDR, FTR = 50, 62
        canvas = np.full((HDR + H + FTR, W, 3), 26, np.uint8)
        canvas[HDR:HDR + H] = vis
        V.text(canvas, "{}   CNN(res2_deep，阈值 {:.2f})   接受 {} 个（圈） / 拒绝 {} 个（小灰点）".format(
            smp, thr, acc, rej), (10, 6), 22, (0, 255, 255), bg=None)
        V.text(canvas, "原图 {}×{}　绿圈=内圈焊点　橙圈=外圈焊点　小灰点=CNN 拒绝的候选　"
                       "本格读数=原图像素坐标".format(W, H), (10, 32), 15, (200, 200, 200), bg=None)
        scalebar(canvas, 12, HDR + H + 16, px=50, h=8)
        cv2.circle(canvas, (W - 72, HDR + H + 24), 5, (0, 255, 255), 1)
        V.text(canvas, "r=5px（焊点黑心半径参考）", (W - 300, HDR + H + 16), 16, (0, 255, 255), bg=None)
        fn = os.path.join(OUT, "cnn_{}.jpg".format(smp))
        cv2.imwrite(fn, canvas, [cv2.IMWRITE_JPEG_QUALITY, 78])
        pages.append(fn)
        meta_json[fn] = dict(label="{}  CNN接受{} 拒绝{}".format(smp, acc, rej),
                             ow=W, oh=H, ox0=0, oy0=0,
                             gx0=0, gy0=HDR, gw=W, gh=H)
        print("{:12} 候选 {:>4} → CNN 接受 {:>3} 拒绝 {:>3}".format(smp, len(kept), acc, rej))
    print("合计：接受 {}（圈）/ 拒绝 {}（小灰点）".format(tot_acc, tot_rej))
    mf = os.path.join(OUT, "meta.json")
    json.dump(meta_json, open(mf, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    inline = os.path.join(HERE, "out", "pack_cnnovl_inline.html")
    full = os.path.join(HERE, "out", "pack_cnnovl_full.html")
    subprocess.run([PY, os.path.join(HERE, "recon", "67_make_viewer6.py"), inline,
                    "--scale", "0.32", "--quality", "60", "--meta", mf,
                    "--pane", os.path.abspath(full)] + pages, check=True)
    subprocess.run([PY, os.path.join(HERE, "recon", "66_make_viewer5.py"), full,
                    "--max", "4000", "--meta", mf] + pages, check=True)
    for p in (inline, full):
        print("{}  {:.2f} MB".format(p, os.path.getsize(p) / 1048576))


if __name__ == "__main__":
    sys.exit(main())
