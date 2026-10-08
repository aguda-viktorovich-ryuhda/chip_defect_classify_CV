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
111_train_cnn.py  —  池化小 CNN：5 折按样品评估（交接文档第七节·第二步）

严格按交接文档第六节的硬要求：
  1 划分按样品（不是按图）：5 折，每折留出 2 个样品，10 个样品各被评估一次，尽量每折含两组
  2 指标分组（1_ / 2_）+ 分类别（焊点 / 非焊点）报
  3 报三个平凡基线对照
  4 逐图（按原图整体统计）归一化，不用全局统计
  5 类别权重（pos_weight = 负/正），否则 2_ 组直接躺平
CPU 预算：单通道灰度 48×48；模型 ≈2.6 万参数；torch.set_num_threads(核数)；
          轻增强（仅亮度/对比度抖动）。

模型结构（按使用者习惯）：输入先过 BatchNorm（防 sigmoid 饱和）→ 卷积块
（Conv-BN-sigmoid）→ 全局平均池化 → 线性层每个后面接 sigmoid → 输出 1 维。
标签 0/1，BCE 损失。
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rss_mb():
    """当前常驻内存 MB（读 /proc/self/statm，纯只读）"""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * 4096 / 1048576
    except Exception:
        return -1.0


def peak_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
sys.path.insert(0, os.path.join(HERE, "lib"))
import pipeline as P   # noqa: E402

S = 48
NTHREAD = min(16, os.cpu_count() or 4)


# ---------------------------------------------------------------- 模型
class Stage(nn.Module):
    """一个阶段 = nconv 个 (Conv-BN) ；可选残差（通道数不同时用 1×1 投影对齐）"""

    def __init__(self, cin, cout, nconv, residual, act):
        super().__init__()
        self.act = act
        self.residual = residual
        self.convs = nn.ModuleList()
        for i in range(nconv):
            self.convs.append(nn.Sequential(
                nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout)))
        self.proj = nn.Conv2d(cin, cout, 1) if (residual and cin != cout) else None

    def forward(self, x):
        h = x
        n = len(self.convs)
        for i, m in enumerate(self.convs):
            h = m(h)
            if not (self.residual and i == n - 1):
                h = self.act(h)                     # 残差时最后一个卷积先不激活，等加了短接再激活
        if self.residual:
            idt = x if self.proj is None else self.proj(x)
            h = self.act(h + idt)
        else:
            h = self.act(h)
        return h


class SmallCNN(nn.Module):
    """arch:
         plain1     每阶段 1 个卷积、无残差（现行基线，25865 参数）
         plain2     每阶段 2 个卷积、无残差（与 res2 同位深、用来隔离"残差"本身的影响）
         res2       每阶段 2 个卷积 + 残差
         res2_deep  4 个阶段、每阶段 2 个卷积 + 残差（看残差能不能撑起更深的网）
    """

    def __init__(self, act="relu", arch="plain1"):
        super().__init__()
        self.in_bn = nn.BatchNorm2d(1)
        self.act = (lambda v: torch.sigmoid(v)) if act == "sigmoid" else (lambda v: torch.relu(v))
        nconv = 1 if arch == "plain1" else 2
        residual = arch.startswith("res")
        nstage = 4 if arch == "res2_deep" else 3
        chans = [16, 32, 64, 64][:nstage]
        self.stages = nn.ModuleList()
        cin = 1
        for cout in chans:
            self.stages.append(Stage(cin, cout, nconv, residual, self.act))
            cin = cout
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.f1 = nn.Linear(cin, 32)
        self.f2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.in_bn(x)
        for st in self.stages:
            x = self.pool(st(x))
        x = self.gap(x).flatten(1)
        x = torch.sigmoid(self.f1(x))
        return torch.sigmoid(self.f2(x)).squeeze(1)


# ---------------------------------------------------------------- 数据
def load_dataset():
    print("  [ld 0] RSS={:.0f}MB 读 dataset.json".format(rss_mb()), flush=True)
    ds = json.load(open(os.path.join(HERE, "out", "cnn", "dataset.json"), encoding="utf-8"))
    print("  [ld 1] RSS={:.0f}MB 条数={}".format(rss_mb(), len(ds)), flush=True)
    xs = np.zeros((len(ds), S, S), np.float32)
    print("  [ld 2] RSS={:.0f}MB 分配 xs".format(rss_mb()), flush=True)
    for i, r in enumerate(ds):
        p = cv2.imread(os.path.join(HERE, r["path"]), cv2.IMREAD_GRAYSCALE)
        xs[i] = p.astype(np.float32)
        if i and i % 300 == 0:
            print("  [ld 3] RSS={:.0f}MB 已读 {} 张 patch".format(rss_mb(), i), flush=True)
    print("  [ld 4] RSS={:.0f}MB patch 读完".format(rss_mb()), flush=True)
    # 逐图（按原图整体）归一化：同一次拍摄的所有候选共用该图的均值/标准差
    print("  [ld 5] RSS={:.0f}MB 开始读原图".format(rss_mb()), flush=True)
    stat = {}
    for r in ds:
        rel = r["image"]
        if rel not in stat:
            g = cv2.imread(os.path.join(P.SRC, rel), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            stat[rel] = (float(g.mean()), float(g.std()) + 1e-6)
        m, s = stat[rel]
        r["norm"] = [m, s]
    print("  [ld 6] RSS={:.0f}MB 原图读完".format(rss_mb()), flush=True)
    # 逐图归一化：统计量必须是【一维】(1326,) 再 [:,None,None] 变 (1326,1,1)；
    # 若写成 [[v] for ...] 得到 (1326,1)，再加两个 None 就成了 4 维 (1326,1,1,1)，
    # 与 (1326,48,48) 广播会左补成 (1,1326,48,48) → 结果 (1326,1326,48,48) ≈ 16TB ✗
    _m = np.array([r["norm"][0] for r in ds], np.float32)
    _s = np.array([r["norm"][1] for r in ds], np.float32)
    assert _m.shape == (len(ds),) and _s.shape == (len(ds),), "统计量必须是一维"
    xs = (xs - _m[:, None, None]) / _s[:, None, None]
    assert xs.shape == (len(ds), S, S), "归一化后形状错误：{}".format(xs.shape)
    print("  [ld 7] RSS={:.0f}MB 归一化完成 shape={}".format(rss_mb(), xs.shape), flush=True)
    # 标签合并（使用者 2026-09-28 裁定：1 正常焊点 与 2 断联焊点 合并为【焊点】，
    # CNN 只判"焊点 / 非焊点"，是否断联交给其它判据）⇒ label>0 一律记 1。
    ys = np.array([1.0 if int(r["label"]) > 0 else 0.0 for r in ds], np.float32)
    return ds, xs, ys, stat


def jitter(batch):
    """轻增强：仅亮度/对比度抖动（压制靠整体亮度分组的捷径）
       注意抖动张量必须建在 batch 所在的设备上（GPU 上跑时否则会 device mismatch）"""
    d = batch.device
    m = torch.empty(batch.size(0), 1, 1, 1, device=d).uniform_(0.9, 1.1)
    a = torch.empty(batch.size(0), 1, 1, 1, device=d).uniform_(-0.25, 0.25)
    return batch * m + a


def folds_by_sample(samples, groups, k=5):
    """按样品分折：5 折 × 2 个样品 = 全部样品各被评估一次，尽量每折含两组。

    ⚠ 必须保证【每个样品都进且只进一折】。早先写法在两两配对后 out[:5] 截断，
    10 个样品（1_ 组 6 + 2_ 组 4）会配出 6 折 → 最后那个样品（OK-1-样例板F）
    永远落在训练集、从未被验证过，指标对它偏乐观。
    """
    g1 = [s for s in samples if groups[s] == "1_"]
    g2 = [s for s in samples if groups[s] == "2_"]
    out = []
    # 先用两组的样品配"含两组"的折，留最后一折给剩下的同组样品
    while g1 and g2 and len(out) < k - 1:
        out.append([g1.pop(0), g2.pop(0)])
    rest = g1 + g2
    while rest and len(out) < k:
        f = rest[:2]
        del rest[:2]
        out.append(f)
    assert sorted(sum(out, [])) == sorted(samples), "有样品没被分到任何折！"
    return out


# ---------------------------------------------------------------- 指标
def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    out = {}
    for cls in (0, 1):
        tp = int(((yhat == cls) & (y == cls)).sum())
        fp = int(((yhat == cls) & (y != cls)).sum())
        fn = int(((yhat != cls) & (y == cls)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[cls] = dict(n=int((y == cls).sum()), tp=tp, fp=fp, fn=fn,
                        precision=prec, recall=rec, f1=f1)
    out["acc"] = float((yhat == y).mean()) if len(y) else 0.0
    out["macro_f1"] = (out[0]["f1"] + out[1]["f1"]) / 2
    return out


def trivial_baselines(y):
    res = {}
    res["全猜非焊点(0)"] = metrics(y, np.zeros_like(y, dtype=float))
    res["全猜焊点(1)"] = metrics(y, np.ones_like(y, dtype=float))
    rng = np.random.default_rng(0)
    res["按先验随机"] = metrics(y, rng.random(len(y)))
    return res


def rule_baseline(ds):
    """现有手写规则的对照：mf > 分圈阈值（外 102 / 内 127）
       （原图/掩码/距离变换按图缓存，别对 1326 条各读一次）"""
    cache = {}
    mfs = []
    for r in ds:
        rel = r["image"]
        if rel not in cache:
            g = cv2.imread(os.path.join(P.SRC, rel), cv2.IMREAD_GRAYSCALE)
            m = P.read_archive_mask(rel)
            _, _, dtin = P.ring_of(m)
            cache[rel] = (g, dtin)
        g, dtin = cache[rel]
        d = float(dtin[r["y"], r["x"]])
        thr = P.RING_TI if d > P.RING_K else P.RING_TO
        mm = P.with_mf(g, [(r["x"], r["y"], r["r"] or 4.0)])[0]
        mfs.append((mm["mf"], thr, d))
    p = np.array([1.0 if m > t else 0.0 for (m, t, _) in mfs])
    return p, mfs


# ---------------------------------------------------------------- 主流程
def start_rss_logger(path=None):
    """后台线程每 0.5s 记一次 RSS 到文件，用于定位内存增长（只读 /proc）"""
    import threading
    path = path or os.path.join(HERE, "out", "cnn", "rss.log")
    open(path, "w").close()

    def loop():
        t0 = time.time()
        while True:
            try:
                with open("/proc/self/statm") as f:
                    r = int(f.read().split()[1]) * 4096 / 1048576
                with open(path, "a") as f:
                    f.write("{:.1f} {:.0f}\n".format(time.time() - t0, r))
            except Exception:
                pass
            time.sleep(0.5)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-folds", type=int, default=0, help="诊断用：只跑前 N 折（0=全部）")
    ap.add_argument("--skip-rules", action="store_true", help="诊断用：跳过规则基线")
    ap.add_argument("--act", default="relu", choices=["sigmoid", "relu"])
    ap.add_argument("--arch", default="plain1", choices=["plain1", "plain2", "res2", "res2_deep"])
    ap.add_argument("--posw-cap", type=float, default=0.0, help="pos_weight 上限（0=不封顶）")
    ap.add_argument("--tag", default="", help="结果文件后缀，便于对照实验")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="auto=有卡用卡（后期部署可强制 cpu）")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    torch.set_num_threads(NTHREAD)
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True        # 输入尺寸固定，让 cudnn 选最快核
        torch.cuda.reset_peak_memory_stats()
        print("设备 = CUDA: {}".format(torch.cuda.get_device_name(0)), flush=True)
    else:
        print("设备 = CPU（{} 线程）".format(NTHREAD), flush=True)
    t_start = time.time()
    rsslog = start_rss_logger()
    print("RSS 曲线 → {}".format(rsslog), flush=True)
    ds, xs, ys, stat = load_dataset()
    samples = sorted(set(r["sample"] for r in ds))
    groups = {r["sample"]: r["group"] for r in ds}
    print("数据 {} 张｜样品 {}｜每图统计缓存 {}".format(len(ds), len(samples), len(stat)))
    folds = folds_by_sample(samples, groups)
    print("按样品分折：")
    for i, f in enumerate(folds, 1):
        n = sum(1 for r in ds if r["sample"] in f)
        pos = sum(1 for r in ds if r["sample"] in f and int(r["label"]) > 0)
        print("  折{} 留出 {}  ({} 张, 正例 {})".format(i, f, n, pos))

    idx_of = {}
    for i, r in enumerate(ds):
        idx_of.setdefault(r["sample"], []).append(i)

    allp = {s: [] for s in args.seeds}
    for seed in args.seeds:
        pred = np.full(len(ds), np.nan)
        for k, held in enumerate(folds[:args.max_folds] if args.max_folds else folds):
            tr = [i for i, r in enumerate(ds) if r["sample"] not in held]
            va = [i for i, r in enumerate(ds) if r["sample"] in held]
            torch.manual_seed(seed * 100 + k)
            np.random.seed(seed * 100 + k)
            net = SmallCNN(act=args.act, arch=args.arch).to(dev)
            npos = float(ys[tr].sum())
            nneg = float(len(tr) - npos)
            _pw = nneg / max(1.0, npos)
            if args.posw_cap > 0:
                _pw = min(_pw, args.posw_cap)
            pos_w = torch.tensor([max(1.0, _pw)]).to(dev)
            lossf = nn.BCELoss(reduction="none")
            opt = torch.optim.Adam(net.parameters(), lr=args.lr)
            Xtr = torch.tensor(xs[tr])[:, None].to(dev)
            ytr = torch.tensor(ys[tr]).to(dev)
            bs = args.batch
            print("      [seed{} 折{}] 训练前 RSS={:.0f}MB 峰值={:.0f}MB 训练样本={} 正例={:.0f}".format(
                seed, k + 1, rss_mb(), peak_mb(), len(tr), npos), flush=True)
            net.train()
            _t0 = time.time()
            for ep in range(args.epochs):
                perm = torch.randperm(len(tr), device=dev)
                for b in range(0, len(tr), bs):
                    sl = perm[b:b + bs]
                    xb = jitter(Xtr[sl])
                    yb = ytr[sl]
                    opt.zero_grad()
                    out = net(xb)
                    w = torch.where(yb > 0.5, pos_w, torch.ones_like(pos_w))
                    loss = (lossf(out, yb) * w).mean()
                    loss.backward()
                    opt.step()
                if ep % 5 == 4:
                    print("        ep{:>3} RSS={:.0f}MB 峰值={:.0f}MB".format(
                        ep + 1, rss_mb(), peak_mb()), flush=True)
            net.eval()
            with torch.no_grad():
                pv = net(torch.tensor(xs[va])[:, None].to(dev)).cpu().numpy()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            _dt = time.time() - _t0
            for i, v in zip(va, pv):
                pred[i] = v
            m = metrics(ys[va], pv)
            print("    折{} held={} acc={:.3f} macroF1={:.3f} 焊点召回={:.2f} 误报={} 用时{:.1f}s".format(
                k + 1, held, m["acc"], m["macro_f1"], m[1]["recall"], m[1]["fp"], _dt))
            torch.save(net.state_dict(), os.path.join(
                HERE, "out", "cnn", "model_s{}_f{}.pt".format(seed, k + 1)))
        allp[seed] = pred.copy()
        json.dump({"seed": seed, "pred": pred.tolist(), "arch": args.arch, "tag": args.tag,
                   "labels": ys.tolist(),
                   "meta": [{k: r[k] for k in ("sample", "group", "x", "y", "label")} for r in ds]},
                  open(os.path.join(HERE, "out", "cnn",
                                    "pred_{}_{}.json".format(args.tag or "run", seed)), "w"))

    # ---------------- 汇总（跨 seed 平均后逐组报）
    print("\n" + "=" * 74)
    print("结果（{} 个随机种子 × 5 折按样品；下表为各 seed 平均 + 多种子集成）".format(
        len(args.seeds)))
    for gname, sel in (("池化", np.ones(len(ds), bool)),
                       ("1_组", np.array([r["group"] == "1_" for r in ds])),
                       ("2_组", np.array([r["group"] == "2_" for r in ds]))):
        print("\n[{}]".format(gname))
        print("  {:>16} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8}".format(
            "方法", "acc", "宏F1", "焊点召回", "焊点F1", "非焊点F1", "误报数"))
        y = ys[sel]
        for tag, f in (("CNN(单种子均)", None), ("CNN(多种子集成)", "ens"), ("全猜非焊点(0)", "b0"),
                       ("全猜焊点(1)", "b1"), ("按先验随机", "br")):
            if f == "ens":
                ens = np.mean([allp[s2][sel] for s2 in args.seeds], axis=0)
                m = metrics(y, ens)
                acc, mf1, rec, f1p, f1n, fp = (m["acc"], m["macro_f1"], m[1]["recall"],
                                               m[1]["f1"], m[0]["f1"], m[1]["fp"])
            elif f is None:
                ms = []
                for seed in args.seeds:
                    ms.append(metrics(y, allp[seed][sel]))
                acc = np.mean([m["acc"] for m in ms])
                mf1 = np.mean([m["macro_f1"] for m in ms])
                rec = np.mean([m[1]["recall"] for m in ms])
                f1p = np.mean([m[1]["f1"] for m in ms])
                f1n = np.mean([m[0]["f1"] for m in ms])
                fp = np.mean([m[1]["fp"] for m in ms])
            else:
                tb = trivial_baselines(y)
                m = {"b0": tb["全猜非焊点(0)"], "b1": tb["全猜焊点(1)"],
                     "br": tb["按先验随机"]}[f]
                acc, mf1, rec, f1p, f1n, fp = (m["acc"], m["macro_f1"], m[1]["recall"],
                                               m[1]["f1"], m[0]["f1"], m[1]["fp"])
            print("  {:>16} {:>7.3f} {:>7.3f} {:>8.2f} {:>8.3f} {:>8.3f} {:>8.1f}".format(
                tag, acc, mf1, rec, f1p, f1n, fp))
    # 现有规则链对照
    if args.skip_rules:
        print("\n（诊断模式：跳过规则基线）")
        return 0
    pr, _ = rule_baseline(ds)
    print("\n[现有手写规则对照]（mf > 分圈阈值 102/127，未含 S8/S9 后处理）")
    for gname, sel in (("池化", np.ones(len(ds), bool)),
                       ("1_组", np.array([r["group"] == "1_" for r in ds])),
                       ("2_组", np.array([r["group"] == "2_" for r in ds]))):
        m = metrics(ys[sel], pr[sel])
        print("  {:>4} acc={:.3f} 宏F1={:.3f} 焊点召回={:.2f} 误报={}".format(
            gname, m["acc"], m["macro_f1"], m[1]["recall"], m[1]["fp"]))
    json.dump(dict(
        tag=args.tag, act=args.act, posw_cap=args.posw_cap,
        folds=[list(map(str, f)) for f in folds],
        preds={str(s): allp[s].tolist() for s in args.seeds},
        labels=ys.tolist(),
        meta=[{k: r[k] for k in ("sample", "group", "x", "y", "label")} for r in ds],
        args=vars(args)), open(os.path.join(HERE, "out", "cnn", "result.json"), "w"),
        ensure_ascii=False, indent=1)
    print("\n总用时 {:.1f}s（设备 {}）".format(time.time() - t_start, dev.type))
    if dev.type == "cuda":
        print("显存峰值 {:.1f} MB".format(torch.cuda.max_memory_allocated() / 1048576))
    # ← 训完立刻把权重【带 tag 归档】✓（原来只有不带 tag 的 model_s*_f*.pt ✗ → 下一次训练直接覆盖 ✗，
    #    实测踩过：09-28 与今天的多版权重互相覆盖，旧版再也无法复现）
    try:
        import hashlib
        import shutil
        _tag = args.tag or "untagged"
        _dst = os.path.join((_PROJ + r"/cnn_models"),
                            time.strftime("%Y%m%d_%H%M") + "_" + _tag)
        os.makedirs(_dst, exist_ok=True)
        _md5, _n = {}, 0
        for _s in args.seeds:
            for _k in range(1, 6):
                _src = os.path.join(HERE, "out", "cnn", "model_s{}_f{}.pt".format(_s, _k))
                if not os.path.exists(_src):
                    continue
                _dstf = os.path.join(_dst, "model_{}_s{}_f{}.pt".format(_tag, _s, _k))
                shutil.copy(_src, _dstf)
                _md5[os.path.basename(_dstf)] = hashlib.md5(open(_dstf, "rb").read()).hexdigest()
                _n += 1
        json.dump({"recorded_at": time.strftime("%Y-%m-%d %H:%M"), "kind": "CNN weights version",
                   "tag": _tag, "n_files": _n, "arch": args.arch, "seeds": list(args.seeds),
                   "dataset": os.path.join(HERE, "out", "cnn", "dataset.json"),
                   "note": "由 111_train_cnn.py 训练结束自动归档 ✓（带 tag，避免覆盖 ✗）",
                   "weights_md5": _md5}, open(os.path.join(_dst, "VERSION.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("权重已归档 {} 个 -> {}".format(_n, _dst))
    except Exception as _e:
        print("权重归档失败（不影响训练结果 ✗）：{}".format(_e))
    print("产物：out/cnn/{model_s*_f*.pt, pred_seed*.json, result.json}")


if __name__ == "__main__":
    sys.exit(main())
