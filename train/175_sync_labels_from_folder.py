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
"""175_sync_labels_from_folder.py — 用【盘上目录归属】重算 dataset.json 的标签

使用者流程：候选预览 png 放在 <交付点>/<样品>/待判定/，人工拖进同级 0/、1/、2/
          0 非焊点 / 1 正常焊点 / 2 断联焊点
本脚本：
  · 备份 dataset.json（.bak_beforelabel<时间戳>）
  · 按 (样品, x, y) 精确匹配，把盘上归属写回 dataset.json 的 label
  · 写标签快照 out/labels/labels_<日期>_final.json（含 meta + 每条 sample/x/y/label）
  · 全量对账：盘上每板 0/1/2 计数 ↔ json 标签分布；逐条核对无遗漏、无多余
不通过则回滚 dataset.json 并退出非零。
"""
import argparse
import collections
import json
import os
import shutil
import sys
import time

H = (_PROJ or _ROOT)
DEFAULT_D = (_PROJ + r"/weak_new_20261005")     # ← 现行分类来源（使用者 2026-10-05 重新分类 ✓）；旧 weak_new/ 仅供参考 ✗


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_D, help="人工分类交付点（含 <样品>/{0,1,2}/）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    D = a.dir

    dsp = os.path.join(H, "out/cnn/dataset.json")
    ds = json.load(open(dsp, encoding="utf-8"))
    print("dataset.json 现有 {} 条".format(len(ds)))

    # 1. 盘上归属
    disk = {}
    per_board = collections.defaultdict(lambda: collections.Counter())
    for s in sorted(os.listdir(D)):
        sd = os.path.join(D, s)
        if not os.path.isdir(sd):
            continue
        for k in "012":
            kd = os.path.join(sd, k)
            if not os.path.isdir(kd):
                continue
            for f in os.listdir(kd):
                if not f.endswith(".png"):
                    continue
                q = f[:-4].split("_")
                x, y = int(q[-2]), int(q[-1])
                disk[(s, x, y)] = int(k)
                per_board[s][int(k)] += 1
    print("盘上已分类 {} 个点（{} 个样品）".format(len(disk), len(per_board)))
    for s in sorted(per_board):
        c = per_board[s]
        print("  {:<11} 0:{:<5} 1:{:<5} 2:{:<5} 合计 {}".format(s, c[0], c[1], c[2], sum(c.values())))
    left = sum(len([f for f in os.listdir(os.path.join(D, s, "待判定")) if f.endswith(".png")])
               for s in os.listdir(D) if os.path.isdir(os.path.join(D, s, "待判定")))
    print("  待判定剩余 {} 个".format(left))

    # 2. 匹配并统计变化
    miss, changed = [], []
    for c in ds:
        key = (c["sample"], int(c["x"]), int(c["y"]))
        if key not in disk:
            miss.append(key); continue
        new = disk.pop(key)
        if int(c["label"]) != new:
            changed.append((c["sample"], key[1], key[2], int(c["label"]), new))
    extra = sorted(disk.keys())
    print()
    print("=== 匹配结果 ===")
    print("  dataset 里在盘上找不到的：{} 个 {}".format(len(miss), miss[:5]))
    print("  盘上有、dataset 里没有的：{} 个 {}".format(len(extra), extra[:5]))
    print("  标签发生变化：{} 个".format(len(changed)))
    for (s, x, y, o, n) in changed:
        print("    {:<11} ({:>4},{:>4})  {} → {}".format(s, x, y, o, n))
    if miss or extra or left:
        print()
        print("✗ 有未对齐的点（缺 {} / 多 {} / 待判定剩余 {}）→ 不出标签，先修数据".format(
            len(miss), len(extra), left))
        return 2
    if a.dry_run:
        print()
        print("（--dry-run：未写任何文件）")
        return 0

    # 3. 备份 + 写回
    bak = dsp + ".bak_beforelabel{}".format(time.strftime("%m%d_%H%M"))
    shutil.copy2(dsp, bak)
    print()
    print("已备份 →", bak)
    for c in ds:
        c["label"] = disk.get((c["sample"], int(c["x"]), int(c["y"])), c["label"])
    # 上面 disk 已被 pop 空，改用直接查盘
    def cls_of(s, x, y):
        for k in "012":
            if os.path.exists(os.path.join(D, s, k, "{}_{:04d}_{:04d}.png".format(
                    {"2_OK-2": "2OK2", "2_OK-3": "2OK3"}.get(s, s.split("-")[-1]), x, y))):
                return int(k)
        return None
    for c in ds:
        v = cls_of(c["sample"], int(c["x"]), int(c["y"]))
        if v is not None:
            c["label"] = v
    json.dump(ds, open(dsp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 4. 快照 + 全量对账
    snap = {"meta": {"created": time.strftime("%Y-%m-%d %H:%M"),
                     "source": D,
                     "classes": {"0": "非焊点", "1": "正常焊点", "2": "断联焊点"},
                     "count": len(ds)},
            "items": [dict(sample=c["sample"], x=int(c["x"]), y=int(c["y"]),
                           r=float(c["r"]), label=int(c["label"]), file=c["file"]) for c in ds]}
    sp = os.path.join(H, "out/labels/labels_{}_final.json".format(time.strftime("%Y%m%d")))
    json.dump(snap, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    back = json.load(open(dsp, encoding="utf-8"))
    jc = collections.Counter(int(c["label"]) for c in back)
    dc, per_board2 = collections.Counter(), collections.defaultdict(collections.Counter)
    for s in sorted(os.listdir(D)):
        sd = os.path.join(D, s)
        if not os.path.isdir(sd):
            continue
        for k in "012":
            kd = os.path.join(sd, k)
            if not os.path.isdir(kd):
                continue
            n = len([f for f in os.listdir(kd) if f.endswith(".png")])
            dc[int(k)] += n
            per_board2[s][int(k)] = n
    ok = (len(back) == len(ds) and jc == dc)
    print()
    print("=== 全量对账 ===")
    print("  json 条数 {}  标签分布 0:{} 1:{} 2:{}".format(len(back), jc[0], jc[1], jc[2]))
    print("  盘上计数 {}      标签分布 0:{} 1:{} 2:{}".format(
        sum(dc.values()), dc[0], dc[1], dc[2]))
    print("  结论：{}".format("一致 ✓" if ok else "不一致 ✗"))
    print("  标签快照 →", sp)
    if not ok:
        shutil.copy2(bak, dsp)
        print("  已回滚 dataset.json")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
