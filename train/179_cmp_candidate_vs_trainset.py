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
"""179_cmp_candidate_vs_trainset.py — 候选集坐标 × 既有训练集坐标 逐点对账（只读 ✓）

使用者（2026-10-05）："你对比之前现有训练集的坐标名单"
口径：
  A = 当前候选集   = out/cnn/dataset.json（1329 条 = S1–S6 的 1326 + 弱反差分支 3 ✓）
  B = 既有训练集   = SharedData/chip_label/（使用者旧标注库，文件名含 <四位>_<x>_<y>.png ✓）
  C = 最近一轮分类 = SharedData/chip_wire_ng/weak_new/<样品>/{0,1,2}/（= 对 A 的全量分类 ✓）
匹配：按【样品 + 坐标 ≤5px】✓（逐图分开 ✓，不同板的同坐标不是同一个物理点 ✗）
输出：逐样品 A/B/C 计数 ✓ + 「只在 A（候选集多出来的）」逐条坐标 ✓ + 「只在 B」逐条 ✓ + JSON 留档 ✓
"""
import glob
import json
import os
import re
import sys

H = (_PROJ or _ROOT)
LAB = (_LAB + r"")
WEAK = (_PROJ + r"/weak_new")
RE = re.compile(r"(\d{4})_(\d{1,4})_(\d{1,4})\.png$")
TOL = 5.0


def board_of_short(sh):
    m = {"1649": "1_NG-1649", "1658": "1_NG-1658", "1705": "1_NG-1705", "1507": "1_OK-1507", "1528": "1_OK-1528",
         "1535": "1_OK-1535", "3414": "2_NG-3414", "3421": "2_NG-3421", "2OK2": "2_OK-2", "2OK3": "2_OK-3",
         "OK2": "2_OK-2", "OK3": "2_OK-3"}
    if sh in m:
        return m[sh]
    t = sh[-4:]
    return m.get(t, sh)


def scan(root):
    """扫 <=> {样品: [(x,y,类/来源)]}，从文件名解析坐标 ✓"""
    out = {}
    if not os.path.isdir(root):
        return out
    for f in glob.glob(os.path.join(root, "**", "*.png"), recursive=True):
        mt = RE.search(os.path.basename(f))
        if not mt:
            continue
        sh, x, y = mt.group(1), int(mt.group(2)), int(mt.group(3))
        rel = os.path.relpath(f, root)
        parts = rel.split(os.sep)
        cls = parts[-2] if len(parts) >= 2 and parts[-2] in ("0", "1", "2", "待判定") else parts[0] if len(parts) > 1 else "?"
        key = board_of_short(sh)
        out.setdefault(key, []).append((x, y, cls, rel))
    return out


def match(a, b, tol=TOL):
    """a 里每个点在 b 里找 ≤tol 的最近点 ✓ 返回 (命中数, 未命中列表)"""
    hit, miss = 0, []
    for (x, y, *_r) in a:
        d = [(abs(x - bx) + abs(y - by), bx, by) for (bx, by, *_q) in b]
        if d and min(d)[0] <= 2 * tol:
            hit += 1
        else:
            miss.append((x, y))
    return hit, miss


def main():
    ds = json.load(open(os.path.join(H, "out/cnn/dataset.json"), encoding="utf-8"))
    A = {}
    for c in ds:
        A.setdefault(c["sample"], []).append((int(c["x"]), int(c["y"]), int(c["label"])))
    B = scan(LAB)
    C = scan(WEAK)
    print("{:<12} {:>8} {:>10} {:>10} {:>8}".format("样品", "A候选集", "B旧训练集", "C弱新分类", "A⊄B"))
    rep = {}
    for s in sorted(A):
        a, b, c = A.get(s, []), B.get(s, []), C.get(s, [])
        h, miss = match(a, b)
        rep[s] = {"A": len(a), "B": len(b), "C": len(c), "A_in_B": h, "A_only": len(miss),
                  "A_only_pts": [[int(x), int(y)] for (x, y) in miss]}
        if not b:
            rep[s]["note"] = "B 里没有该样品的文件（命名/目录口径可能不同 ✗）"
        print("{:<12} {:>8} {:>10} {:>10} {:>8}".format(s, len(a), len(b), len(c), len(miss)))
    print("\n=== 只在候选集 A、不在旧训练集 B 的点（这就是『多出来的候选』✓）===")
    tot = 0
    for s in sorted(rep):
        pts = rep[s]["A_only_pts"]
        if not pts:
            continue
        tot += len(pts)
        print("  {:<11} {} 个：{}".format(s, len(pts), " ".join("({},{})".format(x, y) for x, y in pts[:14])
                                          + (" …" if len(pts) > 14 else "")))
    print("  合计只在 A 的点：{} 个".format(tot))
    print("\n=== B 里存在但不在 A 的点（旧标注已不在候选集 ✓ 不能当 ground truth 用 ✓）===")
    for s in sorted(B):
        if s not in A:
            continue
        h2, miss2 = match(B[s], A.get(s, []))
        if miss2:
            print("  {:<11} {} 个".format(s, len(miss2)))
    print("\n=== B 的类别分布（按目录/来源）===")
    from collections import Counter
    print(" ", Counter(r for s in B for (_x, _y, _c, r) in B[s] if not r.split(os.sep)[0].isdigit())[:0] or "")
    print(" ", {s: Counter(c for (_x, _y, c, _r) in v) for s, v in sorted(B.items())})
    json.dump(rep, open(os.path.join(H, "out/cnn/cand_vs_trainset.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n→ 留档 out/cnn/cand_vs_trainset.json ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
