import os as _os, sys as _sys
# [2026-10-08 适配] 去掉写死路径：按脚本位置定位，全部可用环境变量覆盖
_ROOT = _os.environ.get("CHIP_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_DATA = _os.environ.get("CHIP_DATA") or _os.path.join(_ROOT, "train", "data")
_LIB  = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "lib")
_SRC  = _os.environ.get("CHIP_SRC")  or ""
_PROJ = _os.environ.get("CHIP_PROJ") or ""
_LAB  = _os.environ.get("CHIP_LAB")  or ""
_ANN  = _os.environ.get("CHIP_ANN")  or r"<路径见 本地配置.env>/某电子公司印刷电路samples__20260128"
for _p in (_ROOT, _LIB):
    if _p not in _sys.path: _sys.path.insert(0, _p)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""174_export_pending.py — 把【当前候选集的全部待判定点】输出成人工分类包

口径（使用者 2026-09-28 明确）：不是只输出新增的少数点，而是**扫描出来的待判定点全量**。
当前候选集 = lib/pipeline.py::candidates_union 的 base(原 S1–S6) ∪ extra(弱反差分支)，
条数 = dataset.json 的 1329（原 1326 + 弱反差新增 3）。

输出（SharedData/chip_wire_ng/weak_new/，使用者指定的数据集交付点）：
  <样品>/待判定/<四位>_<x>_<y>.png   全部候选，48×48 裁片 ×4 最近邻 = 192×192
                                     （与使用者标注库 chip_label/ 完全同规格、同命名）
  <样品>/{0,1,2}/                    空目录，分类时把 png 拖进去（0 非焊点 / 1 正常焊点 / 2 断联焊点）
  list.json                          每条：sample/x/y/r/旧标签/file（旧标签来自使用者各板标注库）
  README.txt                         操作说明

规矩：写盘逐个检查 cv2.imwrite 返回值（失败不抛异常）；结束按盘上实数核对总数。
"""
import json
import os
import shutil
import sys

import cv2

H = (_ROOT)
D = (_PROJ + r"/weak_new_20261005")      # ← 新导出目录 ✓ 绝不覆盖旧的 weak_new/ ✗（使用者要人工重新分类 ✓）
SRC = (_SRC + r"")
SMPS = ["1_NG-1649", "1_NG-1658", "1_NG-1705", "1_OK-1507", "1_OK-1528", "1_OK-1535",
        "2_NG-3414", "2_NG-3421", "2_OK-2", "2_OK-3"]
SHORT = {"2_OK-2": "2OK2", "2_OK-3": "2OK3"}


def short_of(s):
    if s in SHORT:
        return SHORT[s]
    t = s.split("-")[-1]
    return t if len(t) == 4 else s.split("_")[-1]


def main():
    src103 = open(os.path.join(H, "103_dump_label_folder.py"), encoding="utf-8").read()
    i = src103.index("def crop_patch"); j = src103.index("\ndef ", i + 1)
    ns = {"cv2": cv2, "S": 48}
    exec(src103[i:j], ns)
    crop_patch = ns["crop_patch"]

    ds = json.load(open(os.path.join(H, "out/cnn/dataset.json"), encoding="utf-8"))
    print("候选集（dataset.json）共 {} 条".format(len(ds)))

    if os.path.isdir(D):
        bak = D.rstrip("/") + "_bak_" + __import__("time").strftime("%Y%m%d_%H%M")
        shutil.copytree(D, bak)                      # ← 改为【备份】✓ 绝不 rmtree ✗（曾经删掉使用者整轮分类成果 ✓）
        print("已存在 → 先整目录备份到 {} ✓（不删除 ✗）".format(bak))
    rows, failed = [], []
    for s in SMPS:
        g = cv2.imread(os.path.join(SRC, next(c["image"] for c in ds if c["sample"] == s)),
                       cv2.IMREAD_GRAYSCALE)
        pend = os.path.join(D, s, "待判定")
        for k in "012":
            os.makedirs(os.path.join(D, s, k), exist_ok=True)
        os.makedirs(pend, exist_ok=True)
        n = 0
        for c in [c for c in ds if c["sample"] == s]:
            x, y = int(c["x"]), int(c["y"])
            big = cv2.resize(crop_patch(g, x, y), (192, 192), interpolation=cv2.INTER_NEAREST)
            fn = "{}_{:04d}_{:04d}.png".format(short_of(s), x, y)
            dst = os.path.join(pend, fn)
            ok = cv2.imwrite(dst, big)
            if not ok or not os.path.exists(dst) or os.path.getsize(dst) == 0:
                failed.append(dst)
            rows.append(dict(sample=s, x=x, y=y, r=float(c["r"]), old_label=int(c["label"]),
                             file=fn, image=c["image"]))
            n += 1
        print("  {:<11} 待判定 {:>4} 个".format(s, n))
    json.dump(rows, open(os.path.join(D, "list.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    open(os.path.join(D, "README.txt"), "w", encoding="utf-8").write(
        "待判定候选集（全量）—— 人工分类用\n\n"
        "内容：当前候选集全部 {} 个点（dataset.json 的 1329 = 原 S1–S6 的 1326 + 弱反差分支新增 3）。\n"
        "      每个点一张 png，48×48 裁片 ×4 最近邻 = 192×192（与你标注库 chip_label/ 同规格同命名）。\n"
        "      命名 <四位>_<x>_<y>.png，四位=图简称末四位；坐标=该点在图块中的【几何正中】。\n\n"
        "怎么分类：把 png 从 <样品>/待判定/ 拖进同级的 0/、1/、2/\n"
        "          0 = 非焊点   1 = 正常焊点   2 = 断联焊点\n\n"
        "list.json 每条含 sample/x/y/r/old_label/file/image；\n"
        "  old_label = 该点在你【旧标注库】里的类别（0/1/2），供对照——本次分类以你新分为准。\n"
        "  其中 label=2 的 3 个点（1_NG-1658 (810,160)、1_NG-1705 (674,23)、(808,162)）是\n"
        "  2026-09-28 由弱反差分支新召回、并经你确认的断联焊点。\n\n"
        "数据源（只读，绝不写入）：{}\n".format(len(rows), SRC))

    # ---- 按盘上实数核对 ----
    print()
    print("=== 写盘核对（独立于脚本计数）===")
    tot = 0
    for s in SMPS:
        n = len([f for f in os.listdir(os.path.join(D, s, "待判定")) if f.endswith(".png")])
        tot += n
        print("  {:<11} 待判定盘上 {:>4} 个".format(s, n))
    print("  合计盘上 {} 个；list.json {} 条；写失败 {} 个".format(tot, len(rows), len(failed)))
    print("  结论：{}".format("一致 ✓" if tot == len(rows) and not failed else "不一致 ✗"))
    print("  目录 →", D)
    return 0 if (tot == len(rows) and not failed) else 1


if __name__ == "__main__":
    sys.exit(main())
