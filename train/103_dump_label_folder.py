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
"""
103_dump_label_folder.py  —  把候选裁图倒成一个平铺文件夹，供使用者用文件管理器缩略图分 1/0

使用者方案（比在网页里点选省事）：
  · 一个平铺文件夹，里面全是候选小图 → 用缩略图浏览
  · 他自己建/用 1（焊点）和 0（非焊点）两个子文件夹，把图拖进去
  · 文件名里带原图坐标 → 我从文件夹归属得到标签，从文件名回读坐标，
    需要训练时再按坐标重裁**原始 48×48**（所以给放大的预览图不丢数据）

输出目录（NTFS 分区，Windows 资源管理器可见）：
  <路径见 本地配置.env>/<图简称>/
      *.png        192×192（48×48 最近邻放大 ×4，便于缩略图看清）
      1/  0/       预建空文件夹
      README.txt   操作说明
      list.json    序号↔坐标 对照（备份用）
"""
import json
import os
import sys

import cv2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "lib"))
import pipeline as P   # noqa: E402

S = 48
UP = 4                     # 预览放大倍数（最近邻）
DEST_ROOT = (_LAB + r"")


def crop_patch(g, px, py, s=S):
    r = s // 2
    y0, x0 = py - r, px - r
    ph, pw = g.shape
    top, left = max(0, -y0), max(0, -x0)
    bot, right = max(0, y0 + s - ph), max(0, x0 + s - pw)
    p = g[max(0, y0):min(ph, y0 + s), max(0, x0):min(pw, x0 + s)]
    if top or bot or left or right:
        p = cv2.copyMakeBorder(p, top, bot, left, right, cv2.BORDER_REPLICATE)
    return p


def short_name(rel):
    grp = os.path.dirname(rel).replace("/", "")
    base = os.path.splitext(os.path.basename(rel))[0].split("_")[-1][-4:]
    return "{}-{}".format(grp, base)


def dump(rel, tag):
    # 保护：目标文件夹里已经有分好的图就跳过，绝不覆盖使用者已完成的标注
    import glob as _g
    guard = os.path.join(DEST_ROOT, short_name(rel))
    if _g.glob(os.path.join(guard, "1", "*.png")) or _g.glob(os.path.join(guard, "0", "*.png")):
        print("跳过 {}（已有人工标注，不覆盖）".format(guard))
        return None, 0
    g = cv2.imread(os.path.join(P.SRC, rel), cv2.IMREAD_GRAYSCALE)
    m4 = P.read_archive_mask(rel)
    st = P.stages(g, m4, prefilter=P.PREFILTERS["median"], d_pct=96)
    _, kept = P.candidates(st["peaks"], m4)
    name = short_name(rel)
    dest = os.path.join(DEST_ROOT, name)
    os.makedirs(os.path.join(dest, "1"), exist_ok=True)
    os.makedirs(os.path.join(dest, "0"), exist_ok=True)
    recs = []
    for i, (x, y, r) in enumerate(sorted(kept, key=lambda v: (v[1] // 40, v[0])), start=1):
        px, py = int(x), int(y)
        p = crop_patch(g, px, py)
        big = cv2.resize(p, (S * UP, S * UP), interpolation=cv2.INTER_NEAREST)
        fn = "{}_{:04d}_{:04d}.png".format(tag, px, py)
        cv2.imwrite(os.path.join(dest, fn), big)
        recs.append(dict(idx=i, x=px, y=py, r=float(r), file=fn, image=rel))
    json.dump(recs, open(os.path.join(dest, "list.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    readme = """焊点裁图人工标注 —— 操作说明

每个小图 = 一个候选点，以该点为中心裁出的 {s}×{s} 像素（此处放大 ×{up} 便于看清，仅预览用）。
文件名格式：  {tag}_x坐标_y坐标.png     例：{tag}_0238_0047.png

怎么标：
  1. 用缩略图视图（大图标/超大图标）浏览本文件夹里的 png；
  2. 是【焊点】的 → 拖进 1 文件夹；
     不是【焊点】的（沟槽／飞线交点／暗斑等）→ 拖进 0 文件夹；
  3. 分完告诉我，我按文件夹归属读标签、按文件名回读坐标，
     再用原图重裁原始 {s}×{s} 图去训练。

判据（你比我清楚，这里只是提示）：焊点 = 焊座上的暗色圆斑（四周被亮焊座包住）。
""".format(s=S, up=UP, tag=tag)
    open(os.path.join(dest, "README.txt"), "w", encoding="utf-8").write(readme)
    print("{:12} → {}  ({} 张，预览 {}×{})".format(name, dest, len(recs), S * UP, S * UP))
    return dest, len(recs)


def main():
    if not os.path.isdir(os.path.dirname(DEST_ROOT)):
        print("目标分区不存在：{}".format(os.path.dirname(DEST_ROOT)))
        return 1
    rels = [("1_OK/微信图片_20240711151507.jpg", "1507"),
            ("1_OK/微信图片_20240711151528.jpg", "1528"),
            ("1_OK/微信图片_20240711151535.jpg", "1535"),
            ("1_NG/微信图片_20240711151649.jpg", "1649"),
            ("1_NG/微信图片_20240711151658.jpg", "1658"),
            ("1_NG/微信图片_20240711151705.jpg", "1705"),
            ("2_OK/2.jpg", "2OK2"),
            ("2_OK/3.jpg", "2OK3"),
            ("2_NG/微信图片_20240711153414.jpg", "3414"),
            ("2_NG/微信图片_20240711153421.jpg", "3421")]
    for rel, tag in rels:
        dump(rel, tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
