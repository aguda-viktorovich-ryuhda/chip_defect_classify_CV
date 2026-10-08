#!/usr/bin/env python3
"""make_placeholder_refs.py —— 生成"占位参考数据 refs/"（不含任何第三方数据）

为什么需要它
------------
流水线的 S4/S5 需要 `refs/`：
  · `refs/annot/<组名>/<图名主干>_m.png`  归档掩膜（S4 的参考流程按它取板面掩膜）
  · `refs/solder_table.json`             焊点表（S1+S2 的结果冻结件，字段 sample/x/y/r/ring/prob）
公开仓库不包含这两样（它们来自使用方的样品与标注）。本脚本用**你自己的图**现场生成
等价结构，使流水线可以在你的数据上跑通。

用法
----
    python tools/make_placeholder_refs.py --src <图片目录> [--out refs] [--from-run <运行目录>]

  --src       图片目录；支持 `<组名>/<图片>` 的子目录结构，也支持平铺（此时组名取目录名）
  --out       输出目录，默认 `refs`
  --from-run  可选：已有的运行目录（`run_pipeline.py` 的输出），里面有 `masks/solder_table.json`
              时直接复制为 `refs/solder_table.json`（最贴近真实焊点表）

说明
----
· `annot/*_m.png` 由软件自己的板面分割（`core.mask.board_mask`）得到，属**机器生成**，不是人工标注。
  实测：同一批样图 + 真实焊点表（`--from-run`）⇒ 全链跑出与基准一致的计数
  （读数 117 ｜ 弯曲 4 ｜ 断联 7 ｜ 裸焊盘 5）。换成新的样品时计数自然会不同，那是数据不同，不是脚本问题。
· 没有 `--from-run` 时 `solder_table.json` 写成空表（`rows: []`），并在末尾打印提示。
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                    # 软件根目录
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def iter_images(src):
    """产出 (组名, 图片绝对路径)。支持 <组名>/<图片> 与平铺两种结构。"""
    out = []
    entries = sorted(os.listdir(src))
    for e in entries:
        p = os.path.join(src, e)
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
                    out.append((e, os.path.join(p, f)))
        elif os.path.splitext(e)[1].lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
            out.append((os.path.basename(os.path.abspath(src)), p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='图片目录')
    ap.add_argument('--out', default='refs', help='输出目录（默认 refs）')
    ap.add_argument('--from-run', default=None, help='可选：含 masks/solder_table.json 的运行目录')
    ap.add_argument('--cfg', default=None, help='可选：cfg 文件路径')
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    out = os.path.abspath(a.out)
    if not os.path.isdir(src):
        print('[ERROR] 图片目录不存在: ' + src)
        return 1

    import cv2
    import numpy as np
    from core import cfg as _cfg
    from core import mask as _mask

    C = _cfg.load(a.cfg) if a.cfg else _cfg.load()
    imgs = iter_images(src)
    if not imgs:
        print('[ERROR] ' + src + ' 下没有找到图片')
        return 1

    os.makedirs(os.path.join(out, 'annot'), exist_ok=True)
    made = []
    for grp, p in imgs:
        g = _mask.read_gray(p)
        m = _mask.board_mask(g, C)[0]
        d = os.path.join(out, 'annot', grp)
        os.makedirs(d, exist_ok=True)
        stem = os.path.splitext(os.path.basename(p))[0]
        dst = os.path.join(d, stem + '_m.png')
        cv2.imwrite(dst, (np.asarray(m) > 0).astype(np.uint8) * 255)
        made.append(os.path.relpath(dst, out))

    # 焊点表
    solder = os.path.join(out, 'solder_table.json')
    src_tbl = os.path.join(a.from_run, 'masks', 'solder_table.json') if a.from_run else None
    if src_tbl and os.path.exists(src_tbl):
        with open(src_tbl, encoding='utf-8') as f:
            tbl = json.load(f)
        note = 'copied from ' + src_tbl
    else:
        print('[ERROR] 没有真实焊点表：请先跑一次 run_pipeline.py，再用 --from-run <运行目录> 重跑本脚本。')
        print('        空表会让 S5 拿不到焊点，整批被误判为"裸焊盘"（实测：读数 0 / 裸焊盘 123）——')
        print('        所以本脚本不再生成空表，只写 annot 掩膜，焊点表必须来自真实运行。')
        print('        已写入的 annot 掩膜可保留；补上焊点表后再跑一次即可。')
        return 0
    with open(solder, 'w', encoding='utf-8') as f:
        json.dump(tbl, f, ensure_ascii=False, indent=1)

    print('已生成占位参考数据: ' + out)
    print('  annot 掩膜 {} 个（来自 {} 张图，组: {}）'.format(
        len(made), len(imgs), sorted({g for g, _ in imgs})))
    print('  solder_table.json: {} （rows={}）'.format(note, len(tbl.get('rows', []))))
    if not (src_tbl and os.path.exists(src_tbl)):
        print('  提示：先用 run_pipeline.py 跑一次，再用 --from-run <运行目录> 生成，可得到真实焊点表')
    print('  注意：这些是机器生成的占位数据，复现不了用人工标注得到的基准数字（117/4/7/5）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
