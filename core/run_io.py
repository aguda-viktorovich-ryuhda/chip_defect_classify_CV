# -*- coding: utf-8 -*-
"""run_io.py - 输入/输出的可配置接口（客户可指向自己的图片目录）

优先级（从高到低）：
  1) 命令行参数  --input <目录或单个图片文件>  --out <输出根目录>  --tag <本次运行名>
  2) cfg/io.input_dir（相对软件根或绝对路径均可）
  3) 默认 chip_samples__20260128（软件自带的示例原图）

只读输入（绝不写入、不改名、不移动客户图片）；所有产物写入 outputs/<YYYYMMDD_HHMM>_<tag>/
"""
import glob
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def resolve_input_dir(C, input_dir=None):
    """返回输入目录的绝对路径（可被 CLI 覆盖）"""
    d = input_dir or C.get("io", {}).get("input_dir") or "chip_samples__20260128"
    d = _abs(d)
    if not os.path.isdir(d):
        raise IOError("输入目录不存在（请用 --input 指定）：" + d)
    return d


def list_boards(C, input_dir=None, glob_pat=None):
    """列出待处理图片。

    支持两种布局：
      A) <输入>/<分组>/<文件.jpg>     例如 NG-1/微信图片_*.jpg   → 样品名 = <分组>-<文件名尾4位>
      B) 扁平：<输入>/<文件.jpg>        → 样品名 = 文件名去扩展名
    返回 [(样品名, 绝对路径), ...]（按样品名排序，便于逐板对照）
    """
    d = resolve_input_dir(C, input_dir)
    pat = glob_pat or C.get("io", {}).get("image_glob", "*.jpg")
    if os.path.isfile(d):
        return [(os.path.splitext(os.path.basename(d))[0], d)]
    subs = sorted([s for s in glob.glob(os.path.join(d, "*")) if os.path.isdir(s)])
    out = []
    for s in subs:
        for f in sorted(glob.glob(os.path.join(s, pat))):
            out.append((os.path.basename(s) + "-" + os.path.splitext(os.path.basename(f))[0][-4:], f))
    if not out:  # 扁平布局
        for f in sorted(glob.glob(os.path.join(d, pat))):
            out.append((os.path.splitext(os.path.basename(f))[0], f))
    if not out:
        raise IOError("输入目录里没找到匹配的图片（glob=" + pat + "）：" + d)
    return out


def make_run_dir(C, out_root=None, tag="run"):
    """创建本次运行目录：outputs/<YYYYMMDD_HHMM>_<tag>/{figures,masks}，返回绝对路径"""
    root = _abs(out_root or C.get("io", {}).get("out_root") or "outputs")
    run = os.path.join(root, time.strftime("%Y%m%d_%H%M") + "_" + tag)
    for sub in ("figures", "masks"):
        os.makedirs(os.path.join(run, sub), exist_ok=True)
    return run


def read_gray(path):
    from core import mask as _mask
    return _mask.read_gray(path)
