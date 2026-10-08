# -*- coding: utf-8 -*-
"""s_inputs.py - 通用输入适配：输入可以是【一张图】或【一个目录（任意张图，可递归）】。

设计目标（使用者 2026-10-06 要求）：
  1. 不再绑定"必须 10 张"；一张、三张、几十张都要能跑。
  2. 为每张图分配稳定编号与显示名，落 输入清单.json，下游一律读清单，
     不再靠"切字符串猜板名"。
命名规则：
  任务名 = 用户给定（界面按时间生成，如 任务_20261006_2055）或目录/文件名
  显示名 = 任务名 + 文件名主干
用法：
  items = collect(src)                       # src 可以是单张图或目录
  write_manifest(run, items, task)           # 写 <run>/inputs/输入清单.json
自测：
  python core/s_inputs.py <路径>
"""
import json
import os
import sys

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def task_name(src=None, explicit=None):
    """任务名：显式给定优先；否则取源（文件主干/目录名）。界面按时间生成后传入。"""
    if explicit:
        return explicit
    if not src:
        import time
        return "任务_" + time.strftime("%Y%m%d_%H%M")
    p = os.path.abspath(src)
    if os.path.isfile(p):
        return os.path.splitext(os.path.basename(p))[0]
    return os.path.basename(p.rstrip("/")) or "任务"


def _is_img(name):
    return name.lower().endswith(IMG_EXT)


def collect(src, task=None, recursive=True):
    """收集图片，返回 [{id, name, file, path}]（按路径排序，编号稳定）。"""
    src = os.path.abspath(src)
    task = task_name(src, task)
    files = []
    if os.path.isfile(src):
        if not _is_img(src):
            raise ValueError("不是支持的图片格式：{}".format(src))
        files = [src]
    elif os.path.isdir(src):
        if recursive:
            for root, _, fs in os.walk(src):
                for f in fs:
                    if _is_img(f):
                        files.append(os.path.join(root, f))
        else:
            for f in sorted(os.listdir(src)):
                p = os.path.join(src, f)
                if os.path.isfile(p) and _is_img(f):
                    files.append(p)
    else:
        raise FileNotFoundError("输入不存在：{}".format(src))
    files = sorted(files)
    items = []
    for i, p in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(p))[0]
        nm = "{}_{:03d}".format(task, i) if stem == task else "{}_{:03d}_{}".format(task, i, stem)
        items.append({"id": i, "name": nm, "stem": stem, "file": os.path.basename(p), "path": p})
    return items


def write_manifest(run, items, task, src=None):
    """把输入清单落到 <run>/inputs/输入清单.json，下游读它。"""
    d = os.path.join(run, "inputs")
    os.makedirs(d, exist_ok=True)
    man = {"task": task, "count": len(items), "src": os.path.abspath(src) if src else None,
           "items": items}
    p = os.path.join(d, "输入清单.json")
    json.dump(man, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return p


def read_manifest(run):
    return json.load(open(os.path.join(run, "inputs", "输入清单.json"), encoding="utf-8"))


def _self_test():
    targets = sys.argv[1:] or ["chip_samples__20260128"]
    for t in targets:
        try:
            items = collect(t, task="任务_20261006_2100" if os.path.isdir(t) else None)
        except Exception as e:
            print("  {} -> 失败：{}".format(t, e))
            continue
        print("输入 {:<28s} 张数 {} ｜ 任务名 {}".format(
            t, len(items), items[0]["name"].rsplit("_", 2)[0] if items else "-"))
        for it in items[:3]:
            print("    #{:<3d} {:<40s} {}".format(it["id"], it["name"], it["file"]))
        if len(items) > 3:
            print("    ... 共 {} 张".format(len(items)))
        if items:
            tmp = os.path.join(os.path.abspath("."), "outputs", "_inputs_selftest")
            p = write_manifest(tmp, items, "自测任务", t)
            print("    清单 -> {} ({})".format(p, "✓" if os.path.exists(p) else "✗"))


if __name__ == "__main__":
    _self_test()
