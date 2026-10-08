# -*- coding: utf-8 -*-
"""cfg.py — 参数读取（唯一入口 ✓）

用法：from core import cfg; C = cfg.load()
优先读 cfg/default.yaml（若有 pyyaml ✓），否则读自动生成的 cfg/default.json（兜底 ✓）。
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE = {}


def load(path=None):
    if path in _CACHE:
        return _CACHE[path]
    y = path or os.path.join(ROOT, "cfg", "default.yaml")
    j = os.path.splitext(y)[0] + ".json"
    d = None
    if os.path.exists(y):
        try:
            import yaml
            d = yaml.safe_load(open(y, encoding="utf-8"))
        except Exception:
            d = None
    if d is None and os.path.exists(j):
        d = json.load(open(j, encoding="utf-8"))
    if d is None:
        raise RuntimeError("cfg 读取失败：既没有可解析的 yaml，也没有 json 兜底")
    _CACHE[path] = d
    return d


def apply_to_pipeline(C):
    """把 s1 参数写进内置的 pipeline 模块（现场生效 ✓ 不改文件 ✗）"""
    import sys as _sys
    # 内置库当年写在 lib/ 下，用的是 `import seg`（同目录导入 ✓）
    # → 先把 core/vendor 挂到 sys.path ✓ 这样它的 import seg 能解析到内置的 seg.py ✓
    _v = os.path.join(ROOT, "core", "vendor")
    if _v not in _sys.path:
        _sys.path.insert(0, _v)
    import core.vendor.pipeline as P
    s1 = C["s1_candidates"]
    for k in ("D_PCT", "KCLOSE", "MASK_ERODE", "RE", "DT_MIN", "RMIN", "RMAX", "SEP", "RING_K",
              "WEAK_MF_T", "WEAK_DEDUP", "WEAK_WIN_BIG", "WEAK_WIN_SMALL"):
        if k in s1:
            setattr(P, k, s1[k])
    return P
