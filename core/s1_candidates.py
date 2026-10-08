# -*- coding: utf-8 -*-
"""s1_candidates.py — 第①步：预处理 → 焊点候选（S1–S6 ∪ 弱反差分支 ✓）

现场生成板框掩膜（CV 轮廓法 ✓ 客户不提供 ✗）。输出：候选点列表（sample/x/y/r/prob_is_weak ✓）
"""
import os

from core import cfg as _cfg


def run_board(img_path, C, sample):
    """单板：读图 → 现场生成板框掩膜 → 候选点。返回 (rows, m4, gray)"""
    from core import mask as _mask
    P = _cfg.apply_to_pipeline(C)
    g = _mask.read_gray(img_path)
    m4, _mids = _mask.board_mask(g, C)
    # 预滤波：cfg 里写的是"人读名"（pre_median = 中值 k5 ✓），内置库的键是 median ✓ → 做一次映射 ✓
    _pf = C["s1_candidates"].get("PREFILTER", "median")
    if _pf not in P.PREFILTERS:
        _pf = {"pre_median": "median"}.get(_pf, sorted(P.PREFILTERS)[0])
    st = P.stages(g, m4, prefilter=P.PREFILTERS[_pf],
                  d_pct=C["s1_candidates"]["D_PCT"])
    _, base = P.candidates(st["peaks"], m4)
    rows = [{"sample": sample, "x": int(x), "y": int(y), "r": float(r), "weak": False} for (x, y, r) in base]
    if C["s1_candidates"].get("use_union", True):
        u = P.candidates_union(g, m4)
        extra = u[-1] if isinstance(u, tuple) else u
        for p in extra:
            x, y = int(p[0]), int(p[1])
            if not any(abs(x - q["x"]) + abs(y - q["y"]) <= C["s1_candidates"]["WEAK_DEDUP"] for q in rows):
                rows.append({"sample": sample, "x": x, "y": y, "r": float(p[2] if len(p) > 2 else 5.0), "weak": True})
    return rows, m4, g
