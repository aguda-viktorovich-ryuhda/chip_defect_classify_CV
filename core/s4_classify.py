# -*- coding: utf-8 -*-
"""s4_classify.py - 第④步：掩膜处理与分类（软件封装）

流程：
  1) 掩膜来源 = S3 现场产出（outputs/<run>/masks/{TTA,v13,v18}/<板>.png）
  2) 甲判据（cfg.s4_classify.bare_pad_rule）在**分类之后**执行：对最终名单里【断联】条目按形态核查
     （只认到 1 个焊点 且 骨架长<20px 或 L/W<1.5 ⇒ 裸焊盘 ⇒ 不计入断联飞线，另项计数）
     注：不可在分类前挖除 —— 实测会切断 S4 补缺/延伸的合并路径，把合格飞线误判成断联（样例板C/样例板F 各 1 条）
  3) 其余（拆分/补缺/认焊点/分类/出图）交给内置 run_reference（原 recon/252 的 main，逐行未改）
  4) 全部路径注入；软件内无"存档掩膜"，故把 P.read_archive_mask 改为现场生成（core/mask.py，同一套 CV 轮廓法）
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V = os.path.join(ROOT, 'core', 'vendor')
if _V not in sys.path:
    sys.path.insert(0, _V)


def _install_on_the_fly_mask(S, C, src_dir):
    """把 252 依赖的"存档掩膜"改成现场生成（客户机器上没有存档）"""
    from core import cfg as _cfg
    from core import mask as _mask
    cache = {}

    def _on_the_fly(rel, *a, **kw):
        base = os.path.basename(rel)[:-6] if os.path.basename(rel).endswith('_m.png') else os.path.splitext(os.path.basename(rel))[0]
        key = base[-4:]
        if key not in cache:
            grp = os.path.dirname(rel)
            d = os.path.join(src_dir, grp)
            cand = [f for f in os.listdir(d) if os.path.splitext(f)[0][-4:] == key]
            assert cand, 'find no image for ' + rel
            g = _mask.read_gray(os.path.join(d, cand[0]))
            m4, _ = _mask.board_mask(g, C)
            cache[key] = m4
        return cache[key]
    if not hasattr(S, 'P') or S.P is None:
        S.P = _cfg.apply_to_pipeline(C)
    S.P.read_archive_mask = _on_the_fly


def run(C, run_dir, src_dir, sold_json=None, annot_dir=None):
    import core.vendor.s4_core as S
    from core import cfg as _cfg
    S.UN = run_dir if os.path.isdir(os.path.join(run_dir, 'TTA')) else os.path.join(run_dir, 'masks')
    os.makedirs(os.path.join(S.UN, 'reports'), exist_ok=True)
    S.SRC = src_dir
    S.PAN = run_dir
    S.SOLD = sold_json or os.path.join(ROOT, 'refs', 'solder_table.json')
    S.ROOT = annot_dir or os.path.join(ROOT, 'refs', 'annot')
    S.ALL = {k: k for k in C['s3_unet_mask']['keep_members']}
    S.KEPT = list(C['s3_unet_mask']['keep_members'])

    S.P = _cfg.apply_to_pipeline(C)
    _install_on_the_fly_mask(S, C, src_dir)
    return S.run_reference()
