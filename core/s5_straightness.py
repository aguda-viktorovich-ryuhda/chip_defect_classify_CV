# -*- coding: utf-8 -*-
"""s5_straightness.py - 第⑤步：掩膜再处理 + 直度判定（软件封装）

口径与 recon/258 链一致：拆分 → 补缺（顺序B）→ 剪端 → 骨架有序路径（最长测地线）→ 直度判据
判据：抖动 > 7.8088px 或 弧弦比 > 1.0062 ⇒ 弯曲不合格

调用方式：进程内依次调用 core/vendor/s5_steps/{s253,s255,s256,s257,s259} 的 main
  （原 258 用 subprocess 调这 5 个脚本；客户机上不保证 python 路径，故改为进程内）
成员目录命名：s255/s253 的 load() 会硬拼 "pred_masks_" 前缀 ⇒ 目录名必须是 pred_masks_<后缀>
  因此 ALL 只给后缀：{'TTA':'tta', 'v13':'v13', 'v18':'v18'} ⇒ 目录 pred_masks_tta 等
"""
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V = os.path.join(ROOT, 'core', 'vendor')
if _V not in sys.path:
    sys.path.insert(0, _V)

STEPS = ('s253_gap_fill', 's255_straightness', 's256_panels', 's257_numbered', 's259_stage')


def _inject(C, run_dir, src_dir, boards, solder_json):
    import core.vendor.s5_core as S   # 若存在则用它承载全局
    return S


def run(C, run_dir, src_dir, boards=None, solder_json=None, boards_pairs=None):
    import importlib
    if boards_pairs is None:
        from core import run_io as _rio
        boards_pairs = _rio.list_boards(C, src_dir)
    UN = os.path.join(run_dir, 'masks') if os.path.isdir(os.path.join(run_dir, 'masks')) else run_dir
    os.makedirs(os.path.join(UN, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'state', 'wires'), exist_ok=True)
    common = dict(
        SRC=src_dir, UN=UN, PAN=run_dir, OUT=run_dir,
        SOLD=solder_json or os.path.join(ROOT, 'refs', 'solder_table.json'),
        # BOARD_FILE：s256/s257 需要的是【板号 -> (组名, 文件名主干)】的映射表（不是路径 ✗）
        # BOARD_FILE：两种键都注入（NG-1-样例板A 与 样例板A），因为 256/257 用前者、259 用后者
        BOARD_FILE=dict(
            [( _b, (os.path.basename(os.path.dirname(_pth)), os.path.splitext(os.path.basename(_pth))[0]) ) for _b, _pth in (boards_pairs or [])]
            + [( os.path.basename(os.path.dirname(_pth)) + '/' + os.path.splitext(os.path.basename(_pth))[0][-4:],
                 (os.path.basename(os.path.dirname(_pth)), os.path.splitext(os.path.basename(_pth))[0]) ) for _b, _pth in (boards_pairs or [])]),
        ROOT=os.path.join(ROOT, 'refs', 'annot'),
        # BOARDS：259 要字符串【组名/尾4位】；257 要三元组 ⇒ 统一给字符串，257 那一行改成两种都收
        BOARDS=[os.path.basename(os.path.dirname(_pth)) + '/' + os.path.splitext(os.path.basename(_pth))[0][-4:]
                for _b, _pth in (boards_pairs or [])],
        KEPT=list(C['s3_unet_mask']['keep_members']),
        ALL={'TTA': 'TTA', 'v13': 'v13', 'v18': 'v18'},   # = S4 生产端实际写出的目录名（masks/<成员>/）
        PY=sys.executable,
    )
    # ---- 注入 P（内置 pipeline ✓）并把 read_archive_mask 改成现场生成（同 s4_classify ✓）----
    from core import cfg as _cfg, mask as _mask
    P = _cfg.apply_to_pipeline(C)
    _cache = {}
    def _on_the_fly(rel, *a, **kw):
        base = os.path.basename(rel)[:-6] if os.path.basename(rel).endswith('_m.png') else os.path.splitext(os.path.basename(rel))[0]
        k = base[-4:]
        if k not in _cache:
            d = os.path.join(src_dir, os.path.dirname(rel))
            cand = [f for f in os.listdir(d) if os.path.splitext(f)[0][-4:] == k]
            assert cand, 'no image for ' + rel
            g = _mask.read_gray(os.path.join(d, cand[0])); _cache[k] = _mask.board_mask(g, C)[0]
        return _cache[k]
    P.read_archive_mask = _on_the_fly
    common['P'] = P
    # ---- 探针：先确认 load() 真能读到成员掩膜（否则链条必然 0 条 ✗ 立刻停）----
    m255 = importlib.import_module('core.vendor.s5_steps.s255_straightness')
    for k, v in common.items():
        setattr(m255, k, v)
    # 探针按"本次输入的第一块板"取名（从 boards_pairs 派生，绝不写死样品名）
    _bp0 = list(boards_pairs or [])
    assert _bp0, '探针无法取名：boards_pairs 为空（本次输入没被接收）'
    _grp0 = os.path.basename(os.path.dirname(_bp0[0][1]))
    _tail0 = os.path.splitext(os.path.basename(_bp0[0][1]))[0][-4:]
    _probe_name = _grp0 + '_' + _tail0 + '.png'
    got = m255.load(m255.ALL['TTA'], _probe_name)
    if got is None:
        raise RuntimeError('探针失败：load() 读不到 pred_masks_tta/' + _probe_name + '（检查 UN/目录名 ✓）')
    print('   探针 ✓ load(tta, {}) 读到掩膜，前景 {:d} px'.format(_probe_name, int(got.sum())))
    # ---- 依次执行 ----
    steps = []
    for mod in STEPS:
        m = importlib.import_module('core.vendor.s5_steps.' + mod)
        for k, v in common.items():
            if hasattr(m, k) or True:
                setattr(m, k, v)
        try:
            rc = m.main()
            steps.append((mod, rc, 'ok'))
        except SystemExit as e:
            steps.append((mod, e.code, 'systemexit'))
        except Exception as e:
            steps.append((mod, repr(e), 'error'))
            open(os.path.join(run_dir, 's5_step_errors.log'), 'a', encoding='utf-8').write(
                mod + ' :: ' + traceback.format_exc() + '\n')
            print('   [驱动兜底] {} 抛异常 {}'.format(mod, repr(e)))
    return steps
