# -*- coding: utf-8 -*-
"""run_pipeline.py —— 飞线异常判定：一条命令跑完 S1–S5（软件交付入口）

架构（与 core/ 实际 API 一致）：
  S1 候选 + S2 CNN 筛选 + S3 U-Net 三成员掩膜 —— 由 core.s4_classify.run() 内部逐板调度
  S4 掩膜分类（合格/断联/焊错/噪音/裸焊盘）
  S5 再处理 + 直度判定（补缺 → 拆分 → 剪端 → 骨架 → 判据）+ 出图

用法：
  python run_pipeline.py --input <待检图片目录> [--out <输出目录>] [--cfg cfg/default.json]
输入目录：其下按子文件夹分组（如 NG-1/、OK-2/），文件名任意（*.jpg/*.png）
输出：<out>/<YYYYMMDD_HHMM>_run/{masks,state,run.log}

依赖：Python + numpy + opencv + torch(CPU)；训练接口保留但不启用（复用 models/ 内权重）。
"""
import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# --- Windows 兼容：cv2.imread / cv2.imwrite 在 Windows 上无法读写中文/非 ASCII 路径 ---
# 仅在 Windows 生效，且只在原生调用失败时回退（numpy 编解码 + Python 文件对象），
# 因此不改变算法（core/ 内的文件一行未动）；Linux 上 cv2 原生支持 UTF-8 路径，行为不变。
if os.name == "nt":
    def _install_utf8_cv2_io():
        try:
            import cv2
            import numpy as np
        except Exception:
            return

        _imread0, _imwrite0 = cv2.imread, cv2.imwrite

        def _is_ascii(path):
            try:
                return os.fspath(path).isascii()
            except Exception:
                return True

        def _imread(path, flags=cv2.IMREAD_COLOR):
            if _is_ascii(path):
                return _imread0(path, flags)
            try:
                with open(os.fspath(path), "rb") as fh:
                    return cv2.imdecode(np.frombuffer(fh.read(), np.uint8), flags)
            except Exception:
                return None

        def _imwrite(path, img, params=None):
            if _is_ascii(path):
                return _imwrite0(path, img, params or [])
            try:
                ext = os.path.splitext(os.fspath(path))[1] or ".png"
                ok, buf = cv2.imencode(ext, img, params or [])
                if not ok:
                    return False
                with open(os.fspath(path), "wb") as fh:
                    fh.write(buf.tobytes())
                return True
            except Exception:
                return False

        cv2.imread, cv2.imwrite = _imread, _imwrite

    _install_utf8_cv2_io()


from core import cfg as C, run_io, s4_classify, s5_straightness, s_report


def run_s1_s3(conf, run, src, pairs, say):
    """S1 候选 -> S2 CNN 筛选 -> S3 三成员掩膜（逐板 ✓ 现场算 ✓ 写进本次运行目录 ✓）

    产出（供 S4 消费）：
      <run>/masks/TTA|v13|v18/<组>_<尾4位>.png   三成员掩膜
      <run>/masks/solder_table.json              本次接受的焊点表 {"rows":[{sample,x,y,...}]}
    """
    import cv2
    import numpy as np
    from core import s1_candidates, s2_cnn, s3_unet_mask
    md = os.path.join(run, 'masks')
    for k in ('TTA', 'v13', 'v18'):
        os.makedirs(os.path.join(md, k), exist_ok=True)
    rows_all = []
    for name, path in pairs:
        g0 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        rows, m4, gray = s1_candidates.run_board(path, conf, name)
        P, mean, acc = s2_cnn.score_board(gray, rows, conf)
        acc = np.asarray(acc).reshape(-1)
        kept = [r for r, a in zip(rows, acc) if a]
        grp, tail = (name.split('/') if '/' in name else name.rsplit('-', 1))
        for r in kept:
            r = dict(r); r['sample'] = grp + '/' + tail   # S4/直度报告用【组名/尾4位】
            r['sample_alt'] = grp + '-' + tail
            rows_all.append(r)
        base = grp + '_' + tail + '.png'                  # 掩膜文件名用【组名_尾4位】（S4 的 load 约定）
        for k in ('TTA', 'v13', 'v18'):
            m = s3_unet_mask.member_mask(gray, conf, k)
            cv2.imwrite(os.path.join(md, k, base), (np.asarray(m) > 0).astype('uint8') * 255)
        say('   %s 候选 %d 接受 %d 掩膜已写' % (name, len(rows), len(kept)))
    json.dump({'rows': rows_all}, open(os.path.join(md, 'solder_table.json'), 'w', encoding='utf-8'),
              ensure_ascii=False)
    say('S1-S3 完成 本次接受焊点合计 %d' % len(rows_all))
    return os.path.join(md, 'solder_table.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', default=os.path.join(HERE, 'outputs'))
    ap.add_argument('--cfg', default=os.path.join(HERE, 'cfg', 'default.json'))
    a = ap.parse_args()

    conf = C.load(a.cfg)
    src = os.path.abspath(a.input)
    run = os.path.join(os.path.abspath(a.out), time.strftime('%Y%m%d_%H%M%S') + '_run')
    for d in ('masks', 'state'):
        os.makedirs(os.path.join(run, d), exist_ok=True)
    log = open(os.path.join(run, 'run.log'), 'w', encoding='utf-8')

    def say(*x):
        s = ' '.join(str(i) for i in x)
        print(s); log.write(s + '\n'); log.flush()

    pairs = run_io.list_boards(conf, src)
    say('输入目录 =', src)
    say('发现 %d 块板' % len(pairs))
    t00 = time.time()
    try:
        sold = run_s1_s3(conf, run, src, pairs, say)
        say('[S1-S3] ok %.1fs' % (time.time() - t00))
    except Exception as e:
        say('[S1-S3] 失败 %r' % e); say(traceback.format_exc()); log.close(); return 1
    t0 = time.time()
    try:
        r4 = s4_classify.run(conf, run, src,
                            sold_json=sold,
                            annot_dir=os.path.join(HERE, 'refs', 'annot'))
        say('[S4 掩膜分类] ok %.1fs %s' % (time.time() - t0, json.dumps(r4, ensure_ascii=False)[:600]))
    except Exception as e:
        say('[S4] 失败 %r' % e); say(traceback.format_exc()); log.close(); return 1
    t1 = time.time()
    try:
        r5 = s5_straightness.run(conf, run, src, boards=[b for b, _ in pairs], boards_pairs=pairs)
        _stagefig(run, src, say)
        try:
            import core.s_barepad as BP
            say("  裸焊盘判定：")
            _bp_res = BP.build(run, os.path.join(run, "masks", "solder_table.json"))
            if _bp_res:
                say("  裸焊盘裁图 -> " + str(BP.crop_sheet(run, src, _bp_res)))
        except Exception as _e:
            say("  裸焊盘判定失败（不阻断）：{}".format(_e))
        say('[S5 再处理+直度] ok %.1fs %s' % (time.time() - t1, r5))
    except Exception as e:
        say('[S5] 失败 %r' % e); say(traceback.format_exc()); log.close(); return 1
    try:
        import core.s_report2 as _SR2
        _rep = _SR2.build(run, src, C=conf)
        say('[产品报告] 产品 %d ｜ 合格 %d ｜ 不合格 %d ｜ CSV %s' % (
            _rep['产品数'], _rep['合格'], _rep['不合格'], _rep['CSV']))
    except Exception as e:
        say('[报告图] 失败 %r' % e); say(traceback.format_exc())
    say('完成  产物 ->', run)
    log.close()
    return 0


def _stagefig(run, src, say):
    try:
        import core.s_stagefig as SF
        boards = []
        for grp in sorted(os.listdir(src)):
            d = os.path.join(src, grp)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    key = os.path.splitext(f)[0]
                    tail = key.split('_')[-1][-4:]
                    boards.append(grp + "/" + tail)
        res = SF.build(run, src, sorted(set(boards)))
        for r in res:
            if r["stage"].startswith("3_"):
                say("  阶段图 {}：掩膜{}px 未覆盖焊点 {}".format(r["board"], r["mask_px"], r["uncovered"]))
        say("  阶段图已出 -> " + os.path.join(run, "state", "stagefig"))
    except Exception as e:
        say("  阶段图失败（不阻断）：{}".format(e))


if __name__ == '__main__':
    sys.exit(main())
