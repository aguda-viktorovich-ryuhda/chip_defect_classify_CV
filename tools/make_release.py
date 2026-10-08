#!/usr/bin/env python3
"""make_release.py —— 从客户交付目录导出一份「开源发布树」（单一源头，避免两份拷贝手工分叉）。

用法（在交付目录 chip_defect_classify/ 内）：
    python tools_make_release.py --src . --dst ../chip_defect_classify_opensource

规则：
  · 白名单：run_pipeline.py、VERSION.md、启动_*.sh/.bat、.gitignore、本地配置.env.example、cfg/、core/、ui/、train/、docs/项目技术文档.md
  · 排除：models/、refs/、chip_samples__20260128/、outputs/、backups/、内部审计文档、__pycache__
  · 文档脱敏：仅替换 .md 文件里的客户板号与组名（代码一律不动，避免改坏判定逻辑）
  · 输出后自动校验：编译全部 .py、扫描机器绝对路径、写入 LICENSE 与 requirements.txt
"""
import argparse
import json, json, os, re, shutil, py_compile, subprocess, sys

KEEP_ROOT = {'run_pipeline.py','VERSION.md','启动_linux.sh','启动_windows.bat','.gitignore','本地配置.env.example','requirements.txt','LICENSE','README.md'}
KEEP_DIRS = {'cfg','core','ui','train','docs','tools'}
KEEP_EXT  = {'.py','.md','.json','.yaml','.yml','.txt'}
SKIP_DIRS = {'__pycache__','outputs','backups','data'}
SKIP_FILES = {'docs/FLOW-AUDIT-20261006.md','docs/_code_inventory.json','docs/VARIANT-AB-TEST.md',
              'docs/OPENSOURCE-EXCLUDE.md','docs/VARIANT-FITDIR-TEST.md'}
EXTRA_DOCS = ['docs/项目技术文档.md']

SCRUB = [('1_NG/1649','样例板A'),('1_NG/1658','样例板B'),('1_NG/1705','样例板C'),
         ('1_OK/1507','样例板D'),('1_OK/1528','样例板E'),('1_OK/1535','样例板F'),
         ('2_NG/3414','样例板G'),('2_NG/3421','样例板H'),('2_OK/2','样例板I'),('2_OK/3','样例板J'),
         ('1649/1658/1705/3414/3421','样例板A–J'),('某电子公司','<公司>')]
# 机器绝对路径检测：不写具体用户名，避免把本机账号信息带进公开仓库
MACHINE = re.compile(r'/(?:data|media|home)/[A-Za-z0-9_.-]+/')

def scrub_md(t):
    for a,b in SCRUB: t = t.replace(a,b)
    return t



# ---- 发布树脱敏②：功能性硬编码表在发布物里换成示例占位（交付目录原样不动）----
def _brace_block(text, varname):
    """返回 (start, end)，end 指向匹配右括号之后"""
    m = re.search(r'(?m)^%s\s*=\s*\{' % re.escape(varname), text)
    if m is None:
        m = re.search(r'%s\s*=\s*\{' % re.escape(varname), text)
    if not m: return None
    i = text.index('{', m.start()); depth = 0
    for j in range(i, len(text)):
        if text[j] == '{': depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0: return (m.start(), j + 1)
    return None

def _apply_placeholders(dst):
    done = []
    # s256：BOARD_FILE → 示例
    p = os.path.join(dst, 'core/vendor/s5_steps/s256_panels.py')
    if os.path.exists(p):
        t = open(p, encoding='utf-8').read(); b = _brace_block(t, 'BOARD_FILE')
        if b:
            t = t[:b[0]] + ('BOARD_FILE = {\n'
                '    # 示例占位（发布版）：键=板名，值=(组名, 原始文件名主干)。请按你的数据集填写。\n'
                '    "GROUP/0001": ("GROUP", "raw_name_0001"),\n'
                '    "GROUP/0002": ("GROUP", "raw_name_0002"),\n'
                '}') + t[b[1]:]
            open(p, 'w', encoding='utf-8').write(t); done.append('s256_panels.BOARD_FILE')
    # s5_core / s255：NG_BOARDS、OKB → 空集合
    for rel, var in [('core/vendor/s5_core.py','NG_BOARDS'), ('core/vendor/s5_core.py','OKB'),
                     ('core/vendor/s5_steps/s255_straightness.py','NG_BOARDS')]:
        p = os.path.join(dst, rel)
        if not os.path.exists(p): continue
        t = open(p, encoding='utf-8').read(); b = _brace_block(t, var)
        if b and ('"' in t[b[0]:b[1]] or "'" in t[b[0]:b[1]]):
            t = t[:b[0]] + ('%s = set()  # 发布版置空：请按你的数据集填写板名' % var) + t[b[1]:]
            open(p, 'w', encoding='utf-8').write(t); done.append(rel + '.' + var)
    # cfg/default.json：板号→序号表 → 示例
    p = os.path.join(dst, 'cfg/default.json')
    if os.path.exists(p):
        try:
            d = json.load(open(p, encoding='utf-8'))
            def walk(o):
                if isinstance(o, dict):
                    keys = [k for k in o if re.search(r'1_NG|1_OK|2_NG|2_OK|1649|1658|1705|3414|3421', str(k))]
                    if len(keys) >= 3:
                        for k in keys: o.pop(k)
                        o['GROUP/0001'] = 1; o['GROUP/0002'] = 2
                        return True
                    return any(walk(v) for v in o.values())
                if isinstance(o, list):
                    return any(walk(v) for v in o)
                return False
            if walk(d):
                json.dump(d, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
                done.append('cfg/default.json 板号表')
        except Exception as e:
            done.append('cfg/default.json 跳过(' + str(e)[:40] + ')')
    # fold_map：发布版清空（公开仓库里没有对应板子）
    p = os.path.join(dst, 'cfg/default.json')
    if os.path.exists(p):
        try:
            d = json.load(open(p, encoding='utf-8'))
            if isinstance(d.get('s2_cnn'), dict) and 'fold_map' in d['s2_cnn']:
                d['s2_cnn']['fold_map'] = {}
                json.dump(d, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
                done.append('cfg fold_map 置空')
        except Exception as e:
            done.append('fold_map 跳过(' + str(e)[:30] + ')')
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='.')
    ap.add_argument('--dst', required=True)
    a = ap.parse_args()
    src, dst = os.path.abspath(a.src), os.path.abspath(a.dst)
    if os.path.exists(dst): shutil.rmtree(dst)
    n = 0
    for root, dirs, fs in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel = os.path.relpath(root, src)
        top = rel.split(os.sep)[0]
        for f in fs:
            if rel == '.':
                if f not in KEEP_ROOT: continue
            else:
                if top not in KEEP_DIRS or os.path.splitext(f)[1] not in KEEP_EXT: continue
            r = os.path.normpath(os.path.join(rel, f))
            if r in SKIP_FILES:
                continue
            if 'models' in r.split(os.sep)[:1] or r.startswith('refs') or r.startswith('chip_samples'):
                continue
            out = os.path.join(dst, r); os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.copy2(os.path.join(root, f), out); n += 1
    # 文档：只带面向公开的技术文档（若上面已带则跳过）
    for d in EXTRA_DOCS:
        p = os.path.join(src, d)
        if os.path.exists(p) and not os.path.exists(os.path.join(dst, d)):
            os.makedirs(os.path.dirname(os.path.join(dst, d)), exist_ok=True)
            shutil.copy2(p, os.path.join(dst, d)); n += 1
    # 文档脱敏
    for root, dirs, fs in os.walk(dst):
        dirs[:] = [x for x in dirs if x != '__pycache__']
        for f in fs:
            if not f.endswith('.md'): continue
            p = os.path.join(root, f); t = open(p, encoding='utf-8').read(); s = scrub_md(t)
            if s != t: open(p, 'w', encoding='utf-8').write(s)
    print('  占位映射替换:', _apply_placeholders(dst) or '无需替换')
    # 公开版 README（由 tools/README.public.md 安装而来，避免手工维护两份）
    pub = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.public.md')
    if os.path.exists(pub):
        shutil.copy2(pub, os.path.join(dst, 'README.md'))
    # 许可证与依赖
    gpl = '/usr/share/common-licenses/GPL-3'
    if os.path.exists(gpl): shutil.copy2(gpl, os.path.join(dst, 'LICENSE'))
    open(os.path.join(dst, 'requirements.txt'), 'w', encoding='utf-8').write(
        '# 仅这三个依赖；opencv 必须是 contrib 版（cv2.ximgproc.thinning 不在主包里）\nnumpy\nopencv-contrib-python\n'
        '# torch 建议 CPU 版：pip install torch --index-url https://download.pytorch.org/whl/cpu\ntorch\n')
    # 校验
    py = [os.path.join(r,f) for r,_,fs in os.walk(dst) for f in fs if f.endswith('.py')]
    bad = []
    for f in py:
        try: py_compile.compile(f, doraise=True, cfile='/tmp/_chk.pyc')
        except Exception as e: bad.append((f, str(e)[:60]))
    mach = []
    for r, dirs, fs in os.walk(dst):
        dirs[:] = [x for x in dirs if x != '__pycache__']
        for f in fs:
            if not f.endswith(('.py','.json','.yaml','.md','.sh','.bat','.txt')): continue
            p = os.path.join(r, f)
            for i, l in enumerate(open(p, encoding='utf-8', errors='ignore').read().splitlines(), 1):
                if MACHINE.search(l): mach.append('{}:{}'.format(os.path.relpath(p, dst), i)); break
    for r, dirs, fs in os.walk(dst):
        if '__pycache__' in r: shutil.rmtree(r, ignore_errors=True)
    print('导出完成: {} （{} 个文件）'.format(dst, n))
    print('  编译: {}/{} 通过'.format(len(py)-len(bad), len(py)))
    if bad: print('  失败:', bad)
    print('  机器绝对路径:', mach or '无 ✓')
    return 0 if not bad and not mach else 1

if __name__ == '__main__':
    sys.exit(main())
