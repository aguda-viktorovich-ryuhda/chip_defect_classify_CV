# -*- coding: utf-8 -*-
"""s_paths.py —— 路径与命名解析的**唯一出处**（2026-10-07 立）

背景：板名/路径解析曾散落在 s4_core、s255、s253、报告、阶段图各处，各写各的，
导致"同一块板在不同模块里名字不一样"——连坏数轮的根因。此处收敛为一份契约。

契约（由 docs/PATHS-CONTRACT.md 机械抽取得出，勿凭记忆改）：
  1. 板枚举：`<ROOT>/<组名>/<base>_m.png`；`base = 文件名去掉 _m.png`
  2. 板标识：`key = <组名>/<base[-4:]>`（斜杠形式）｜`name = <组名>-<base[-4:]>`（连字符形式）
  3. 原图  ：`<SRC>/<组名>/<base>.jpg`（用**完整 base**）
  4. 三成员掩膜：`<UN>/<成员>/<组名>_<base[-4:]>.png`
  5. 飞线掩膜  ：`<UN>/pred_masks_wires/<file>`；判定阶段块：`<UN>/pred_masks_all/<file>`
  6. 焊点表：写入时同时登记 `sample`（斜杠）与 `sample_alt`（连字符）；查表一律经 solder_key()

用法：
    import core.s_paths as SP
    for b in SP.iter_boards(ROOT):
        g, base = b.group, b.base
        img = b.image(SRC); mem = b.member_mask(UN, "TTA")
"""
import glob
import os

_FONT_CANDS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",   # Linux（本项目历来用的）
    r"C:\Windows\Fonts\msyh.ttc",                             # Windows 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",                           # Windows 黑体
    "/System/Library/Fonts/PingFang.ttc",                       # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",          # 兜底（无中文）
)


def font_path():
    """跨平台中文字体：依次尝试，返回第一个存在的路径。"""
    for c in _FONT_CANDS:
        if os.path.exists(c):
            return c
    return _FONT_CANDS[0]


class Board(object):
    """一块板（= 一张图）的解析结果。所有路径都从这里取，不在别处拼名字。"""

    __slots__ = ("group", "base", "annot")

    def __init__(self, group, base, annot=None):
        self.group = group
        self.base = base
        self.annot = annot

    # ---- 标识 ----
    @property
    def tail(self):
        """尾号：引擎约定的 4 位/末 4 字符（掩膜命名与焊点键都用它）。"""
        return self.base[-4:]

    @property
    def key(self):
        """斜杠形式：组名/尾号（报告、焊点表 sample、S4/S255 的 key）。"""
        return self.group + "/" + self.tail

    @property
    def name(self):
        """连字符形式（S4/S255 查焊点表用的 name）。"""
        return self.group + "-" + self.tail

    def __repr__(self):
        return "Board(%s)" % self.key

    def __eq__(self, other):
        return isinstance(other, Board) and other.key == self.key

    def __hash__(self):
        return hash(self.key)

    # ---- 路径 ----
    def image(self, src):
        """原图（完整 base，不是末 4 位）。不存在时返回 None。"""
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
            p = os.path.join(src, self.group, self.base + ext)
            if os.path.exists(p):
                return p
        d = os.path.join(src, self.group)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if self.tail and self.tail in os.path.splitext(f)[0]:
                    return os.path.join(d, f)
        return None

    def member_mask(self, un, member):
        """三成员掩膜：<UN>/<成员>/<组名>_<尾号>.png"""
        return os.path.join(un, member, "{}_{}.png".format(self.group, self.tail))

    def wire_mask(self, un, file_name):
        """最终飞线掩膜：<UN>/pred_masks_wires/<file>"""
        return os.path.join(un, "pred_masks_wires", file_name)

    def stage_mask(self, un, file_name):
        """判定阶段块掩膜：<UN>/pred_masks_all/<file>"""
        return os.path.join(un, "pred_masks_all", file_name)

    def barepad_mask(self, un, file_name):
        return os.path.join(un, "pred_masks_barepad", file_name)


def iter_boards(root):
    """按契约枚举板：<root>/<组名>/<base>_m.png（S4/S255 的枚举口径）。"""
    for f in sorted(glob.glob(os.path.join(root, "*", "*_m.png"))):
        grp = os.path.basename(os.path.dirname(f))
        base = os.path.basename(f)[:-6]
        yield Board(grp, base, annot=f)


def from_annot(annot_path):
    """从标注文件路径反推 Board（供仅拿到 f 的旧代码迁移用）。"""
    grp = os.path.basename(os.path.dirname(annot_path))
    base = os.path.basename(annot_path)[:-6] if annot_path.endswith("_m.png") else \
        os.path.splitext(os.path.basename(annot_path))[0]
    return Board(grp, base, annot=annot_path)


def solder_index(rows):
    """焊点表索引：**两种形式都登记**，返回 {key: [(x,y), ...]}（同一行只登记一次，不重复计数）。
    这是"s4_core 与 s255 各自建表、只登记一种形式导致查不到"的根治点。"""
    idx = {}
    seen = set()
    for e in rows:
        xy = (float(e["x"]), float(e["y"]))
        keys = set()
        for k in (str(e.get("sample", "")), str(e.get("sample_alt", ""))):
            if not k:
                continue
            keys.add(k)
            keys.add(k.replace("/", "-"))
            keys.add(k.replace("-", "/"))
        for k in keys:
            tag = (k, xy)
            if tag in seen:
                continue
            seen.add(tag)
            idx.setdefault(k, []).append(xy)
    return idx


def solder_key(sample):
    """把任一形式的焊点键规范成两种形式（返回 tuple），供查表。"""
    s = str(sample)
    return (s, s.replace("/", "-"), s.replace("-", "/"))


def self_test(root, src, un, solder_json=None):
    """自检：按契约解析一遍，逐项核对文件是否存在。返回 (ok, 明细)。"""
    import json
    detail = []
    ok = True
    boards = list(iter_boards(root))
    detail.append("枚举板数 = %d" % len(boards))
    rows = []
    if solder_json and os.path.exists(solder_json):
        rows = json.load(open(solder_json, encoding="utf-8")).get("rows", [])
    idx = solder_index(rows) if rows else {}
    for b in boards:
        img = b.image(src)
        mem = {m: os.path.exists(b.member_mask(un, m)) for m in ("TTA", "v13", "v18")}
        hit = {k: len(idx.get(k, [])) for k in solder_key(b.name)}
        good = (img is not None) and all(mem.values())
        ok = ok and good
        detail.append("  %-28s 原图 %s ｜ 掩膜 %s ｜ 焊点命中 %s" % (
            b.key, "有" if img else "无✗", {k: ("有" if v else "无✗") for k, v in mem.items()}, hit))
    return ok, detail


if __name__ == "__main__":
    import json
    import sys
    r = sys.argv[1] if len(sys.argv) > 1 else sorted(
        glob.glob("outputs/*_run"), key=os.path.getmtime)[-1]
    run = os.path.abspath(r)
    root = os.path.join(run, "inputs", "annot", "auto")
    src = os.path.join(run, "inputs", "src")
    un = os.path.join(run, "masks")
    if not os.path.isdir(root):          # 老结构：直接用输入目录 + 旧标注目录
        root = os.path.join(".", "refs", "annot")
        src = os.path.join(".", "chip_samples__20260128")
        un = os.path.join(".", "masks")
    ok, detail = self_test(root, src, un, os.path.join(run, "masks", "solder_table.json"))
    print("自检运行目录:", run)
    print("ROOT =", root)
    print("SRC  =", src)
    print("UN   =", un)
    for d in detail[:14]:
        print(d)
    print("自检结论:", "通过 ✓" if ok else "有项目不通过 ✗")
