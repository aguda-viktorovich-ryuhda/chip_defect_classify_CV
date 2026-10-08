# -*- coding: utf-8 -*-
"""s3_unet_mask.py - 第③步：U-Net 现场出飞线掩膜（并集三成员 TTA / v13 / v18）

合成口径（自 recon/208_tta_and_v12.py 第 165-190 行原样搬入）：
   每个模型： p = tta_prob(net, prep(g, mode))（4 视图先平均） -> m = (p > 0.5)（各自阈值化）
   TTA 成员： 5 个模型的布尔图逐像素【逻辑或】-> 成员掩膜
   v13/v18  ： 各自脚本的整板推理块（transporting in progress）
   并集     ： 保留成员逐像素逻辑或
权重： dict 形式，权重在 ck["sd"]，levels = ck.get("levels", 3)
"""
import os
import sys

import cv2
import numpy as np
import torch

_V = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "vendor")
if _V not in sys.path:
    sys.path.insert(0, _V)
import unet_tta as T  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load(net_file, C):
    ck = torch.load(os.path.join(ROOT, "models", "unet", net_file), map_location=DEV)
    if C["s3_unet_mask"].get("ckpt_format") == "dict_sd":
        net = T.U(16, 1, ck.get("levels", 3)).to(DEV)
        net.load_state_dict(ck["sd"])
    else:
        net = T.U(16, 1, 3).to(DEV)
        net.load_state_dict(ck)
    net.eval()
    return net


def tta_member(g, C, tag=""):
    """TTA 成员：5 个模型 x 4 视图 -> 各模型 p>0.5 -> 逐像素或"""
    pool = np.zeros(g.shape, bool)
    for w in C["s3_unet_mask"]["weights"]["TTA"]:
        if w["mode"] not in ("raw", "pct", "clahe"):
            raise ValueError("TTA 这个权重没有可用的输入模式：" + str(w))
        net = _load(w["file"], C)
        p = T.tta_prob(net, T.prep(g, w["mode"]))
        pool |= (p > 0.5)
        print("    {} 输入{:<6} 前景 {:>6.2f}%".format(w["file"], w["mode"], 100.0 * pool.mean()))
    return pool


def member_mask(g, C, member):
    """成员掩膜：TTA=5 模型 x 4 视图取或；v13/v18=各自整板推理块（重叠 50%）"""
    if member == "TTA":
        return tta_member(g, C)
    import torch
    def _wp(f):
        """cfg 里存的是裸文件名 ⇒ 解析到 models/unet/ 下（与 TTA 成员一致）"""
        cands = [f, os.path.join(ROOT, os.path.basename(str(f))),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'unet', os.path.basename(str(f)))]
        for c in cands:
            if os.path.exists(c):
                return c
        raise FileNotFoundError("权重找不到：" + str(f))
    if member == "v13":
        from core.vendor import unet_v13 as V
        ck = torch.load(_wp(C["s3_unet_mask"]["weights"]["v13"][0]["file"]), map_location=DEV)
        net = V.U(c=16, cin=1, levels=ck.get("levels", 3))
    elif member == "v18":
        from core.vendor import unet_v18 as V
        ck = torch.load(_wp(C["s3_unet_mask"]["weights"]["v18"][0]["file"]), map_location=DEV)
        net = V.Net(c=16, cin=1, levels=ck.get("levels", 3))
    else:
        raise ValueError("未知成员：" + str(member))
    net.load_state_dict(ck["sd"] if "sd" in ck else ck)
    net = net.to(DEV)
    net.eval()
    with torch.no_grad():
        # v13：infer_v13 内部不做归一化 ⇒ 调用方必须给【原图/255】（见 unet_v13.py 说明 ✓）
        ch_in = (g.astype("float32") / 255.0) if member == "v13" else g
        m = V.infer_v13(net, ch_in) if member == "v13" else V.infer_v18(net, g)
    return np.asarray(m) > 0


def union_mask(g, C, only=None):
    members = only or C["s3_unet_mask"]["keep_members"]
    out = np.zeros(g.shape, bool)
    for m in members:
        mm = member_mask(g, C, m)
        print("  成员 {} 前景 {:>6.2f}%".format(m, 100.0 * mm.mean()))
        out |= mm
    return out
