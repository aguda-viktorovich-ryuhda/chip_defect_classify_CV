"""飞线方向估计：2D 傅里叶频谱亮脊法（供追踪器与校验脚本共用）

原理：窗口内一条直线/条纹，其 2D 频谱的亮脊方向 与 线方向 垂直；
      故「沿各角度对频谱径向求和 → 取最大」得到的是线的法向，+90° 即线方向。
要点：
  * Hann 窗（必须）—— 否则方块边界的硬边会在频谱留下十字伪影；
  * 去低频（highpass）—— 去掉照度梯度这类大尺度成分；
  * 传入 band 掩膜时，把黑框像素用局部背景中位数替换 —— 避免人为硬边，也让黑框不参与投票；
  * 方向天然有 ±180° 歧义，调用方必须用行进方向定朝向；
  * 返回各向异性 R ∈ [0,1]：R 大 = 窗口内方向单一（可信），R 小 = 各方向都有（不可信）。
"""

import numpy as np


def fft_dir(g, cx, cy, half=32, highpass=3.0, band=None):
    """返回 (方向角_度 ∈ [0,180), 各向异性 R) ；窗口越界返回 None。"""
    cx, cy = int(round(cx)), int(round(cy))
    x0, y0 = cx - half, cy - half
    if x0 < 0 or y0 < 0 or x0 + 2 * half >= g.shape[1] or y0 + 2 * half >= g.shape[0]:
        return None
    patch = g[y0:y0 + 2 * half, x0:x0 + 2 * half].astype(np.float32)
    if band is not None:
        m = band[y0:y0 + 2 * half, x0:x0 + 2 * half]
        if m.any() and (~m).any():
            patch = patch.copy()
            patch[m] = np.median(patch[~m])
    patch = patch - patch.mean()
    win = np.outer(np.hanning(2 * half), np.hanning(2 * half)).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(patch * win))
    mag = np.abs(F)
    yy, xx = np.mgrid[-half:half, -half:half]
    mag[np.hypot(xx, yy) < highpass] = 0.0
    best_th, best_s, prof = 0.0, -1.0, []
    r = np.arange(highpass, half)
    for th in np.arange(0.0, 180.0, 0.5):
        t = np.deg2rad(th)
        xs = np.clip(np.round(half + r * np.cos(t)).astype(int), 0, 2 * half - 1)
        ys = np.clip(np.round(half + r * np.sin(t)).astype(int), 0, 2 * half - 1)
        s = float(mag[ys, xs].sum())
        prof.append(s)
        if s > best_s:
            best_s, best_th = s, th
    ss = np.array(prof)
    aniso = float((ss.max() - np.median(ss)) / (ss.max() + 1e-9))
    return float((best_th + 90.0) % 180.0), aniso


def to_heading(ang180, ref):
    """把 ±180° 歧义的方向，按参考朝向 ref（单位向量）定成有向角（度）。"""
    v = np.array([np.cos(np.deg2rad(ang180)), np.sin(np.deg2rad(ang180))])
    if float(np.dot(v, ref)) < 0:
        v = -v
    return float(np.rad2deg(np.arctan2(v[1], v[0])))
