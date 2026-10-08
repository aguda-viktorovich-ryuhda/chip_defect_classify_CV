# -*- coding: utf-8 -*-
"""cnn_model.py - 焊点 CNN 网络定义（自 recon/111_train_cnn.py 第 53..117 行原样搬入）

class Stage + class SmallCNN（arch=res2_deep；act=relu）
不许改这里的结构 - 与训练时完全一致才能直接复用权重；要改变体请新加类，别动旧类。
"""
import torch
import torch.nn as nn


class Stage(nn.Module):
    """一个阶段 = nconv 个 (Conv-BN) ；可选残差（通道数不同时用 1×1 投影对齐）"""

    def __init__(self, cin, cout, nconv, residual, act):
        super().__init__()
        self.act = act
        self.residual = residual
        self.convs = nn.ModuleList()
        for i in range(nconv):
            self.convs.append(nn.Sequential(
                nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout)))
        self.proj = nn.Conv2d(cin, cout, 1) if (residual and cin != cout) else None

    def forward(self, x):
        h = x
        n = len(self.convs)
        for i, m in enumerate(self.convs):
            h = m(h)
            if not (self.residual and i == n - 1):
                h = self.act(h)                     # 残差时最后一个卷积先不激活，等加了短接再激活
        if self.residual:
            idt = x if self.proj is None else self.proj(x)
            h = self.act(h + idt)
        else:
            h = self.act(h)
        return h


class SmallCNN(nn.Module):
    """arch:
         plain1     每阶段 1 个卷积、无残差（现行基线，25865 参数）
         plain2     每阶段 2 个卷积、无残差（与 res2 同位深、用来隔离"残差"本身的影响）
         res2       每阶段 2 个卷积 + 残差
         res2_deep  4 个阶段、每阶段 2 个卷积 + 残差（看残差能不能撑起更深的网）
    """

    def __init__(self, act="relu", arch="plain1"):
        super().__init__()
        self.in_bn = nn.BatchNorm2d(1)
        self.act = (lambda v: torch.sigmoid(v)) if act == "sigmoid" else (lambda v: torch.relu(v))
        nconv = 1 if arch == "plain1" else 2
        residual = arch.startswith("res")
        nstage = 4 if arch == "res2_deep" else 3
        chans = [16, 32, 64, 64][:nstage]
        self.stages = nn.ModuleList()
        cin = 1
        for cout in chans:
            self.stages.append(Stage(cin, cout, nconv, residual, self.act))
            cin = cout
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.f1 = nn.Linear(cin, 32)
        self.f2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.in_bn(x)
        for st in self.stages:
            x = self.pool(st(x))
        x = self.gap(x).flatten(1)
        x = torch.sigmoid(self.f1(x))
        return torch.sigmoid(self.f2(x)).squeeze(1)


# ---------------------------------------------------------------- 数据


def build_and_load(weight_path, arch="res2_deep", act="relu", device=None):
    """按训练时的结构建网并载入权重（strict=True ✓ 结构不符会立刻报错 ✗）"""
    dev = device or torch.device("cpu")
    net = SmallCNN(act=act, arch=arch).to(dev)
    sd = torch.load(weight_path, map_location=dev)
    net.load_state_dict(sd)
    net.eval()
    return net
