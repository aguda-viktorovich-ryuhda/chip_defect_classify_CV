# train/ —— 训练与数据准备脚本（保留接口，默认不启用）

这些脚本从项目工作目录原样搬入，用于：准备训练数据、训练 CNN（焊点二分类）、训练 U-Net（飞线掩膜分割）、合并/导出权重。
**交付版不会自动跑训练**：`run_pipeline.py` 只做推理，复用 `models/` 里的现成权重。

## 一、路径约定（2026-10-08 适配）

所有脚本已去掉写死的绝对路径，改为按脚本位置自动定位，并可用环境变量覆盖：

| 环境变量 | 含义 | 默认值 |
|---|---|---|
| `CHIP_ROOT` | 软件根目录 | 本文件的上一级（即 `chip_defect_classify/`） |
| `CHIP_DATA` | 训练数据根目录 | `<CHIP_ROOT>/train/data` |
| `CHIP_PY`   | 训练时调用的解释器 | 当前解释器 `sys.executable` |
| `CHIP_SRC`  | 原始样品图目录（只读） | （不设默认，必须显式指定） |
| `CHIP_PROJ` | 工程数据目录（掩膜/面板/权重等） | （不设默认，必须显式指定） |

脚本头部会自动把 `CHIP_ROOT` 与 `train/lib` 加入 `sys.path`，因此直接 `python train/xxx.py` 即可。

## 二、客户要准备什么数据

按脚本实际读取的目录整理（`<CHIP_DATA>` 下）：

```
<CHIP_DATA>/
├── dataset/            图片（待裁块的原图或已裁块），文件名任意
├── dataset.json        CNN 训练集索引：每条含图片路径、候选点坐标、标签（0=非焊点 1=焊点）
├── labels/             CNN 标签相关的中间产物（由 174/175 生成或同步）
└── out/cnn/            CNN 流程的中间与结果文件
    ├── dataset.json              训练集索引（同上，实际路径以脚本为准）
    ├── cand_vs_trainset.json    179 的比对结果
    └── trainset_points_new.json 184 的核查输出

U-Net 训练数据（人工逐像素掩膜）：
├── pred_masks_v3/ … pred_masks_v6/   各版本掩膜（不同前处理/配方对应的训练目标）
```

要点：**U-Net 训练必须有"图片 + 人工掩膜"成对的标注**；没有掩膜，脚本无法训练分割模型。

## 三、典型流程与脚本对应

| 目的 | 脚本 |
|---|---|
| CNN 训练集制作/核查 | `174_export_pending.py`、`175_sync_labels_from_folder.py`、`179_cmp_candidate_vs_trainset.py`、`182_trainset_check_sheet.py`、`184_trainset_points_on_images.py` |
| CNN 训练 / 套用 / 叠加出图 | `111_train_cnn.py`、`168_train_cnn3.py`、`114_apply_cnn.py`、`121_cnn_overlays.py` |
| U-Net 训练（多种配方） | `199_train_unet_preproc.py`、`200_train_unet_A_canny.py`、`201_train_unet_contrast_norm.py`、`203_train_unet_multichan.py`、`204_train_unet_chan4.py`、`205_train_unet_cldice.py`、`206_train_unet_cldice_v2.py`、`207_train_unet_raw_cldice.py`、`210_train_raw_overlap50.py`、`211_train_prelinear_residual.py`、`212_train_premlp3.py`、`215_conv3_sig_resnorm.py`、`216_train_inverted.py` |
| 用全部标签重训 | `198_retrain_with_all_labels.py`、`199_train_unet_preproc.py` |
| 权重合并（TTA 成员等） | `208_tta_and_v12.py` |

## 三·补、安全提示（先读再跑）

1. **部分脚本没有命令行参数**（例如 `210_train_raw_overlap50.py`）：**直接运行就等于开始训练**，不要为了"看一眼帮助"而运行它。运行前先读脚本头部确认它会读写哪些目录。
2. **`210` 一类脚本的 `torch.save` 不带 tag**，直接覆盖 `models/unet/unet_v13_raw_overlap50.pt`。重训前**先备份该权重文件**，或手动给输出文件名加版本标记。

## 四、三条硬规矩

1. **写权重必须带 tag（版本标记）**，否则会覆盖 `models/` 里的既有权重——这是当前代码的已知隐患。
2. **新权重启用后必须跑基准验收**：10 张样品应仍为
   **读数 117 ｜ 弯曲 4 ｜ 断联 7 ｜ 裸焊盘 5**；数字变了必须人工逐项核对，不能直接发布。
3. **权重与 cfg 都可回滚**：替换文件即可回到旧版本。
