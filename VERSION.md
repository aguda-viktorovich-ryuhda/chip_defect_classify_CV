# chip_defect_classify 版本

- 算法版本号：V-20261005-003（流水线算法版本定格）
  ./versions/V-20261005-003-pipeline-algorithm-versions.md
- 交付口径：V-20261006-001（三成员并集 + 7 个 U-Net 权重；含 TTA 0.5% 差异的如实记录）
  ./versions/V-20261006-001-union-members-and-weights.md
- U-Net 权重（7 个，models/unet/）：TTA = unet_v3_unet8(raw) / unet_v6_A_pct(pct) / unet_v6_B_clahe(clahe) / unet_v10_cldice3(pct) / unet_v11_raw_cldice(raw)；v13；v18
- CNN 权重（25 个，models/cnn/）：来自 cnn_models/20261005_1521_1005union
- 参数：cfg/default.yaml（唯一来源）
- 训练接口：保留但不开启（train/，需 --enable-training）
- 血统总表：unet/reports/WEIGHTS_PROVENANCE.md（182 个 .pt 全清点）
- 本次写入时间：2026-10-06 10:31

- 新增：V-20261006-002（至 S4 的算法版本固化 + 代码备份 backups/chip_defect_classify_20261006_1100 + 裸焊盘剔除判据 甲）| 2026-10-06 11:00
