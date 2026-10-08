# core/vendor/ — 内置冻结库（原样拷贝 ✓ 不许改 ✗）

- `pipeline.py`：焊点候选链（S1–S6 + 弱反差分支 ✓）sha256_16 = f79d916fd54ca325
- `seg.py`：板框掩膜（CV 轮廓法：σ6 → Otsu → 闭运算25 → 开运算5 → 最大轮廓 ✓）sha256_16 = fc0fe25088123f3e
- 来源：<项目根>/chip_wire_ng/lib/（定格于版本 V-20261005-003 ✓）

改参数请改 `cfg/default.yaml` ✓，不要改这里的文件 ✗（改了就无法与定格版本对账 ✗）。

- 2026-10-06：`pipeline.py` 的 `SRC` 由外部绝对路径改为软件内相对路径 `chip_samples__20260128`；软件实际读图走 `core/mask.py` + `core/run_io.py`，不依赖该常量。
