#!/usr/bin/env bash
# 芯片质检（焊点与飞线检查）· Linux 启动
# 解释器三级探测：CHIP_PY 环境变量 → 本机配置 ./本地配置.env → 自动找带 cv2 的 python
set -u
cd "$(dirname "$0")"
[ -f ./本地配置.env ] && . ./本地配置.env
PORT="${PORT:-8000}"
PY="${CHIP_PY:-}"
if [ -z "$PY" ]; then
  for c in python3 python "$HOME/.local/bin/python3"; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import numpy,cv2,torch" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "[ERROR] 找不到同时装有 numpy/opencv-contrib-python/torch 的 Python。"
  echo "        请先安装依赖，然后任选一种方式指定解释器："
  echo "        1) export CHIP_PY=/path/to/python    2) 在 ./本地配置.env 写 CHIP_PY=/path/to/python"
  exit 1
fi
echo "使用解释器：$PY"
[ -z "${BROWSER:-}" ] || true
echo "打开浏览器访问 http://127.0.0.1:$PORT  （Ctrl+C 停止）"
exec "$PY" ui/app.py --port "$PORT"
