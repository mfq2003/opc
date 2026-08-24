#!/usr/bin/env bash
# 本脚本在 Ubuntu 云端创建隔离虚拟环境、固定 OpenILT 提交并安装 GPU 实验依赖。
# 输入为项目根目录和可访问公网的 Python 3.8/ git 环境；输出为 .venv 与 third_party/OpenILT。
# 脚本不会读取 .env、写入密钥、安装系统包或修改 OpenILT 上游源码。
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENILT_DIR="$PROJECT_ROOT/third_party/OpenILT"
OPENILT_REPOSITORY="https://github.com/OpenOPC/OpenILT.git"
OPENILT_COMMIT="dabb97c6ca3dfd159362e48273c436444c77353b"

command -v git >/dev/null || { echo "缺少 git" >&2; exit 1; }
command -v python3.8 >/dev/null || { echo "缺少 python3.8；请先由管理员安装" >&2; exit 1; }

cd "$PROJECT_ROOT"
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==23.3.2 setuptools==68.2.2 wheel==0.41.3
python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.0.1+cu118 torchvision==0.15.2+cu118

if [[ ! -d "$OPENILT_DIR/.git" ]]; then
  mkdir -p "$(dirname "$OPENILT_DIR")"
  git clone "$OPENILT_REPOSITORY" "$OPENILT_DIR"
fi
git -C "$OPENILT_DIR" fetch --depth 1 origin "$OPENILT_COMMIT"
git -C "$OPENILT_DIR" checkout --detach "$OPENILT_COMMIT"
test "$(git -C "$OPENILT_DIR" rev-parse HEAD)" = "$OPENILT_COMMIT"

python -m pip install -r requirements-lock.txt
python -m pip install -e "$OPENILT_DIR/thirdparty/adaptive-boxes"
python -m pip install -e .
python - <<'PY'
import torch
assert torch.cuda.is_available(), "未检测到 CUDA；请核查 GPU 驱动与 CUDA 版 PyTorch"
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
PY
echo "部署完成：$PROJECT_ROOT"

