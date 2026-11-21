#!/usr/bin/env bash
# Set up all model envs, repos and weights.
#
#   ./setup.sh cuda     # GPU servers (CUDA 12.8 torch wheels + flash-attn & friends)
#   ./setup.sh cpu      # CPU-only machines (CPU torch wheels, no CUDA kernels)
#
# Needs: git, ffmpeg, an `hf auth login` session, and ~45 GB of disk.
set -euo pipefail
MODE="${1:-cuda}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

if [[ "$MODE" == "cuda" ]]; then
  TORCH_INDEX=(--index-url https://download.pytorch.org/whl/cu128)
else
  TORCH_INDEX=(--index-url https://download.pytorch.org/whl/cpu)
fi

# ---------------------------------------------------------------- upstream repos
mkdir -p third_party weights envs logs
clone() { [[ -d "third_party/$2" ]] || git clone -q "$1" "third_party/$2"; git -C "third_party/$2" checkout -q "$3"; }
clone https://github.com/NVlabs/alpamayo2-edge          alpamayo2-edge 02f55a7
clone https://github.com/QwenLM/Qwen-Drive-1.0           Qwen-Drive-1.0 28091c1
clone https://github.com/ahydchh/Impromptu-VLA           Impromptu-VLA  HEAD
clone https://github.com/Applied-Intuition-Open-Source/nord nord        HEAD

# ---------------------------------------------------------------- core (renderer)
uv venv -q --allow-existing --python 3.12 .venv
uv pip install -q --python .venv/bin/python numpy opencv-python-headless pillow

# ---------------------------------------------------------------- Alpamayo 2 Edge
# Upstream lockfile (torch 2.8.0 + CUDA, transformers 5.17.0). flash extra on GPU.
( cd third_party/alpamayo2-edge && UV_PROJECT_ENVIRONMENT=.venv \
    uv sync --locked $([[ "$MODE" == "cuda" ]] && echo --extra flash) )

# ---------------------------------------------------------------- Qwen-Drive-1.0
# Upstream requirements.txt pins (Python 3.12, CUDA 12.8). The CUDA-only kernels
# (flash-attn, causal-conv1d, flash-linear-attention) are skipped in cpu mode;
# transformers then uses its torch fallback for the linear-attention layers.
QD=third_party/Qwen-Drive-1.0
uv venv -q --allow-existing --python 3.12 "$QD/.venv"
uv pip install -q --python "$QD/.venv/bin/python" "${TORCH_INDEX[@]}" torch==2.8.0 torchvision==0.23.0
uv pip install -q --python "$QD/.venv/bin/python" transformers==5.14.1 accelerate==1.12.0 \
  safetensors==0.8.0 numpy==2.2.6 pillow==12.0.0 opencv-python-headless matplotlib==3.10.7 \
  tqdm==4.67.1 pyarrow
if [[ "$MODE" == "cuda" ]]; then
  uv pip install -q --python "$QD/.venv/bin/python" --no-build-isolation \
    flash-attn==2.8.3 causal-conv1d==1.6.2.post1 flash-linear-attention==0.5.1
fi
uv pip install -q --python "$QD/.venv/bin/python" --no-deps -e "$QD"

# ---------------------------------------------------------------- Qwen2.5-VL (Impromptu + NoRD)
uv venv -q --allow-existing --python 3.12 envs/qwen25vl
uv pip install -q --python envs/qwen25vl/bin/python "${TORCH_INDEX[@]}" torch==2.8.0 torchvision==0.23.0
uv pip install -q --python envs/qwen25vl/bin/python transformers==4.57.1 accelerate pillow numpy
if [[ "$MODE" == "cuda" ]]; then
  uv pip install -q --python envs/qwen25vl/bin/python --no-build-isolation flash-attn==2.8.3
fi
uv pip install -q --python envs/qwen25vl/bin/python --no-deps -e third_party/nord

# ---------------------------------------------------------------- weights (~36 GB)
hf download nvidia/Alpamayo2-Edge --revision f8951d095de3589b3face1dec3f4394c8ecedb38 \
  --local-dir weights/Alpamayo2-Edge
hf download Qwen/Qwen-Drive-1.0-4B --local-dir weights/Qwen-Drive-1.0-4B
# Impromptu ships DeepSpeed optimizer states too (~38 GB); fetch only the weights.
hf download aaaaaap/ImpromptuVLAModel --local-dir weights/ImpromptuVLA \
  --include "3B_AD_finetune/*.json" --include "3B_AD_finetune/*.txt" \
  --include "3B_AD_finetune/*.safetensors"
hf download AppliedIntuitionResearch/nord --local-dir weights/nord

echo "setup complete ($MODE)"
