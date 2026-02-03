#!/usr/bin/env bash
set -e

ENV_NAME=rfm
PYTHON_VERSION=3.10

echo "=== Create conda environment: ${ENV_NAME} (python ${PYTHON_VERSION}) ==="

# conda가 없는 경우 대비
if ! command -v conda &> /dev/null; then
  echo "conda not found. Please install Miniconda/Anaconda first."
  exit 1
fi

# 기존 env 있으면 경고
if conda env list | grep -q "^${ENV_NAME}\s"; then
  echo "Environment '${ENV_NAME}' already exists."
  echo "Remove it with: conda remove -n ${ENV_NAME} --all"
  exit 1
fi

# env 생성
conda create -y -n ${ENV_NAME} python=${PYTHON_VERSION}

echo "=== Activate environment ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo "=== Upgrade pip ==="
python -m pip install --upgrade pip

echo "=== Install core numeric & vision packages ==="
pip install \
  numpy \
  scipy \
  matplotlib \
  opencv-python

echo "=== Install PyTorch (CUDA if available) ==="
# CUDA 11.8 기준 (A6000, RTX30/40 계열에 무난)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU만 쓸 경우 아래로 대체 가능
# pip install torch torchvision torchaudio

echo "=== Install ROS Python bridge packages ==="
# rclpy 자체는 ROS에서 제공 → conda에 설치하지 않음
pip install \
  cv_bridge \
  transforms3d

echo "=== Install misc utilities ==="
pip install \
  tqdm \
  pyyaml \
  easydict

echo "=== DONE ==="
echo ""
echo "Next steps:"
echo "  conda activate ${ENV_NAME}"
echo "  source /opt/ros/humble/setup.bash"
echo "  python your_script.py"

