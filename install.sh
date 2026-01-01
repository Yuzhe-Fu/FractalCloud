#!/usr/bin/env bash
set -e

echo "========== FractalCloud Environment Setup =========="

############################
# 1. Check conda
############################
if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda is not found on your system."
    echo "Please install Miniconda or Anaconda before continuing."
    echo "Miniconda download: https://docs.conda.io/en/latest/miniconda.html"
    return 1
fi

# Ensure conda is initialized
source "$(conda info --base)/etc/profile.d/conda.sh"

############################
# 2. Create conda env
############################
ENV_NAME=openpoints

if conda env list | grep -q "$ENV_NAME"; then
    echo "[INFO] Removing existing conda environment: $ENV_NAME"
    conda deactivate || true
    conda env remove -n $ENV_NAME -y
fi

echo "[INFO] Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME -y python=3.7 numpy=1.20 numba
conda activate $ENV_NAME

############################
# 3. Install PyTorch + CUDA
# if cuda version is not 11.3, please revise the script accordingly.
############################
echo "[INFO] Installing PyTorch with recommended CUDA runtime (11.3)"
conda install -y \
    pytorch=1.10.1 torchvision cudatoolkit=11.3 \
    -c pytorch -c nvidia

############################
# 4. Install torch-scatter
# if cuda version is not 11.3, please revise the script accordingly.
############################
echo "[INFO] Installing torch-scatter"
pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-1.10.1+cu113.html \
  --only-binary=:all:
# pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html

############################
# 5. Python dependencies
############################
echo "[INFO] Installing Python dependencies"
pip install -r requirements.txt

echo "[INFO] Installing gdown for artifact download"
pip install gdown

############################
# 6. Compile C++ extensions
# if cuda version is not 11.3, please revise the script accordingly.
############################
echo "[INFO] Building PointNet++ extensions"
cd openpoints/cpp/pointnet2_batch
python setup.py install
cd -

echo "========== Great! Installation Complete =========="
