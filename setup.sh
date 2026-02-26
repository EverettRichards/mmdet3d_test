#!/usr/bin/env bash
# ============================================================
# setup.sh – Create and configure a conda environment for
#            MMDetection3D with CUDA 12.4 (compatible with
#            RTX 5060Ti via NVIDIA forward-compatibility).
#
# RTX 5060Ti (Blackwell, sm_120) notes:
#   - Requires NVIDIA driver >= 570 for CUDA 12.8 support.
#   - CUDA 12.4 runs on Blackwell through NVIDIA's forward-
#     compatibility package when driver >= 570 is installed.
#   - Native sm_120 CUDA-extension compilation (for custom ops)
#     requires CUDA 12.8+; see the optional section below.
#
# pkg_resources / setuptools note:
#   setuptools >= 82 removes the legacy pkg_resources namespace.
#   Several OpenMMLab packages still rely on pkg_resources, so
#   we pin setuptools < 82 and build mmcv with --no-build-isolation
#   to reuse the pinned setuptools instead of the isolated build env.
#
# numpy / opencv note:
#   mmdet3d 1.4.0 and mmcv 2.1.0 link against the NumPy 1.x C API.
#   Run with numpy<2.0. opencv-python >= 4.10 requires numpy>=2, so
#   we use opencv-python-headless<4.10 instead.
#
# mmcv version note:
#   mmdet3d 1.4.0 requires mmcv>=2.0.0rc4,<2.2.0 → use mmcv 2.1.0.
# ============================================================

set -e

ENV_NAME="${1:-mmdet3d}"
PYTHON_VERSION="3.10"

echo "==> Creating conda environment: ${ENV_NAME}"
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"

# Activate inside the script
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# -----------------------------------------------------------------
# 1.  Pin setuptools < 82 BEFORE anything else.
#     setuptools >= 82 drops pkg_resources, which breaks many
#     OpenMMLab install scripts that call pkg_resources.get_info().
# -----------------------------------------------------------------
echo "==> Pinning setuptools < 82"
pip install "setuptools<82" --upgrade --quiet

# -----------------------------------------------------------------
# 2.  PyTorch with CUDA 12.4 (satisfies CUDA 12.1+ requirement).
#     Wheel index ships cu124 builds for PyTorch 2.5.x.
# -----------------------------------------------------------------
echo "==> Installing PyTorch 2.5.1 + CUDA 12.4"
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# -----------------------------------------------------------------
# 3.  mmengine (no CUDA extensions – install from PyPI).
# -----------------------------------------------------------------
echo "==> Installing mmengine"
pip install "mmengine>=0.10.0"

# -----------------------------------------------------------------
# 4.  mmcv 2.1.0 – mmdet3d 1.4.0 requires mmcv>=2.0.0rc4,<2.2.0.
#
#     Use openmim (mim) to install mmcv.  openmim detects the active
#     PyTorch/CUDA version and fetches the matching prebuilt wheel
#     from the OpenMMLab CDN, ensuring the compiled _ext module is
#     included.  A plain 'pip install mmcv' from PyPI installs only
#     the source distribution, which requires a CUDA-capable compiler
#     and may silently omit the _ext extension (causing the error
#     "No module named 'mmcv._ext'").
# -----------------------------------------------------------------
echo "==> Installing openmim"
pip install openmim --quiet

echo "==> Installing mmcv 2.1.0 (prebuilt wheel via openmim)"
mim install "mmcv==2.1.0"

# -----------------------------------------------------------------
# 5.  MMDetection and MMDetection3D.
# -----------------------------------------------------------------
echo "==> Installing mmdet and mmdet3d"
pip install "mmdet>=3.2.0,<=3.3.0"
# mmdet3d's setup.py uses pkg_resources; use --no-build-isolation
# to pick up our pinned setuptools and avoid ModuleNotFoundError.
pip install mmdet3d==1.4.0 --no-build-isolation

# -----------------------------------------------------------------
# 6.  Runtime dependencies.
#
#     numpy<2.0  – mmcv/mmdet3d link against numpy 1.x C API.
#     opencv-python-headless<4.10 – >=4.10 requires numpy>=2.
# -----------------------------------------------------------------
echo "==> Installing runtime dependencies"
pip install \
    "numpy<2.0" \
    "opencv-python-headless<4.10" \
    scipy \
    scikit-learn \
    pyquaternion \
    shapely \
    matplotlib \
    pillow \
    pyyaml \
    tqdm \
    addict \
    yapf \
    termcolor \
    rich \
    open3d \
    "networkx>=2.2,<3.0" \
    nuscenes-devkit \
    lyft_dataset_sdk

# -----------------------------------------------------------------
# 7.  Verify pkg_resources is still importable.
# -----------------------------------------------------------------
echo "==> Verifying pkg_resources availability"
python - <<'PYEOF'
try:
    import pkg_resources
    print("  [OK] pkg_resources is available")
except ImportError as e:
    print(f"  [WARN] pkg_resources not available: {e}")
    print("  Hint: run  pip install 'setuptools<82'  to fix this.")
PYEOF

echo ""
echo "================================================================"
echo "  Setup complete!  Activate with:  conda activate ${ENV_NAME}"
echo "  Run verification: python verify_install.py"
echo "================================================================"

# ----  Optional: native Blackwell (sm_120) CUDA extension build  -----
# Uncomment if you need to compile CUDA ops natively on RTX 5060Ti:
#
#   CUDA_HOME=/usr/local/cuda-12.8 \
#   TORCH_CUDA_ARCH_LIST="12.0" \
#   pip install mmcv==2.1.0 --no-build-isolation --no-binary mmcv
#
# Ensure CUDA 12.8 toolkit is installed first:
#   conda install cuda-toolkit=12.8 -c nvidia
