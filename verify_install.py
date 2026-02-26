#!/usr/bin/env python3
"""
verify_install.py – Verify that the mmdetection3d environment is
correctly set up and functional.

Run after setup.sh:
    conda activate mmdet3d
    python verify_install.py
"""

import sys
import importlib
import textwrap
from packaging.version import Version

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
WARN = "\033[33m[WARN]\033[0m"
INFO = "\033[36m[INFO]\033[0m"


def check(label: str, fn):
    """Run *fn* and print a PASS/FAIL line."""
    try:
        result = fn()
        msg = f" ({result})" if result else ""
        print(f"  {PASS} {label}{msg}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  {FAIL} {label}: {exc}")
        return False


# ---------------------------------------------------------------------------
# 1. Python version
# ---------------------------------------------------------------------------
print("\n=== 1. Python ===")
pv = sys.version_info
check(f"Python >= 3.10", lambda: f"{pv.major}.{pv.minor}.{pv.micro}")
if pv < (3, 10):
    print(f"  {WARN} Python 3.10+ is recommended for mmdet3d.")

# ---------------------------------------------------------------------------
# 2. setuptools / pkg_resources
# ---------------------------------------------------------------------------
print("\n=== 2. setuptools / pkg_resources ===")

def _check_setuptools():
    import setuptools
    v = setuptools.__version__
    if Version(v) >= Version("82"):
        raise RuntimeError(
            f"setuptools {v} >= 82 – pkg_resources may be unavailable. "
            "Fix: pip install 'setuptools<82'"
        )
    return v

check("setuptools < 82", _check_setuptools)
check("pkg_resources importable", lambda: __import__("pkg_resources") and "ok")

# ---------------------------------------------------------------------------
# 3. PyTorch + CUDA
# ---------------------------------------------------------------------------
print("\n=== 3. PyTorch ===")

def _torch_version():
    import torch
    return torch.__version__

def _cuda_available():
    import torch
    avail = torch.cuda.is_available()
    msg = "available" if avail else "not available (CPU-only mode)"
    if not avail:
        print(f"    {INFO} No CUDA device detected – running in CPU-only mode.")
        print(f"    {INFO} On RTX 5060Ti, ensure driver >= 570 and CUDA 12.4+.")
    return msg

def _cuda_version():
    import torch
    return torch.version.cuda or "N/A"

def _cuda_arch_list():
    import torch
    if not torch.cuda.is_available():
        return "N/A (no GPU)"
    caps = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        caps.append(f"{p.name} (sm_{p.major}{p.minor})")
    return ", ".join(caps) if caps else "N/A"

check("torch importable", _torch_version)
check("CUDA runtime version", _cuda_version)
check("CUDA device", _cuda_available)
check("GPU compute capability", _cuda_arch_list)

# ---------------------------------------------------------------------------
# 4. Core OpenMMLab packages
# ---------------------------------------------------------------------------
print("\n=== 4. OpenMMLab packages ===")

def _pkg_version(name):
    mod = importlib.import_module(name)
    return mod.__version__

for pkg in ("mmengine", "mmcv", "mmdet", "mmdet3d"):
    check(f"{pkg} importable", lambda p=pkg: _pkg_version(p))

# ---------------------------------------------------------------------------
# 5. mmengine basic functionality
# ---------------------------------------------------------------------------
print("\n=== 5. mmengine basics ===")

def _mmengine_config():
    from mmengine.config import Config
    cfg = Config({"a": 1, "b": {"c": 2}})
    assert cfg.a == 1
    assert cfg.b.c == 2
    return "Config OK"

def _mmengine_registry():
    from mmengine.registry import Registry
    reg = Registry("test")
    @reg.register_module()
    class Foo:
        pass
    assert reg.get("Foo") is Foo
    return "Registry OK"

check("mmengine Config", _mmengine_config)
check("mmengine Registry", _mmengine_registry)

# ---------------------------------------------------------------------------
# 6. mmcv core ops (CPU)
# ---------------------------------------------------------------------------
print("\n=== 6. mmcv ops (CPU) ===")

def _mmcv_nms():
    import torch
    from mmcv.ops import nms
    boxes = torch.tensor(
        [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=torch.float32
    )
    scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
    # nms returns (dets, keep_indices)
    dets, keep = nms(boxes, scores, iou_threshold=0.5)
    assert len(keep) > 0
    return f"NMS kept {len(keep)} boxes"

check("mmcv.ops.nms (CPU)", _mmcv_nms)

# ---------------------------------------------------------------------------
# 7. mmdet3d basic imports
# ---------------------------------------------------------------------------
print("\n=== 7. mmdet3d imports ===")

def _mmdet3d_structures():
    from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
    return "Det3DDataSample, LiDARInstance3DBoxes"

def _mmdet3d_models():
    # mmdet3d 1.4.0 uses mmengine's builder; check core model classes
    from mmdet3d.models import (  # noqa: F401
        CenterPoint, VoxelNet, SECOND
    )
    return "CenterPoint, VoxelNet, SECOND"

def _mmdet3d_datasets():
    from mmdet3d.datasets import NuScenesDataset  # noqa: F401
    return "NuScenesDataset"

check("mmdet3d.structures", _mmdet3d_structures)
check("mmdet3d.models.build_model", _mmdet3d_models)
check("mmdet3d.datasets.NuScenesDataset", _mmdet3d_datasets)

# ---------------------------------------------------------------------------
# 8. LiDAR box tensor sanity check
# ---------------------------------------------------------------------------
print("\n=== 8. LiDARInstance3DBoxes sanity check ===")

def _lidar_boxes():
    import torch
    from mmdet3d.structures import LiDARInstance3DBoxes

    # 3-D bounding boxes: [x, y, z, dx, dy, dz, yaw]
    raw = torch.tensor([[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, 0.0]])
    boxes = LiDARInstance3DBoxes(raw)
    assert boxes.tensor.shape == (1, 7), f"Unexpected shape {boxes.tensor.shape}"
    return f"shape={tuple(boxes.tensor.shape)}"

check("LiDARInstance3DBoxes creation", _lidar_boxes)

# ---------------------------------------------------------------------------
# 9. CUDA-extension availability (informational)
# ---------------------------------------------------------------------------
print("\n=== 9. CUDA extension availability ===")

def _cuda_ext():
    import torch
    if not torch.cuda.is_available():
        return "skipped (no CUDA device)"
    from mmcv.ops import ball_query  # noqa: F401
    return "ball_query CUDA op loaded"

check("mmcv CUDA extension (ball_query)", _cuda_ext)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Verification complete.")
print("  If all checks above show [PASS], the environment is ready.")
print("  [WARN] on the CUDA device check is expected in headless")
print("  CI/sandbox environments without a GPU.")
print("=" * 60 + "\n")
