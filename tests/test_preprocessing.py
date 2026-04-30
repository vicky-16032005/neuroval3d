from __future__ import annotations

import numpy as np


def test_preprocessor_handles_missing_files():
    from neuroval3d.data import PreprocessingConfig, Stage1Preprocessor

    cfg = PreprocessingConfig(target_shape=(64, 64, 64), bias_correct=False)
    pre = Stage1Preprocessor(cfg)
    paths = {
        "T1": "missing/T1.nii.gz",
        "T1ce": "missing/T1ce.nii.gz",
        "T2": "missing/T2.nii.gz",
        "FLAIR": "missing/FLAIR.nii.gz",
    }
    out = pre.run(paths)
    assert out.volume.shape == (4, 64, 64, 64)
    assert out.volume.dtype == np.float32


def test_preprocessor_reshapes_oversize_input(tmp_path):
    pytest_skip_if_no_nibabel()
    import nibabel as nib

    from neuroval3d.data import PreprocessingConfig, Stage1Preprocessor

    base = tmp_path
    paths: dict[str, str] = {}
    for m in ("T1", "T1ce", "T2", "FLAIR"):
        arr = np.random.RandomState(0).rand(160, 160, 160).astype("float32")
        p = base / f"{m}.nii.gz"
        nib.save(nib.Nifti1Image(arr, affine=np.eye(4)), str(p))
        paths[m] = str(p)
    pre = Stage1Preprocessor(PreprocessingConfig(target_shape=(96, 96, 96), bias_correct=False))
    out = pre.run(paths)
    assert out.volume.shape == (4, 96, 96, 96)


def pytest_skip_if_no_nibabel() -> None:
    try:
        import nibabel  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("nibabel not installed")
