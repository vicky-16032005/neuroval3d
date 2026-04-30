from __future__ import annotations


def test_synthetic_report_is_deterministic(small_seg_mask):
    from neuroval3d.data import SyntheticReportGenerator

    gen = SyntheticReportGenerator()
    a = gen.from_mask(small_seg_mask).text
    b = gen.from_mask(small_seg_mask).text
    assert a == b


def test_synthetic_report_handles_empty_mask():
    import numpy as np
    from neuroval3d.data import SyntheticReportGenerator

    empty = np.zeros((32, 32, 32), dtype=np.int16)
    rep = SyntheticReportGenerator().from_mask(empty).text
    assert "Unremarkable" in rep or "No discrete" in rep
