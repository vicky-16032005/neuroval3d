from __future__ import annotations

from pathlib import Path

import pytest


def _radgenome_present() -> bool:
    return Path("data/raw/RadGenome-BrainMRI/BraTS_GLI/global_finding.json").exists()


def test_radgenome_loader_subsets():
    if not _radgenome_present():
        pytest.skip("RadGenome reports not downloaded; run scripts/download_radgenome_reports.py")
    from neuroval3d.data import load_radgenome
    records = load_radgenome(section="global_finding")
    subsets = {r["subset"] for r in records}
    # Expect at least 4 of the 5 disease subsets
    assert len(subsets) >= 4
    for r in records[:5]:
        assert r["report"]
        assert r["subset"] in {"BraTS_GLI", "BraTS_MEN", "BraTS_MET", "ISLES22", "WMH"}
        assert r["disease"] in {"glioma", "meningioma", "metastasis", "infarction",
                                "white_matter_hyperintensity"}


def test_radgenome_modality_keywords_present():
    if not _radgenome_present():
        pytest.skip("RadGenome reports not downloaded")
    from neuroval3d.data import radgenome_reports_only
    reports = radgenome_reports_only(limit=20)
    text = " ".join(reports)
    # RadGenome reports should mention modality keywords
    assert any(m in text for m in ("T1", "T2", "FLAIR"))


def test_radgenome_section_switch():
    if not _radgenome_present():
        pytest.skip("RadGenome reports not downloaded")
    from neuroval3d.data import load_radgenome
    findings = load_radgenome(section="global_finding")
    impressions = load_radgenome(section="impression")
    assert len(findings) >= len(impressions) > 0
    # Different section → different text
    if findings and impressions:
        f0 = next(iter(findings))
        i0 = next((r for r in impressions if r["subject_id"] == f0["subject_id"]), None)
        if i0:
            assert f0["report"] != i0["report"]
