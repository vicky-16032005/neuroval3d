from __future__ import annotations

import pytest


def test_textbrats_loader_smoke():
    from pathlib import Path

    if not Path("data/raw/TextBraTS/reports").exists():
        pytest.skip("TextBraTS reports not downloaded yet — run scripts/download_textbrats_reports.py")

    from neuroval3d.data import load_textbrats, textbrats_reports_only

    records = load_textbrats(limit=5)
    assert len(records) == 5
    for r in records:
        assert r["subject_id"].startswith("BraTS20_Training_")
        assert r["report"]
        assert r["source"] == "TextBraTS"
        assert r["license"] == "MIT"

    flat = textbrats_reports_only(limit=10)
    assert len(flat) == 10
    assert all(isinstance(r, str) for r in flat)


def test_textbrats_loader_missing_root_raises(tmp_path):
    from neuroval3d.data import load_textbrats

    with pytest.raises(FileNotFoundError):
        load_textbrats(root=tmp_path / "nonexistent")
