from __future__ import annotations


def test_numeric_extracts_cm_and_mm():
    from neuroval3d.validators import NumericValidator

    nv = NumericValidator()
    s_self = nv.score("a 3.5 cm mass and a 12 mm nodule",
                      "a 3.5 cm mass and a 12 mm nodule")
    assert s_self == 1.0

    s_size_flip = nv.score("a 3.5 cm mass", "a 1.0 cm mass")
    assert s_size_flip < 0.5

    s_unit_match = nv.score("a 3.5 cm mass", "a 35 mm mass")  # equivalent: 35mm == 3.5cm
    assert s_unit_match == 1.0


def test_numeric_handles_no_measurements():
    from neuroval3d.validators import NumericValidator

    nv = NumericValidator()
    assert nv.score("no measurements", "still no measurements") == 1.0
    assert nv.score("a 3 cm mass", "no measurements") == 0.0


def test_modality_detects_t1_t2_flair():
    from neuroval3d.validators import ModalityValidator

    mv = ModalityValidator()
    assert "t1" in mv.detect("the T1 sequence demonstrates")
    assert "t2" in mv.detect("on T2-weighted imaging")
    assert "flair" in mv.detect("FLAIR hyperintensity is present")
    assert "dwi" in mv.detect("diffusion-weighted imaging shows")


def test_modality_score_jaccard():
    from neuroval3d.validators import ModalityValidator

    mv = ModalityValidator()
    assert mv.score("T1 and T2 imaging", "T1 and T2 imaging") == 1.0
    assert mv.score("T1 imaging", "T2 imaging") == 0.0
    assert 0.0 < mv.score("T1 and T2", "T1 only") < 1.0
