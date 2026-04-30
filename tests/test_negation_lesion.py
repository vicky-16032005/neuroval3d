from __future__ import annotations


def test_negation_self_match():
    from neuroval3d.validators import NegationValidator

    nv = NegationValidator()
    s = nv.score("There is marked oedema. No hemorrhage.", "There is marked oedema. No hemorrhage.")
    assert s == 1.0


def test_negation_polarity_flip():
    from neuroval3d.validators import NegationValidator

    nv = NegationValidator()
    s = nv.score("There is marked oedema.", "There is no oedema.")
    assert s < 0.5


def test_negation_extract_pairs():
    from neuroval3d.validators import NegationValidator

    nv = NegationValidator()
    pairs = nv.extract("No oedema, marked hemorrhage, mass effect present.")
    polarities = {term: neg for term, neg in pairs}
    assert polarities.get("oedema") is True
    assert polarities.get("hemorrhage") is False


def test_lesion_type_self_match():
    from neuroval3d.validators import LesionTypeValidator

    lv = LesionTypeValidator()
    s = lv.score("high-grade glioma in the frontal lobe", "consistent with a glioblastoma")
    assert s == 1.0  # both fall in the glioma family


def test_lesion_type_family_flip():
    from neuroval3d.validators import LesionTypeValidator

    lv = LesionTypeValidator()
    s = lv.score("high-grade glioma in the frontal lobe", "meningioma in the frontal lobe")
    assert s < 0.5
