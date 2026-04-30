from __future__ import annotations


def test_ratescore_lite_self_match_high():
    from neuroval3d.validators import RaTEScoreLite

    rs = RaTEScoreLite()
    assert rs.score("a b c d", "a b c d") == 1.0


def test_ratescore_lite_disjoint_zero():
    from neuroval3d.validators import RaTEScoreLite

    rs = RaTEScoreLite()
    assert rs.score("apple banana", "carrot daikon") == 0.0


def test_extended_vasari_vocab_has_modality_terms():
    from neuroval3d.grounding.vasari import vasari_vocabulary

    vocab = vasari_vocabulary()
    assert "t1" in vocab
    assert "t2" in vocab
    assert "flair" in vocab
    assert "dwi" in vocab
    # numerics + counts
    assert "cm" in vocab
    assert "two" in vocab
    assert "multiple" in vocab
