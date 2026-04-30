from __future__ import annotations


def test_paraphrase_is_deterministic():
    from neuroval3d.evaluation import paraphrase

    text = "There is a 3.5 cm intra-axial mass with avid enhancement and marked oedema."
    a = paraphrase(text, seed=42)
    b = paraphrase(text, seed=42)
    assert a == b


def test_paraphrase_preserves_length_roughly():
    from neuroval3d.evaluation import paraphrase

    text = "There is a 3.5 cm intra-axial mass with avid enhancement and marked oedema."
    out = paraphrase(text, seed=1)
    assert 0.5 < len(out) / len(text) < 2.0


def test_paraphrase_changes_something_when_substitutions_exist():
    from neuroval3d.evaluation import paraphrase

    text = "There is avid enhancement and marked oedema."
    out = paraphrase(text, seed=2, max_changes=3)
    # At minimum one of the canonical substitutions should kick in.
    assert out != text
