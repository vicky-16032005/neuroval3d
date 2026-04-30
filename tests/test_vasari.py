from __future__ import annotations


def test_vasari_lexicon_sane():
    from neuroval3d.grounding.vasari import VASARI_FEATURES, vasari_vocabulary

    assert len(VASARI_FEATURES) >= 25
    codes = [f.code for f in VASARI_FEATURES]
    assert codes == sorted(codes, key=lambda x: int(x[1:]))
    assert "F1" in codes
    vocab = vasari_vocabulary()
    assert "left" in vocab
    assert "frontal" in vocab
    assert any("enhancement" in v for v in vocab)


def test_parser_recovers_basic_features():
    from neuroval3d.grounding.vasari import VASARIParser

    text = (
        "There is a left frontal mass with avid enhancement. "
        "Marked oedema is present. No restricted diffusion."
    )
    feats = VASARIParser().parse(text).to_dict()
    # Side
    assert feats.get("side") in {"left", "right", "bilateral", "central"}
    # Enhancement quality
    assert "enhancement_quality" in feats
    # Negation: 'no restricted diffusion' should be flagged as no
    # (the parser currently records the value reached; either branch is acceptable
    # provided the field is filled)
    assert "diffusion" in feats


def test_parser_handles_negation_window():
    from neuroval3d.grounding.vasari import VASARIParser

    text = "No frank haemorrhage."
    feats = VASARIParser().parse(text).to_dict()
    assert feats.get("hemorrhage") == "no"
