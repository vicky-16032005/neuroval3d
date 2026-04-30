from __future__ import annotations

import numpy as np

from neuroval3d.validators import FusionValidator, LexicalValidator, SemanticValidator, StructuralValidator


def test_lexical_negation_is_penalized(report_pair):
    gen, _ = report_pair
    flipped = gen.replace("No restricted diffusion", "Restricted diffusion is present")
    flipped = flipped.replace("no hemorrhage", "areas of hemorrhage")
    lex = LexicalValidator().fit([gen, flipped])
    s_clean = lex.score(gen, gen)
    s_flip = lex.score(gen, flipped)
    assert s_clean >= s_flip


def test_structural_f1_self_is_perfect(report_pair):
    gen, _ = report_pair
    st = StructuralValidator()
    assert st.score(gen, gen) >= 0.95


def test_fusion_logistic_trains(report_pair):
    gen, ref = report_pair
    sem = SemanticValidator().score(gen, ref)
    lex = LexicalValidator().fit([gen, ref]).score(gen, ref)
    st = StructuralValidator().score(gen, ref)

    rng = np.random.default_rng(0)
    sub_scores = []
    labels = []
    for _ in range(40):
        sub_scores.append((sem * rng.uniform(0.7, 1.0), lex * rng.uniform(0.7, 1.0), st * rng.uniform(0.7, 1.0)))
        labels.append(1)
        sub_scores.append((sem * rng.uniform(0.0, 0.3), lex * rng.uniform(0.0, 0.3), st * rng.uniform(0.0, 0.3)))
        labels.append(0)

    fusion = FusionValidator().fit(sub_scores, labels)
    pos = fusion.predict(sem, lex, st).fused
    neg = fusion.predict(sem * 0.1, lex * 0.1, st * 0.1).fused
    assert pos > neg
