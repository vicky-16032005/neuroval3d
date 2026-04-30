from __future__ import annotations

from neuroval3d.evaluation.perturbation import (
    PerturbationOp,
    build_perturbation_set,
    perturb,
)


def test_each_perturbation_op_has_a_chance():
    text = (
        "FINDINGS:\n"
        "- There is a 3.5 cm intra-axial lesion in the left frontal lobe with avid enhancement.\n"
        "- Marked vasogenic oedema is present.\n"
        "- Two satellite lesions are noted.\n"
        "- No restricted diffusion on T1 or T2.\n"
        "IMPRESSION:\nLeft frontal high-grade glioma."
    )
    hits = 0
    for op in PerturbationOp:
        rec = perturb(text, op)
        if rec is not None and rec.perturbed != text:
            hits += 1
    # We expect most ops to find a candidate in this rich seed text.
    assert hits >= 5


def test_build_set_keeps_clean_originals():
    reports = [
        "Right frontal mass with marked enhancement and moderate oedema.",
        "Left cerebellar lesion measuring 2.0 cm without restricted diffusion.",
    ]
    pset = build_perturbation_set(reports, n_per_report=3, seed=0)
    cleans = [r for r in pset.records if r.op_detail == "<clean>"]
    assert len(cleans) == len(reports)
    assert len(pset.records) >= len(reports) * 2
