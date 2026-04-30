"""End-to-end smoke test: import every module, exercise the perturbation benchmark on a tiny set."""
from __future__ import annotations


def test_import_all():
    import neuroval3d  # noqa: F401
    from neuroval3d import cli  # noqa: F401
    from neuroval3d.data import (  # noqa: F401
        BrainMRIDatasetSpec,
        DatasetRegistry,
        PreprocessingConfig,
        Stage1Preprocessor,
        SyntheticReportGenerator,
    )
    from neuroval3d.evaluation import (  # noqa: F401
        BenchmarkResult,
        PerturbationOp,
        build_perturbation_set,
        compute_nlg_metrics,
        run_benchmark,
    )
    from neuroval3d.grounding import (  # noqa: F401
        VASARI_FEATURES,
        AnatomicalAnchorer,
        VASARIParser,
    )
    from neuroval3d.models import build_encoder, MLPProjector, ReportDecoder  # noqa: F401
    from neuroval3d.utils import CheckpointManager, get_logger, save_checkpoint  # noqa: F401
    from neuroval3d.validators import (  # noqa: F401
        FusionValidator,
        LexicalValidator,
        SemanticValidator,
        StructuralValidator,
    )
    from neuroval3d.viz import GradCAM3D, save_triptych  # noqa: F401


def test_synthetic_report_roundtrip(small_seg_mask):
    from neuroval3d.data import SyntheticReportGenerator
    from neuroval3d.grounding import VASARIParser

    gen = SyntheticReportGenerator()
    rep = gen.from_mask(small_seg_mask)
    assert "FINDINGS:" in rep.text
    assert "IMPRESSION:" in rep.text

    parser = VASARIParser()
    feats = parser.parse(rep.text)
    assert feats.values  # at least one VASARI feature recovered


def test_validator_scores_are_bounded(report_pair):
    from neuroval3d.validators import LexicalValidator, SemanticValidator, StructuralValidator

    gen, ref = report_pair
    sem = SemanticValidator().score(gen, ref)
    lex = LexicalValidator().fit([gen, ref]).score(gen, ref)
    st = StructuralValidator().score(gen, ref)
    for x in (sem, lex, st):
        assert 0.0 <= x <= 1.0 + 1e-6


def test_perturbation_benchmark_runs():
    from neuroval3d.evaluation import run_benchmark

    result = run_benchmark(use_synthetic=True, n_samples=8, n_per_report=2)
    assert "fusion" in result.auroc_overall
    assert result.n_records > 0
