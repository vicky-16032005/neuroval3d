"""`neuroval3d` CLI entrypoint."""
from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from neuroval3d import __version__
from neuroval3d.utils.logging import get_logger

app = typer.Typer(no_args_is_help=True, add_completion=False)
log = get_logger("cli")


@app.command()
def version() -> None:
    """Print the installed NeuroVal-3D version."""
    rprint(f"neuroval3d v{__version__}")


@app.command()
def info() -> None:
    """Print environment + import status for each stage."""
    import importlib

    rprint(f"[bold]NeuroVal-3D[/bold] v{__version__}")
    for mod in [
        "neuroval3d.data.preprocessing",
        "neuroval3d.grounding.vasari",
        "neuroval3d.validators.lexical",
        "neuroval3d.validators.semantic",
        "neuroval3d.validators.fusion",
        "neuroval3d.evaluation.perturbation",
    ]:
        try:
            importlib.import_module(mod)
            rprint(f"  [green]OK[/green]  {mod}")
        except Exception as e:  # noqa: BLE001
            rprint(f"  [red]FAIL[/red] {mod} :: {e!s}")


@app.command()
def benchmark(
    synthetic: bool = typer.Option(False, help="Use built-in synthetic reports (no download)."),
    textbrats: bool = typer.Option(False, help="Use real TextBraTS reports from data/raw/TextBraTS/."),
    radgenome: bool = typer.Option(False, help="Use real RadGenome-Brain MRI reports."),
    n_samples: int = typer.Option(120, help="Max base reports to perturb (cap)."),
    train_frac: float = typer.Option(0.7, help="Fraction of base reports for fusion training."),
    out_dir: Path = typer.Option(Path("outputs/results"), help="Where to write the AUROC table."),
) -> None:
    """Run the perturbation benchmark with held-out evaluation."""
    from neuroval3d.evaluation.benchmark import run_benchmark

    out_dir.mkdir(parents=True, exist_ok=True)

    if textbrats:
        from neuroval3d.data import textbrats_reports_only
        reports = textbrats_reports_only(limit=n_samples)
        log.info("Running held-out benchmark on %d TextBraTS reports", len(reports))
        result = run_benchmark(reports=reports, use_synthetic=False, n_samples=len(reports),
                               train_frac=train_frac, out_dir=out_dir)
    elif radgenome:
        from neuroval3d.data import radgenome_reports_only
        reports = radgenome_reports_only(limit=n_samples)
        log.info("Running held-out benchmark on %d RadGenome reports", len(reports))
        result = run_benchmark(reports=reports, use_synthetic=False, n_samples=len(reports),
                               train_frac=train_frac, out_dir=out_dir)
    else:
        log.info("Running synthetic benchmark (n=%d)", n_samples)
        result = run_benchmark(use_synthetic=True, n_samples=n_samples,
                               train_frac=train_frac, out_dir=out_dir)
    rprint(result.summary_table())


@app.command()
def cross_dataset(
    train: str = typer.Option(..., help="Train dataset: textbrats | radgenome | synthetic"),
    test: str = typer.Option(..., help="Test dataset: textbrats | radgenome | synthetic"),
    n_train: int = typer.Option(369, help="Cap on train base reports."),
    n_test: int = typer.Option(369, help="Cap on test base reports."),
    out_dir: Path = typer.Option(Path("outputs/results"), help="Where to write."),
) -> None:
    """Cross-dataset transfer: train fusion on one dataset's perturbations, evaluate on another."""
    from neuroval3d.evaluation.benchmark import run_cross_dataset_benchmark

    def _load(name: str, n: int):
        if name == "textbrats":
            from neuroval3d.data import textbrats_reports_only
            return textbrats_reports_only(limit=n)
        if name == "radgenome":
            from neuroval3d.data import radgenome_reports_only
            return radgenome_reports_only(limit=n)
        if name == "synthetic":
            from neuroval3d.evaluation.benchmark import _synthetic_reports
            return _synthetic_reports(n)
        raise ValueError(f"unknown dataset: {name}")

    train_reports = _load(train, n_train)
    test_reports = _load(test, n_test)
    log.info("Cross-dataset: train=%s (%d) → test=%s (%d)",
             train, len(train_reports), test, len(test_reports))
    result = run_cross_dataset_benchmark(
        train_reports=train_reports, test_reports=test_reports,
        train_label=train, test_label=test, out_dir=out_dir,
    )
    rprint(result.summary_table())


@app.command()
def vasari_demo(report: str = typer.Argument(..., help="Free-text radiology report")) -> None:
    """Parse a free-text report into VASARI features."""
    from neuroval3d.grounding.vasari import VASARIParser

    parser = VASARIParser()
    feats = parser.parse(report)
    rprint(feats.to_dict())


if __name__ == "__main__":
    app()
