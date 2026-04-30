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
    n_samples: int = typer.Option(120, help="Max base reports to perturb (cap)."),
    out_dir: Path = typer.Option(Path("outputs/results"), help="Where to write the AUROC table."),
) -> None:
    """Run the perturbation benchmark and print the AUROC table."""
    from neuroval3d.evaluation.benchmark import run_benchmark

    out_dir.mkdir(parents=True, exist_ok=True)

    if textbrats:
        from neuroval3d.data import textbrats_reports_only
        reports = textbrats_reports_only(limit=n_samples)
        log.info("Running benchmark on %d real TextBraTS reports", len(reports))
        result = run_benchmark(reports=reports, use_synthetic=False, n_samples=len(reports),
                               out_dir=out_dir)
    else:
        log.info("Running benchmark (synthetic=%s, n=%d)", synthetic or True, n_samples)
        result = run_benchmark(use_synthetic=synthetic or True, n_samples=n_samples, out_dir=out_dir)
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
