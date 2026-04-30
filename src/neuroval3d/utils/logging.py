"""Console + file logging via rich, with run-id stamps."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.logging import RichHandler

_DEFAULT_LOG_DIR = Path("outputs/logs")
_CONFIGURED: dict[str, logging.Logger] = {}


def get_logger(name: str = "neuroval3d", log_dir: Path | None = None) -> logging.Logger:
    if name in _CONFIGURED:
        return _CONFIGURED[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    rich_handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rich_handler)

    log_dir = log_dir or _DEFAULT_LOG_DIR
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(log_dir / f"{name}_{run_id}.log", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(file_handler)
    except OSError as e:  # noqa: BLE001
        print(f"[neuroval3d] could not attach file logger: {e}", file=sys.stderr)

    logger.propagate = False
    _CONFIGURED[name] = logger
    return logger
