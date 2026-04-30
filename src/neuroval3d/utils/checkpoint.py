"""Checkpoint manager — persists model state + metadata, registers in CHECKPOINTS.md."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from neuroval3d.utils.io import write_yaml


@dataclass
class CheckpointMeta:
    """Metadata block accompanying every saved checkpoint."""

    cp_id: str
    phase: str
    description: str
    metric_name: str
    metric_value: float
    config_hash: str
    git_sha: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    artifact_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Saves model weights + metadata + registers entries in docs/CHECKPOINTS.md."""

    def __init__(self, root: str | Path = "outputs/checkpoints") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger_path = Path("docs/CHECKPOINTS.md")

    def save(
        self,
        meta: CheckpointMeta,
        artifact: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> Path:
        cp_dir = self.root / meta.cp_id
        cp_dir.mkdir(parents=True, exist_ok=True)

        if config is not None:
            write_yaml(cp_dir / "config.yaml", config)

        artifact_path: Path | None = None
        if artifact is not None:
            artifact_path = self._save_artifact(cp_dir, artifact)
            meta.artifact_path = str(artifact_path.relative_to(Path.cwd())) if artifact_path else None

        with open(cp_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2)

        self._append_ledger(meta)
        return cp_dir

    @staticmethod
    def _save_artifact(cp_dir: Path, artifact: Any) -> Path | None:
        try:
            import torch
            if isinstance(artifact, torch.nn.Module):
                path = cp_dir / "model.pt"
                torch.save(artifact.state_dict(), path)
                return path
            if isinstance(artifact, dict):
                path = cp_dir / "state.pt"
                torch.save(artifact, path)
                return path
        except ImportError:
            pass
        if isinstance(artifact, (str, Path)):
            src = Path(artifact)
            if src.exists():
                dst = cp_dir / src.name
                shutil.copy2(src, dst)
                return dst
        path = cp_dir / "artifact.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, default=str)
        return path

    def _append_ledger(self, meta: CheckpointMeta) -> None:
        if not self.ledger_path.exists():
            return
        line = (
            f"| {meta.cp_id} | {meta.timestamp[:10]} | {meta.phase} | "
            f"{meta.description} (`{meta.metric_name}`={meta.metric_value:.4f}) | "
            f"`{meta.artifact_path or self.root / meta.cp_id}` |"
        )
        existing = self.ledger_path.read_text(encoding="utf-8")
        if meta.cp_id in existing:
            return
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write("\n" + line + "\n")


def save_checkpoint(
    cp_id: str,
    phase: str,
    description: str,
    metric_name: str,
    metric_value: float,
    artifact: Any | None = None,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    root: str | Path = "outputs/checkpoints",
) -> Path:
    config_hash = (
        hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]
        if config
        else "no-config"
    )
    meta = CheckpointMeta(
        cp_id=cp_id,
        phase=phase,
        description=description,
        metric_name=metric_name,
        metric_value=float(metric_value),
        config_hash=config_hash,
        extra=extra or {},
    )
    return CheckpointManager(root).save(meta, artifact=artifact, config=config)
