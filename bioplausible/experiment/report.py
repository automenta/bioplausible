"""Append-only JSONL Report + resume index (architecture §6.7).

The Report is the experiment's artifact *and* resume index: every probe is
appended as one JSONL line, keyed by ``(stage, model, config_key, seed)``. On
relaunch, probes recorded with ``status != "error"`` are skipped, giving
crash-resume, incremental extension, and exact reproducibility.

Storage is deliberately plain JSONL (the Optuna study layer lives in
``cli.run --db`` / SQLite); the JSONL-vs-SQLite divergence is declared, not
silent (RESEARCH §4.3).
"""

from __future__ import annotations

import json
from pathlib import Path

from bioplausible.experiment.probe import ProbeResult

__all__ = ["Report", "probe_index_key"]


def probe_index_key(stage: str, result: ProbeResult) -> str:
    """Return the resume-index key for a probe record."""
    return f"{stage}:{result.model}:{result.config_key}:{result.seed}"


class Report:
    """Append-only JSONL of probes plus an in-memory resume index."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._finished: set[str] = set()
        self._stage_records: dict[str, list[ProbeResult]] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        """Index any existing probe lines so relaunch is a no-op for them."""
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            stage = record.get("stage", "")
            key = (
                f"{stage}:{record.get('model', '')}:{record.get('config_key', '')}"
                f":{record.get('seed', '')}"
            )
            if record.get("status") != "error":
                self._finished.add(key)
            self._stage_records.setdefault(stage, []).append(
                _result_from_record(record)
            )

    def is_finished(self, stage: str, result: ProbeResult) -> bool:
        """Return whether this probe's resume key is already recorded (ok)."""
        return probe_index_key(stage, result) in self._finished

    def append(self, stage: str, result: ProbeResult) -> None:
        """Append one probe to the JSONL and update the resume index."""
        record = result.to_dict()
        record["stage"] = stage
        key = probe_index_key(stage, result)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        if result.status != "error":
            self._finished.add(key)
        self._stage_records.setdefault(stage, []).append(result)

    def finished_keys(self) -> set[str]:
        """Return the set of completed probe keys (for the producer skip)."""
        return set(self._finished)

    def stage_results(self, stage: str) -> list[ProbeResult]:
        """Return all recorded probes for a stage, in append order."""
        return list(self._stage_records.get(stage, []))

    def stage_names(self) -> list[str]:
        """Return the recorded stage names, in append order."""
        return list(self._stage_records)


def _result_from_record(record: dict[str, object]) -> ProbeResult:
    """Rebuild a :class:`ProbeResult` from a decoded JSONL line."""
    config = record.get("config") or {}
    if not isinstance(config, dict):
        config = {}
    return ProbeResult(
        model=str(record.get("model", "")),
        task=str(record.get("task", "")),
        config=config,
        config_key=str(record.get("config_key", "")),
        seed=int(record.get("seed", 0)),
        status=str(record.get("status", "error")),
        final_acc=float(record.get("final_acc", 0.0)),
        final_train_loss=float(record.get("final_train_loss", 0.0)),
        epoch_time_s=float(record.get("epoch_time_s", 0.0)),
        param_count=int(record.get("param_count", 0)),
        forward_flops=int(record.get("forward_flops", 0)),
        backward_flops=int(record.get("backward_flops", 0)),
        peak_memory_mb=float(record.get("peak_memory_mb", 0.0)),
        wall_time_s=float(record.get("wall_time_s", 0.0)),
        error=str(record.get("error", "")),
    )
