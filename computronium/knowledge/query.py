"""
Structured query operations for the knowledge base.

Handles SQL-based filtering, conditional queries, and flagship selection.
"""

import json
import sqlite3
from dataclasses import dataclass

from pydantic import ValidationError

from computronium.core._paths import db_path
from computronium.core.exceptions import ConditionalQueryError
from computronium.core.logging import get_logger
from computronium.knowledge.entries import (
    ConditionalQuery,
    ConditionalResult,
    FlagshipCandidate,
    FlagshipDecision,
    KnowledgeEntry,
    _ConditionalQueryModel,
    _entry_accuracy,
    _entry_flops,
    _entry_memory,
    _entry_substrate,
    _entry_task,
)

logger = get_logger()


@dataclass(frozen=True, slots=True)
class QueryConfig:
    """Configuration for query operations."""

    db_path: str = db_path("computronium_kb.db")
    default_limit: int = 100


class QueryEngine:
    """
    Structured query engine for the knowledge base.

    Provides SQL-based filtering, conditional queries (P2 read-half),
    and flagship selection (P3a).
    """

    def __init__(self, config: QueryConfig | None = None):
        self.config = config or QueryConfig()

    def query(
        self,
        tag: str | None = None,
        model_family: str | None = None,
        topic: str | None = None,
        source: str | None = None,
        min_confidence: float | None = None,
        experiment_id: str | None = None,
        limit: int | None = None,
    ) -> list[KnowledgeEntry]:
        """Query knowledge base with structured filters."""
        limit = limit or self.config.default_limit
        conditions = []
        params = []

        if tag:
            conditions.append("tags LIKE ?")
            params.append(f"%{tag}%")
        if model_family:
            conditions.append("model_family = ?")
            params.append(model_family)
        if topic:
            conditions.append("topic = ?")
            params.append(topic)
        if source:
            conditions.append("source = ?")
            params.append(source)
        if min_confidence is not None:
            conditions.append("confidence >= ?")
            params.append(min_confidence)
        if experiment_id:
            conditions.append("experiment_id = ?")
            params.append(experiment_id)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        with sqlite3.connect(self.config.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sql = (
                f"SELECT * FROM knowledge{where_clause} ORDER BY timestamp DESC LIMIT ?"  # ruff: ignore[hardcoded-sql-expression]
            )
            cursor = conn.execute(sql, params + [limit])  # ruff: ignore[collection-literal-concatenation]
            rows = cursor.fetchall()

        return [self._row_to_entry(row) for row in rows]

    def query_conditionals(
        self, query: ConditionalQuery, limit: int = 100
    ) -> list[ConditionalResult]:
        """Return verified positive conditionals matching a P2 read-half query.

        Filters stored experiment results by model, task, substrate, and any
        accuracy/memory/flops caps provided in ``query``. This is the flywheel's
        *read* side: the proposer consumes it to avoid re-characterizing probes
        that a prior run already answered.

        Args:
            query: The filter (see :class:`ConditionalQuery`). Validated with
                Pydantic v2 at this boundary.
            limit: Maximum number of results.

        Returns:
            Matching conditionals, best-accuracy first.

        Raises:
            ConditionalQueryError: If ``query`` is malformed (e.g. negative cap
                or an accuracy target outside [0, 1]).
        """
        try:
            q = _ConditionalQueryModel(**query)
        except ValidationError as exc:
            raise ConditionalQueryError(f"invalid conditional query: {exc}") from exc

        entries = self.query(
            model_family=q.model,
            source="experiment",
            limit=max(limit * 4, 64),
        )
        if q.task is not None:
            entries = [e for e in entries if _entry_task(e) == q.task]

        results: list[ConditionalResult] = []
        for e in entries:
            metrics = e.metrics or {}
            acc = _entry_accuracy(metrics)
            if q.accuracy_target is not None and acc < q.accuracy_target:
                continue
            mem = _entry_memory(metrics)
            if q.memory_cap is not None and mem > q.memory_cap:
                continue
            flops = _entry_flops(metrics)
            if q.flops_cap is not None and flops > q.flops_cap:
                continue
            substrate = _entry_substrate(e)
            if q.substrate is not None and substrate != q.substrate:
                continue
            results.append(
                ConditionalResult(
                    model=str(e.model_family),
                    task=_entry_task(e),
                    accuracy=acc,
                    memory_mb=mem,
                    flops=flops,
                    wall_time_s=float(metrics.get("wall_time_s", 0.0)),
                    config=tuple(sorted((e.hyperparameters or {}).items())),
                    substrate=substrate,
                    entry_id=e.id,
                )
            )
        results.sort(key=lambda r: r.accuracy, reverse=True)
        return results[:limit]

    def select_flagship(
        self,
        *,
        task: str = "mnist",
        accuracy_gap: float = 0.15,
        substrate: str | None = None,
    ) -> FlagshipDecision:
        """Select the flagship rule by querying the KB (P3a).

        Codified selection rule, implemented as a query rather than a judgment
        call: among **validated** families (P0a surface record present and
        honest), pick the one with minimal ``cost_of_plausibility`` — the
        geometric mean of its FLOPs/memory/time ratios to the backprop reference
        on the same task — subject to:

        1. **non-phantom space** — the family passed P0a (honest surface record).
        2. **substrate-eligible** — if ``substrate`` is given, only families with
           a matching substrate-conditional candidate.
        3. **minimal accuracy-to-backprop gap** — the candidate's accuracy must
           be within ``accuracy_gap`` of the backprop reference, so a trivially
           cheap-but-awful rule is never chosen.

        Args:
            task: Task the frontier/conditionals were measured on.
            accuracy_gap: Maximum accuracy deficit vs the backprop reference.
            substrate: If set, only consider conditionals on this substrate.

        Returns:
            A :class:`FlagshipDecision` ranking the candidate families.
        """
        import math

        bp = self._best_conditional("backprop_mlp", task, substrate)
        if bp is None:
            return FlagshipDecision(task=task, chosen=None, ranked=())

        validated = {
            e.model_family
            for e in self.query(topic="rule_space_surface", source="validator")
            if e.finding == "honest"
        }

        ranked: list[FlagshipCandidate] = []
        for family in validated:
            cand = self._best_conditional(family, task, substrate)
            if cand is None:
                continue
            if bp.accuracy - cand.accuracy > accuracy_gap:
                continue
            cost = math.pow(
                (cand.flops / max(bp.flops, 1e-12))
                * (cand.memory_mb / max(bp.memory_mb, 1e-12))
                * (cand.wall_time_s / max(bp.wall_time_s, 1e-12)),
                1.0 / 3.0,
            )
            ranked.append(
                FlagshipCandidate(
                    model=family,
                    accuracy=cand.accuracy,
                    memory_mb=cand.memory_mb,
                    flops=cand.flops,
                    wall_time_s=cand.wall_time_s,
                    cost_of_plausibility=cost,
                    substrate=cand.substrate,
                )
            )

        ranked.sort(key=lambda c: (c.cost_of_plausibility, -c.accuracy))
        chosen = ranked[0].model if ranked else None
        return FlagshipDecision(task=task, chosen=chosen, ranked=tuple(ranked))

    def _best_conditional(
        self, model: str, task: str, substrate: str | None
    ) -> ConditionalResult | None:
        """Best (highest-accuracy) verified conditional for a model on a task."""
        results = self.query_conditionals({
            "model": model,
            "task": task,
            "accuracy_target": 0.0,
            "substrate": substrate,
            "memory_cap": None,
            "flops_cap": None,
        })
        return results[0] if results else None

    def _row_to_entry(self, row: sqlite3.Row) -> KnowledgeEntry:
        """Convert SQLite row to KnowledgeEntry."""
        return KnowledgeEntry(
            id=row["id"],
            topic=row["topic"],
            model_family=row["model_family"],
            finding=row["finding"],
            details=row["details"] or "",
            confidence=row["confidence"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            timestamp=row["timestamp"],
            source=row["source"],
            experiment_id=row["experiment_id"],
            metrics=json.loads(row["metrics"]) if row["metrics"] else {},
            hyperparameters=(
                json.loads(row["hyperparameters"]) if row["hyperparameters"] else {}
            ),
            extra=json.loads(row["extra"]) if row["extra"] else {},
        )

    def get_by_id(self, entry_id: str) -> KnowledgeEntry | None:
        """Get entry by ID."""
        with sqlite3.connect(self.config.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM knowledge WHERE id = ?", (entry_id,))
            row = cursor.fetchone()

        if row:
            return self._row_to_entry(row)
        return None

    def get_experiment(self, experiment_id: str) -> dict[str, object] | None:
        """Get experiment by ID."""
        with sqlite3.connect(self.config.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            )
            row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    def list_experiments(
        self,
        model_family: str | None = None,
        task: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """List experiments with optional filters."""
        conditions = []
        params = []

        if model_family:
            conditions.append("model_family = ?")
            params.append(model_family)
        if task:
            conditions.append("task = ?")
            params.append(task)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        with sqlite3.connect(self.config.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sql = (
                "SELECT * FROM experiments"  # ruff: ignore[hardcoded-sql-expression]
                f"{where_clause} ORDER BY timestamp DESC LIMIT ?"
            )
            cursor = conn.execute(sql, params + [limit])  # ruff: ignore[collection-literal-concatenation]
            return [dict(row) for row in cursor]


__all__ = [
    "QueryConfig",
    "QueryEngine",
]
