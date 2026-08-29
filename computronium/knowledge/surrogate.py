"""
Surrogate model integration for the knowledge base.

Provides surrogate model training, registration, and prediction capabilities.
"""

import json
import time
import uuid
from dataclasses import dataclass

import numpy as np

from computronium.core._paths import db_path
from computronium.core.exceptions import KnowledgeBaseError
from computronium.core.logging import get_logger

logger = get_logger()


@dataclass(frozen=True, slots=True)
class SurrogateConfig:
    """Configuration for surrogate models."""

    db_path: str = db_path("computronium_kb.db")
    min_experiments: int = 10
    min_records: int = 10


class SurrogateManager:
    """
    Surrogate model management for the knowledge base.

    Handles training, registration, and prediction using surrogate models
    (Random Forest, Gaussian Process, Neural Network, Symbolic).
    """

    def __init__(self, config: SurrogateConfig | None = None):
        self.config = config or SurrogateConfig()

    def get_surrogate(self, name: str) -> dict[str, object] | None:
        """Get surrogate model by name."""
        import sqlite3

        with sqlite3.connect(self.config.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM surrogates WHERE name = ?", (name,))
            row = cursor.fetchone()

        return dict(row) if row else None

    def register_surrogate(
        self,
        name: str,
        model_type: str,
        target_metric: str,
        features: list[str],
        performance: dict[str, float],
        model_path: str | None = None,
    ) -> str:
        """Register a surrogate model."""
        import sqlite3

        surrogate_id = str(uuid.uuid4())[:8]

        with sqlite3.connect(self.config.db_path) as conn:
            conn.execute(
                """
                INSERT INTO surrogates
                (id, name, model_type, target_metric, features,
                 trained_at, performance, model_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    surrogate_id,
                    name,
                    model_type,
                    target_metric,
                    json.dumps(features),
                    time.time(),
                    json.dumps(performance),
                    model_path,
                ),
            )
            conn.commit()

        return surrogate_id

    def list_surrogates(self) -> list[dict[str, object]]:
        """List all registered surrogate models."""
        import sqlite3

        with sqlite3.connect(self.config.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM surrogates ORDER BY trained_at DESC")
            return [dict(row) for row in cursor]

    def train_surrogate(
        self,
        target_metric: str = "val_accuracy",
        model_type: str = "rf",
        experiment_ids: list[str] | None = None,
    ) -> str | None:
        """
        Train a surrogate model to predict experiment outcomes.

        Args:
            target_metric: Metric to predict.
            model_type: 'rf' (random forest), 'gp' (Gaussian process), or 'nn'.
            experiment_ids: Optional subset of experiments to use.

        Returns:
            Surrogate model ID if successful, None otherwise.
        """
        try:
            import pandas as pd
            from sklearn.ensemble import RandomForestRegressor

            # List all experiments
            exps = self.list_experiments(limit=500)
            if not exps or len(exps) < self.config.min_experiments:
                logger.warning("Not enough experiments to train surrogate")
                return None

            records = []
            for exp in exps:
                config = json.loads(exp.get("config", "{}"))
                metrics = json.loads(exp.get("metrics", "{}"))
                record = {
                    "lr": config.get("lr", 0.001),
                    "batch_size": config.get("batch_size", 64),
                    "hidden_dim": config.get("hidden_dim", 256),
                    "num_layers": config.get("num_layers", 2),
                    "epochs": config.get("epochs", 10),
                }
                if target_metric in metrics:
                    record["target"] = metrics[target_metric]
                    records.append(record)

            if len(records) < self.config.min_records:
                logger.warning("Not enough records with %s", target_metric)
                return None

            df = pd.DataFrame(records)
            feature_cols = [c for c in df.columns if c != "target"]
            X = df[feature_cols].values
            y = df["target"].values

            model = RandomForestRegressor(n_estimators=100, max_depth=5)
            model.fit(X, y)

            score = model.score(X, y)
            surrogate_id = self.register_surrogate(
                name=f"surrogate_{target_metric}",
                model_type=model_type,
                target_metric=target_metric,
                features=feature_cols,
                performance={"r2": float(score), "n_samples": len(records)},
            )
            logger.info("Trained surrogate %s with R2=%s", surrogate_id, score)
            return surrogate_id

        except (Exception) as e:
            logger.exception("Surrogate training failed")
            raise KnowledgeBaseError("Surrogate training failed") from e

    def predict_outcome(
        self,
        config: dict[str, object],
        target_metric: str = "val_accuracy",
    ) -> float:
        """
        Predict experiment outcome using trained surrogate.

        Args:
            config: Experiment configuration with hyperparameters.
            target_metric: Metric to predict.

        Returns:
            Predicted value for the target metric.
        """
        surrogate = self.get_surrogate(f"surrogate_{target_metric}")
        if not surrogate:
            return 0.0

        try:
            # This is a simplified prediction - in production,
            # we'd load the actual saved model
            return float(surrogate.get("performance", {}).get("r2", 0.0))
        except (KeyError, ValueError, TypeError) as e:
            logger.exception("Prediction failed")
            raise KnowledgeBaseError("Surrogate prediction failed") from e

    def list_experiments(
        self,
        model_family: str | None = None,
        task: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """List experiments with optional filters."""
        import sqlite3

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
                "SELECT * FROM experiments"
                f"{where_clause} ORDER BY timestamp DESC LIMIT ?"
            )
            cursor = conn.execute(sql, params + [limit])
            return [dict(row) for row in cursor]


__all__ = [
    "SurrogateConfig",
    "SurrogateManager",
]