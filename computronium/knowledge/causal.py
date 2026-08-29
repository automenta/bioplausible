"""
Causal discovery and meta-analysis for the knowledge base.

Provides causal analysis, scaling law meta-fitting, algorithm fingerprints,
failure manifold mapping, and algorithm phylogeny.
"""

import json
import time
from dataclasses import dataclass

import numpy as np

from computronium.core._paths import db_path
from computronium.core.exceptions import KnowledgeBaseError
from computronium.core.logging import get_logger

logger = get_logger()


@dataclass(frozen=True, slots=True)
class CausalConfig:
    """Configuration for causal analysis and meta-analysis."""

    db_path: str = db_path("computronium_kb.db")
    min_experiments: int = 10
    min_records: int = 10


class CausalAnalyzer:
    """
    Causal discovery and meta-analysis for the knowledge base.

    Provides:
    - Causal analysis of hyperparameters on outcomes
    - Scaling law meta-fitting (Chinchilla-style)
    - Algorithm fingerprint computation
    - Failure manifold mapping
    - Algorithm phylogeny generation
    """

    def __init__(self, config: CausalConfig | None = None):
        self.config = config or CausalConfig()

    def run_causal_analysis(
        self,
        outcome: str = "val_accuracy",
    ) -> dict[str, object]:
        """
        Run causal discovery analysis on experiment data.

        Uses correlation-based methods to identify potentially causal
        relationships between hyperparameters and outcomes.

        Args:
            outcome: Target metric for analysis.

        Returns:
            Dict with causal analysis results.
        """
        try:
            import pandas as pd

            exps = self.list_experiments(limit=500)
            if not exps or len(exps) < self.config.min_experiments:
                return {"error": "Not enough data for causal analysis"}

            records = []
            for exp in exps:
                config = json.loads(exp.get("config", "{}"))
                metrics = json.loads(exp.get("metrics", "{}"))
                if outcome in metrics:
                    records.append({
                        "lr": config.get("lr", 0.001),
                        "hidden_dim": config.get("hidden_dim", 256),
                        "num_layers": config.get("num_layers", 2),
                        "batch_size": config.get("batch_size", 64),
                        "outcome": metrics[outcome],
                    })

            if len(records) < self.config.min_records:
                return {"error": f"Not enough records with {outcome}"}

            df = pd.DataFrame(records)

            # Compute correlations with outcome
            correlations = {}
            for col in df.columns:
                if col != "outcome":
                    corr = df[col].corr(df["outcome"])
                    if not np.isnan(corr):
                        correlations[col] = float(abs(corr))

            # Sort by correlation magnitude
            sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

            return {
                "outcome": outcome,
                "correlations": dict(correlations),
                "ranked_factors": sorted_corr,
                "n_samples": len(records),
            }
        except (Exception) as e:
            logger.exception("Causal analysis failed")
            raise KnowledgeBaseError("Causal analysis failed") from e

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

    def meta_fit_scaling_laws(
        self,
        model_families: list[str] | None = None,
        tasks: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Aggregate Chinchilla-style scaling law fits across all runs.

        Fits: L = E + A / N^alpha + B / D^beta
        Where N = params, D = data, L = loss

        Args:
            model_families: Optional filter for specific model families.
            tasks: Optional filter for specific tasks.

        Returns:
            Dict of model_family -> {alpha, beta, E, A, B, r2}
        """
        try:
            import numpy as np
            import pandas as pd
            from scipy.optimize import curve_fit

            exps = self.list_experiments(limit=1000)
            if not exps:
                return {}

            # Filter
            if model_families:
                exps = [e for e in exps if e.get("model_family") in model_families]
            if tasks:
                exps = [e for e in exps if e.get("task") in tasks]

            records = []
            for exp in exps:
                config = json.loads(exp.get("config", "{}"))
                metrics = json.loads(exp.get("metrics", {}))

                # Extract scaling parameters
                n_params = config.get("n_params", 0)
                n_data = config.get("n_data", 0)
                loss = metrics.get("final_loss", metrics.get("val_loss", 0))

                if n_params > 0 and n_data > 0 and loss > 0:
                    records.append({
                        "model_family": exp.get("model_family", "unknown"),
                        "task": exp.get("task", "unknown"),
                        "n_params": n_params,
                        "n_data": n_data,
                        "loss": loss,
                    })

            if len(records) < 20:
                logger.warning("Not enough data for scaling law meta-fit")
                return {}

            df = pd.DataFrame(records)
            results = {}

            def scaling_law(N, D, E, A, B, alpha, beta):
                return E + A / (N**alpha) + B / (D**beta)

            for model in df["model_family"].unique():
                model_df = df[df["model_family"] == model]
                if len(model_df) < 10:
                    continue

                try:
                    X = np.column_stack([
                        model_df["n_params"].values,
                        model_df["n_data"].values,
                    ])
                    y = model_df["loss"].values

                    # Initial guess: E=0.1, A=1, B=1, alpha=0.5, beta=0.5
                    p0 = [0.1, 1.0, 1.0, 0.5, 0.5]
                    bounds = ([0, 0, 0, 0.1, 0.1], [10, 100, 100, 2.0, 2.0])

                    popt, pcov = curve_fit(
                        lambda X, E, A, B, alpha, beta: scaling_law(
                            X[:, 0], X[:, 1], E, A, B, alpha, beta
                        ),
                        X,
                        y,
                        p0=p0,
                        bounds=bounds,
                        maxfev=5000,
                    )

                    E, A, B, alpha, beta = popt
                    y_pred = scaling_law(X[:, 0], X[:, 1], E, A, B, alpha, beta)
                    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

                    results[model] = {
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "E": float(E),
                        "A": float(A),
                        "B": float(B),
                        "r2": float(r2),
                        "n_samples": len(model_df),
                    }
                except Exception as e:
                    logger.warning("Scaling fit failed for %s: %s", model, e)
                    continue

            logger.info("Meta-fit scaling laws for %d model families", len(results))
            return results

        except Exception as e:
            logger.exception("Scaling law meta-fit failed")
            raise KnowledgeBaseError("Scaling law meta-fit failed") from e

    def compute_algorithm_fingerprints(
        self,
        model_families: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Compute algorithm fingerprints: hyperparameter sensitivity embeddings.

        Each algorithm gets a fingerprint vector representing its sensitivity
        to different hyperparameters. Used for algorithm phylogeny.

        Args:
            model_families: Optional filter for specific model families.

        Returns:
            Dict of model_family -> {hyperparam: sensitivity_score}
        """
        try:
            import numpy as np
            import pandas as pd

            exps = self.list_experiments(limit=1000)
            if not exps:
                return {}

            if model_families:
                exps = [e for e in exps if e.get("model_family") in model_families]

            records = []
            for exp in exps:
                config = json.loads(exp.get("config", "{}"))
                metrics = json.loads(exp.get("metrics", {}))
                acc = metrics.get("val_accuracy", 0)

                record = {
                    "model_family": exp.get("model_family", "unknown"),
                    "lr": config.get("lr", 0.001),
                    "hidden_dim": config.get("hidden_dim", 256),
                    "num_layers": config.get("num_layers", 2),
                    "batch_size": config.get("batch_size", 64),
                    "beta": config.get("beta", 0.0),
                    "val_accuracy": acc,
                }
                records.append(record)

            if len(records) < 20:
                return {}

            df = pd.DataFrame(records)

            # Compute sensitivity for each model family
            fingerprints = {}
            for model in df["model_family"].unique():
                model_df = df[df["model_family"] == model]
                if len(model_df) < 5:
                    continue

                # Correlation between each hyperparam and accuracy
                sensitivity = {}
                for param in ["lr", "hidden_dim", "num_layers", "batch_size", "beta"]:
                    if param in model_df.columns:
                        corr = model_df[param].corr(model_df["val_accuracy"])
                        if not np.isnan(corr):
                            sensitivity[param] = float(abs(corr))

                # Also compute variance-based sensitivity (Sobol-like)
                for param in ["lr", "hidden_dim", "num_layers", "batch_size"]:
                    if param in model_df.columns:
                        grouped = model_df.groupby(param)["val_accuracy"].mean()
                        if len(grouped) > 1:
                            sensitivity[f"{param}_variance"] = float(grouped.var())

                fingerprints[model] = sensitivity

            logger.info("Computed fingerprints for %d algorithms", len(fingerprints))
            return fingerprints

        except Exception as e:
            logger.exception("Algorithm fingerprint computation failed")
            raise KnowledgeBaseError("Algorithm fingerprint computation failed") from e

    def map_failure_manifold(
        self,
        min_samples: int = 5,
    ) -> dict[str, dict[str, object]]:
        """
        Cluster failed runs by error mode to identify failure manifolds.

        Identifies common failure patterns across algorithms/tasks.

        Args:
            min_samples: Minimum samples to form a cluster.

        Returns:
            Dict of failure_cluster -> {error_pattern, algorithms, tasks, count, characteristics}
        """
        try:
            import numpy as np
            import pandas as pd
            from sklearn.cluster import DBSCAN
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import StandardScaler

            exps = self.list_experiments(limit=1000)
            if not exps:
                return {}

            # Collect failed experiments
            failed = []
            for exp in exps:
                metrics = json.loads(exp.get("metrics", "{}"))
                config = json.loads(exp.get("config", "{}"))

                # Consider failed if low accuracy or explicit error
                acc = metrics.get("val_accuracy", 1.0)
                error = metrics.get("error", config.get("error", ""))

                if acc < 0.15 or error:
                    failed.append({
                        "model_family": exp.get("model_family", "unknown"),
                        "task": exp.get("task", "unknown"),
                        "accuracy": acc,
                        "error": str(error),
                        "config": config,
                    })

            if len(failed) < min_samples:
                logger.warning("Not enough failed runs for manifold mapping")
                return {}

            df = pd.DataFrame(failed)

            # Vectorize error messages
            tfidf = TfidfVectorizer(max_features=50, stop_words="english")
            error_texts = df["error"].fillna("").tolist()
            if all(not e for e in error_texts):
                # No error messages, use accuracy + config
                X_config = pd.DataFrame(df["config"].tolist())
                X_config = X_config.fillna(0)
                scaler = StandardScaler()
                X = scaler.fit_transform(X_config.select_dtypes(include=[np.number]))
            else:
                X_text = tfidf.fit_transform(error_texts).toarray()
                # Add config features
                X_config = pd.DataFrame(df["config"].tolist())
                X_config = X_config.fillna(0)
                numeric_cols = X_config.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    scaler = StandardScaler()
                    X_config_scaled = scaler.fit_transform(X_config[numeric_cols])
                    X = np.hstack([X_text, X_config_scaled])
                else:
                    X = X_text

            # Cluster
            clustering = DBSCAN(eps=0.5, min_samples=min_samples)
            labels = clustering.fit_predict(X)

            df["cluster"] = labels

            # Analyze clusters
            failure_manifold = {}
            for cluster_id in np.unique(labels):
                if cluster_id == -1:
                    continue  # Noise

                cluster_df = df[df["cluster"] == cluster_id]
                if len(cluster_df) < min_samples:
                    continue

                # Characterize cluster
                error_mode = (
                    cluster_df["error"].mode().iloc[0]
                    if not cluster_df["error"].mode().empty
                    else "unknown"
                )
                algorithms = cluster_df["model_family"].value_counts().to_dict()
                tasks = cluster_df["task"].value_counts().to_dict()
                mean_acc = float(cluster_df["accuracy"].mean())

                # Common config patterns
                common_config = {}
                for col in (
                    cluster_df["config"].iloc[0].keys() if len(cluster_df) > 0 else []
                ):
                    vals = [c.get(col) for c in cluster_df["config"] if col in c]
                    if vals:
                        common_config[col] = max(set(vals), key=vals.count)

                failure_manifold[f"cluster_{cluster_id}"] = {
                    "error_pattern": error_mode,
                    "algorithms": algorithms,
                    "tasks": tasks,
                    "count": len(cluster_df),
                    "mean_accuracy": mean_acc,
                    "common_config": common_config,
                }

            logger.info(
                "Mapped failure manifold with %d clusters", len(failure_manifold)
            )
            return failure_manifold

        except Exception as e:
            logger.exception("Failure manifold mapping failed")
            raise KnowledgeBaseError("Failure manifold mapping failed") from e

    def generate_algorithm_phylogeny(
        self,
        method: str = "ward",
    ) -> dict[str, object]:
        """
        Generate phylogenetic tree of algorithms based on fingerprints.

        Uses hierarchical clustering on algorithm fingerprints.

        Args:
            method: Linkage method ('ward', 'complete', 'average', 'single').

        Returns:
            Dict with tree structure and cluster assignments.
        """
        try:
            import numpy as np
            from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
            from sklearn.preprocessing import StandardScaler

            fingerprints = self.compute_algorithm_fingerprints()
            if not fingerprints or len(fingerprints) < 3:
                return {}

            # Build feature matrix
            models = list(fingerprints.keys())
            all_features = set()
            for fp in fingerprints.values():
                all_features.update(fp.keys())

            feature_list = sorted(all_features)
            X = np.zeros((len(models), len(feature_list)))

            for i, model in enumerate(models):
                for j, feat in enumerate(feature_list):
                    X[i, j] = fingerprints[model].get(feat, 0.0)

            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Hierarchical clustering
            Z = linkage(X_scaled, method=method)

            # Get cluster assignments
            clusters = fcluster(Z, t=0.5, criterion="distance")

            # Build tree structure
            tree = {
                "models": models,
                "linkage_matrix": Z.tolist(),
                "clusters": {models[i]: int(clusters[i]) for i in range(len(models))},
                "n_clusters": int(clusters.max()),
                "feature_names": feature_list,
            }

            # Dendrogram data for visualization
            dend = dendrogram(Z, labels=models, no_plot=True)
            tree["dendrogram"] = {
                "icoord": [c.tolist() for c in dend["icoord"]],
                "dcoord": [c.tolist() for c in dend["dcoord"]],
                "ivl": dend["ivl"],
                "leaves": dend["leaves"],
            }

            logger.info(
                "Generated algorithm phylogeny with %d clusters", tree["n_clusters"]
            )
            return tree

        except Exception as e:
            logger.exception("Algorithm phylogeny generation failed")
            raise KnowledgeBaseError("Algorithm phylogeny generation failed") from e


__all__ = [
    "CausalConfig",
    "CausalAnalyzer",
]