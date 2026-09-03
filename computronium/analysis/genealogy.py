"""Algorithm Genealogy — Hyperparameter Fingerprints → Embeddings → Phylogeny.

Constructs algorithm maps for paper figures by embedding hyperparameter configurations
and building phylogenetic trees of algorithm evolution.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from scipy.cluster.hierarchy import LinkageMatrix

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class AlgorithmFingerprint:
    """Hyperparameter fingerprint for an algorithm configuration."""

    algorithm: str
    config: dict
    metadata: dict
    embedding: np.ndarray | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary (excluding embedding)."""
        return {
            "algorithm": self.algorithm,
            "config": self.config,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AlgorithmFingerprint:
        """Create from dictionary."""
        return cls(
            algorithm=d["algorithm"],
            config=d["config"],
            metadata=d["metadata"],
            embedding=None,
        )


@dataclass(frozen=True, slots=True)
class PhylogenyNode:
    """Node in algorithm phylogeny tree."""

    name: str
    fingerprint: AlgorithmFingerprint | None = None
    children: tuple[PhylogenyNode, ...] = field(default_factory=tuple)
    distance: float = 0.0
    support: float = 1.0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_leaves(self) -> list[PhylogenyNode]:
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves


# =============================================================================
# Fingerprint Extraction
# =============================================================================


FINGERPRINT_FEATURES = [
    # Core hyperparameters
    "lr",
    "batch_size",
    "epochs",
    "hidden_dim",
    "num_layers",
    "dropout",
    "weight_decay",
    # Algorithm-specific
    "beta",
    "gamma",
    "settle_steps",
    "nudge_steps",
    "feedback_scale",
    "rank_frac",
    "fisher_damping",
    "muon_lr",
    "spectral_norm",
    "constraint_weight",
    # Architecture
    "activation",
    "normalization",
    "skip_connections",
    # Sparsity
    "sparsity",
    "pruning_method",
    # Training mode
    "mode",
    "algorithm",
    "propagator",
]


def extract_fingerprint(config: dict, metadata: dict | None = None) -> np.ndarray:
    """Extract normalized hyperparameter fingerprint vector.

    Args:
        config: Full configuration dict
        metadata: Optional registry metadata

    Returns:
        Normalized fingerprint vector (n_features,)
    """
    features = []
    for key in FINGERPRINT_FEATURES:
        value = config.get(key)
        if value is None and metadata:
            value = metadata.get(key)
        if value is None:
            features.append(0.0)
        elif isinstance(value, bool):
            features.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            features.append(float(value))
        elif isinstance(value, str):
            # Hash string to float in [0, 1]
            features.append(hash(value) % 1000 / 1000.0)
        else:
            features.append(0.0)

    arr = np.array(features, dtype=float)

    # Normalize: log-scale for positive values, center around 0
    for i, key in enumerate(FINGERPRINT_FEATURES):
        if arr[i] > 0:
            arr[i] = np.log1p(arr[i])
        elif arr[i] < 0:
            arr[i] = -np.log1p(-arr[i])

    # Standardize
    mean = np.mean(arr[arr != 0]) if np.any(arr != 0) else 0
    std = np.std(arr[arr != 0]) if np.any(arr != 0) else 1
    arr = (arr - mean) / (std + 1e-8)

    return arr


def build_fingerprint_matrix(
    configs: list[dict],
    metadatas: list[dict] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build fingerprint matrix from list of configs.

    Args:
        configs: List of configuration dicts
        metadatas: Optional list of metadata dicts

    Returns:
        (matrix: n_configs x n_features, labels: list of algorithm names)
    """
    if metadatas is None:
        metadatas = [{} for _ in configs]

    matrix = []
    labels = []
    for config, meta in zip(configs, metadatas):
        fp = extract_fingerprint(config, meta)
        matrix.append(fp)
        label = (
            config.get("algorithm")
            or config.get("model")
            or config.get("propagator")
            or "unknown"
        )
        labels.append(label)

    return np.vstack(matrix), labels


# =============================================================================
# Embedding & Dimensionality Reduction
# =============================================================================


def reduce_dimensions(
    matrix: np.ndarray,
    method: Literal["pca", "tsne", "umap"] = "pca",
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce fingerprint matrix dimensions for visualization.

    Args:
        matrix: Fingerprint matrix (n_samples, n_features)
        method: Dimensionality reduction method
        n_components: Target dimensions
        random_state: Random seed

    Returns:
        Reduced matrix (n_samples, n_components)
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(matrix)
    elif method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=min(30, len(matrix) - 1),
        )
        return reducer.fit_transform(matrix)
    elif method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=n_components, random_state=random_state)
            return reducer.fit_transform(matrix)
        except ImportError:
            logger.warning("UMAP not available, falling back to PCA")
            return reduce_dimensions(matrix, "pca", n_components, random_state)
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Phylogenetic Tree Construction
# =============================================================================


def build_phylogeny(
    fingerprints: list[AlgorithmFingerprint],
    method: Literal["ward", "complete", "average", "single"] = "ward",
    metric: str = "euclidean",
) -> tuple[LinkageMatrix, list[str]]:
    """Build hierarchical clustering (phylogeny) from fingerprints.

    Args:
        fingerprints: List of AlgorithmFingerprint objects
        method: Linkage method
        metric: Distance metric

    Returns:
        (linkage_matrix, labels)
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    matrix = np.vstack([
        fp.embedding for fp in fingerprints if fp.embedding is not None
    ])
    labels = [fp.algorithm for fp in fingerprints if fp.embedding is not None]

    if len(matrix) < 2:
        raise ValueError("Need at least 2 fingerprints for phylogeny")

    # pdist metric can be string or callable; use string for standard metrics
    distances = pdist(matrix, metric=metric)  # type: ignore[arg-type]
    linkage_matrix = linkage(distances, method=method)

    return linkage_matrix, labels


def linkage_to_tree(
    linkage_matrix: LinkageMatrix,
    labels: list[str],
) -> PhylogenyNode:
    """Convert scipy linkage matrix to PhylogenyNode tree.

    Args:
        linkage_matrix: Output from scipy.cluster.hierarchy.linkage
        labels: Original leaf labels

    Returns:
        Root PhylogenyNode
    """
    n = len(labels)
    nodes: list[PhylogenyNode] = [PhylogenyNode(name=labels[i]) for i in range(n)]

    for i, row in enumerate(linkage_matrix):
        left_idx = int(row[0])
        right_idx = int(row[1])
        distance = float(row[2])

        left = nodes[left_idx]
        right = nodes[right_idx]

        new_node = PhylogenyNode(
            name=f"cluster_{n + i}",
            children=(left, right),
            distance=distance,
        )
        nodes.append(new_node)

    return nodes[-1]


# =============================================================================
# Algorithm Map Generation
# =============================================================================


@dataclass(frozen=True, slots=True)
class AlgorithmMapConfig:
    """Configuration for algorithm map generation."""

    reduction_method: Literal["pca", "tsne", "umap"] = "pca"
    phylogeny_method: Literal["ward", "complete", "average"] = "ward"
    color_by: str = "family"  # "family", "locality", "bio_score", "algorithm"
    size_by: str | None = None
    annotate: bool = True


def generate_algorithm_map(
    configs: list[dict],
    metadatas: list[dict] | None = None,
    config: AlgorithmMapConfig | None = None,
) -> tuple[np.ndarray, list[str], LinkageMatrix | None]:
    """Generate algorithm map data for visualization.

    Args:
        configs: List of algorithm configurations
        metadatas: Optional metadata for each config
        config: Generation configuration

    Returns:
        (embeddings_2d, labels, linkage_matrix)
    """
    if config is None:
        config = AlgorithmMapConfig()

    # Build fingerprint matrix
    matrix, labels = build_fingerprint_matrix(configs, metadatas)

    # Reduce dimensions
    embeddings_2d = reduce_dimensions(
        matrix, method=config.reduction_method, n_components=2
    )

    # Build phylogeny
    fingerprints = [
        AlgorithmFingerprint(
            algorithm=labels[i],
            config=configs[i],
            metadata=metadatas[i] if metadatas else {},
            embedding=embeddings_2d[i],
        )
        for i in range(len(configs))
    ]

    linkage_matrix, _ = build_phylogeny(fingerprints, method=config.phylogeny_method)

    return embeddings_2d, labels, linkage_matrix


# =============================================================================
# Visualization
# =============================================================================


def plot_algorithm_map(  # ruff: ignore[complex-structure, too-many-branches]
    embeddings: np.ndarray,
    labels: list[str],
    metadatas: list[dict] | None = None,
    linkage_matrix: LinkageMatrix | None = None,
    color_by: str = "family",
    size_by: str | None = None,
    output_path: str | Path | None = None,
) -> go.Figure:
    """Plot algorithm map with phylogeny overlay.

    Args:
        embeddings: 2D embeddings (n, 2)
        labels: Algorithm labels
        metadatas: Optional metadata for coloring/sizing
        linkage_matrix: Optional phylogeny for tree overlay
        color_by: Metadata field to color by
        size_by: Metadata field for point size
        output_path: Optional path to save HTML

    Returns:
        Plotly Figure
    """
    import plotly.graph_objects as go

    if metadatas is None:
        metadatas = [{} for _ in labels]

    # Color mapping
    color_values = []
    for i, meta in enumerate(metadatas):
        val = meta.get(color_by, labels[i])
        if isinstance(val, str):
            # Map string to color index
            color_values.append(hash(val) % 100 / 100.0)
        else:
            color_values.append(float(val) if val is not None else 0.0)

    # Size mapping
    sizes = None
    if size_by:
        sizes = []
        for meta in metadatas:
            val = meta.get(size_by, 10)
            sizes.append(float(val) if val is not None else 10)

    fig = go.Figure()

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            mode="markers+text",
            marker={
                "size": sizes if sizes else 12,
                "color": color_values,
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": color_by},
                "opacity": 0.8,
                "line": {"width": 1, "color": "white"},
            },
            text=labels if len(labels) < 50 else None,
            textposition="top center",
            hovertemplate=(
                "%{text}<br>"
                "Dim 1=%{x:.3f}<br>"
                "Dim 2=%{y:.3f}<br>"
                f"{color_by}=%{{marker.color:.3f}}<extra></extra>"
            ),
        )
    )

    # Phylogeny tree overlay (simplified: connect nearest neighbors)
    if linkage_matrix is not None:  # ruff: ignore[too-many-nested-blocks]
        from scipy.cluster.hierarchy import fcluster

        # Get cluster assignments at various thresholds
        for threshold in np.linspace(0, linkage_matrix[:, 2].max(), 10):
            clusters = fcluster(linkage_matrix, threshold, criterion="distance")
            for cluster_id in np.unique(clusters):
                indices = np.where(clusters == cluster_id)[0]
                if len(indices) > 1:
                    # Draw convex hull or connections
                    cluster_points = embeddings[indices]
                    for i in range(len(cluster_points)):
                        for j in range(i + 1, len(cluster_points)):
                            fig.add_trace(
                                go.Scatter(
                                    x=[cluster_points[i, 0], cluster_points[j, 0]],
                                    y=[cluster_points[i, 1], cluster_points[j, 1]],
                                    mode="lines",
                                    line={
                                        "color": "rgba(128,128,128,0.2)",
                                        "width": 0.5,
                                    },
                                    showlegend=False,
                                    hoverinfo="skip",
                                )
                            )

    fig.update_layout(
        title="Algorithm Genealogy Map",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        template="plotly_white",
        showlegend=False,
    )

    if output_path:
        fig.write_html(output_path)
        logger.info("Saved algorithm map to %s", output_path)

    return fig


def plot_phylogeny_tree(
    tree: PhylogenyNode,
    output_path: str | Path | None = None,
) -> go.Figure:
    """Plot phylogenetic tree as dendrogram.

    Args:
        tree: Root PhylogenyNode
        output_path: Optional path to save HTML

    Returns:
        Plotly Figure
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    def add_node(node: PhylogenyNode, x: float, y: float, depth: int):
        # Add node marker
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker={"size": 10, "color": "blue" if node.is_leaf() else "red"},
                text=[node.name if node.is_leaf() else ""],
                textposition="top center",
                showlegend=False,
            )
        )

        if not node.is_leaf():
            # Draw children
            n_children = len(node.children)
            for i, child in enumerate(node.children):
                child_x = x + (i - (n_children - 1) / 2) * 0.5 / (depth + 1)
                child_y = y - 1.0

                # Connect parent to child
                fig.add_trace(
                    go.Scatter(
                        x=[x, child_x],
                        y=[y, child_y],
                        mode="lines",
                        line={"color": "gray", "width": 1},
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

                add_node(child, child_x, child_y, depth + 1)

    add_node(tree, 0, 0, 0)

    fig.update_layout(
        title="Algorithm Phylogeny",
        xaxis_title="",
        yaxis_title="Distance",
        template="plotly_white",
        showlegend=False,
    )

    if output_path:
        fig.write_html(output_path)
        logger.info("Saved phylogeny tree to %s", output_path)

    return fig


# =============================================================================
# High-Level Pipeline
# =============================================================================


def run_genealogy_analysis(
    experiment_results: list[dict],
    output_dir: str | Path,
    config: AlgorithmMapConfig | None = None,
) -> dict:
    """Run full genealogy analysis on experiment results.

    Args:
        experiment_results: List of experiment result dicts with config and metadata
        output_dir: Output directory for plots and data
        config: Analysis configuration

    Returns:
        Dictionary with embeddings, labels, phylogeny tree
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [r.get("config", {}) for r in experiment_results]
    metadatas = [r.get("metadata", {}) for r in experiment_results]

    # Generate algorithm map
    embeddings, labels, linkage_matrix = generate_algorithm_map(
        configs, metadatas, config
    )

    # Save embeddings
    np.save(output_dir / "embeddings.npy", embeddings)
    with (output_dir / "labels.json").open("w") as f:
        json.dump(labels, f)

    # Build and save phylogeny tree
    if linkage_matrix is not None:
        tree = linkage_to_tree(linkage_matrix, labels)
        # Save linkage matrix
        np.save(output_dir / "linkage_matrix.npy", linkage_matrix)

        # Plot phylogeny
        plot_phylogeny_tree(tree, output_dir / "phylogeny.html")

    # Plot algorithm map
    plot_algorithm_map(
        embeddings,
        labels,
        metadatas,
        linkage_matrix,
        color_by=config.color_by if config else "family",
        size_by=config.size_by if config else None,
        output_path=output_dir / "algorithm_map.html",
    )

    return {
        "embeddings": embeddings,
        "labels": labels,
        "linkage_matrix": linkage_matrix,
        "output_dir": output_dir,
    }


__all__ = [
    "AlgorithmFingerprint",
    "AlgorithmMapConfig",
    "PhylogenyNode",
    "build_fingerprint_matrix",
    "build_phylogeny",
    "extract_fingerprint",
    "generate_algorithm_map",
    "linkage_to_tree",
    "plot_algorithm_map",
    "plot_phylogeny_tree",
    "reduce_dimensions",
    "run_genealogy_analysis",
]
