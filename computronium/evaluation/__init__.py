"""Evaluation package: standardized benchmarks and MetricSuite."""

from computronium.evaluation.base import (
    BenchmarkResult,
    EvaluatorBase,
    MetricSuite,
    cross_validate,
    evaluate_model_on_task,
    registry_evaluator,
)
from computronium.evaluation.benchmarks import (
    BenchmarkRegistry,
    cifar10_benchmark,
    get_benchmark,
    list_benchmarks,
    mnist_benchmark,
    tiny_shakespeare_benchmark,
)
from computronium.evaluation.cross_domain import (
    BenchmarkSuiteConfig,
    BenchmarkSuiteResult,
    CrossDomainBenchmarkSuite,
    run_cross_domain_benchmark,
)
from computronium.evaluation.fairness import (
    BenchmarkRunner,
    FairnessContract,
    ResourceAwareBenchmarkRunner,
    validate_fairness,
)

__all__ = [  # ruff: ignore[unsorted-dunder-all]
    # Base
    "EvaluatorBase",
    "MetricSuite",
    "BenchmarkResult",
    "evaluate_model_on_task",
    "cross_validate",
    "registry_evaluator",
    # Benchmarks
    "BenchmarkRegistry",
    "get_benchmark",
    "list_benchmarks",
    "mnist_benchmark",
    "cifar10_benchmark",
    "tiny_shakespeare_benchmark",
    # Cross-domain suite
    "BenchmarkSuiteConfig",
    "BenchmarkSuiteResult",
    "CrossDomainBenchmarkSuite",
    "run_cross_domain_benchmark",
    # Fairness (PR-6)  # ruff: ignore[commented-out-code]
    "FairnessContract",
    "validate_fairness",
    "BenchmarkRunner",
    "ResourceAwareBenchmarkRunner",
]
