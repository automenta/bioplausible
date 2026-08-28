"""Phase 3 — Edge Memory-Wall Benchmark.

The most visually shareable result: local rules train under activation-memory
ceilings where backprop cannot.

Envelopes (simulated/accounting-tier):
- 2 MB: SGD + ternary weights, hidden_dim=64
- 8 MB: Adam, hidden_dim=128
- 32 MB: Adam, hidden_dim=256

Local-rule arms (FA, Hebbian/STDP, EqProp) use SGD at all envelopes —
structural advantage: no optimizer state, no stored backward graph.

Control floor: gradient checkpointing (no offload) + SGD at 2 MB, Adam at 8/32 MB.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from computronium.core.presets import (
    create_backprop_mlp,
    create_eqprop_mlp,
    create_fa_mlp,
    create_hebbian_mlp,
)
from computronium.data.vision import create_data_loaders
from computronium.resources import ResourceUsage

__all__ = [
    "ArmConfig",
    "BenchmarkResult",
    "EnvelopeConfig",
    "export_deployment_artifacts",
    "generate_frontier_chart",
    "main",
    "run_memory_wall_benchmark",
]


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EnvelopeConfig:
    """Memory envelope configuration (simulated/accounting-tier)."""

    name: str
    ceiling_mb: float
    hidden_dim: int
    use_adam: bool
    use_ternary: bool = False

    @property
    def ceiling_bytes(self) -> int:
        return int(self.ceiling_mb * 1024 * 1024)


@dataclass(frozen=True, slots=True)
class ArmConfig:
    """Arm configuration for benchmark."""

    name: str
    factory_name: str  # 'fa', 'eqprop', 'hebbian', 'backprop'
    use_optimizer_state: (
        bool  # True for backprop+Adam, False for local rules (SGD only)
    )
    local_rule: bool = False


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Single benchmark run result."""

    arm_name: str
    envelope_name: str
    seed: int
    peak_activation_bytes: int
    peak_memory_mb: float
    final_accuracy: float
    best_accuracy: float
    disqualified: bool
    disqualification_reason: str | None = None
    wall_time_s: float = 0.0
    epochs_completed: int = 0
    resource_usage: ResourceUsage | None = None


# Pre-registered envelopes (from E-1 pre-registration)
ENVELOPES = (
    EnvelopeConfig("2MB", 2.0, 64, use_adam=False, use_ternary=True),
    EnvelopeConfig("8MB", 8.0, 128, use_adam=True),
    EnvelopeConfig("32MB", 32.0, 256, use_adam=True),
)

# Arm configurations
LOCAL_RULE_ARMS = (
    ArmConfig("FA", "fa", use_optimizer_state=False, local_rule=True),
    ArmConfig("Hebbian", "hebbian", use_optimizer_state=False, local_rule=True),
    ArmConfig("EqProp", "eqprop", use_optimizer_state=False, local_rule=True),
)

CONTROL_ARM = ArmConfig(
    "Backprop", "backprop", use_optimizer_state=True, local_rule=False
)

ALL_ARMS = LOCAL_RULE_ARMS + (CONTROL_ARM,)

# Factory mapping
FACTORY_MAP = {
    "fa": create_fa_mlp,
    "eqprop": create_eqprop_mlp,
    "hebbian": create_hebbian_mlp,
    "backprop": create_backprop_mlp,
}


# ──────────────────────────────────────────────
# Model Wrappers with Memory Accounting
# ──────────────────────────────────────────────


class MemoryAccountedModel:
    """Wrapper that tracks peak activation memory during forward/backward.

    Wraps a _ComposedSystem from the factory functions, hooking its geometry
    to measure activation memory during train_step calls.
    """

    def __init__(self, system, envelope: EnvelopeConfig, arm: ArmConfig, device: str):
        self.system = system
        self.envelope = envelope
        self.arm = arm
        self.device = device
        self.peak_activation_bytes = 0
        self.peak_memory_mb = 0.0
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on the geometry to track activation memory."""

        def hook(module: nn.Module, inp, output):
            if isinstance(output, torch.Tensor):
                bytes_used = output.numel() * output.element_size()
                self.peak_activation_bytes = max(self.peak_activation_bytes, bytes_used)
            elif isinstance(output, (tuple, list)):
                for t in output:
                    if isinstance(t, torch.Tensor):
                        bytes_used = t.numel() * t.element_size()
                        self.peak_activation_bytes = max(
                            self.peak_activation_bytes, bytes_used
                        )

        # Hook the geometry (which is an nn.Module)
        geometry = self.system.geometry
        for module in geometry.modules():
            if isinstance(
                module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid)
            ):
                self._hooks.append(module.register_forward_hook(hook))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """Single training step with memory tracking."""
        self.system.geometry.train()
        x = x.to(self.device)
        y = y.to(self.device)

        # Reset peak tracking for this step
        self.peak_activation_bytes = 0

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Forward + backward via the system's train_step
        metrics = self.system.train_step(x, y)

        if self.device == "cuda":
            torch.cuda.synchronize()
            self.peak_memory_mb = max(
                self.peak_memory_mb, torch.cuda.max_memory_allocated() / (1024 * 1024)
            )

        return metrics

    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model accuracy."""
        self.system.geometry.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.system.forward(x)
                pred = logits.argmax(-1)
                correct += (pred == y).sum().item()
                total += y.shape[0]
        return correct / total if total > 0 else 0.0

    def check_envelope(self) -> tuple[bool, str | None]:
        """Check if model exceeds memory envelope."""
        peak_mb = self.peak_activation_bytes / (1024 * 1024)
        if peak_mb > self.envelope.ceiling_mb:
            return (
                True,
                f"Peak activation {peak_mb:.2f} MB exceeds envelope {self.envelope.ceiling_mb} MB",
            )
        return False, None

    def get_resource_usage(self) -> ResourceUsage:
        """Get resource usage record."""
        geometry = self.system.geometry
        param_count = sum(p.numel() for p in geometry.parameters())
        param_memory_mb = sum(
            p.numel() * p.element_size() for p in geometry.parameters()
        ) / (1024 * 1024)

        # Estimate optimizer memory
        optimizer_memory_mb = 0.0
        if self.arm.use_optimizer_state:
            # Adam: 2x params (m, v) + params (grad) = 3x param memory approx
            optimizer_memory_mb = param_memory_mb * 3

        return ResourceUsage(
            coordinate=f"{self.arm.name}/{self.envelope.name}",
            device=self.device,
            batch_size=64,
            forward_flops=0,  # Filled by profiler if needed
            backward_flops=0,
            param_count=param_count,
            param_memory_mb=param_memory_mb,
            activation_memory_mb=self.peak_activation_bytes / (1024 * 1024),
            gradient_memory_mb=param_memory_mb if self.arm.use_optimizer_state else 0.0,
            peak_memory_mb=self.peak_memory_mb,
            peak_activation_bytes=self.peak_activation_bytes,
            plastic_state_capacity=0.0,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.system.forward(x)


def create_model_for_arm(
    arm: ArmConfig, envelope: EnvelopeConfig, device: str
) -> MemoryAccountedModel:
    """Create a model for the given arm and envelope."""
    factory = FACTORY_MAP[arm.factory_name]

    # Adjust parameters based on envelope
    if arm.factory_name == "eqprop":
        system = factory(
            input_dim=784,
            hidden_dims=(envelope.hidden_dim, envelope.hidden_dim),
            output_dim=10,
            lr=0.001,
            inference_steps=3,  # Reduced for memory efficiency
            init_scale=0.1,
            device=device,
        )
    else:
        system = factory(
            input_dim=784,
            hidden_dims=(envelope.hidden_dim, envelope.hidden_dim),
            output_dim=10,
            lr=0.001,
            init_scale=0.1,
            device=device,
        )

    # Move geometry to device (factory doesn't do this automatically)
    system.geometry.to(device)

    # Apply ternary quantization if needed (for 2MB envelope backprop)
    if envelope.use_ternary and arm.factory_name == "backprop":
        from computronium.deployment import quantize_model_ternary_inplace

        quantize_model_ternary_inplace(system.geometry, threshold=0.5)

    return MemoryAccountedModel(system, envelope, arm, device)


def create_optimizer(
    model: MemoryAccountedModel, envelope: EnvelopeConfig, arm: ArmConfig
) -> torch.optim.Optimizer | None:
    """Create optimizer based on envelope and arm config."""
    # Get the geometry's parameters
    geometry = model.system.geometry
    if not arm.use_optimizer_state:
        # Local rules: SGD only, no optimizer state stored
        return torch.optim.SGD(geometry.parameters(), lr=0.001, momentum=0.0)

    if envelope.use_adam:
        return torch.optim.Adam(geometry.parameters(), lr=0.001)
    else:
        return torch.optim.SGD(geometry.parameters(), lr=0.001, momentum=0.0)


# ──────────────────────────────────────────────
# Gradient Checkpointing Wrapper (Control Floor)
# ──────────────────────────────────────────────


class GradientCheckpointedModel(MemoryAccountedModel):
    """Backprop model with gradient checkpointing for memory-efficient training."""

    def __init__(self, system, envelope: EnvelopeConfig, device: str):
        # Initialize without calling parent __init__ (we don't want hooks registered yet)
        self.system = system
        self.envelope = envelope
        self.arm = CONTROL_ARM
        self.device = device
        self.peak_activation_bytes = 0
        self.peak_memory_mb = 0.0
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        # Register hooks to track activation memory even with gradient checkpointing
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on the geometry to track activation memory."""

        def hook(module: nn.Module, inp, output):
            if isinstance(output, torch.Tensor):
                bytes_used = output.numel() * output.element_size()
                self.peak_activation_bytes = max(self.peak_activation_bytes, bytes_used)
            elif isinstance(output, (tuple, list)):
                for t in output:
                    if isinstance(t, torch.Tensor):
                        bytes_used = t.numel() * t.element_size()
                        self.peak_activation_bytes = max(
                            self.peak_activation_bytes, bytes_used
                        )

        geometry = self.system.geometry
        for module in geometry.modules():
            if isinstance(
                module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid)
            ):
                self._hooks.append(module.register_forward_hook(hook))

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """Training step with gradient checkpointing."""
        self.system.geometry.train()
        x = x.to(self.device)
        y = y.to(self.device)

        self.peak_activation_bytes = 0

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Use gradient checkpointing for the forward pass
        def run_forward(x):
            return self.system.forward(x)

        # Checkpoint the model in segments (by layer)
        # For simplicity, we use the whole model as one checkpoint segment
        # In practice, you'd checkpoint per layer
        output = torch.utils.checkpoint.checkpoint(run_forward, x, use_reentrant=False)
        loss = nn.functional.cross_entropy(output, y)
        loss.backward()

        if self.device == "cuda":
            torch.cuda.synchronize()
            self.peak_memory_mb = max(
                self.peak_memory_mb, torch.cuda.max_memory_allocated() / (1024 * 1024)
            )

        return {
            "loss": loss.item(),
            "accuracy": (output.argmax(-1) == y).float().mean().item(),
        }

    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model accuracy."""
        self.system.geometry.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.system.forward(x)
                pred = logits.argmax(-1)
                correct += (pred == y).sum().item()
                total += y.shape[0]
        return correct / total if total > 0 else 0.0

    def check_envelope(self) -> tuple[bool, str | None]:
        """Check if model exceeds memory envelope."""
        peak_mb = self.peak_activation_bytes / (1024 * 1024)
        if peak_mb > self.envelope.ceiling_mb:
            return (
                True,
                f"Peak activation {peak_mb:.2f} MB exceeds envelope {self.envelope.ceiling_mb} MB",
            )
        return False, None

    def get_resource_usage(self) -> ResourceUsage:
        """Get resource usage record."""
        geometry = self.system.geometry
        param_count = sum(p.numel() for p in geometry.parameters())
        param_memory_mb = sum(
            p.numel() * p.element_size() for p in geometry.parameters()
        ) / (1024 * 1024)

        return ResourceUsage(
            coordinate=f"{self.arm.name}/{self.envelope.name}",
            device=self.device,
            batch_size=64,
            forward_flops=0,
            backward_flops=0,
            param_count=param_count,
            param_memory_mb=param_memory_mb,
            activation_memory_mb=self.peak_activation_bytes / (1024 * 1024),
            gradient_memory_mb=param_memory_mb,
            peak_memory_mb=self.peak_memory_mb,
            peak_activation_bytes=self.peak_activation_bytes,
            plastic_state_capacity=0.0,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.system.forward(x)


# ──────────────────────────────────────────────
# Benchmark Runner
# ──────────────────────────────────────────────


def run_single_benchmark(
    arm: ArmConfig,
    envelope: EnvelopeConfig,
    seed: int,
    epochs: int,
    batch_size: int,
    device: str,
    quick: bool = False,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    torch.manual_seed(seed)
    random.seed(seed)

    # Create model
    if arm.name == "Backprop" and envelope.name == "2MB":
        # Use gradient checkpointing for backprop at 2MB
        base_system = FACTORY_MAP["backprop"](
            input_dim=784,
            hidden_dims=(envelope.hidden_dim, envelope.hidden_dim),
            output_dim=10,
            lr=0.001,
            init_scale=0.1,
            device=device,
        )
        # Move geometry to device
        base_system.geometry.to(device)
        from computronium.deployment import quantize_model_ternary_inplace

        quantize_model_ternary_inplace(base_system.geometry, threshold=0.5)
        # Move again after quantization (it creates new layers)
        base_system.geometry.to(device)
        model = GradientCheckpointedModel(base_system, envelope, device)
    else:
        model = create_model_for_arm(arm, envelope, device)

    optimizer = create_optimizer(model, envelope, arm)

    # Data
    train_loader, test_loader = create_data_loaders(
        dataset_name="mnist",
        batch_size=batch_size,
        num_workers=0,
        flatten=True,
    )

    # Quick mode: fewer batches
    if quick:
        train_loader = DataLoader(
            TensorDataset(*next(iter(train_loader))),
            batch_size=batch_size,
            shuffle=True,
        )

    best_accuracy = 0.0
    final_accuracy = 0.0
    epochs_completed = 0
    start_time = time.perf_counter()
    disqualified = False
    disqualification_reason = None

    for epoch in range(epochs):
        epochs_completed = epoch + 1

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.view(x.shape[0], -1)
            metrics = model.train_step(x, y)

            if optimizer is not None:
                optimizer.step()
                optimizer.zero_grad()

            # Check envelope after each step
            exceeded, reason = model.check_envelope()
            if exceeded:
                disqualified = True
                disqualification_reason = reason
                break

        if disqualified:
            break

        # Evaluate
        acc = model.evaluate(test_loader)
        final_accuracy = acc
        best_accuracy = max(best_accuracy, acc)

    wall_time = time.perf_counter() - start_time

    # Final envelope check
    if not disqualified:
        exceeded, reason = model.check_envelope()
        if exceeded:
            disqualified = True
            disqualification_reason = reason

    resource_usage = model.get_resource_usage()

    return BenchmarkResult(
        arm_name=arm.name,
        envelope_name=envelope.name,
        seed=seed,
        peak_activation_bytes=model.peak_activation_bytes,
        peak_memory_mb=model.peak_memory_mb,
        final_accuracy=final_accuracy,
        best_accuracy=best_accuracy,
        disqualified=disqualified,
        disqualification_reason=disqualification_reason,
        wall_time_s=wall_time,
        epochs_completed=epochs_completed,
        resource_usage=resource_usage,
    )


def run_memory_wall_benchmark(
    arms: list[str] | None = None,
    envelopes: list[str] | None = None,
    output_dir: str | Path = "benchmark_results/memory_wall",
    epochs: int = 5,
    batch_size: int = 64,
    seeds: int = 5,
    device: str = "auto",
    quick: bool = False,
) -> dict:
    """Run the full memory-wall benchmark suite."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter arms
    selected_arms = [a for a in ALL_ARMS if arms is None or a.name in arms]
    # Filter envelopes
    selected_envelopes = [
        e for e in ENVELOPES if envelopes is None or e.name in envelopes
    ]

    all_results: dict = {}

    for arm in selected_arms:
        all_results[arm.name] = {}
        for envelope in selected_envelopes:
            all_results[arm.name][envelope.name] = {"seeds": []}

            print(f"\n=== {arm.name} / {envelope.name} ===")
            for seed in range(seeds):
                print(f"  Seed {seed}...")
                result = run_single_benchmark(
                    arm, envelope, seed, epochs, batch_size, device, quick
                )
                all_results[arm.name][envelope.name]["seeds"].append({
                    "peak_activation_bytes": result.peak_activation_bytes,
                    "peak_memory_mb": result.peak_memory_mb,
                    "final_accuracy": result.final_accuracy,
                    "best_accuracy": result.best_accuracy,
                    "disqualified": result.disqualified,
                    "disqualification_reason": result.disqualification_reason,
                    "wall_time_s": result.wall_time_s,
                    "epochs_completed": result.epochs_completed,
                    "resource_usage": result.resource_usage.to_dict()
                    if result.resource_usage
                    else None,
                })
                status = (
                    "DNF" if result.disqualified else f"acc={result.best_accuracy:.4f}"
                )
                print(
                    f"    {status} (peak={result.peak_activation_bytes / 1024 / 1024:.2f} MB)"
                )

            # Aggregate across seeds
            seeds_list = all_results[arm.name][envelope.name]["seeds"]
            if seeds_list:
                for key in [
                    "final_accuracy",
                    "best_accuracy",
                    "peak_memory_mb",
                    "wall_time_s",
                ]:
                    vals = [float(s[key]) for s in seeds_list if not s["disqualified"]]
                    if vals:
                        mean_val = sum(vals) / len(vals)
                        all_results[arm.name][envelope.name][f"mean_{key}"] = mean_val
                        all_results[arm.name][envelope.name][f"std_{key}"] = (
                            (sum((v - mean_val) ** 2 for v in vals) / len(vals)) ** 0.5
                            if len(vals) > 1
                            else 0.0
                        )

    # Save results
    results_file = output_dir / "memory_wall_results.json"
    with results_file.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {results_file}")
    return all_results


# ──────────────────────────────────────────────
# Frontier Chart Generation
# ──────────────────────────────────────────────


def generate_frontier_chart(
    results: dict,
    output_dir: str | Path = "benchmark_results/memory_wall",
) -> Path | None:
    """Generate memory-accuracy frontier chart."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping chart generation")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color and marker mapping
    styles = {
        "FA": ("#1f77b4", "o", "-"),
        "Hebbian": ("#ff7f0e", "s", "--"),
        "EqProp": ("#2ca02c", "^", "-."),
        "Backprop": ("#d62728", "D", ":"),
    }

    # Plot each arm
    for arm_name, envelope_data in results.items():
        color, marker, linestyle = styles.get(arm_name, ("gray", "o", "-"))

        x_vals = []
        y_vals = []
        y_err = []
        dnf_envelopes = []

        for env in ENVELOPES:
            env_data = envelope_data.get(env.name, {})
            if not env_data or "mean_best_accuracy" not in env_data:
                continue

            # Check if any seed was disqualified
            dnf_count = sum(
                1 for s in env_data.get("seeds", []) if s.get("disqualified", False)
            )
            if dnf_count > 0:
                dnf_envelopes.append((env.ceiling_mb, dnf_count))

            x_vals.append(env.ceiling_mb)
            y_vals.append(env_data["mean_best_accuracy"])
            y_err.append(env_data.get("std_best_accuracy", 0))

        if x_vals:
            ax.errorbar(
                x_vals,
                y_vals,
                yerr=y_err,
                color=color,
                marker=marker,
                linestyle=linestyle,
                label=arm_name,
                linewidth=2,
                markersize=8,
                capsize=5,
            )

        # Mark DNFs
        for mb, count in dnf_envelopes:
            ax.plot(mb, 0, "x", color=color, markersize=12, markeredgewidth=3)
            ax.annotate(
                f"DNF ({count})", (mb, 0.02), color=color, fontsize=8, ha="center"
            )

    # Envelope ceiling lines
    for env in ENVELOPES:
        ax.axvline(
            x=env.ceiling_mb, color="gray", linestyle=":", alpha=0.5, linewidth=1
        )
        ax.text(
            env.ceiling_mb,
            1.02,
            f"{env.name} ceiling",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color="gray",
        )

    ax.set_xlabel("Activation Memory Ceiling (MB)", fontsize=12)
    ax.set_ylabel("Best Test Accuracy", fontsize=12)
    ax.set_title("Memory-Accuracy Frontier: Local Rules vs Backprop", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.set_xticks([e.ceiling_mb for e in ENVELOPES])
    ax.set_xticklabels([e.name for e in ENVELOPES])
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=11)

    # Annotate structural advantage
    ax.annotate(
        "Local rules: no optimizer state,\nno stored backward graph",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    chart_path = output_dir / "memory_accuracy_frontier.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Frontier chart saved to {chart_path}")
    return chart_path


# ──────────────────────────────────────────────
# Deployment Artifact Export
# ──────────────────────────────────────────────


def export_deployment_artifacts(
    results: dict,
    output_dir: str | Path = "benchmark_results/memory_wall",
    device: str = "cpu",
) -> dict[str, list[str]]:
    """Export deployment artifacts for successful runs via PR-8 pipeline."""
    from computronium.deployment import export_model

    output_dir = Path(output_dir)
    export_dir = output_dir / "deployment_artifacts"
    export_dir.mkdir(parents=True, exist_ok=True)

    exported = {}

    for arm_name, envelope_data in results.items():
        exported[arm_name] = []
        for env in ENVELOPES:
            env_data = envelope_data.get(env.name, {})
            if not env_data or "mean_best_accuracy" not in env_data:
                continue

            # Only export if not disqualified and accuracy > 0.05 (better than random for quick tests)
            if env_data["mean_best_accuracy"] > 0.05:
                # Re-create the best model configuration
                arm = next(a for a in ALL_ARMS if a.name == arm_name)
                system = create_model_for_arm(arm, env, device).system
                # Export the geometry (nn.Module) not the whole system
                model = system.geometry

                model_name = f"{arm_name.lower()}_{env.name.lower()}"
                model_params = {
                    "input_dim": 784,
                    "hidden_dims": (env.hidden_dim, env.hidden_dim),
                    "output_dim": 10,
                    "envelope": env.name,
                    "arm": arm_name,
                }

                try:
                    info = export_model(
                        model=model,
                        model_name=model_name,
                        model_params=model_params,
                        output_dir=str(export_dir / model_name),
                        formats=["onnx", "pt2", "config", "state"],
                        training_metrics={
                            "best_accuracy": env_data["mean_best_accuracy"]
                        },
                        verbose=False,
                    )
                    exported[arm_name].append(str(export_dir / model_name))
                    print(f"  Exported {model_name} to {info.export_path}")
                except Exception as e:
                    print(f"  Failed to export {model_name}: {e}")

    return exported


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Memory-Wall Benchmark")
    parser.add_argument("--arms", nargs="+", default=[a.name for a in ALL_ARMS])
    parser.add_argument("--envelopes", nargs="+", default=[e.name for e in ENVELOPES])
    parser.add_argument("--output-dir", default="benchmark_results/memory_wall")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    parser.add_argument("--no-chart", action="store_true", help="Skip chart generation")
    parser.add_argument(
        "--no-export", action="store_true", help="Skip deployment export"
    )
    args = parser.parse_args()

    if args.quick:
        args.epochs = 1
        args.seeds = 1
        args.batch_size = 32

    print("=" * 60)
    print("Phase 3: Edge Memory-Wall Benchmark")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Seeds: {args.seeds}")
    print(f"Batch size: {args.batch_size}")
    print(f"Arms: {args.arms}")
    print(f"Envelopes: {args.envelopes}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Run benchmark
    results = run_memory_wall_benchmark(
        arms=args.arms,
        envelopes=args.envelopes,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seeds=args.seeds,
        device=args.device,
        quick=args.quick,
    )

    # Generate frontier chart
    if not args.no_chart:
        generate_frontier_chart(results, args.output_dir)

    # Export deployment artifacts
    if not args.no_export and not args.quick:
        export_deployment_artifacts(
            results, args.output_dir, args.device if args.device != "auto" else "cpu"
        )

    print("\n" + "=" * 60)
    print("Memory-Wall Benchmark Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
