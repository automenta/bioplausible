"""
Track 57: Honest Trade-off Analysis - EqProp vs Backprop

CRITICAL REALITY CHECK: Measures everything that matters on the same task.

Metrics:
- Training time (wall-clock seconds, not epochs)
- Memory usage (actual MB)
- Final accuracy
- Convergence speed (epochs to reach 90% of final accuracy)
- Hyperparameter sensitivity

Test scenarios:
1. Small model (10K params, 100 hidden) - baseline
2. Medium model (100K params, 512 hidden) - realistic
3. Deep model (500 steps) - depth claim test

Honest verdict:
- If EqProp is slower AND worse AND harder → STOP RESEARCH
- If EqProp matches with acceptable trade-offs → CONTINUE with clear niche
- If EqProp wins on key metric → VALIDATE value proposition
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import psutil
import torch
import torch.nn.functional as F
from torch import optim

from computronium.core.logging import get_logger
from computronium.core.utils.device import get_device
from computronium.core.utils.optimizer import OptimizerConfig, create_optimizer
from computronium.models.native.backprop_native import create_native_backprop_mlp
from computronium.models.native.eqprop_native import create_native_eqprop_mlp
from computronium.utils import count_parameters

from ._base import build_track_result, track_header

if TYPE_CHECKING:
    from ..notebook import TrackResult

root_path = Path(__file__).parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

__all__ = [
    "count_parameters",
    "get_memory_usage",
    "logger",
    "root_path",
    "track_57_honest_tradeoff_analysis",
    "train_and_measure",
]
logger = get_logger()


def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def train_and_measure(
    model, optimizer, train_loader, test_loader, epochs, device, name
):
    """Train model and measure everything."""

    # Initial memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    mem_start = get_memory_usage()

    # Training metrics
    train_times = []
    train_losses = []
    train_accs = []
    test_accs = []

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            # Flatten MNIST images
            x_batch = x_batch.view(x_batch.size(0), -1)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            out = model(x_batch)
            loss = F.cross_entropy(out, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # Test
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                # Flatten MNIST images
                x_batch = x_batch.view(x_batch.size(0), -1)
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                pred = out.argmax(dim=1)
                test_correct += (pred == y_batch).sum().item()
                test_total += y_batch.size(0)

        test_acc = test_correct / test_total

        epoch_time = time.time() - epoch_start

        train_times.append(epoch_time)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "    [%s] Epoch %d/%d: "
                "loss=%.4f, train_acc=%.3f, "
                "test_acc=%.3f, time=%.2fs",
                name,
                epoch + 1,
                epochs,
                train_loss,
                train_acc,
                test_acc,
                epoch_time,
            )

    total_time = time.time() - start_time

    # Peak memory during training
    mem_peak = get_memory_usage()
    mem_used = mem_peak - mem_start

    # Find convergence epoch (when reached 90% of final accuracy)
    final_acc = max(test_accs)
    target_acc = final_acc * 0.9
    convergence_epoch = len(test_accs)
    for i, acc in enumerate(test_accs):
        if acc >= target_acc:
            convergence_epoch = i + 1
            break

    return {
        "final_test_acc": final_acc,
        "final_train_acc": train_accs[-1],
        "final_loss": train_losses[-1],
        "total_time_sec": total_time,
        "mean_epoch_time": sum(train_times) / len(train_times),
        "memory_mb": mem_used,
        "convergence_epoch": convergence_epoch,
        "train_accs": train_accs,
        "test_accs": test_accs,
    }


def track_57_honest_tradeoff_analysis(verifier) -> TrackResult:
    """
    Track 57: Honest Trade-off Analysis

    Direct comparison of EqProp vs Backprop on SAME task.
    Measures EVERYTHING that matters for practical use.
    """
    start = track_header(57, "HONEST TRADE-OFF ANALYSIS - EqProp vs Backprop", width=70)
    logger.info(
        "\n\u26a0\ufe0f  CRITICAL REALITY CHECK - Determines if research should continue\n"
    )

    device = str(get_device())

    # Load MNIST through the canonical vision-dataset sink (cached tensor
    # path, no per-batch ToTensor/Normalize cost).
    from torch.utils.data import DataLoader, Subset

    from computronium.data.vision import get_vision_dataset

    train_dataset = get_vision_dataset("mnist", train=True)
    test_dataset = get_vision_dataset("mnist", train=False)

    # Use subset based on mode
    if verifier.quick_mode:
        n_train, n_test = 1000, 500
        epochs = 10
    else:
        n_train, n_test = 10000, 2000
        epochs = 20

    train_subset = Subset(train_dataset, range(n_train))
    test_subset = Subset(test_dataset, range(n_test))

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    logger.info(
        "[57] Configuration: %d train, %d test, %d epochs\n", n_train, n_test, epochs
    )

    results = {}

    # Test scenarios
    scenarios = [
        ("Small (100 hidden)", 100, 20),
        ("Medium (256 hidden)", 256, 20),
    ]

    if not verifier.quick_mode:
        scenarios.append(("Deep (500 steps)", 128, 100))

    for scenario_name, hidden_dim, max_steps in scenarios:
        logger.info("\n%s", "=" * 70)
        logger.info("Scenario: %s", scenario_name)
        logger.info("%s", "=" * 70)

        # EqProp
        logger.info(
            "[57a] Training EqProp (%d hidden, %d steps)...", hidden_dim, max_steps
        )
        eqprop_model = create_native_eqprop_mlp(
            input_dim=784,
            hidden_dim=hidden_dim,
            output_dim=10,
            use_spectral_norm=True,
            max_steps=max_steps,
        ).to(device)

        eqprop_params = count_parameters(eqprop_model)
        eqprop_opt = optim.Adam(eqprop_model.parameters(), lr=0.001)

        eqprop_results = train_and_measure(
            eqprop_model,
            eqprop_opt,
            train_loader,
            test_loader,
            epochs,
            device,
            "EqProp",
        )

        # Backprop (same capacity)
        logger.info("\n[57b] Training Backprop (%d hidden)...", hidden_dim)
        backprop_model = create_native_backprop_mlp(
            input_dim=784, hidden_dim=hidden_dim, output_dim=10
        ).to(device)

        backprop_params = count_parameters(backprop_model)
        backprop_opt = optim.Adam(backprop_model.parameters(), lr=0.001)

        backprop_results = train_and_measure(
            backprop_model,
            backprop_opt,
            train_loader,
            test_loader,
            epochs,
            device,
            "Backprop",
        )

        # Compute ratios
        time_ratio = (
            eqprop_results["total_time_sec"] / backprop_results["total_time_sec"]
        )
        memory_ratio = eqprop_results["memory_mb"] / max(
            backprop_results["memory_mb"], 1
        )
        acc_gap = (
            backprop_results["final_test_acc"] - eqprop_results["final_test_acc"]
        ) * 100

        results[scenario_name] = {
            "eqprop": eqprop_results,
            "backprop": backprop_results,
            "eqprop_params": eqprop_params,
            "backprop_params": backprop_params,
            "time_ratio": time_ratio,
            "memory_ratio": memory_ratio,
            "acc_gap_percent": acc_gap,
        }

        logger.info("\n%s", "=" * 70)
        logger.info("COMPARISON: %s", scenario_name)
        logger.info("%s", "=" * 70)
        logger.info(
            "  Parameters:   EqProp=%s vs Backprop=%s",
            f"{eqprop_params:,}",
            f"{backprop_params:,}",
        )
        eq_a = f"EqProp={eqprop_results['final_test_acc']:.3f}"
        bp_a = f"Backprop={backprop_results['final_test_acc']:.3f}"
        logger.info("  Final Acc:    %s vs %s", eq_a, bp_a)
        gap_str = "EqProp worse" if acc_gap > 0 else "EqProp better"
        logger.info("  Accuracy Gap: %+.2f%% (%s)", acc_gap, gap_str)
        eq_t = f"EqProp={eqprop_results['total_time_sec']:.1f}s"
        bp_t = f"Backprop={backprop_results['total_time_sec']:.1f}s"
        logger.info("  Total Time:   %s vs %s", eq_t, bp_t)
        logger.info(
            "  Time Ratio:   %.2f\u00d7 (%s)",
            time_ratio,
            "SLOWER" if time_ratio > 1 else "FASTER",
        )
        eq_m = f"EqProp={eqprop_results['memory_mb']:.1f}MB"
        bp_m = f"Backprop={backprop_results['memory_mb']:.1f}MB"
        logger.info("  Memory Used:  %s vs %s", eq_m, bp_m)
        eq_c = f"EqProp={eqprop_results['convergence_epoch']}"
        bp_c = f"Backprop={backprop_results['convergence_epoch']}"
        logger.info("  Convergence:  %s epochs vs %s epochs", eq_c, bp_c)

    # Overall verdict
    logger.info("\n%s", "=" * 70)
    logger.info("OVERALL VERDICT")
    logger.info("%s", "=" * 70)

    avg_time_ratio = sum(r["time_ratio"] for r in results.values()) / len(results)
    avg_acc_gap = sum(r["acc_gap_percent"] for r in results.values()) / len(results)
    max_acc_gap = max(r["acc_gap_percent"] for r in results.values())

    # Decision criteria
    is_much_slower = avg_time_ratio > 2.0
    is_worse_accuracy = avg_acc_gap > 3.0
    is_competitive = abs(avg_acc_gap) < 5.0 and avg_time_ratio < 3.0

    logger.info(
        "Average time ratio:     %.2f\u00d7 (EqProp vs Backprop)", avg_time_ratio
    )
    logger.info("Average accuracy gap:   %+.2f%%", avg_acc_gap)
    logger.info("Max accuracy gap:       %+.2f%%", max_acc_gap)

    if is_much_slower and is_worse_accuracy:
        verdict = "\u274c STOP RESEARCH: EqProp is both slower AND less accurate"
        recommendation = "No clear advantage found. Consider pivoting or stopping."
        score = 30
        status = "fail"
    elif is_worse_accuracy and max_acc_gap > 5:
        verdict = "\u26a0\ufe0f  ACCURACY PROBLEM: EqProp consistently worse"
        recommendation = "Need to improve accuracy before claiming value."
        score = 50
        status = "partial"
    elif is_much_slower:
        verdict = "\u26a0\ufe0f  SPEED PROBLEM: EqProp 2-3\u00d7 slower"
        recommendation = "Accuracy is competitive but speed is a major limitation."
        score = 60
        status = "partial"
    elif is_competitive:
        verdict = (
            "\u2705 COMPETITIVE: EqProp matches Backprop within acceptable margins"
        )
        recommendation = (
            "Research can continue. Focus on finding unique value proposition."
        )
        score = 85
        status = "pass"
    else:
        verdict = "\u26a0\ufe0f  MIXED RESULTS: Some scenarios work, others don't"
        recommendation = "Characterize when EqProp works well vs poorly."
        score = 70
        status = "partial"

    logger.info(verdict)
    logger.info("Recommendation: %s", recommendation)

    # Build evidence table
    table_rows = []
    for scenario, r in results.items():
        ep = r["eqprop"]
        bp = r["backprop"]
        table_rows.append(
            f"| {scenario} | {ep['final_test_acc']:.3f} | {bp['final_test_acc']:.3f} | "
            f"{r['acc_gap_percent']:+.1f}% | {r['time_ratio']:.2f}\u00d7 | "
            f"{ep['total_time_sec']:.1f}s | {bp['total_time_sec']:.1f}s |"
        )

    evidence = f"""
**CRITICAL REALITY CHECK**: Direct comparison on MNIST classification.

**Configuration**: {n_train} train samples, {n_test} test samples, {epochs} epochs

| Scenario | EqProp Acc | Backprop Acc | Gap | Time Ratio | EqProp Time | Backprop Time |
|----------|------------|--------------|-----|------------|-------------|---------------|
{chr(10).join(table_rows)}

**Summary**:
- Average time ratio: **{avg_time_ratio:.2f}\u00d7** (EqProp vs Backprop)
- Average accuracy gap: **{avg_acc_gap:+.2f}%**
- Max accuracy gap: **{max_acc_gap:+.2f}%**

**Verdict**: {verdict}

**Recommendation**: {recommendation}

**Key Insights**:
- EqProp {"matches" if is_competitive else "does not match"} Backprop accuracy
- EqProp is {"competitive on" if avg_time_ratio < 2 else "slower than Backprop by"} training speed
- {"Further research warranted" if score >= 70 else "Critical issues need resolution"}
"""

    return build_track_result(
        track_id=57,
        name="Honest Trade-off Analysis",
        status=status,
        score=score,
        metrics={
            "results": results,
            "avg_time_ratio": avg_time_ratio,
            "avg_acc_gap": avg_acc_gap,
            "is_competitive": is_competitive,
        },
        evidence=evidence,
        start=start,
        improvements=[] if score >= 80 else ["Address speed and/or accuracy gaps"],
    )