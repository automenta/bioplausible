"""P4-lite surrogate sanity (plan §P4-lite step 1): does a quantized-forward
fpga facade materially distort accuracy vs the float base on matched configs?

Compares ``looped_mlp`` (float base) against ``quantized_looped_mlp`` (fpga
facade, bits=8) and the analog facade (``noisy_looped_mlp``) on identical
configs so the substrate distortion is isolated from search variance.

Emits one JSON verdict: per-config accuracy/flops deltas and a binary
"distorts materially" flag using the parity_threshold (0.05).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from computronium.experiment.probe import CoreTrainerDriver

logger = logging.getLogger("p4lite_surrogate_sanity")

_DEFAULT_TASK = "mnist"
_EPOCHS = 1
_DEVICE = "cuda"
_SEED = 0
_PARITY_THRESHOLD = 0.05
_OUT = "logs/p4lite_surrogate_sanity.json"

_CONFIGS: tuple[dict[str, object], ...] = (
    {
        "hidden_dim": 128,
        "max_steps": 20,
        "num_layers": 2,
        "lr": 1e-3,
        "use_spectral_norm": True,
    },
    {
        "hidden_dim": 256,
        "max_steps": 40,
        "num_layers": 3,
        "lr": 3e-3,
        "use_spectral_norm": True,
    },
)


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    driver = CoreTrainerDriver(
        num_workers=0,
        batch_size=128,
        track_energy=False,
        track_flops=True,
        track_memory=True,
        record_results=True,
    )
    start = time.time()
    rows: list[dict[str, object]] = []
    for model in ("eqprop_mlp", "quantized_looped_mlp", "noisy_looped_mlp"):
        for cfg in _CONFIGS:
            metrics = driver.train(
                model=model,
                task=_DEFAULT_TASK,
                config=dict(cfg),
                seed=_SEED,
                epochs=_EPOCHS,
                device=_DEVICE,
            )
            rows.append({"model": model, "config": cfg, **metrics})
            logger.info(
                "%s cfg%d -> acc=%.4f flops=%d",
                model,
                _CONFIGS.index(cfg),
                metrics["final_acc"],
                metrics["forward_flops"],
            )

    base = {c: m for c, m in enumerate(rows) if m["model"] == "eqprop_mlp"}
    rel = [r for r in rows if r["model"] != "eqprop_mlp"]
    verdicts: list[dict[str, object]] = []
    distorted = False
    for r in rel:
        cfg_idx = _CONFIGS.index(dict(r["config"]))
        b = base[cfg_idx]
        delta = r["final_acc"] - b["final_acc"]
        flops_delta = (r["forward_flops"] - b["forward_flops"]) / max(
            1, b["forward_flops"]
        )
        flag = abs(delta) > _PARITY_THRESHOLD  # parity_threshold
        distorted = distorted or flag
        verdicts.append({
            "facade": r["model"],
            "config_idx": cfg_idx,
            "base_acc": b["final_acc"],
            "facade_acc": r["final_acc"],
            "acc_delta": delta,
            "flops_delta_frac": flops_delta,
            "distorts": flag,
        })
        logger.info(
            "%s cfg%d acc_delta=%.4f flops_delta=%.2f distort=%s",
            r["model"],
            cfg_idx,
            delta,
            flops_delta,
            flag,
        )

    report = {
        "task": _DEFAULT_TASK,
        "epochs": _EPOCHS,
        "device": _DEVICE,
        "seed": _SEED,
        "budget_parallel_units_approx_s": round(time.time() - start, 1),
        "verdicts": verdicts,
        "distorts_materially": distorted,
    }
    out = Path(_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    logger.info("report written to %s  distort=%s", out, distorted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
