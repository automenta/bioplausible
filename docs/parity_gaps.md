# Biological Parity Gap Documentation

*Sprint 1.5.3.* Every model whose `parity_threshold` exceeds the default
`0.05` must have a section here explaining the biological trade-off. This is
enforced by `test_parity_threshold_documented` in `test_backprop_parity.py`
and by the `biopl-registry-audit` check.

A **parity gap** = the bio-plausible model achieving lower accuracy than
standard backprop on the same task. An *elevated* threshold is a deliberate,
data-driven acceptance of that gap because closing it would require abandoning
the biological constraint the model exists to honor.

---

## pepita

- **Threshold**: `parity_threshold: 0.2` (see `hyperparams/pepita.yaml`)
- **Measured gap**: ~0.196 on the synthetic 64-dim parity task (from `sweep_results.json`).
- **Biological rationale**:
  - PEPITA uses a **single forward perturbation** (`x + σ·P·e`) to extract a
    layer-local, forward-only learning signal. It performs **no backward pass
    at all** — the error signal is transmitted forward through a fixed random
    projection `P`.
  - Information-theoretic ceiling: the learning signal is a noisy, compressed
    proxy of the true gradient. There is no mechanism to recover the exact
    backprop gradient, unlike EqProp (which converges to it via settling) or
    FA (which aligns `B → Wᵀ` over training).
  - This is a genuine **forward-only / zero-backward trade-off**: the model is
    the most neuromorphically faithful (no weight-transpose, no error
    backpropagation), which is precisely why its gap cannot be tuned away
    without making it a different algorithm.
- **Why not lower the threshold**: sweeping `lr` (sweep_results.json) showed
  the gap saturates near ~0.2 regardless of learning rate. The remaining
  distance is algorithmic, not hyperparameter noise.

---

## (No other models currently exceed 0.05)

`eqprop_mlp`, `directed_ep`, `forward_forward`, and `equitile` all carry the
default `parity_threshold: 0.05`. If any future tuning pushes one above `0.05`,
add its section here before adjusting its YAML.
