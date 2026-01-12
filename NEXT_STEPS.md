# Next Steps for EqProp Development

## Current Status
- **Smoke Tests Passed**: All models in `bioplausible/models/` can be instantiated and run a forward pass using the default `gradient_method='bptt'`.
- **Equilibrium Mode Issue**: The `gradient_method='equilibrium'` (Implicit Differentiation) mode in `bioplausible/models/eqprop_base.py` is currently unstable.
  - It encounters `RuntimeError: Trying to backward through the graph a second time` or hangs/timeouts when attempting to fix the graph retention.
  - The `EquilibriumFunction` needs to be carefully debugged to handle the Vector-Jacobian Product (VJP) loop without leaking memory or retaining unnecessary graph segments.
  - **Action Item**: Debug `EquilibriumFunction.backward`. Isolate the VJP loop. Ensure `delta` is detached and `retain_graph` is handled correctly.

## Verification
- Run `bioplausible/tests/test_all_models.py` to ensure no regressions in standard BPTT mode.
- Tracks 52 (DFA) and 53 (CHL) are marked as "Partial" in `AGENTS.md`. Investigate why they are not fully passing (accuracy/Lipschitz targets).

## Refactoring
- The `gradient_method` argument was added to `EqPropModel` and `LoopedMLP` but might need to be propagated to other subclasses if they are to support Implicit Differentiation.

## Optimization
- Once `equilibrium` mode works, benchmark memory usage against `bptt` mode to verify O(1) memory claims.
