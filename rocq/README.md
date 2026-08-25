# Computronium formal verification (Rocq)

Machine-checked statements for `EnergyMinimizationDynamics` and the
EqProp control-Lyapunov claim, ported from the original Lean scaffold.

## Setup (one time)
- System package: `apt install rocq` (v9.x), or
- opam: `opam install rocq coq-stdlib`

## Build
```
make            # compiles Utils -> EnergyDynamics
make clean      # removes build artifacts
```

## Interactive proving
- VS Code: install **VsCoq 2** extension; open any `.v` file.
- Emacs: Proof General.
- Logical path is `-Q . Computronium`; import via
  `From Computronium Require Import Utils.`

## Adding a proof
1. State the theorem with full explicit hypotheses.
2. If not proved immediately: write `Admitted.` plus a STUB comment
   describing exactly what's missing (see EnergyDynamics.v for style).
3. Record it in the status table at the top of the module.

## Status conventions
Every module header carries a Proved / Admitted / Stub table. Nothing
may claim more than its proof delivers — no vacuous `Qed`s, no hidden
admits. Property-test counterparts live in
`tests/property/test_eqprop_locality.py`.
