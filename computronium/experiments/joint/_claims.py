"""Claims-scope markers for the shakedown suites (per-suite audit, session 5).

Scopes are AUDIT STATUS, not verdicts:
- ``plumbing_only``: forward loop contains no ψ mediation today, so results
  cannot evidence ψ-mediated behavior. (L3.5's own code comments document the
  full-training simplification; L3 is a plain-MLP damage test.)
- ``psi_wired_uncontrolled``: ψ IS stepped inside forward and modulates
  computation (empirically confirmed to differentiate plasticity types), but
  θ trains concurrently, ``plasticity.step`` receives ``None`` context, and
  there is no frozen-θ control — so these are suggestive, not clean, ψ
  evidence. Rewiring with frozen-θ phases + ``ThetaInvarianceAudit`` remains
  open and would upgrade these suites rather than invalidate them.
"""

CLAIMS_SCOPE_PLUMBING_ONLY = "plumbing_only"
CLAIMS_SCOPE_PSI_WIRED_UNCONTROLLED = "psi_wired_uncontrolled"
