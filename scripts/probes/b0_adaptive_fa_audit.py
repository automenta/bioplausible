"""B0 — legacy AdaptiveFA audit: the alignment is soft weight transport
(TODO12 Workstream B, step B0; TODO12 row "audit the legacy seam first").

`computronium/core/local_learning/rules/fa.py` `AdaptiveFA` (Akrout et
al. 2019 port) updates feedback via
``alignment_grad = param.data - fb`` (or ``param.data.T - fb``) and
``fb += feedback_lr * alignment_grad`` — it READS the forward weights.
Akrout's rule also has a non-transport term (pre/post-activity
correlation direction); the legacy port kept only the transport term.

Decisive demonstration: freeze W entirely, supply NO activity, run only
``_update_feedback_weights`` — if cos(B, Wᵀ) still climbs to ~1, the
alignment cannot be credit/learning-driven: it is pure weight transport.

Reusable pieces for B1 (LearnedFeedbackCredit): the SLOW feedback
timescale (feedback_lr 1e-4 ≪ lr 1e-2), the cos(B, Wᵀ) alignment metric,
feedback_scale. The ontology port must NOT inherit the transport; the
biology xfail (`tests/property/biology/test_biology_axioms.py`, "feedback
LR too small to show alignment in 50 steps") stays xfail until a
transport-free rule passes it — and under the L3 lock (‖B − Wᵀ‖ > 1e-3,
separate storage) it never will, by construction.
"""

import torch

from computronium.core.local_learning.rules.fa import AdaptiveFA


def cos_b_wt(fb: torch.Tensor, w: torch.Tensor) -> float:
    b = w.T.reshape(fb.shape).flatten()
    a = fb.flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def main() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    )
    params = [p for p in model.parameters() if p.ndim >= 2]
    rule = AdaptiveFA(list(model.parameters()), model)
    w_frozen = [p.detach().clone() for p in params]
    # The rule's guard requires param.grad is not None, but the B update
    # never reads the gradient — supply zeros once to enable the branch.
    for p in model.parameters():
        p.grad = torch.zeros_like(p)

    checkpoints = (0, 10, 100, 1000, 5000)
    done = 0
    pairs = [
        (fb, w)
        for fb, w in zip(rule.feedback_weights, list(model.parameters()), strict=True)
        if fb is not None
    ]
    print("updates  cos(B0,W0^T)  cos(B1,W1^T)  max|W-W_frozen|")
    for target in checkpoints:
        for _ in range(target - done):
            rule._update_feedback_weights()
        done = target
        drift = max(
            float((p - w).norm()) for p, w in zip(params, w_frozen, strict=True)
        )
        cosines = [cos_b_wt(fb, w) for fb, w in pairs]
        print(f"{target:>7}  {cosines[0]:>12.4f}  {cosines[1]:>12.4f}  {drift:.6f}")
    print(
        "W never updated, no activity ever supplied — alignment is pure "
        "weight transport (soft W read)."
    )


if __name__ == "__main__":
    main()
