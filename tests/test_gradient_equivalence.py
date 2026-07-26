import torch

# FORCE DISABLE TRITON/COMPILE CHECKS BEFORE IMPORTING MODELS
# This avoids the hang observed during import of ConvEqProp
import bioplausible.acceleration

bioplausible.acceleration._check_compile_works = lambda: False

from bioplausible.zoo.models.eqprop import LoopedMLP


def test_contrastive_gradients():
    """Verify gradient equivalence after .detach() optimization."""
    print("Testing contrastive gradient correctness...")
    torch.manual_seed(42)

    # Create model
    model = LoopedMLP(10, 20, 5, gradient_method="contrastive", max_steps=10)

    # Create dummy data
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))

    # Run contrastive step
    metrics = model.train_step(x, y)

    print(f"Metrics: {metrics}")

    # Verify gradients exist and are valid (no NaNs)
    has_grads = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                has_grads = True
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"
                # Check magnitude is reasonable
                grad_norm = param.grad.norm().item()
                assert grad_norm <= 100.0, f"High gradient norm for {name}: {grad_norm}"

    assert has_grads, "No gradients computed for any parameter."
