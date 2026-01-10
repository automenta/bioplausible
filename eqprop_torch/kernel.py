"""
EqProp Kernel: Pure NumPy/CuPy Equilibrium Propagation

Standalone implementation without PyTorch autograd. Can use CuPy for GPU
acceleration or fall back to NumPy for CPU/portability.

Key advantages:
- No computation graph overhead
- O(1) memory training via contrastive Hebbian
- Direct portability to HLS/Verilog for FPGA

Usage:
    from eqprop_torch.kernel import EqPropKernel
    kernel = EqPropKernel(784, 256, 10, use_gpu=True)
    kernel.train_step(x_batch, y_batch)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try to import CuPy for GPU
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


def get_backend(use_gpu: bool) -> Any:
    """Return appropriate array library (CuPy or NumPy)."""
    if use_gpu and HAS_CUPY:
        return cp
    return np


def to_numpy(arr: Any) -> np.ndarray:
    """Convert array to NumPy (handles both NumPy and CuPy arrays)."""
    if HAS_CUPY and cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr


def spectral_normalize(W: np.ndarray, num_iters: int = 1, u: Optional[np.ndarray] = None, xp=np) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """Power iteration spectral normalization.

    Normalizes W by its largest singular value (spectral norm).
    This ensures the operator norm ‖W‖ ≈ 1, maintaining Lipschitz < 1.

    Args:
        W: Weight matrix [out_dim, in_dim]
        num_iters: Power iteration steps (1 is usually enough)
        u: Previous u vector for warm start
        xp: Array module (np or cp)

    Returns:
        W_normalized: Normalized weight matrix
        u_new: Updated u vector for next call
        sigma: Estimated spectral norm
    """
    out_dim, in_dim = W.shape

    u = _initialize_u_vector(u, out_dim, W.dtype, xp)

    for _ in range(num_iters):
        v = _compute_v_vector(W, u, xp)
        u = _compute_u_vector(W, v, xp)

    sigma = _compute_spectral_norm(W, u, v)
    W_normalized = W / (_add_epsilon(sigma))

    return W_normalized, u, sigma


def _add_epsilon(value: float, epsilon: float = 1e-12) -> float:
    """Add small epsilon to prevent division by zero."""
    return value + epsilon


def _initialize_u_vector(u: Optional[np.ndarray], out_dim: int, dtype: np.dtype, xp) -> np.ndarray:
    """Initialize or validate the u vector for power iteration."""
    if u is None:
        u = xp.random.randn(out_dim).astype(dtype)
    return u / xp.linalg.norm(u)


def _compute_v_vector(W: np.ndarray, u: np.ndarray, xp) -> np.ndarray:
    """Compute v vector in power iteration: v = W.T @ u, normalized."""
    v = W.T @ u
    norm = xp.linalg.norm(v)
    return v / _add_epsilon(norm)


def _compute_u_vector(W: np.ndarray, v: np.ndarray, xp) -> np.ndarray:
    """Compute u vector in power iteration: u = W @ v, normalized."""
    u = W @ v
    norm = xp.linalg.norm(u)
    return u / _add_epsilon(norm)


def _compute_spectral_norm(W: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    """Compute the spectral norm (largest singular value) of W."""
    return u @ W @ v


def softmax(x: np.ndarray, xp=np) -> np.ndarray:
    """Stable softmax."""
    x_max = xp.max(x, axis=-1, keepdims=True)
    exp_x = xp.exp(x - x_max)
    return exp_x / xp.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray, xp=np) -> float:
    """Cross-entropy loss from logits."""
    batch_size = logits.shape[0]
    probs = softmax(logits, xp)
    probs = xp.clip(probs, 1e-10, 1.0)
    log_probs = xp.log(probs)
    loss = -xp.sum(log_probs[xp.arange(batch_size), targets]) / batch_size
    return loss


def tanh_deriv(x: np.ndarray, xp=np) -> np.ndarray:
    """Derivative of tanh: 1 - tanh(x)^2"""
    return 1 - xp.tanh(x) ** 2


class EqPropKernel:
    """Pure NumPy/CuPy Equilibrium Propagation kernel.

    Implements:
    - Forward pass to equilibrium
    - Free and nudged phases
    - Contrastive Hebbian weight updates
    - Spectral normalization for stability
    - Adam optimizer

    Example:
        >>> kernel = EqPropKernel(784, 256, 10, use_gpu=True)
        >>> for x_batch, y_batch in data_loader:
        ...     metrics = kernel.train_step(x_batch, y_batch)
        ...     print(f"Loss: {metrics['loss']:.4f}")
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        gamma: float = 0.5,
        beta: float = 0.22,
        max_steps: int = 10,
        epsilon: float = 1e-3,
        lr: float = 0.001,
        use_spectral_norm: bool = True,
        use_gpu: bool = False,
        adaptive_epsilon: bool = True,
    ) -> None:
        """Initialize EqProp kernel.

        Args:
            input_dim: Input dimension (e.g., 784 for MNIST)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (e.g., 10 for classification)
            gamma: Damping factor for equilibrium dynamics
            beta: Nudge strength for contrastive learning
            max_steps: Maximum equilibrium steps
            epsilon: Convergence threshold
            lr: Learning rate
            use_spectral_norm: Whether to apply spectral normalization
            use_gpu: Whether to use CuPy (GPU)
            adaptive_epsilon: Use relaxed epsilon after step 5
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.beta = beta
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.lr = lr
        self.use_spectral_norm = use_spectral_norm
        self.use_gpu = use_gpu and HAS_CUPY
        self.adaptive_epsilon = adaptive_epsilon

        self.xp = get_backend(self.use_gpu)

        # Initialize weights
        scale = 0.5
        self.weights = {
            'embed': self._init_weight(input_dim, hidden_dim, scale),
            'W1': self._init_weight(hidden_dim, hidden_dim * 4, scale),
            'W2': self._init_weight(hidden_dim * 4, hidden_dim, scale),
            'head': self._init_weight(hidden_dim, output_dim, scale),
        }

        self.biases = {
            'embed': self.xp.zeros(hidden_dim, dtype=np.float32),
            'W1': self.xp.zeros(hidden_dim * 4, dtype=np.float32),
            'W2': self.xp.zeros(hidden_dim, dtype=np.float32),
            'head': self.xp.zeros(output_dim, dtype=np.float32),
        }

        self.sn_state = {'W1_u': None, 'W2_u': None}

        # Adam state
        self.adam_state = {
            'm': {k: self.xp.zeros_like(v) for k, v in self.weights.items()},
            'v': {k: self.xp.zeros_like(v) for k, v in self.weights.items()},
            't': 0,
        }

    def _init_weight(self, in_dim: int, out_dim: int, scale: float = 0.5) -> np.ndarray:
        """Initialize weight matrix with Xavier-like initialization."""
        xp = self.xp
        std = scale * np.sqrt(2.0 / (in_dim + out_dim))
        W = xp.random.randn(out_dim, in_dim).astype(np.float32) * std
        return W

    def _get_normalized_weights(self) -> Dict[str, np.ndarray]:
        """Get spectral-normalized weights."""
        if not self.use_spectral_norm:
            return self.weights.copy()

        weights = self.weights.copy()

        # Normalize W1 and W2 with spectral normalization
        weights['W1'] = self._normalize_weight('W1', 'W1_u')
        weights['W2'] = self._normalize_weight('W2', 'W2_u')

        return weights

    def _should_normalize_weight(self, weight_key: str) -> bool:
        """Check if a weight should be normalized."""
        return self.use_spectral_norm and weight_key in ['W1', 'W2']

    def _normalize_weight(self, weight_key: str, sn_state_key: str) -> np.ndarray:
        """Normalize a specific weight matrix using spectral normalization."""
        weight = self.weights[weight_key]
        u_state = self.sn_state[sn_state_key]

        normalized_weight, new_u_state, _ = spectral_normalize(
            weight, u=u_state, xp=self.xp
        )

        self.sn_state[sn_state_key] = new_u_state
        return normalized_weight

    def _get_weight_by_key(self, weight_key: str) -> np.ndarray:
        """Get a weight matrix by its key.

        Args:
            weight_key: Key identifying the weight matrix

        Returns:
            The corresponding weight matrix
        """
        return self.weights[weight_key]

    def forward_step(self, h: np.ndarray, x_emb: np.ndarray, weights: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Single equilibrium step.

        Args:
            h: Current hidden state
            x_emb: Embedded input
            weights: Model weights dictionary

        Returns:
            Next hidden state and activation dictionary
        """
        xp = self.xp

        h_mean = xp.mean(h, axis=-1, keepdims=True)
        h_std = xp.std(h, axis=-1, keepdims=True) + 1e-5
        h_norm = (h - h_mean) / h_std

        ffn_hidden = xp.tanh(h_norm @ weights['W1'].T + self.biases['W1'])
        ffn_out = ffn_hidden @ weights['W2'].T + self.biases['W2']

        h_next = (1 - self.gamma) * h + self.gamma * (ffn_out + x_emb)

        activations = {
            'h_norm': h_norm,
            'ffn_hidden': ffn_hidden,
            'h': h,
            'h_next': h_next,
        }

        return h_next, activations

    def solve_equilibrium(self, x: np.ndarray, nudge_grad: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]], Dict[str, Any]]:
        """
        Find equilibrium state h* via fixed-point iteration.

        Args:
            x: Input data
            nudge_grad: Optional gradient for nudged phase

        Returns:
            Equilibrium state, activation log, and convergence info
        """
        xp = self.xp
        batch_size = x.shape[0]

        x = self._prepare_input(x)
        x_emb = self._compute_embedded_input(x)
        weights = self._get_normalized_weights()
        h = xp.zeros((batch_size, self.hidden_dim), dtype=np.float32)

        activations_log = []

        for t in range(self.max_steps):
            h, activations = self._perform_equilibrium_step(h, x_emb, weights, nudge_grad)
            activations_log.append(activations)

            if self._check_convergence(h, activations_log[-1]['h'], t):
                return h, activations_log, {'steps': t + 1, 'converged': True}

        return h, activations_log, {'steps': self.max_steps, 'converged': False}

    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        """Prepare input for processing on the appropriate device."""
        if self.use_gpu and not isinstance(x, self.xp.ndarray):
            return self.xp.asarray(x)
        return x

    def _compute_embedded_input(self, x: np.ndarray) -> np.ndarray:
        """Compute embedded input representation."""
        return x @ self.weights['embed'].T + self.biases['embed']

    def _perform_equilibrium_step(self, h: np.ndarray, x_emb: np.ndarray, weights: Dict[str, np.ndarray],
                                 nudge_grad: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Perform a single equilibrium step, applying nudge if provided."""
        h, activations = self.forward_step(h, x_emb, weights)

        if nudge_grad is not None:
            h = h - self.beta * nudge_grad

        return h, activations

    def _check_convergence(self, h: np.ndarray, h_prev: np.ndarray, step: int) -> bool:
        """Check if the equilibrium has converged."""
        diff = self.xp.max(self.xp.linalg.norm(h - h_prev, axis=1))
        threshold = self._get_convergence_threshold(step)
        return diff < threshold

    def _get_convergence_threshold(self, step: int) -> float:
        """Get the convergence threshold based on the current step."""
        multiplier = 2.0 if self.adaptive_epsilon and step > 5 else 1.0
        return self.epsilon * multiplier

    def compute_output(self, h: np.ndarray) -> np.ndarray:
        """
        Compute output logits from hidden state.

        Args:
            h: Hidden state

        Returns:
            Output logits
        """
        return h @ self.weights['head'].T + self.biases['head']

    def compute_hebbian_update(self, act_free: Dict[str, np.ndarray], act_nudged: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute contrastive Hebbian weight updates.

        Args:
            act_free: Activations from free phase
            act_nudged: Activations from nudged phase

        Returns:
            Weight updates dictionary

        Formula:
            ΔW = (1/β) * (A_nudged ⊗ A_nudged.T - A_free ⊗ A_free.T)
        """
        batch_size = act_free['h'].shape[0]

        grads = {}

        grad_free_W2 = act_free['h_next'].T @ act_free['ffn_hidden'] / batch_size
        grad_nudged_W2 = act_nudged['h_next'].T @ act_nudged['ffn_hidden'] / batch_size
        grads['W2'] = (1.0 / self.beta) * (grad_nudged_W2 - grad_free_W2)

        grad_free_W1 = act_free['ffn_hidden'].T @ act_free['h_norm'] / batch_size
        grad_nudged_W1 = act_nudged['ffn_hidden'].T @ act_nudged['h_norm'] / batch_size
        grads['W1'] = (1.0 / self.beta) * (grad_nudged_W1 - grad_free_W1)

        return grads

    def adam_update(self, grads: Dict[str, np.ndarray], beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        """
        Apply Adam optimizer update.

        Args:
            grads: Gradient dictionary
            beta1: Adam beta1 parameter
            beta2: Adam beta2 parameter
            eps: Adam epsilon parameter
        """
        self.adam_state['t'] += 1
        t = self.adam_state['t']

        for key in grads:
            if key not in self.weights:
                continue

            g = grads[key]
            self.adam_state['m'][key] = beta1 * self.adam_state['m'][key] + (1 - beta1) * g
            self.adam_state['v'][key] = beta2 * self.adam_state['v'][key] + (1 - beta2) * (g ** 2)

            m_hat = self.adam_state['m'][key] / (1 - beta1 ** t)
            v_hat = self.adam_state['v'][key] / (1 - beta2 ** t)

            self.weights[key] -= self.lr * m_hat / (self.xp.sqrt(v_hat) + eps)

    def train_step(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Full EqProp training step.

        Args:
            x: Input batch [batch, input_dim]
            y: Target labels [batch]

        Returns:
            Dict with loss, accuracy, free_steps, nudged_steps
        """
        xp = self.xp

        # Prepare inputs
        x, y = self._prepare_inputs(x, y)

        # Free Phase
        h_free, act_log_free, info_free = self.solve_equilibrium(x)

        # Compute gradients for nudging
        logits, d_logits, nudge_grad = self._compute_gradients_for_nudging(h_free, y, xp)

        # Nudged Phase
        h_nudged, act_log_nudged, info_nudged = self.solve_equilibrium(x, nudge_grad)

        # Compute Updates
        grads = self.compute_hebbian_update(act_log_free[-1], act_log_nudged[-1])
        grads['head'] = d_logits.T @ h_free / self._get_batch_size(x)

        self.adam_update(grads)

        # Compute metrics
        metrics = self._compute_training_metrics(logits, y, info_free, info_nudged, xp)

        return metrics

    def _prepare_inputs(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare input tensors for processing."""
        xp = self.xp

        if not isinstance(x, np.ndarray) and not (HAS_CUPY and cp is not None and isinstance(x, cp.ndarray)):
            x = np.asarray(x)
        if self.use_gpu:
            x = xp.asarray(x)
            y = xp.asarray(y)

        return x, y

    def _compute_gradients_for_nudging(self, h_free: np.ndarray, y: np.ndarray, xp) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute gradients needed for the nudging phase."""
        logits = self.compute_output(h_free)
        probs = softmax(logits, xp)
        batch_size = self._get_batch_size(logits)

        one_hot = xp.zeros_like(probs)
        one_hot[xp.arange(batch_size), y] = 1.0
        d_logits = probs - one_hot
        nudge_grad = d_logits @ self.weights['head']

        return logits, d_logits, nudge_grad

    def _get_batch_size(self, tensor: np.ndarray) -> int:
        """Get the batch size from a tensor."""
        return tensor.shape[0]

    def _compute_training_metrics(self, logits: np.ndarray, y: np.ndarray,
                                info_free: Dict[str, Any], info_nudged: Dict[str, Any], xp) -> Dict[str, float]:
        """Compute training metrics."""
        loss = cross_entropy_loss(logits, y, xp)
        preds = xp.argmax(logits, axis=1)
        accuracy = xp.mean(preds == y)

        return {
            'loss': float(to_numpy(loss)),
            'accuracy': float(to_numpy(accuracy)),
            'free_steps': info_free['steps'],
            'nudged_steps': info_nudged['steps'],
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run inference on input.

        Args:
            x: Input data

        Returns:
            Predicted class indices
        """
        xp = self.xp
        if self.use_gpu:
            x = xp.asarray(x)

        h_star, _, _ = self.solve_equilibrium(x)
        logits = self.compute_output(h_star)
        return to_numpy(xp.argmax(logits, axis=1))

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate accuracy on a dataset.

        Args:
            x: Input data
            y: True labels

        Returns:
            Accuracy score
        """
        preds = self.predict(x)
        y_np = to_numpy(y) if not isinstance(y, np.ndarray) else y
        return float(np.mean(preds == y_np))


# Alias for backward compatibility with BPTT kernel
class EqPropKernelBPTT(EqPropKernel):
    """Alias for backward compatibility."""
    pass


__all__ = [
    'EqPropKernel',
    'EqPropKernelBPTT',
    'HAS_CUPY',
    'get_backend',
    'to_numpy',
    'spectral_normalize',
]
