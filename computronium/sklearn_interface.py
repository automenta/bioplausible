"""
Scikit-Learn Compatible Wrapper for EqProp Models

Allows using EqProp models in Scikit-Learn pipelines with .fit() and .predict().
Supports incremental learning via .partial_fit().
"""

from typing import TYPE_CHECKING, cast

import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from computronium.core.losses import compute_accuracy
from computronium.core.trainer import dispatch_train_step
from computronium.core.utils.device import get_device
from computronium.core.utils.optimizer import OptimizerConfig, create_optimizer
from computronium.utils import seed_everything

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "EqPropClassifier",
]


class EqPropClassifier(BaseEstimator, ClassifierMixin):
    """
    Equilibrium Propagation Classifier compatible with Scikit-Learn.

    Supports incremental learning via partial_fit().

    Parameters
    ----------
    model_name : str, default="eqprop_mlp"
        Name of the native model composition to use.
    hidden_dim : int, default=256
        Number of neurons in the hidden layer.
    steps : int, default=30
        Number of equilibrium steps during training.
    learning_rate : float, default=0.001
        Learning rate for the optimizer.
    batch_size : int, default=128
        Batch size for training.
    epochs : int, default=10
        Number of training epochs (for fit()).
    use_spectral_norm : bool, default=True
        Whether to use spectral normalization (required for stability).
    device : str | None, default=None
        Device to train on ('cpu' or 'cuda').
    random_state : int | None, default=None
        Random seed for reproducibility.
    **kwargs
        Additional arguments passed to model constructor.
    """

    def __init__(
        self,
        model_name: str = "eqprop_mlp",
        hidden_dim: int = 256,
        steps: int = 30,
        learning_rate: float = 0.001,
        batch_size: int = 128,
        epochs: int = 10,
        use_spectral_norm: bool = True,
        device: str | None = None,
        random_state: int | None = None,
        **kwargs: object,
    ):
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.steps = steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_spectral_norm = use_spectral_norm
        self.device = device
        self.random_state = random_state
        self.kwargs = kwargs

        self.classes_: np.ndarray | None = None
        self.n_classes_: int | None = None
        self.n_features_in_: int | None = None
        self.model_: nn.Module | None = None
        self.optimizer_: torch.optim.Optimizer | None = None

    def _initialize(
        self, X: np.ndarray, y: np.ndarray | None = None, classes: object = None
    ) -> None:
        """Initialize the model and optimizer if not already initialized."""
        if self.model_ is not None:
            return

        if classes is not None:
            self.classes_ = unique_labels(classes)
        elif y is not None:
            self.classes_ = unique_labels(y)
        else:
            raise ValueError(
                "Classes must be provided for initialization if y is None."
            )

        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        if self.random_state is not None:
            seed_everything(self.random_state)

        if self.device is None:
            self.device = str(get_device())

        from computronium.core.construction import construct_model
        from computronium.experiment.param_estimator import resolve_native_model

        model_cls = resolve_native_model(self.model_name)

        factory_kwargs = self.kwargs.copy()
        factory_kwargs.setdefault("hidden_dim", self.hidden_dim)

        self.model_ = construct_model(
            model_cls,
            factory_kwargs,
            input_dim=int(self.n_features_in_),
            output_dim=int(self.n_classes_),
            model_name=self.model_name,
        )
        self.model_ = self.model_.to(self.device)

        if hasattr(self.model_, "max_steps"):
            self.model_.max_steps = self.steps
        if hasattr(self.model_, "eq_steps"):
            self.model_.eq_steps = self.steps

        self.optimizer_ = create_optimizer(
            self.model_,
            OptimizerConfig(name="adam", lr=self.learning_rate, weight_decay=0.0),
        )

    def _train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """Single training step, routed through the shared dispatch seam."""
        metrics = dispatch_train_step(
            model=self.model_,
            x=x,
            y=y,
            adapt_input=lambda x: x,
            optimizer=self.optimizer_,
            config=None,
        )
        if "logits" in metrics:
            logits = metrics["logits"]
            metrics["accuracy"] = compute_accuracy(logits, y)
        return cast("dict[str, float]", metrics)

    def fit(self, X: np.ndarray, y: np.ndarray) -> EqPropClassifier:
        """
        Train the EqProp model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : EqPropClassifier
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self._initialize(X, y)

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                self._train_step(batch_x.to(self.device), batch_y.to(self.device))

        return self

    def partial_fit(
        self, X: np.ndarray, y: np.ndarray, classes: object = None
    ) -> EqPropClassifier:
        """
        Incremental fit on a batch of samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        classes : array-like, optional
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit.

        Returns
        -------
        self : EqPropClassifier
            Returns self.
        """
        X = check_array(X)

        if self.model_ is None:
            self._initialize(X, y, classes=classes)

        if self.classes_ is None and classes is None:
            raise ValueError("classes must be passed on the first call to partial_fit.")

        x = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)

        self.model_.train()
        self._train_step(x, y_t)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X = check_array(X)

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()
