"""
Scikit-Learn Compatible Wrapper for EqProp Models

Allows using EqProp models in Scikit-Learn pipelines with .fit() and .predict().
"""

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from torch.utils.data import DataLoader, TensorDataset

from .core import EqPropTrainer
from .models import LoopedMLP


class EqPropClassifier(BaseEstimator, ClassifierMixin):
    """
    Equilibrium Propagation Classifier compatible with Scikit-Learn.

    Parameters
    ----------
    hidden_dim : int, default=256
        Number of neurons in the hidden layer.
    steps : int, default=30
        Number of equilibrium steps during training.
    learning_rate : float, default=0.001
        Learning rate for the optimizer.
    batch_size : int, default=128
        Batch size for training.
    epochs : int, default=10
        Number of training epochs.
    use_spectral_norm : bool, default=True
        Whether to use spectral normalization (required for stability).
    device : str, default='cpu'
        Device to train on ('cpu' or 'cuda').
    random_state : int, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        hidden_dim=256,
        steps=30,
        learning_rate=0.001,
        batch_size=128,
        epochs=10,
        use_spectral_norm=True,
        device=None,
        random_state=None,
    ):
        self.hidden_dim = hidden_dim
        self.steps = steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_spectral_norm = use_spectral_norm
        self.device = device
        self.random_state = random_state

    def fit(self, X, y):
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
        self : object
            Fitted estimator.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = len(self.classes_)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize Model
        self.model_ = LoopedMLP(
            input_dim=self.n_features_in_,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_classes_,
            max_steps=self.steps,
            use_spectral_norm=self.use_spectral_norm,
        )

        # Initialize Trainer
        self.trainer_ = EqPropTrainer(
            model=self.model_,
            lr=self.learning_rate,
            device=self.device,
            use_compile=False, # Disable compile for simple sklearn usage to avoid overhead
        )

        # Train
        self.trainer_.fit(loader, epochs=self.epochs, progress_bar=False)

        return self

    def predict(self, X):
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

        X_tensor = torch.FloatTensor(X).to(self.trainer_.device)

        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()

    def predict_proba(self, X):
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

        X_tensor = torch.FloatTensor(X).to(self.trainer_.device)

        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()
