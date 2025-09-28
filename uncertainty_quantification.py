"""
Uncertainty quantification methods for polymer property prediction.

This module implements state-of-the-art uncertainty quantification techniques:
- Bayesian Neural Networks with MC Dropout
- Deep Ensembles
- Gaussian Process regression
- Conformal prediction
- Evidential deep learning
- Variational inference
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Union, Callable
import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Gamma
import torch.distributions as dist

try:
    import gpytorch
    from gpytorch.models import ExactGP
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
    from gpytorch.means import ConstantMean
    GPYTORCH_AVAILABLE = True
except ImportError:
    GPYTORCH_AVAILABLE = False
    logging.warning("GPyTorch not available. Install with: pip install gpytorch")


class UncertaintyEstimator(ABC):
    """Abstract base class for uncertainty estimation methods."""

    @abstractmethod
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimates.
        Returns: (predictions, uncertainties)
        """
        pass

    @abstractmethod
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Fit the uncertainty model."""
        pass


class MCDropoutModel(nn.Module, UncertaintyEstimator):
    """Monte Carlo Dropout for uncertainty quantification."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 output_dim: int = 1, dropout_rate: float = 0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_samples = 100  # Number of MC samples

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Forward pass with optional training mode for dropout."""
        if training:
            self.train()
        return self.network(x)

    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with MC Dropout uncertainty."""
        self.train()  # Enable dropout during inference
        predictions = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                pred = self.forward(x, training=True)
                predictions.append(pred)

        predictions = torch.stack(predictions)  # [num_samples, batch_size, output_dim]

        # Calculate mean and uncertainty
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)

        return mean_pred, uncertainty

    def fit(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 100,
            lr: float = 0.001) -> None:
        """Train the MC Dropout model."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.forward(x, training=True)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


class DeepEnsemble(UncertaintyEstimator):
    """Deep ensemble for uncertainty quantification."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 output_dim: int = 1, num_models: int = 5):
        self.num_models = num_models
        self.models = []

        for i in range(num_models):
            model = self._create_model(input_dim, hidden_dims, output_dim)
            self.models.append(model)

    def _create_model(self, input_dim: int, hidden_dims: List[int], output_dim: int) -> nn.Module:
        """Create a single model for the ensemble."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def fit(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 100,
            lr: float = 0.001) -> None:
        """Train all models in the ensemble."""
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{self.num_models}")

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            # Add random weight initialization diversity
            for layer in model.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)

            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with ensemble uncertainty."""
        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)  # [num_models, batch_size, output_dim]

        # Calculate mean and uncertainty
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)

        return mean_pred, uncertainty


class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight uncertainty."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_logvar = nn.Parameter(torch.randn(out_features) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample weights from posterior
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)

        weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
        bias = self.bias_mu + bias_std * torch.randn_like(bias_std)

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence for variational inference."""
        # KL divergence between posterior and prior (assuming standard normal prior)
        kl_weight = 0.5 * torch.sum(
            self.weight_mu**2 + torch.exp(self.weight_logvar) - self.weight_logvar - 1
        )
        kl_bias = 0.5 * torch.sum(
            self.bias_mu**2 + torch.exp(self.bias_logvar) - self.bias_logvar - 1
        )
        return kl_weight + kl_bias


class BayesianNeuralNetwork(nn.Module, UncertaintyEstimator):
    """Bayesian Neural Network with variational inference."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 output_dim: int = 1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(BayesianLinear(prev_dim, output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence of the network."""
        kl_total = 0
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                kl_total += layer.kl_divergence()
        return kl_total

    def fit(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 100,
            lr: float = 0.001, kl_weight: float = 1e-5) -> None:
        """Train the Bayesian neural network."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            pred = self.forward(x)

            # Likelihood loss
            likelihood_loss = F.mse_loss(pred, y)

            # KL divergence
            kl_loss = self.kl_divergence()

            # Total loss
            total_loss = likelihood_loss + kl_weight * kl_loss

            total_loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Likelihood: {likelihood_loss.item():.4f}, "
                      f"KL: {kl_loss.item():.4f}")

    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with Bayesian uncertainty."""
        self.eval()
        predictions = []

        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)

        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)

        return mean_pred, uncertainty


class EvidentialRegression(nn.Module, UncertaintyEstimator):
    """Evidential regression for uncertainty quantification."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 output_dim: int = 1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # Output 4 parameters: gamma, nu, alpha, beta
        layers.append(nn.Linear(prev_dim, 4 * output_dim))
        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = self.network(x)
        output = output.view(-1, 4, self.output_dim)

        # Ensure positive parameters
        gamma = output[:, 0, :]  # Mean
        nu = F.softplus(output[:, 1, :]) + 1  # Degrees of freedom
        alpha = F.softplus(output[:, 2, :]) + 1  # Shape parameter
        beta = F.softplus(output[:, 3, :])  # Rate parameter

        return {'gamma': gamma, 'nu': nu, 'alpha': alpha, 'beta': beta}

    def evidential_loss(self, params: Dict[str, torch.Tensor], y: torch.Tensor,
                       coeff: float = 1.0) -> torch.Tensor:
        """Evidential loss function."""
        gamma, nu, alpha, beta = params['gamma'], params['nu'], params['alpha'], params['beta']

        # NLL loss
        nll = 0.5 * torch.log(np.pi / nu) \
              - alpha * torch.log(beta) \
              + (alpha + 0.5) * torch.log(nu * (y - gamma)**2 + 2 * beta) \
              + torch.lgamma(alpha) \
              - torch.lgamma(alpha + 0.5)

        # Regularization term
        error = torch.abs(y - gamma)
        reg = error * (2 * nu + alpha)

        return torch.mean(nll + coeff * reg)

    def fit(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 100,
            lr: float = 0.001) -> None:
        """Train evidential regression model."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            params = self.forward(x)
            loss = self.evidential_loss(params, y)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with evidential uncertainty."""
        self.eval()
        with torch.no_grad():
            params = self.forward(x)
            gamma, nu, alpha, beta = params['gamma'], params['nu'], params['alpha'], params['beta']

            # Epistemic and aleatoric uncertainties
            epistemic = beta / (alpha - 1)
            aleatoric = beta / (nu * (alpha - 1))

            total_uncertainty = torch.sqrt(epistemic + aleatoric)

            return gamma, total_uncertainty


class ConformalPredictor:
    """Conformal prediction for distribution-free uncertainty quantification."""

    def __init__(self, model: nn.Module, alpha: float = 0.1):
        self.model = model
        self.alpha = alpha  # Miscoverage level
        self.calibration_scores = None

    def calibrate(self, x_cal: torch.Tensor, y_cal: torch.Tensor) -> None:
        """Calibrate the conformal predictor."""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x_cal)
            # Absolute residuals as conformity scores
            self.calibration_scores = torch.abs(y_cal - predictions).flatten()

    def predict_intervals(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict with conformal prediction intervals."""
        if self.calibration_scores is None:
            raise ValueError("Must calibrate predictor first")

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x)

        # Compute quantile
        n = len(self.calibration_scores)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        quantile = torch.quantile(self.calibration_scores, quantile_level)

        # Prediction intervals
        lower = predictions - quantile
        upper = predictions + quantile

        return predictions, lower, upper


if GPYTORCH_AVAILABLE:
    class GaussianProcessRegressor(ExactGP, UncertaintyEstimator):
        """Gaussian Process regression with uncertainty quantification."""

        def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor):
            likelihood = GaussianLikelihood()
            super().__init__(train_x, train_y, likelihood)

            self.mean_module = ConstantMean()
            self.covar_module = ScaleKernel(RBFKernel())

        def forward(self, x: torch.Tensor):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        def fit(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 100,
                lr: float = 0.1) -> None:
            """Train the Gaussian Process."""
            self.train()
            self.likelihood.train()

            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

            for epoch in range(epochs):
                optimizer.zero_grad()
                output = self(x)
                loss = -mll(output, y.flatten())
                loss.backward()
                optimizer.step()

                if epoch % 20 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Predict with GP uncertainty."""
            self.eval()
            self.likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self(x))
                mean = observed_pred.mean
                uncertainty = observed_pred.stddev

            return mean.unsqueeze(-1), uncertainty.unsqueeze(-1)


class UncertaintyBenchmark:
    """Benchmark different uncertainty quantification methods."""

    def __init__(self, methods: List[str] = None):
        if methods is None:
            methods = ['mc_dropout', 'deep_ensemble', 'bayesian_nn', 'evidential']
        self.methods = methods
        self.models = {}

    def train_all_methods(self, x_train: torch.Tensor, y_train: torch.Tensor,
                         input_dim: int) -> None:
        """Train all uncertainty quantification methods."""

        for method in self.methods:
            print(f"\nTraining {method}...")

            if method == 'mc_dropout':
                model = MCDropoutModel(input_dim)
            elif method == 'deep_ensemble':
                model = DeepEnsemble(input_dim)
            elif method == 'bayesian_nn':
                model = BayesianNeuralNetwork(input_dim)
            elif method == 'evidential':
                model = EvidentialRegression(input_dim)
            elif method == 'gp' and GPYTORCH_AVAILABLE:
                model = GaussianProcessRegressor(x_train, y_train)
            else:
                continue

            model.fit(x_train, y_train)
            self.models[method] = model

    def evaluate_uncertainty(self, x_test: torch.Tensor, y_test: torch.Tensor) -> Dict:
        """Evaluate uncertainty quality using various metrics."""
        results = {}

        for method, model in self.models.items():
            pred_mean, pred_std = model.predict_with_uncertainty(x_test)

            # Calculate metrics
            rmse = torch.sqrt(torch.mean((pred_mean - y_test)**2))
            mae = torch.mean(torch.abs(pred_mean - y_test))

            # Negative log-likelihood (assuming Gaussian)
            nll = 0.5 * torch.log(2 * np.pi * pred_std**2) + \
                  0.5 * (y_test - pred_mean)**2 / pred_std**2
            nll = torch.mean(nll)

            results[method] = {
                'rmse': rmse.item(),
                'mae': mae.item(),
                'nll': nll.item(),
                'mean_uncertainty': torch.mean(pred_std).item()
            }

        return results


def demo_uncertainty_quantification():
    """Demonstrate uncertainty quantification methods."""
    print("Uncertainty Quantification Demo")
    print("=" * 50)

    # Generate synthetic data
    torch.manual_seed(42)
    n_samples = 1000
    input_dim = 5

    x = torch.randn(n_samples, input_dim)
    # True function with noise
    y_true = torch.sum(x**2, dim=1, keepdim=True) + 0.1 * torch.sin(10 * x[:, 0:1])
    noise = 0.1 * torch.randn(n_samples, 1)
    y = y_true + noise

    # Split data
    train_size = int(0.8 * n_samples)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Test different methods
    methods = ['mc_dropout', 'deep_ensemble', 'bayesian_nn', 'evidential']

    print(f"\nTraining data: {x_train.shape}")
    print(f"Test data: {x_test.shape}")

    for method in methods:
        print(f"\n{method.upper()}:")

        if method == 'mc_dropout':
            model = MCDropoutModel(input_dim)
        elif method == 'deep_ensemble':
            model = DeepEnsemble(input_dim, num_models=3)
        elif method == 'bayesian_nn':
            model = BayesianNeuralNetwork(input_dim)
        elif method == 'evidential':
            model = EvidentialRegression(input_dim)

        try:
            # Train
            model.fit(x_train, y_train, epochs=50)

            # Predict with uncertainty
            pred_mean, pred_std = model.predict_with_uncertainty(x_test[:10])

            print(f"Predictions: {pred_mean[:5].flatten()}")
            print(f"Uncertainties: {pred_std[:5].flatten()}")
            print(f"True values: {y_test[:5].flatten()}")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    demo_uncertainty_quantification()