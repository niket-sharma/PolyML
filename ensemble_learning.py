"""
Advanced ensemble learning framework for polymer property prediction.

This module implements sophisticated ensemble methods including:
- Multi-level stacking ensembles
- Dynamic ensemble weighting
- Boosting with custom base learners
- Multi-task ensemble learning
- Hierarchical ensemble architectures
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Union, Callable, Any
import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

try:
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    GRADIENT_BOOSTING_AVAILABLE = True
except ImportError:
    GRADIENT_BOOSTING_AVAILABLE = False
    logging.warning("Gradient boosting libraries not available")


class BaseEnsemble(ABC):
    """Abstract base class for ensemble methods."""

    def __init__(self):
        self.models = []
        self.weights = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseEnsemble':
        """Fit the ensemble to training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the ensemble."""
        pass

    def get_model_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all base models."""
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
            else:
                # Handle PyTorch models
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    pred = model(X_tensor).numpy()
            predictions.append(pred.flatten())
        return np.column_stack(predictions)


class StackingEnsemble(BaseEnsemble):
    """Multi-level stacking ensemble with cross-validation."""

    def __init__(self, base_models: List[Any], meta_model: Any = None,
                 cv_folds: int = 5, use_base_features: bool = True):
        super().__init__()
        self.base_models = base_models
        self.meta_model = meta_model or self._default_meta_model()
        self.cv_folds = cv_folds
        self.use_base_features = use_base_features
        self.models = base_models

    def _default_meta_model(self):
        """Default meta-learner (Ridge regression)."""
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """Fit stacking ensemble with cross-validation."""
        n_samples = X.shape[0]
        n_models = len(self.base_models)

        # Generate out-of-fold predictions
        oof_predictions = np.zeros((n_samples, n_models))
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train = y[train_idx]

            for i, model in enumerate(self.base_models):
                # Clone and fit model
                model_clone = self._clone_model(model)
                model_clone.fit(X_fold_train, y_fold_train)

                # Predict on validation fold
                if hasattr(model_clone, 'predict'):
                    val_pred = model_clone.predict(X_fold_val)
                else:
                    # PyTorch model
                    model_clone.eval()
                    with torch.no_grad():
                        X_val_tensor = torch.FloatTensor(X_fold_val)
                        val_pred = model_clone(X_val_tensor).numpy().flatten()

                oof_predictions[val_idx, i] = val_pred

        # Fit base models on full data
        for model in self.base_models:
            model.fit(X, y)

        # Prepare meta-features
        if self.use_base_features:
            meta_features = np.column_stack([oof_predictions, X])
        else:
            meta_features = oof_predictions

        # Fit meta-model
        self.meta_model.fit(meta_features, y)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using stacking ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        # Get base model predictions
        base_predictions = self.get_model_predictions(X)

        # Prepare meta-features
        if self.use_base_features:
            meta_features = np.column_stack([base_predictions, X])
        else:
            meta_features = base_predictions

        # Meta-model prediction
        return self.meta_model.predict(meta_features)

    def _clone_model(self, model):
        """Clone a model for cross-validation."""
        if hasattr(model, 'get_params'):
            # Scikit-learn style
            from sklearn.base import clone
            return clone(model)
        else:
            # For PyTorch models, return a copy
            import copy
            return copy.deepcopy(model)


class DynamicEnsemble(BaseEnsemble):
    """Ensemble with dynamic weighting based on input characteristics."""

    def __init__(self, base_models: List[Any], weighting_network: Optional[nn.Module] = None):
        super().__init__()
        self.base_models = base_models
        self.models = base_models
        self.weighting_network = weighting_network or self._create_weighting_network()

    def _create_weighting_network(self, input_dim: int = None) -> nn.Module:
        """Create default weighting network."""
        if input_dim is None:
            input_dim = 10  # Will be adjusted during fit

        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(self.base_models)),
            nn.Softmax(dim=-1)
        )

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> 'DynamicEnsemble':
        """Fit dynamic ensemble."""
        # First fit base models
        for model in self.base_models:
            model.fit(X, y)

        # Adjust weighting network input dimension
        if X.shape[1] != self.weighting_network[0].in_features:
            self.weighting_network = self._create_weighting_network(X.shape[1])

        # Get base model predictions for training weighting network
        base_predictions = self.get_model_predictions(X)

        # Train weighting network
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1) if y.ndim == 1 else torch.FloatTensor(y)
        base_pred_tensor = torch.FloatTensor(base_predictions)

        optimizer = torch.optim.Adam(self.weighting_network.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.weighting_network.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Get dynamic weights
            weights = self.weighting_network(X_tensor)

            # Weighted ensemble prediction
            weighted_pred = torch.sum(weights * base_pred_tensor, dim=1, keepdim=True)

            # Loss
            loss = criterion(weighted_pred, y_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using dynamic ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        # Get base predictions
        base_predictions = self.get_model_predictions(X)

        # Get dynamic weights
        self.weighting_network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            weights = self.weighting_network(X_tensor).numpy()

        # Weighted ensemble prediction
        weighted_pred = np.sum(weights * base_predictions, axis=1)
        return weighted_pred


class AdaBoostRegressor(BaseEnsemble):
    """Custom AdaBoost for regression with neural network weak learners."""

    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimator_weights = []
        self.estimator_errors = []

    def _weak_learner(self, input_dim: int) -> nn.Module:
        """Create weak learner (shallow neural network)."""
        return nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaBoostRegressor':
        """Fit AdaBoost ensemble."""
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples

        self.models = []
        self.estimator_weights = []
        self.estimator_errors = []

        for i in range(self.n_estimators):
            # Create and train weak learner
            weak_model = self._weak_learner(X.shape[1])
            self._train_weak_learner(weak_model, X, y, sample_weights)

            # Get predictions
            weak_model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions = weak_model(X_tensor).numpy().flatten()

            # Compute weighted error
            errors = np.abs(predictions - y)
            avg_error = np.average(errors, weights=sample_weights)

            # Compute estimator weight
            if avg_error <= 0:
                estimator_weight = 1.0
            else:
                estimator_weight = self.learning_rate * np.log((1.0 - avg_error) / avg_error)

            # Update sample weights
            sample_weights *= np.exp(estimator_weight * errors / np.max(errors))
            sample_weights /= np.sum(sample_weights)

            self.models.append(weak_model)
            self.estimator_weights.append(estimator_weight)
            self.estimator_errors.append(avg_error)

            if avg_error <= 0:
                break

        self.is_fitted = True
        return self

    def _train_weak_learner(self, model: nn.Module, X: np.ndarray, y: np.ndarray,
                          sample_weights: np.ndarray, epochs: int = 50):
        """Train weak learner with sample weights."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        weights_tensor = torch.FloatTensor(sample_weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(X_tensor)

            # Weighted MSE loss
            loss = torch.mean(weights_tensor * (pred.squeeze() - y_tensor.squeeze())**2)
            loss.backward()
            optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using AdaBoost ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        predictions = np.zeros(X.shape[0])
        X_tensor = torch.FloatTensor(X)

        for model, weight in zip(self.models, self.estimator_weights):
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor).numpy().flatten()
                predictions += weight * pred

        # Normalize by total weight
        total_weight = sum(self.estimator_weights)
        return predictions / total_weight if total_weight > 0 else predictions


class MultiTaskEnsemble(BaseEnsemble):
    """Ensemble for multi-task learning across different polymer properties."""

    def __init__(self, base_models: List[Any], task_weights: Optional[List[float]] = None):
        super().__init__()
        self.base_models = base_models
        self.models = base_models
        self.task_weights = task_weights
        self.n_tasks = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiTaskEnsemble':
        """
        Fit multi-task ensemble.
        y should be of shape (n_samples, n_tasks)
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.n_tasks = y.shape[1]

        if self.task_weights is None:
            self.task_weights = [1.0] * self.n_tasks

        # Fit models for each task
        self.task_models = []
        for task_idx in range(self.n_tasks):
            task_models = []
            y_task = y[:, task_idx]

            for model in self.base_models:
                model_clone = self._clone_model(model)
                model_clone.fit(X, y_task)
                task_models.append(model_clone)

            self.task_models.append(task_models)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, task_idx: Optional[int] = None) -> np.ndarray:
        """
        Make predictions for specific task or all tasks.
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        if task_idx is not None:
            # Predict for specific task
            task_predictions = []
            for model in self.task_models[task_idx]:
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                else:
                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X)
                        pred = model(X_tensor).numpy().flatten()
                task_predictions.append(pred)

            # Average predictions
            return np.mean(task_predictions, axis=0)

        else:
            # Predict for all tasks
            all_predictions = []
            for task_idx in range(self.n_tasks):
                task_pred = self.predict(X, task_idx)
                all_predictions.append(task_pred)

            return np.column_stack(all_predictions)

    def _clone_model(self, model):
        """Clone a model."""
        if hasattr(model, 'get_params'):
            from sklearn.base import clone
            return clone(model)
        else:
            import copy
            return copy.deepcopy(model)


class HierarchicalEnsemble(BaseEnsemble):
    """Hierarchical ensemble with specialized sub-ensembles."""

    def __init__(self, molecular_models: List[Any], process_models: List[Any],
                 fusion_model: Optional[Any] = None):
        super().__init__()
        self.molecular_models = molecular_models
        self.process_models = process_models
        self.fusion_model = fusion_model or self._default_fusion_model()
        self.models = molecular_models + process_models

    def _default_fusion_model(self):
        """Default fusion model."""
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, random_state=42)

    def fit(self, X_molecular: np.ndarray, X_process: np.ndarray,
            y: np.ndarray) -> 'HierarchicalEnsemble':
        """
        Fit hierarchical ensemble.
        X_molecular: molecular features
        X_process: process features
        """
        # Fit molecular models
        for model in self.molecular_models:
            model.fit(X_molecular, y)

        # Fit process models
        for model in self.process_models:
            model.fit(X_process, y)

        # Get predictions from sub-ensembles
        mol_predictions = self._get_ensemble_predictions(self.molecular_models, X_molecular)
        proc_predictions = self._get_ensemble_predictions(self.process_models, X_process)

        # Combine predictions for fusion model
        fusion_features = np.column_stack([
            mol_predictions.mean(axis=1),
            mol_predictions.std(axis=1),
            proc_predictions.mean(axis=1),
            proc_predictions.std(axis=1),
            X_molecular.mean(axis=1),
            X_process.mean(axis=1)
        ])

        # Fit fusion model
        self.fusion_model.fit(fusion_features, y)
        self.is_fitted = True

        return self

    def predict(self, X_molecular: np.ndarray, X_process: np.ndarray) -> np.ndarray:
        """Make predictions using hierarchical ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        # Get sub-ensemble predictions
        mol_predictions = self._get_ensemble_predictions(self.molecular_models, X_molecular)
        proc_predictions = self._get_ensemble_predictions(self.process_models, X_process)

        # Prepare fusion features
        fusion_features = np.column_stack([
            mol_predictions.mean(axis=1),
            mol_predictions.std(axis=1),
            proc_predictions.mean(axis=1),
            proc_predictions.std(axis=1),
            X_molecular.mean(axis=1),
            X_process.mean(axis=1)
        ])

        # Final prediction
        return self.fusion_model.predict(fusion_features)

    def _get_ensemble_predictions(self, models: List[Any], X: np.ndarray) -> np.ndarray:
        """Get predictions from ensemble of models."""
        predictions = []
        for model in models:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
            else:
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    pred = model(X_tensor).numpy().flatten()
            predictions.append(pred)
        return np.column_stack(predictions)


class EnsembleOptimizer:
    """Optimizer for ensemble weights and hyperparameters."""

    def __init__(self, ensemble: BaseEnsemble, optimization_method: str = 'genetic'):
        self.ensemble = ensemble
        self.optimization_method = optimization_method

    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray,
                        n_generations: int = 50) -> np.ndarray:
        """Optimize ensemble weights using genetic algorithm."""
        if self.optimization_method == 'genetic':
            return self._genetic_optimization(X_val, y_val, n_generations)
        elif self.optimization_method == 'grid_search':
            return self._grid_search_optimization(X_val, y_val)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")

    def _genetic_optimization(self, X_val: np.ndarray, y_val: np.ndarray,
                            n_generations: int) -> np.ndarray:
        """Genetic algorithm for weight optimization."""
        n_models = len(self.ensemble.models)
        population_size = 100

        # Initialize population
        population = np.random.dirichlet(np.ones(n_models), population_size)

        for generation in range(n_generations):
            # Evaluate fitness
            fitness_scores = []
            for weights in population:
                score = self._evaluate_weights(weights, X_val, y_val)
                fitness_scores.append(score)

            # Selection
            fitness_scores = np.array(fitness_scores)
            sorted_indices = np.argsort(fitness_scores)
            elite_size = population_size // 4

            # Keep elite
            new_population = population[sorted_indices[:elite_size]]

            # Crossover and mutation
            while len(new_population) < population_size:
                parent1 = population[np.random.choice(sorted_indices[:elite_size])]
                parent2 = population[np.random.choice(sorted_indices[:elite_size])]

                # Crossover
                alpha = np.random.random()
                child = alpha * parent1 + (1 - alpha) * parent2

                # Mutation
                if np.random.random() < 0.1:
                    mutation = np.random.normal(0, 0.1, n_models)
                    child += mutation

                # Normalize
                child = np.abs(child)
                child /= np.sum(child)

                new_population = np.vstack([new_population, child])

            population = new_population

        # Return best weights
        best_weights = population[np.argmin(fitness_scores)]
        return best_weights

    def _evaluate_weights(self, weights: np.ndarray, X_val: np.ndarray,
                         y_val: np.ndarray) -> float:
        """Evaluate ensemble with given weights."""
        predictions = self.ensemble.get_model_predictions(X_val)
        weighted_pred = np.sum(weights * predictions, axis=1)
        return mean_squared_error(y_val, weighted_pred)


def create_polymer_ensemble(ensemble_type: str = "stacking", **kwargs) -> BaseEnsemble:
    """Factory function for creating polymer-specific ensembles."""

    # Default base models for polymer applications
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR

    base_models = [
        RandomForestRegressor(n_estimators=100, random_state=42),
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        Ridge(alpha=1.0),
        SVR(kernel='rbf')
    ]

    if GRADIENT_BOOSTING_AVAILABLE:
        base_models.extend([
            xgb.XGBRegressor(n_estimators=100, random_state=42),
            lgb.LGBMRegressor(n_estimators=100, random_state=42),
            CatBoostRegressor(n_estimators=100, random_state=42, verbose=False)
        ])

    if ensemble_type == "stacking":
        return StackingEnsemble(base_models, **kwargs)
    elif ensemble_type == "dynamic":
        return DynamicEnsemble(base_models, **kwargs)
    elif ensemble_type == "adaboost":
        return AdaBoostRegressor(**kwargs)
    elif ensemble_type == "multitask":
        return MultiTaskEnsemble(base_models, **kwargs)
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


def demo_ensemble_learning():
    """Demonstrate ensemble learning capabilities."""
    print("Ensemble Learning Demo")
    print("=" * 50)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :3]**2, axis=1) + 0.1 * np.random.randn(n_samples)

    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Test different ensemble methods
    ensemble_types = ["stacking", "dynamic", "adaboost"]

    for ensemble_type in ensemble_types:
        print(f"\n{ensemble_type.upper()} ENSEMBLE:")

        try:
            ensemble = create_polymer_ensemble(ensemble_type)

            if ensemble_type == "dynamic":
                ensemble.fit(X_train, y_train, epochs=50)
            else:
                ensemble.fit(X_train, y_train)

            # Predictions
            y_pred = ensemble.predict(X_test)

            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"RMSE: {rmse:.4f}")
            print(f"R²: {r2:.4f}")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    demo_ensemble_learning()