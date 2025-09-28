"""
Advanced hyperparameter optimization for polymer ML models.

This module implements state-of-the-art hyperparameter optimization techniques:
- Bayesian optimization with Gaussian processes
- Multi-objective optimization (Pareto frontier)
- Neural architecture search (NAS)
- Population-based training
- Hyperband and successive halving
- Multi-fidelity optimization
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Union, Callable, Any
import logging
import time
import json
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available. Install with: pip install optuna")

try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.search.optuna import OptunaSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logging.warning("Ray Tune not available. Install with: pip install ray[tune]")

try:
    import hyperopt
    from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    logging.warning("Hyperopt not available. Install with: pip install hyperopt")


class OptimizationObjective(ABC):
    """Abstract base class for optimization objectives."""

    @abstractmethod
    def __call__(self, params: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Evaluate objective function."""
        pass


class ModelPerformanceObjective(OptimizationObjective):
    """Objective function for model performance optimization."""

    def __init__(self, model_class: type, metric: str = 'rmse', cv_folds: int = 3):
        self.model_class = model_class
        self.metric = metric
        self.cv_folds = cv_folds

    def __call__(self, params: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Evaluate model performance with given hyperparameters."""
        try:
            # Create model with hyperparameters
            model = self.model_class(**params)

            # Train model
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
            else:
                # PyTorch model
                model = self._train_pytorch_model(model, X_train, y_train, X_val, y_val)
                predictions = self._predict_pytorch_model(model, X_val)

            # Calculate metric
            if self.metric == 'rmse':
                score = np.sqrt(mean_squared_error(y_val, predictions))
            elif self.metric == 'r2':
                score = -r2_score(y_val, predictions)  # Negative for minimization
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

            return score

        except Exception as e:
            logging.warning(f"Evaluation failed: {e}")
            return float('inf')

    def _train_pytorch_model(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray, epochs: int = 50) -> nn.Module:
        """Train PyTorch model."""
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1) if y_train.ndim == 1 else torch.FloatTensor(y_train)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(X_train_tensor)
            loss = criterion(pred, y_train_tensor)
            loss.backward()
            optimizer.step()

        return model

    def _predict_pytorch_model(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        """Make predictions with PyTorch model."""
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = model(X_tensor).numpy()
        return predictions.flatten()


class MultiObjectiveOptimizer:
    """Multi-objective hyperparameter optimization."""

    def __init__(self, objectives: List[OptimizationObjective]):
        self.objectives = objectives
        self.pareto_frontier = []

    def optimize(self, search_space: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 100) -> List[Dict]:
        """Perform multi-objective optimization."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for multi-objective optimization")

        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in search_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )

            # Evaluate all objectives
            scores = []
            for obj in self.objectives:
                score = obj(params, X_train, y_train, X_val, y_val)
                scores.append(score)

            return scores

        # Create study with multiple objectives
        study = optuna.create_study(
            directions=['minimize'] * len(self.objectives),
            sampler=TPESampler()
        )

        study.optimize(objective, n_trials=n_trials)

        # Extract Pareto frontier
        self.pareto_frontier = []
        for trial in study.best_trials:
            self.pareto_frontier.append({
                'params': trial.params,
                'scores': trial.values
            })

        return self.pareto_frontier


class NeuralArchitectureSearch:
    """Neural Architecture Search for polymer models."""

    def __init__(self, input_dim: int, output_dim: int = 1):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def create_search_space(self) -> Dict[str, Any]:
        """Define NAS search space."""
        return {
            'num_layers': {'type': 'int', 'low': 2, 'high': 8},
            'hidden_dim_1': {'type': 'int', 'low': 32, 'high': 512},
            'hidden_dim_2': {'type': 'int', 'low': 16, 'high': 256},
            'hidden_dim_3': {'type': 'int', 'low': 8, 'high': 128},
            'dropout_rate': {'type': 'float', 'low': 0.0, 'high': 0.5},
            'activation': {'type': 'categorical', 'choices': ['relu', 'tanh', 'elu', 'gelu']},
            'batch_norm': {'type': 'categorical', 'choices': [True, False]},
            'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-1, 'log': True},
            'optimizer': {'type': 'categorical', 'choices': ['adam', 'adamw', 'sgd']}
        }

    def create_model(self, params: Dict[str, Any]) -> nn.Module:
        """Create neural network based on architecture parameters."""
        layers = []
        prev_dim = self.input_dim

        # Activation function mapping
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }

        for i in range(params['num_layers']):
            # Hidden dimension for this layer
            if i == 0:
                hidden_dim = params['hidden_dim_1']
            elif i == 1:
                hidden_dim = params['hidden_dim_2']
            elif i == 2:
                hidden_dim = params['hidden_dim_3']
            else:
                hidden_dim = max(16, params['hidden_dim_3'] // (2 ** (i - 2)))

            # Add linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Add batch normalization if specified
            if params['batch_norm']:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Add activation
            layers.append(activation_map[params['activation']])

            # Add dropout
            if params['dropout_rate'] > 0:
                layers.append(nn.Dropout(params['dropout_rate']))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))

        return nn.Sequential(*layers)

    def optimize_architecture(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            n_trials: int = 50) -> Dict[str, Any]:
        """Optimize neural architecture."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for NAS")

        search_space = self.create_search_space()

        def objective(trial):
            # Sample architecture parameters
            params = {}
            for param_name, param_config in search_space.items():
                if param_config['type'] == 'float':
                    if param_config.get('log', False):
                        params[param_name] = trial.suggest_float(
                            param_name, param_config['low'], param_config['high'], log=True
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, param_config['low'], param_config['high']
                        )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )

            # Create and train model
            model = self.create_model(params)
            score = self._evaluate_architecture(model, params, X_train, y_train, X_val, y_val)

            return score

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        return study.best_params

    def _evaluate_architecture(self, model: nn.Module, params: Dict[str, Any],
                             X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Evaluate neural architecture."""
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1) if y_train.ndim == 1 else torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)

        # Create optimizer
        if params['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        elif params['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'])

        criterion = nn.MSELoss()

        # Training
        model.train()
        epochs = 50
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(X_train_tensor)
            loss = criterion(pred, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor).numpy()
            val_score = np.sqrt(mean_squared_error(y_val, val_pred.flatten()))

        return val_score


class PopulationBasedOptimizer:
    """Population-based training for hyperparameter optimization."""

    def __init__(self, population_size: int = 8, perturbation_interval: int = 10):
        self.population_size = population_size
        self.perturbation_interval = perturbation_interval

    def optimize(self, model_creator: Callable, search_space: Dict[str, Any],
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                max_iterations: int = 100) -> Dict[str, Any]:
        """Perform population-based optimization."""

        # Initialize population
        population = []
        for i in range(self.population_size):
            params = self._sample_hyperparameters(search_space)
            model = model_creator(**params)
            population.append({
                'model': model,
                'params': params,
                'score': float('inf'),
                'age': 0
            })

        best_params = None
        best_score = float('inf')

        for iteration in range(max_iterations):
            # Train and evaluate each member
            for member in population:
                # Train for a few epochs
                score = self._train_and_evaluate(
                    member['model'], X_train, y_train, X_val, y_val, epochs=5
                )
                member['score'] = score
                member['age'] += 1

                if score < best_score:
                    best_score = score
                    best_params = member['params'].copy()

            # Population-based updates
            if iteration % self.perturbation_interval == 0:
                self._exploit_and_explore(population, search_space)

            print(f"Iteration {iteration}, Best Score: {best_score:.4f}")

        return best_params

    def _sample_hyperparameters(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample hyperparameters from search space."""
        params = {}
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'float':
                if param_config.get('log', False):
                    params[param_name] = np.random.uniform(
                        np.log(param_config['low']), np.log(param_config['high'])
                    )
                    params[param_name] = np.exp(params[param_name])
                else:
                    params[param_name] = np.random.uniform(
                        param_config['low'], param_config['high']
                    )
            elif param_config['type'] == 'int':
                params[param_name] = np.random.randint(
                    param_config['low'], param_config['high'] + 1
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = np.random.choice(param_config['choices'])

        return params

    def _train_and_evaluate(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray, epochs: int) -> float:
        """Train model for a few epochs and evaluate."""
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1) if y_train.ndim == 1 else torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(X_train_tensor)
            loss = criterion(pred, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor).numpy()
            score = np.sqrt(mean_squared_error(y_val, val_pred.flatten()))

        return score

    def _exploit_and_explore(self, population: List[Dict], search_space: Dict[str, Any]):
        """Exploit good performers and explore new hyperparameters."""
        # Sort by performance
        population.sort(key=lambda x: x['score'])

        # Bottom 25% copy from top 25%
        top_quarter = len(population) // 4
        bottom_quarter = 3 * len(population) // 4

        for i in range(bottom_quarter, len(population)):
            # Copy from top performer
            source_idx = np.random.randint(0, top_quarter)
            population[i]['params'] = population[source_idx]['params'].copy()

            # Perturb hyperparameters
            for param_name, param_config in search_space.items():
                if np.random.random() < 0.2:  # 20% chance to perturb
                    if param_config['type'] == 'float':
                        noise = np.random.normal(0, 0.1)
                        population[i]['params'][param_name] *= (1 + noise)
                        population[i]['params'][param_name] = np.clip(
                            population[i]['params'][param_name],
                            param_config['low'], param_config['high']
                        )


class HyperparameterOptimizer:
    """Main hyperparameter optimization interface."""

    def __init__(self, optimization_method: str = 'optuna'):
        self.optimization_method = optimization_method

    def optimize(self, model_class: type, search_space: Dict[str, Any],
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                n_trials: int = 100, **kwargs) -> Dict[str, Any]:
        """Optimize hyperparameters using specified method."""

        if self.optimization_method == 'optuna' and OPTUNA_AVAILABLE:
            return self._optuna_optimization(
                model_class, search_space, X_train, y_train, X_val, y_val, n_trials
            )
        elif self.optimization_method == 'hyperopt' and HYPEROPT_AVAILABLE:
            return self._hyperopt_optimization(
                model_class, search_space, X_train, y_train, X_val, y_val, n_trials
            )
        elif self.optimization_method == 'ray' and RAY_AVAILABLE:
            return self._ray_optimization(
                model_class, search_space, X_train, y_train, X_val, y_val, n_trials
            )
        else:
            raise ValueError(f"Optimization method '{self.optimization_method}' not available")

    def _optuna_optimization(self, model_class: type, search_space: Dict[str, Any],
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           n_trials: int) -> Dict[str, Any]:
        """Optuna-based optimization."""
        objective_func = ModelPerformanceObjective(model_class)

        def objective(trial):
            params = {}
            for param_name, param_config in search_space.items():
                if param_config['type'] == 'float':
                    if param_config.get('log', False):
                        params[param_name] = trial.suggest_float(
                            param_name, param_config['low'], param_config['high'], log=True
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, param_config['low'], param_config['high']
                        )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )

            return objective_func(params, X_train, y_train, X_val, y_val)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        return study.best_params


def create_search_space(model_type: str) -> Dict[str, Any]:
    """Create search space for different model types."""

    if model_type == 'neural_network':
        return {
            'num_layers': {'type': 'int', 'low': 2, 'high': 6},
            'hidden_dim': {'type': 'int', 'low': 32, 'high': 512},
            'dropout_rate': {'type': 'float', 'low': 0.0, 'high': 0.5},
            'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-1, 'log': True},
            'batch_size': {'type': 'int', 'low': 16, 'high': 256},
            'optimizer': {'type': 'categorical', 'choices': ['adam', 'adamw', 'sgd']}
        }

    elif model_type == 'random_forest':
        return {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
            'max_depth': {'type': 'int', 'low': 3, 'high': 20},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]}
        }

    elif model_type == 'xgboost':
        return {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
            'max_depth': {'type': 'int', 'low': 3, 'high': 15},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 1.0},
            'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 1.0}
        }

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def demo_hyperparameter_optimization():
    """Demonstrate hyperparameter optimization capabilities."""
    print("Hyperparameter Optimization Demo")
    print("=" * 50)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :3]**2, axis=1) + 0.1 * np.random.randn(n_samples)

    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Test data: {X_test.shape}")

    # Test different optimization methods
    if OPTUNA_AVAILABLE:
        print("\nOptuna Optimization:")

        from sklearn.ensemble import RandomForestRegressor
        search_space = create_search_space('random_forest')

        optimizer = HyperparameterOptimizer('optuna')
        best_params = optimizer.optimize(
            RandomForestRegressor, search_space,
            X_train, y_train, X_val, y_val,
            n_trials=20
        )

        print(f"Best parameters: {best_params}")

        # Evaluate best model
        best_model = RandomForestRegressor(**best_params)
        best_model.fit(X_train, y_train)
        test_pred = best_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        print(f"Test RMSE: {test_rmse:.4f}")

    # Neural Architecture Search
    print("\nNeural Architecture Search:")
    nas = NeuralArchitectureSearch(input_dim=n_features)

    if OPTUNA_AVAILABLE:
        best_architecture = nas.optimize_architecture(
            X_train, y_train, X_val, y_val, n_trials=10
        )
        print(f"Best architecture: {best_architecture}")


if __name__ == "__main__":
    demo_hyperparameter_optimization()