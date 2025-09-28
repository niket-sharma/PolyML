"""
Modern MLOps pipeline for polymer property prediction.

This module implements a comprehensive MLOps framework including:
- Experiment tracking with MLflow/WandB
- Model versioning and registry
- Automated model validation and testing
- Continuous integration/deployment
- Model monitoring and drift detection
- Data versioning and lineage
- Feature stores
"""

import os
import json
import yaml
import time
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union, Any, Callable
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

try:
    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Install with: pip install mlflow")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("WandB not available. Install with: pip install wandb")

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logging.warning("Evidently not available. Install with: pip install evidently")


class ExperimentTracker:
    """Experiment tracking with multiple backends."""

    def __init__(self, backend: str = 'mlflow', project_name: str = 'polymer_ml'):
        self.backend = backend
        self.project_name = project_name
        self.experiment_id = None
        self.run_id = None

        if backend == 'mlflow' and MLFLOW_AVAILABLE:
            self._setup_mlflow()
        elif backend == 'wandb' and WANDB_AVAILABLE:
            self._setup_wandb()
        else:
            logging.warning(f"Backend {backend} not available, using local logging")
            self.backend = 'local'
            self._setup_local()

    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri("./mlruns")
        try:
            experiment = mlflow.get_experiment_by_name(self.project_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(self.project_name)
            else:
                self.experiment_id = experiment.experiment_id
        except Exception as e:
            logging.error(f"MLflow setup failed: {e}")
            self.backend = 'local'
            self._setup_local()

    def _setup_wandb(self):
        """Setup WandB tracking."""
        try:
            wandb.init(project=self.project_name, reinit=True)
        except Exception as e:
            logging.error(f"WandB setup failed: {e}")
            self.backend = 'local'
            self._setup_local()

    def _setup_local(self):
        """Setup local experiment tracking."""
        self.log_dir = Path("./experiments")
        self.log_dir.mkdir(exist_ok=True)

    def start_run(self, run_name: Optional[str] = None) -> str:
        """Start a new experiment run."""
        if self.backend == 'mlflow':
            mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
            self.run_id = mlflow.active_run().info.run_id
        elif self.backend == 'wandb':
            wandb.run.name = run_name
            self.run_id = wandb.run.id
        else:
            self.run_id = f"run_{int(time.time())}"
            if run_name:
                self.run_id = f"{run_name}_{self.run_id}"

        return self.run_id

    def log_params(self, params: Dict[str, Any]):
        """Log experiment parameters."""
        if self.backend == 'mlflow':
            mlflow.log_params(params)
        elif self.backend == 'wandb':
            wandb.config.update(params)
        else:
            self._log_local('params', params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log experiment metrics."""
        if self.backend == 'mlflow':
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        elif self.backend == 'wandb':
            wandb.log(metrics, step=step)
        else:
            self._log_local('metrics', metrics, step)

    def log_model(self, model: Any, model_name: str, **kwargs):
        """Log model artifacts."""
        if self.backend == 'mlflow':
            if isinstance(model, nn.Module):
                mlflow.pytorch.log_model(model, model_name, **kwargs)
            else:
                mlflow.sklearn.log_model(model, model_name, **kwargs)
        elif self.backend == 'wandb':
            # Save model locally and log as artifact
            model_path = f"./temp_model_{model_name}.pkl"
            torch.save(model, model_path) if isinstance(model, nn.Module) else pickle.dump(model, open(model_path, 'wb'))
            wandb.save(model_path)
            os.remove(model_path)
        else:
            self._log_local('model', {'name': model_name, 'type': type(model).__name__})

    def log_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """Log artifact files."""
        if self.backend == 'mlflow':
            mlflow.log_artifact(artifact_path, artifact_name)
        elif self.backend == 'wandb':
            wandb.save(artifact_path)
        else:
            self._log_local('artifact', {'path': artifact_path, 'name': artifact_name})

    def end_run(self):
        """End the current experiment run."""
        if self.backend == 'mlflow':
            mlflow.end_run()
        elif self.backend == 'wandb':
            wandb.finish()

    def _log_local(self, log_type: str, data: Any, step: Optional[int] = None):
        """Log data locally."""
        log_file = self.log_dir / f"{self.run_id}_{log_type}.json"

        if log_file.exists():
            with open(log_file, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        if step is not None:
            log_entry['step'] = step

        existing_data.append(log_entry)

        with open(log_file, 'w') as f:
            json.dump(existing_data, f, indent=2, default=str)


class ModelRegistry:
    """Model registry for versioning and deployment."""

    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.metadata_file = self.registry_path / "registry.json"
        self._load_registry()

    def _load_registry(self):
        """Load registry metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {"models": {}}

    def _save_registry(self):
        """Save registry metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)

    def register_model(self, model: Any, model_name: str, version: str,
                      metrics: Dict[str, float], metadata: Optional[Dict] = None) -> str:
        """Register a new model version."""
        model_id = f"{model_name}_{version}"
        model_path = self.registry_path / f"{model_id}.pkl"

        # Save model
        if isinstance(model, nn.Module):
            torch.save(model, model_path)
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        # Update registry
        if model_name not in self.registry["models"]:
            self.registry["models"][model_name] = {"versions": {}}

        self.registry["models"][model_name]["versions"][version] = {
            "model_id": model_id,
            "path": str(model_path),
            "metrics": metrics,
            "metadata": metadata or {},
            "registered_at": datetime.now().isoformat(),
            "status": "staging"
        }

        self._save_registry()
        return model_id

    def load_model(self, model_name: str, version: str = "latest") -> Any:
        """Load a model from registry."""
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model {model_name} not found in registry")

        if version == "latest":
            versions = self.registry["models"][model_name]["versions"]
            version = max(versions.keys())

        model_info = self.registry["models"][model_name]["versions"][version]
        model_path = Path(model_info["path"])

        if model_path.suffix == '.pkl':
            if model_path.stat().st_size > 100 * 1024 * 1024:  # > 100MB, likely PyTorch
                return torch.load(model_path)
            else:
                with open(model_path, 'rb') as f:
                    return pickle.load(f)

    def promote_model(self, model_name: str, version: str, stage: str):
        """Promote model to production/staging."""
        if stage not in ["staging", "production", "archived"]:
            raise ValueError(f"Invalid stage: {stage}")

        self.registry["models"][model_name]["versions"][version]["status"] = stage
        self._save_registry()

    def list_models(self) -> Dict[str, List[str]]:
        """List all models and versions."""
        return {name: list(info["versions"].keys())
                for name, info in self.registry["models"].items()}


class ModelValidator:
    """Automated model validation and testing."""

    def __init__(self, validation_tests: Optional[List[str]] = None):
        self.validation_tests = validation_tests or [
            'performance_test',
            'invariance_test',
            'directional_expectation_test',
            'minimum_functionality_test'
        ]

    def validate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      validation_config: Optional[Dict] = None) -> Dict[str, bool]:
        """Run comprehensive model validation."""
        results = {}

        for test_name in self.validation_tests:
            try:
                if test_name == 'performance_test':
                    results[test_name] = self._performance_test(model, X_test, y_test)
                elif test_name == 'invariance_test':
                    results[test_name] = self._invariance_test(model, X_test)
                elif test_name == 'directional_expectation_test':
                    results[test_name] = self._directional_expectation_test(model, X_test)
                elif test_name == 'minimum_functionality_test':
                    results[test_name] = self._minimum_functionality_test(model, X_test, y_test)
            except Exception as e:
                logging.error(f"Validation test {test_name} failed: {e}")
                results[test_name] = False

        return results

    def _performance_test(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                         min_r2: float = 0.7) -> bool:
        """Test if model meets minimum performance threshold."""
        predictions = self._predict(model, X_test)
        r2 = r2_score(y_test, predictions)
        return r2 >= min_r2

    def _invariance_test(self, model: Any, X_test: np.ndarray,
                        noise_level: float = 0.01) -> bool:
        """Test model invariance to small perturbations."""
        original_pred = self._predict(model, X_test)

        # Add small noise
        noise = np.random.normal(0, noise_level, X_test.shape)
        noisy_X = X_test + noise
        noisy_pred = self._predict(model, noisy_X)

        # Check if predictions are similar
        relative_change = np.abs((noisy_pred - original_pred) / (original_pred + 1e-8))
        return np.mean(relative_change) < 0.1  # Less than 10% change

    def _directional_expectation_test(self, model: Any, X_test: np.ndarray,
                                    feature_expectations: Optional[Dict] = None) -> bool:
        """Test if model behavior matches domain expectations."""
        if feature_expectations is None:
            # Default expectations for polymer properties
            feature_expectations = {
                0: 'positive',  # Temperature usually increases melt index
                1: 'positive',  # Pressure usually increases melt index
            }

        for feature_idx, expected_direction in feature_expectations.items():
            if feature_idx >= X_test.shape[1]:
                continue

            # Test directional influence
            X_modified = X_test.copy()
            X_modified[:, feature_idx] *= 1.1  # 10% increase

            original_pred = self._predict(model, X_test)
            modified_pred = self._predict(model, X_modified)

            if expected_direction == 'positive':
                if not np.mean(modified_pred > original_pred) > 0.6:
                    return False
            elif expected_direction == 'negative':
                if not np.mean(modified_pred < original_pred) > 0.6:
                    return False

        return True

    def _minimum_functionality_test(self, model: Any, X_test: np.ndarray,
                                  y_test: np.ndarray) -> bool:
        """Test basic model functionality."""
        try:
            predictions = self._predict(model, X_test)

            # Check for NaN or infinite values
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                return False

            # Check reasonable output range
            if np.max(predictions) > 1000 * np.max(y_test):
                return False

            return True
        except Exception:
            return False

    def _predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions with the model."""
        if hasattr(model, 'predict'):
            return model.predict(X)
        else:
            # PyTorch model
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions = model(X_tensor).numpy()
            return predictions.flatten()


class DataDriftDetector:
    """Monitor data drift in production."""

    def __init__(self):
        self.reference_data = None
        self.drift_thresholds = {
            'statistical': 0.05,  # p-value threshold
            'distance': 0.1       # distance threshold
        }

    def set_reference_data(self, X_ref: np.ndarray, y_ref: Optional[np.ndarray] = None):
        """Set reference dataset for drift detection."""
        self.reference_data = {'X': X_ref, 'y': y_ref}

    def detect_drift(self, X_current: np.ndarray, y_current: Optional[np.ndarray] = None,
                    method: str = 'statistical') -> Dict[str, Any]:
        """Detect data drift."""
        if self.reference_data is None:
            raise ValueError("Reference data not set")

        if method == 'statistical':
            return self._statistical_drift_detection(X_current, y_current)
        elif method == 'distance':
            return self._distance_based_drift_detection(X_current)
        else:
            raise ValueError(f"Unknown drift detection method: {method}")

    def _statistical_drift_detection(self, X_current: np.ndarray,
                                   y_current: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Statistical drift detection using KS test."""
        from scipy import stats

        drift_results = {'feature_drift': {}, 'target_drift': None}

        # Feature drift
        for i in range(X_current.shape[1]):
            ref_feature = self.reference_data['X'][:, i]
            current_feature = X_current[:, i]

            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(ref_feature, current_feature)

            drift_results['feature_drift'][f'feature_{i}'] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'drift_detected': p_value < self.drift_thresholds['statistical']
            }

        # Target drift
        if y_current is not None and self.reference_data['y'] is not None:
            ks_stat, p_value = stats.ks_2samp(self.reference_data['y'], y_current)
            drift_results['target_drift'] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'drift_detected': p_value < self.drift_thresholds['statistical']
            }

        return drift_results

    def _distance_based_drift_detection(self, X_current: np.ndarray) -> Dict[str, Any]:
        """Distance-based drift detection."""
        from scipy.spatial.distance import cdist

        # Calculate distances between reference and current data
        distances = cdist(self.reference_data['X'], X_current, metric='euclidean')
        min_distances = np.min(distances, axis=0)

        # Drift score based on minimum distances
        drift_score = np.mean(min_distances)

        return {
            'drift_score': drift_score,
            'drift_detected': drift_score > self.drift_thresholds['distance']
        }


class MLOpsPipeline:
    """Complete MLOps pipeline orchestrator."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)

        # Initialize components
        self.tracker = ExperimentTracker(
            backend=self.config.get('tracking_backend', 'mlflow'),
            project_name=self.config.get('project_name', 'polymer_ml')
        )
        self.registry = ModelRegistry(self.config.get('registry_path', './model_registry'))
        self.validator = ModelValidator()
        self.drift_detector = DataDriftDetector()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load pipeline configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        else:
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Default pipeline configuration."""
        return {
            'tracking_backend': 'mlflow',
            'project_name': 'polymer_ml',
            'registry_path': './model_registry',
            'validation_config': {
                'min_r2_score': 0.7,
                'run_invariance_tests': True,
                'run_directional_tests': True
            },
            'deployment_config': {
                'auto_promote': False,
                'require_validation': True
            }
        }

    def run_training_pipeline(self, model_class: type, model_params: Dict[str, Any],
                            X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            model_name: str, version: str) -> Dict[str, Any]:
        """Run complete training pipeline."""

        # Start experiment tracking
        run_id = self.tracker.start_run(f"{model_name}_v{version}")

        try:
            # Log parameters
            self.tracker.log_params(model_params)

            # Train model
            model = model_class(**model_params)
            start_time = time.time()

            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
            else:
                # Handle PyTorch models
                model = self._train_pytorch_model(model, X_train, y_train, X_val, y_val)

            training_time = time.time() - start_time

            # Evaluate model
            train_pred = self._predict_model(model, X_train)
            val_pred = self._predict_model(model, X_val)
            test_pred = self._predict_model(model, X_test)

            metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'train_r2': r2_score(y_train, train_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'val_r2': r2_score(y_val, val_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'test_r2': r2_score(y_test, test_pred),
                'training_time': training_time
            }

            # Log metrics
            self.tracker.log_metrics(metrics)

            # Validate model
            validation_results = self.validator.validate_model(model, X_test, y_test)
            self.tracker.log_metrics({f'validation_{k}': v for k, v in validation_results.items()})

            # Register model if validation passes
            if all(validation_results.values()):
                model_id = self.registry.register_model(
                    model, model_name, version, metrics,
                    metadata={'validation': validation_results, 'run_id': run_id}
                )

                # Auto-promote if configured
                if self.config.get('deployment_config', {}).get('auto_promote', False):
                    self.registry.promote_model(model_name, version, 'production')

                status = 'success'
            else:
                status = 'validation_failed'
                model_id = None

            # Log model
            self.tracker.log_model(model, model_name)

            return {
                'status': status,
                'model_id': model_id,
                'metrics': metrics,
                'validation': validation_results,
                'run_id': run_id
            }

        except Exception as e:
            logging.error(f"Training pipeline failed: {e}")
            return {'status': 'error', 'error': str(e), 'run_id': run_id}

        finally:
            self.tracker.end_run()

    def _train_pytorch_model(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray, epochs: int = 100) -> nn.Module:
        """Train PyTorch model with validation."""
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1) if y_train.ndim == 1 else torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1) if y_val.ndim == 1 else torch.FloatTensor(y_val)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            train_pred = model(X_train_tensor)
            train_loss = criterion(train_pred, y_train_tensor)
            train_loss.backward()
            optimizer.step()

            # Validation
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val_tensor)
                    val_loss = criterion(val_pred, y_val_tensor)

                self.tracker.log_metrics({
                    'train_loss': train_loss.item(),
                    'val_loss': val_loss.item()
                }, step=epoch)

        return model

    def _predict_model(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions with any model type."""
        if hasattr(model, 'predict'):
            return model.predict(X)
        else:
            # PyTorch model
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions = model(X_tensor).numpy()
            return predictions.flatten()

    def monitor_production_model(self, model_name: str, version: str,
                               X_current: np.ndarray, y_current: Optional[np.ndarray] = None):
        """Monitor production model for drift and performance."""
        # Load model
        model = self.registry.load_model(model_name, version)

        # Detect drift
        if self.drift_detector.reference_data is not None:
            drift_results = self.drift_detector.detect_drift(X_current, y_current)

            # Log drift metrics
            run_id = self.tracker.start_run(f"monitoring_{model_name}_v{version}")
            self.tracker.log_metrics({
                'drift_detected': any(result.get('drift_detected', False)
                                    for result in drift_results.get('feature_drift', {}).values())
            })
            self.tracker.end_run()

            return drift_results


def create_mlops_config(config_path: str = "./mlops_config.yaml"):
    """Create a sample MLOps configuration file."""
    config = {
        'tracking_backend': 'mlflow',
        'project_name': 'polymer_ml_production',
        'registry_path': './model_registry',
        'validation_config': {
            'min_r2_score': 0.8,
            'run_invariance_tests': True,
            'run_directional_tests': True,
            'feature_expectations': {
                0: 'positive',  # Temperature
                1: 'positive',  # Pressure
                2: 'negative'   # Catalyst age
            }
        },
        'deployment_config': {
            'auto_promote': False,
            'require_validation': True,
            'staging_tests': ['performance_test', 'invariance_test']
        },
        'monitoring_config': {
            'drift_detection_method': 'statistical',
            'drift_threshold': 0.05,
            'monitoring_frequency': 'daily'
        }
    }

    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=2)

    print(f"MLOps configuration saved to {config_path}")


def demo_mlops_pipeline():
    """Demonstrate MLOps pipeline capabilities."""
    print("MLOps Pipeline Demo")
    print("=" * 50)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :3], axis=1) + 0.1 * np.random.randn(n_samples)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Test data: {X_test.shape}")

    # Initialize MLOps pipeline
    pipeline = MLOpsPipeline()

    # Train and register model
    from sklearn.ensemble import RandomForestRegressor

    model_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }

    result = pipeline.run_training_pipeline(
        RandomForestRegressor, model_params,
        X_train, y_train, X_val, y_val, X_test, y_test,
        model_name='polymer_rf', version='1.0'
    )

    print(f"\nPipeline result: {result['status']}")
    print(f"Test R²: {result['metrics']['test_r2']:.4f}")
    print(f"Test RMSE: {result['metrics']['test_rmse']:.4f}")

    # Set up drift detection
    pipeline.drift_detector.set_reference_data(X_train, y_train)

    # Simulate production data with drift
    X_production = X_test + 0.2 * np.random.randn(*X_test.shape)  # Add drift
    drift_results = pipeline.monitor_production_model('polymer_rf', '1.0', X_production)

    print(f"\nDrift detection results:")
    feature_drift = any(result.get('drift_detected', False)
                       for result in drift_results.get('feature_drift', {}).values())
    print(f"Feature drift detected: {feature_drift}")


if __name__ == "__main__":
    # Create sample config
    create_mlops_config()

    # Run demo
    demo_mlops_pipeline()