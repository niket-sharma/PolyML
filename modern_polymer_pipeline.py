"""
Modern polymer property prediction pipeline integrating all state-of-the-art techniques.

This script demonstrates the complete modernized pipeline including:
- Advanced molecular representations
- Transformer models
- Uncertainty quantification
- Ensemble learning
- Hyperparameter optimization
- MLOps integration
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# Import our modern modules
from molecular_representations import AdvancedMolecularFeaturizer
from transformer_models import create_transformer_pipeline, SMILESTokenizer
from advanced_timeseries import create_advanced_timeseries_model
from uncertainty_quantification import MCDropoutModel, DeepEnsemble, BayesianNeuralNetwork
from ensemble_learning import create_polymer_ensemble
from hyperparameter_optimization import HyperparameterOptimizer, create_search_space
from mlops_pipeline import MLOpsPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModernPolymerPredictor:
    """
    State-of-the-art polymer property prediction system.

    Integrates multiple advanced ML/NLP techniques for optimal performance.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()

        # Initialize components
        self.molecular_featurizer = AdvancedMolecularFeaturizer()
        self.smiles_tokenizer = SMILESTokenizer()
        self.mlops_pipeline = MLOpsPipeline()

        # Model components
        self.models = {}
        self.ensembles = {}
        self.uncertainty_models = {}

        logger.info("Modern Polymer Predictor initialized")

    def _default_config(self) -> Dict:
        """Default configuration for the predictor."""
        return {
            'use_molecular_transformers': True,
            'use_time_series_models': True,
            'use_uncertainty_quantification': True,
            'use_ensemble_methods': True,
            'enable_hyperparameter_optimization': True,
            'enable_mlops': True,
            'molecular_features': {
                'use_fingerprints': True,
                'use_descriptors': True,
                'use_graphs': True
            },
            'model_types': ['transformer', 'ensemble', 'uncertainty'],
            'optimization': {
                'method': 'optuna',
                'n_trials': 50
            }
        }

    def prepare_molecular_data(self, smiles_list: List[str]) -> Dict[str, Any]:
        """
        Prepare comprehensive molecular features from SMILES.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Dictionary containing different molecular representations
        """
        logger.info(f"Processing {len(smiles_list)} molecules")

        # Advanced molecular features
        molecular_features = self.molecular_featurizer.featurize_dataset(smiles_list)

        # Extract different representation types
        fingerprints = []
        descriptors = []
        tokenized_smiles = []

        for features in molecular_features:
            if features.fingerprints is not None:
                fingerprints.append(features.fingerprints)
            if features.descriptors is not None:
                descriptors.append(features.descriptors)

            # Tokenize SMILES for transformer models
            encoded = self.smiles_tokenizer.encode(features.smiles, max_length=128)
            tokenized_smiles.append({
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask']
            })

        return {
            'fingerprints': np.array(fingerprints) if fingerprints else None,
            'descriptors': np.array(descriptors) if descriptors else None,
            'tokenized_smiles': tokenized_smiles,
            'raw_smiles': smiles_list
        }

    def prepare_process_data(self, process_data: np.ndarray) -> Dict[str, Any]:
        """
        Prepare process data for time-series models.

        Args:
            process_data: Process variables array

        Returns:
            Dictionary containing processed data
        """
        # For time-series models, reshape to sequence format
        if len(process_data.shape) == 2:
            # Assume each row is a time step, reshape for sequences
            n_samples, n_features = process_data.shape
            sequence_length = min(10, n_samples // 10)  # Adaptive sequence length

            sequences = []
            for i in range(0, n_samples - sequence_length + 1, sequence_length):
                seq = process_data[i:i + sequence_length]
                sequences.append(seq)

            time_series_data = np.array(sequences)
        else:
            time_series_data = process_data

        return {
            'raw_data': process_data,
            'time_series': time_series_data,
            'scaled_data': self._scale_features(process_data)
        }

    def _scale_features(self, data: np.ndarray) -> np.ndarray:
        """Scale features using robust scaling."""
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        return scaler.fit_transform(data)

    def build_transformer_models(self, molecular_data: Dict, process_data: Dict,
                                targets: np.ndarray) -> Dict[str, Any]:
        """
        Build and train transformer-based models.

        Args:
            molecular_data: Molecular feature dictionary
            process_data: Process data dictionary
            targets: Target values

        Returns:
            Dictionary of trained transformer models
        """
        models = {}

        if self.config['use_molecular_transformers']:
            logger.info("Building molecular transformer models")

            # Molecular transformer
            mol_transformer = create_transformer_pipeline("molecular")

            # Prepare training data
            if molecular_data['tokenized_smiles']:
                # Convert to tensors for batch processing
                input_ids = torch.stack([item['input_ids'] for item in molecular_data['tokenized_smiles']])
                attention_masks = torch.stack([item['attention_mask'] for item in molecular_data['tokenized_smiles']])

                # Simple training loop (in practice, use more sophisticated training)
                models['molecular_transformer'] = self._train_transformer_model(
                    mol_transformer, input_ids, attention_masks, targets
                )

        if self.config['use_time_series_models']:
            logger.info("Building time-series transformer models")

            # Time-series transformer
            if process_data['time_series'].size > 0:
                ts_transformer = create_advanced_timeseries_model(
                    "adaptive", input_dim=process_data['raw_data'].shape[1]
                )

                models['timeseries_transformer'] = self._train_timeseries_model(
                    ts_transformer, process_data['time_series'], targets
                )

        return models

    def build_uncertainty_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Build uncertainty quantification models.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            Dictionary of uncertainty models
        """
        models = {}

        if not self.config['use_uncertainty_quantification']:
            return models

        logger.info("Building uncertainty quantification models")

        # MC Dropout model
        mc_dropout = MCDropoutModel(
            input_dim=X.shape[1],
            hidden_dims=[128, 64, 32],
            dropout_rate=0.2
        )
        mc_dropout.fit(torch.FloatTensor(X), torch.FloatTensor(y), epochs=50)
        models['mc_dropout'] = mc_dropout

        # Deep ensemble
        deep_ensemble = DeepEnsemble(
            input_dim=X.shape[1],
            hidden_dims=[128, 64, 32],
            num_models=5
        )
        deep_ensemble.fit(torch.FloatTensor(X), torch.FloatTensor(y), epochs=50)
        models['deep_ensemble'] = deep_ensemble

        # Bayesian neural network
        bayesian_nn = BayesianNeuralNetwork(
            input_dim=X.shape[1],
            hidden_dims=[128, 64]
        )
        bayesian_nn.fit(torch.FloatTensor(X), torch.FloatTensor(y), epochs=50)
        models['bayesian_nn'] = bayesian_nn

        return models

    def build_ensemble_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Build ensemble learning models.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            Dictionary of ensemble models
        """
        models = {}

        if not self.config['use_ensemble_methods']:
            return models

        logger.info("Building ensemble models")

        # Stacking ensemble
        stacking_ensemble = create_polymer_ensemble("stacking")
        stacking_ensemble.fit(X, y)
        models['stacking'] = stacking_ensemble

        # Dynamic ensemble
        dynamic_ensemble = create_polymer_ensemble("dynamic")
        dynamic_ensemble.fit(X, y, epochs=50)
        models['dynamic'] = dynamic_ensemble

        return models

    def optimize_hyperparameters(self, model_class, X: np.ndarray, y: np.ndarray,
                                model_type: str = 'neural_network') -> Dict[str, Any]:
        """
        Optimize hyperparameters for models.

        Args:
            model_class: Model class to optimize
            X: Feature matrix
            y: Target values
            model_type: Type of model for search space

        Returns:
            Optimized hyperparameters
        """
        if not self.config['enable_hyperparameter_optimization']:
            return {}

        logger.info(f"Optimizing hyperparameters for {model_type}")

        # Split data for optimization
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create optimizer
        optimizer = HyperparameterOptimizer('optuna')
        search_space = create_search_space(model_type)

        # Optimize
        best_params = optimizer.optimize(
            model_class, search_space,
            X_train, y_train, X_val, y_val,
            n_trials=self.config['optimization']['n_trials']
        )

        return best_params

    def train_complete_pipeline(self, smiles_data: List[str], process_data: np.ndarray,
                              targets: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the complete modern pipeline.

        Args:
            smiles_data: SMILES strings
            process_data: Process variables
            targets: Target values
            test_size: Fraction for test set

        Returns:
            Complete pipeline results
        """
        logger.info("Starting complete pipeline training")

        # Prepare data
        molecular_data = self.prepare_molecular_data(smiles_data)
        process_features = self.prepare_process_data(process_data)

        # Combine features for traditional ML models
        combined_features = []
        if molecular_data['fingerprints'] is not None:
            combined_features.append(molecular_data['fingerprints'])
        if molecular_data['descriptors'] is not None:
            combined_features.append(molecular_data['descriptors'])
        if process_features['scaled_data'] is not None:
            combined_features.append(process_features['scaled_data'])

        if combined_features:
            X_combined = np.hstack(combined_features)
        else:
            # Fallback to process data only
            X_combined = process_features['raw_data']

        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, targets, test_size=test_size, random_state=42
        )

        results = {
            'data_info': {
                'n_samples': len(targets),
                'n_features': X_combined.shape[1],
                'n_molecules': len(smiles_data)
            },
            'models': {},
            'metrics': {}
        }

        # Build and train models
        try:
            # Transformer models
            transformer_models = self.build_transformer_models(
                molecular_data, process_features, y_train
            )
            results['models'].update(transformer_models)

            # Uncertainty models
            uncertainty_models = self.build_uncertainty_models(X_train, y_train)
            results['models'].update(uncertainty_models)

            # Ensemble models
            ensemble_models = self.build_ensemble_models(X_train, y_train)
            results['models'].update(ensemble_models)

            # Evaluate all models
            results['metrics'] = self._evaluate_all_models(
                results['models'], X_test, y_test
            )

            # MLOps integration
            if self.config['enable_mlops']:
                self._integrate_mlops(results, X_train, y_train, X_test, y_test)

        except Exception as e:
            logger.error(f"Pipeline training failed: {e}")
            results['error'] = str(e)

        logger.info("Pipeline training completed")
        return results

    def _train_transformer_model(self, model: nn.Module, input_ids: torch.Tensor,
                                attention_masks: torch.Tensor, targets: np.ndarray,
                                epochs: int = 50) -> nn.Module:
        """Train transformer model."""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        targets_tensor = torch.FloatTensor(targets).unsqueeze(1)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(input_ids, attention_masks)
            loss = criterion(outputs, targets_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                logger.debug(f"Transformer training epoch {epoch}, loss: {loss.item():.4f}")

        return model

    def _train_timeseries_model(self, model: nn.Module, time_series_data: np.ndarray,
                              targets: np.ndarray, epochs: int = 50) -> nn.Module:
        """Train time-series model."""
        if time_series_data.size == 0:
            return model

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Use last sequence for each target
        X_tensor = torch.FloatTensor(time_series_data[:len(targets)])
        y_tensor = torch.FloatTensor(targets[:len(time_series_data)]).unsqueeze(1)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                logger.debug(f"Time-series training epoch {epoch}, loss: {loss.item():.4f}")

        return model

    def _evaluate_all_models(self, models: Dict[str, Any], X_test: np.ndarray,
                           y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models."""
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        metrics = {}

        for model_name, model in models.items():
            try:
                # Make predictions
                if hasattr(model, 'predict_with_uncertainty'):
                    # Uncertainty model
                    X_tensor = torch.FloatTensor(X_test)
                    predictions, uncertainties = model.predict_with_uncertainty(X_tensor)
                    predictions = predictions.numpy().flatten()

                    metrics[model_name] = {
                        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                        'mae': mean_absolute_error(y_test, predictions),
                        'r2': r2_score(y_test, predictions),
                        'mean_uncertainty': float(np.mean(uncertainties.numpy()))
                    }

                elif hasattr(model, 'predict'):
                    # Scikit-learn style model
                    predictions = model.predict(X_test)

                    metrics[model_name] = {
                        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                        'mae': mean_absolute_error(y_test, predictions),
                        'r2': r2_score(y_test, predictions)
                    }

                else:
                    # PyTorch model
                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_test)
                        predictions = model(X_tensor).numpy().flatten()

                    metrics[model_name] = {
                        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                        'mae': mean_absolute_error(y_test, predictions),
                        'r2': r2_score(y_test, predictions)
                    }

            except Exception as e:
                logger.error(f"Failed to evaluate model {model_name}: {e}")
                metrics[model_name] = {'error': str(e)}

        return metrics

    def _integrate_mlops(self, results: Dict, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray):
        """Integrate with MLOps pipeline."""
        try:
            # Register best performing model
            best_model_name = max(
                results['metrics'].keys(),
                key=lambda k: results['metrics'][k].get('r2', -float('inf'))
                if 'error' not in results['metrics'][k] else -float('inf')
            )

            best_model = results['models'][best_model_name]
            best_metrics = results['metrics'][best_model_name]

            # Use MLOps pipeline for model management
            if hasattr(best_model, 'get_params'):
                # Scikit-learn model
                from sklearn.ensemble import RandomForestRegressor
                pipeline_result = self.mlops_pipeline.run_training_pipeline(
                    RandomForestRegressor, best_model.get_params(),
                    X_train, y_train, X_test[:50], y_test[:50], X_test[50:], y_test[50:],
                    model_name='polymer_predictor', version='1.0'
                )
                results['mlops_result'] = pipeline_result

        except Exception as e:
            logger.error(f"MLOps integration failed: {e}")
            results['mlops_error'] = str(e)

    def predict_with_uncertainty(self, smiles_data: List[str], process_data: np.ndarray,
                                model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Make predictions with uncertainty estimates.

        Args:
            smiles_data: SMILES strings
            process_data: Process variables
            model_name: Specific model to use (if None, use best available)

        Returns:
            Predictions with uncertainty estimates
        """
        # Prepare data
        molecular_data = self.prepare_molecular_data(smiles_data)
        process_features = self.prepare_process_data(process_data)

        # Combine features
        combined_features = []
        if molecular_data['fingerprints'] is not None:
            combined_features.append(molecular_data['fingerprints'])
        if molecular_data['descriptors'] is not None:
            combined_features.append(molecular_data['descriptors'])
        if process_features['scaled_data'] is not None:
            combined_features.append(process_features['scaled_data'])

        X = np.hstack(combined_features) if combined_features else process_features['raw_data']

        # Use uncertainty models if available
        uncertainty_models = {k: v for k, v in self.models.items()
                            if 'uncertainty' in k or 'ensemble' in k}

        if uncertainty_models and model_name is None:
            # Use best uncertainty model
            model_name = list(uncertainty_models.keys())[0]
            model = uncertainty_models[model_name]
        elif model_name and model_name in self.models:
            model = self.models[model_name]
        else:
            raise ValueError("No suitable model found")

        # Make predictions
        X_tensor = torch.FloatTensor(X)
        if hasattr(model, 'predict_with_uncertainty'):
            predictions, uncertainties = model.predict_with_uncertainty(X_tensor)
            return {
                'predictions': predictions.numpy().flatten(),
                'uncertainties': uncertainties.numpy().flatten(),
                'model_used': model_name
            }
        else:
            if hasattr(model, 'predict'):
                predictions = model.predict(X)
            else:
                model.eval()
                with torch.no_grad():
                    predictions = model(X_tensor).numpy().flatten()

            return {
                'predictions': predictions,
                'uncertainties': None,
                'model_used': model_name
            }


def load_polymer_data() -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Load polymer data from CSV files.

    Returns:
        Tuple of (SMILES, process_data, targets)
    """
    # Load HDPE plant data
    hdpe_data = pd.read_csv('HDPE_LG_Plant_Data.csv')

    # Load polymer SMILES data
    polymer_data = pd.read_csv('Polymer_Tg_SMILES.csv')

    # Extract features and targets
    process_features = hdpe_data.iloc[:, :-1].values  # All columns except last
    melt_index = hdpe_data.iloc[:, -1].values  # Last column

    # Get SMILES data
    smiles_list = polymer_data['SMILES Structure'].tolist()
    tg_values = polymer_data['Tg'].values

    # For demonstration, use process data with HDPE targets
    # In practice, you'd need to properly align molecular and process data
    min_samples = min(len(smiles_list), len(process_features))

    return (
        smiles_list[:min_samples],
        process_features[:min_samples],
        melt_index[:min_samples]
    )


def main():
    """
    Demonstrate the complete modern polymer prediction pipeline.
    """
    print("🧪 Modern Polymer Property Prediction Pipeline")
    print("=" * 60)

    try:
        # Load data
        print("📊 Loading polymer data...")
        smiles_data, process_data, targets = load_polymer_data()

        print(f"✅ Data loaded:")
        print(f"   - {len(smiles_data)} molecules")
        print(f"   - {process_data.shape[1]} process features")
        print(f"   - {len(targets)} target values")

        # Initialize modern predictor
        print("\n🤖 Initializing modern predictor...")
        config = {
            'use_molecular_transformers': True,
            'use_uncertainty_quantification': True,
            'use_ensemble_methods': True,
            'enable_hyperparameter_optimization': False,  # Disable for demo speed
            'enable_mlops': True,
            'optimization': {'n_trials': 10}  # Reduced for demo
        }

        predictor = ModernPolymerPredictor(config)

        # Train complete pipeline
        print("\n🚀 Training complete pipeline...")
        results = predictor.train_complete_pipeline(
            smiles_data, process_data, targets, test_size=0.2
        )

        # Display results
        print("\n📈 Pipeline Results:")
        print(f"   - Data info: {results['data_info']}")
        print(f"   - Models trained: {list(results['models'].keys())}")

        if 'metrics' in results:
            print("\n📊 Model Performance:")
            for model_name, metrics in results['metrics'].items():
                if 'error' not in metrics:
                    print(f"   {model_name}:")
                    print(f"     - RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                    print(f"     - R²: {metrics.get('r2', 'N/A'):.4f}")
                    if 'mean_uncertainty' in metrics:
                        print(f"     - Mean Uncertainty: {metrics['mean_uncertainty']:.4f}")

        # Demonstrate prediction with uncertainty
        print("\n🔮 Making predictions with uncertainty...")
        test_smiles = smiles_data[:5]
        test_process = process_data[:5]

        # Store models for prediction
        predictor.models = results['models']

        pred_results = predictor.predict_with_uncertainty(test_smiles, test_process)

        print(f"✅ Predictions completed using: {pred_results['model_used']}")
        print(f"   Sample predictions: {pred_results['predictions'][:3]}")
        if pred_results['uncertainties'] is not None:
            print(f"   Sample uncertainties: {pred_results['uncertainties'][:3]}")

        print("\n🎉 Pipeline demonstration completed successfully!")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logger.error(f"Pipeline error: {e}", exc_info=True)


if __name__ == "__main__":
    main()