"""
Modern Random Forest pipeline for HDPE melt index prediction.

This script demonstrates an updated workflow using modern ML practices including:
- Advanced feature engineering
- Hyperparameter optimization
- Cross-validation
- Model interpretability
- Uncertainty quantification
"""

from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.inspection import permutation_importance

# Try to import advanced libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent / "HDPE_LG_Plant_Data.csv"


class ModernRandomForestPipeline:
    """
    Modern Random Forest pipeline with advanced ML practices.
    """

    def __init__(self, use_optimization: bool = True, use_interpretability: bool = True):
        self.use_optimization = use_optimization
        self.use_interpretability = use_interpretability
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.best_params = None

    def load_and_preprocess_data(self, path: Path) -> tuple[pd.DataFrame, pd.Series]:
        """Load and preprocess data with advanced feature engineering."""
        logger.info(f"Loading data from {path}")

        df = pd.read_csv(path)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Store feature names for interpretability
        self.feature_names = X.columns.tolist()

        # Advanced feature engineering
        X_engineered = self._engineer_features(X)

        logger.info(f"Data loaded: {X_engineered.shape[0]} samples, {X_engineered.shape[1]} features")
        return X_engineered, y

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features from raw process variables."""
        X_new = X.copy()

        # Polynomial features for key variables
        if 'C2' in X.columns and 'H2' in X.columns:
            X_new['C2_H2_ratio'] = X['C2'] / (X['H2'] + 1e-8)
            X_new['C2_H2_interaction'] = X['C2'] * X['H2']

        if 'T' in X.columns and 'P' in X.columns:
            X_new['T_P_interaction'] = X['T'] * X['P']
            X_new['T_squared'] = X['T'] ** 2

        # Rolling statistics (if time-series nature)
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X_new[f'{col}_rolling_mean'] = X[col].rolling(window=5, min_periods=1).mean()
                X_new[f'{col}_rolling_std'] = X[col].rolling(window=5, min_periods=1).std().fillna(0)

        return X_new

    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """Optimize hyperparameters using Optuna or GridSearch."""
        if OPTUNA_AVAILABLE and self.use_optimization:
            return self._optuna_optimization(X_train, y_train)
        else:
            return self._grid_search_optimization(X_train, y_train)

    def _optuna_optimization(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """Optimize using Optuna."""
        logger.info("Starting Optuna hyperparameter optimization")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42
            }

            model = RandomForestRegressor(**params)
            scores = cross_val_score(model, X_train, y_train, cv=5,
                                   scoring='neg_mean_squared_error')
            return -scores.mean()

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)

        logger.info(f"Best parameters: {study.best_params}")
        return study.best_params

    def _grid_search_optimization(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """Optimize using GridSearchCV."""
        logger.info("Starting GridSearch hyperparameter optimization")

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_params_

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the Random Forest model with optimized parameters."""
        if self.use_optimization:
            self.best_params = self.optimize_hyperparameters(X_train, y_train)
            self.model = RandomForestRegressor(**self.best_params)
        else:
            # Use reasonable default parameters
            self.model = RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, max_features='sqrt', random_state=42
            )

        logger.info("Training Random Forest model")
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Comprehensive model evaluation."""
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5,
                                  scoring='neg_mean_squared_error')

        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'cv_rmse_mean': np.sqrt(-cv_scores.mean()),
            'cv_rmse_std': np.sqrt(cv_scores.std()),
            'overfitting_score': abs(
                np.sqrt(mean_squared_error(y_train, y_train_pred)) -
                np.sqrt(mean_squared_error(y_test, y_test_pred))
            )
        }

        return metrics

    def interpret_model(self, X_test: np.ndarray, y_test: np.ndarray):
        """Model interpretability analysis."""
        if not self.use_interpretability:
            return

        logger.info("Analyzing model interpretability")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n🔍 Feature Importance (Top 10):")
        print(feature_importance.head(10).to_string(index=False))

        # Permutation importance
        perm_importance = permutation_importance(
            self.model, X_test, y_test, n_repeats=10, random_state=42
        )

        perm_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)

        print("\n🔄 Permutation Importance (Top 10):")
        print(perm_df.head(10).to_string(index=False))

        # SHAP analysis if available
        if SHAP_AVAILABLE:
            self._shap_analysis(X_test[:100])  # Use subset for speed

    def _shap_analysis(self, X_sample: np.ndarray):
        """SHAP analysis for model interpretability."""
        try:
            logger.info("Running SHAP analysis")
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)

            # Summary statistics
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            shap_importance = pd.DataFrame({
                'feature': self.feature_names,
                'shap_importance': mean_abs_shap
            }).sort_values('shap_importance', ascending=False)

            print("\n📊 SHAP Feature Importance (Top 10):")
            print(shap_importance.head(10).to_string(index=False))

        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")

    def get_prediction_intervals(self, X: np.ndarray, confidence: float = 0.95) -> tuple:
        """Get prediction intervals using ensemble predictions."""
        # Get predictions from individual trees
        tree_predictions = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ])

        # Calculate prediction intervals
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        predictions = np.mean(tree_predictions, axis=0)
        lower_bound = np.percentile(tree_predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(tree_predictions, upper_percentile, axis=0)

        return predictions, lower_bound, upper_bound

    def run_complete_pipeline(self):
        """Run the complete modern Random Forest pipeline."""
        print("🌲 Modern Random Forest Pipeline for HDPE Melt Index Prediction")
        print("=" * 70)

        # Load and preprocess data
        X, y = self.load_and_preprocess_data(DATA_PATH)

        # Train-test split with stratification consideration
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features using RobustScaler (more robust to outliers)
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Update feature names after scaling
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()

        # Train model
        self.train_model(X_train_scaled, y_train)

        # Evaluate model
        metrics = self.evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test)

        # Display results
        print("\n📈 Model Performance Metrics:")
        print(f"  Training RMSE: {metrics['train_rmse']:.4f}")
        print(f"  Training R²:   {metrics['train_r2']:.4f}")
        print(f"  Test RMSE:     {metrics['test_rmse']:.4f}")
        print(f"  Test R²:       {metrics['test_r2']:.4f}")
        print(f"  Test MAE:      {metrics['test_mae']:.4f}")
        print(f"  CV RMSE:       {metrics['cv_rmse_mean']:.4f} ± {metrics['cv_rmse_std']:.4f}")
        print(f"  Overfitting:   {metrics['overfitting_score']:.4f}")

        # Model interpretability
        self.interpret_model(X_test_scaled, y_test)

        # Prediction intervals demo
        print("\n🎯 Prediction Intervals (first 5 test samples):")
        pred_mean, pred_lower, pred_upper = self.get_prediction_intervals(X_test_scaled[:5])

        for i in range(5):
            print(f"  Sample {i+1}: {pred_mean[i]:.3f} "
                  f"[{pred_lower[i]:.3f}, {pred_upper[i]:.3f}] "
                  f"(Actual: {y_test.iloc[i]:.3f})")

        # Best parameters
        if self.best_params:
            print(f"\n⚙️  Best Parameters:")
            for param, value in self.best_params.items():
                print(f"    {param}: {value}")

        print("\n✅ Pipeline completed successfully!")

        return {
            'model': self.model,
            'scaler': self.scaler,
            'metrics': metrics,
            'feature_names': self.feature_names
        }


def main() -> None:
    """Main function to run the modern Random Forest pipeline."""
    # Initialize pipeline with modern features
    pipeline = ModernRandomForestPipeline(
        use_optimization=OPTUNA_AVAILABLE,  # Use optimization if Optuna available
        use_interpretability=True
    )

    # Run complete pipeline
    results = pipeline.run_complete_pipeline()

    return results


if __name__ == "__main__":
    main()
