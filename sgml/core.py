"""
Science-Guided Machine Learning (SGML) Framework - Core

This is the main framework that merges first-principle modeling with machine learning
to optimize polyolefin processes.

Key Innovation:
- Integrates physics-based models with neural networks
- Uses inductive bias to ensure predictions respect fundamental laws
- Enables hybrid predictions combining both approaches

Author: Niket Sharma (Ph.D. Research, Virginia Tech)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging

from .physics import (
    PolymerizationKinetics,
    ThermodynamicConstraints,
    MassBalanceModel
)
from .inductive_bias import (
    PhysicsInformedLayer,
    ThermodynamicConsistencyLayer,
    MassBalanceConstraintLayer
)

logger = logging.getLogger(__name__)


@dataclass
class SGMLConfig:
    """Configuration for SGML framework."""
    
    # Model architecture
    input_dim: int = 10
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    output_dim: int = 1
    
    # Physics integration
    use_physics_loss: bool = True
    physics_loss_weight: float = 0.5
    use_thermodynamic_constraints: bool = True
    use_mass_balance_constraints: bool = True
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    
    # Hybrid prediction
    physics_weight: float = 0.3  # Weight for physics-based predictions
    ml_weight: float = 0.7  # Weight for ML predictions
    
    # Optimization
    optimize_process: bool = False
    optimization_target: str = "conversion"  # or "molecular_weight", "yield"


class SGMLFramework(nn.Module):
    """
    Science-Guided Machine Learning Framework.
    
    Integrates:
    1. First-principle physics models (kinetics, thermodynamics, mass balance)
    2. Machine learning models with inductive bias
    3. Hybrid prediction combining both approaches
    
    This framework ensures predictions respect fundamental laws of chemistry
    and physics while leveraging data-driven learning.
    """
    
    def __init__(self, config: Optional[SGMLConfig] = None):
        """
        Initialize SGML framework.
        
        Args:
            config: SGML configuration
        """
        super().__init__()
        self.config = config or SGMLConfig()
        
        # Initialize physics models
        self.kinetics_model = PolymerizationKinetics()
        self.thermo_model = ThermodynamicConstraints()
        self.mass_balance_model = MassBalanceModel()
        
        # Build neural network with inductive bias
        self.ml_model = self._build_ml_model()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.ml_model.parameters(),
            lr=self.config.learning_rate
        )
        
        logger.info("SGML Framework initialized")
        logger.info(f"Physics loss weight: {self.config.physics_loss_weight}")
        logger.info(f"Hybrid weights: Physics={self.config.physics_weight}, ML={self.config.ml_weight}")
    
    def _build_ml_model(self) -> nn.Module:
        """
        Build physics-informed neural network.
        
        Returns:
            Neural network with inductive bias layers
        """
        layers = []
        
        # Input layer
        prev_dim = self.config.input_dim
        
        # Hidden layers
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Output layer with physics constraints
        if self.config.use_thermodynamic_constraints:
            layers.append(
                ThermodynamicConsistencyLayer(
                    prev_dim,
                    self.config.output_dim,
                    physics_weight=self.config.physics_loss_weight
                )
            )
        elif self.config.use_mass_balance_constraints:
            layers.append(
                MassBalanceConstraintLayer(
                    prev_dim,
                    self.config.output_dim,
                    physics_weight=self.config.physics_loss_weight
                )
            )
        else:
            layers.append(nn.Linear(prev_dim, self.config.output_dim))
        
        model = nn.Sequential(*layers)
        
        logger.info(f"ML model architecture: {self.config.input_dim} → "
                   f"{self.config.hidden_dims} → {self.config.output_dim}")
        
        return model
    
    def physics_prediction(self, x: Dict[str, float]) -> Dict[str, float]:
        """
        Make prediction using first-principle physics models.
        
        Args:
            x: Dictionary with process variables:
               - temperature (K)
               - pressure (bar)
               - monomer_concentration (mol/L)
               - catalyst_concentration (mol/L)
               - active_sites (mol/L)
        
        Returns:
            Dictionary with physics-based predictions
        """
        T = x.get('temperature', 350.0)
        C_M = x.get('monomer_concentration', 1.0)
        C_P = x.get('active_sites', 0.01)
        
        # Calculate using kinetics model
        R_p = self.kinetics_model.propagation_rate(T, C_M, C_P)
        Mn, Mw = self.kinetics_model.calculate_molecular_weight(T, C_M, C_P)
        
        # Calculate using thermodynamics
        conversion_eq = self.thermo_model.equilibrium_conversion(
            T,
            x.get('pressure', 10.0)
        )
        delta_G = self.thermo_model.gibbs_free_energy(T, conversion_eq)
        
        return {
            'propagation_rate': R_p,
            'number_average_MW': Mn,
            'weight_average_MW': Mw,
            'equilibrium_conversion': conversion_eq,
            'gibbs_energy': delta_G,
            'polydispersity': Mw / Mn if Mn > 0 else 1.0
        }
    
    def ml_prediction(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make prediction using machine learning model.
        
        Args:
            x: Input tensor [batch, features]
        
        Returns:
            Tuple of (prediction, physics_loss)
        """
        # Forward pass through physics-informed network
        output = None
        physics_loss = torch.tensor(0.0)
        
        for layer in self.ml_model:
            if isinstance(layer, PhysicsInformedLayer):
                output, physics_loss = layer(x if output is None else output)
            else:
                x = layer(x) if output is None else layer(output)
                output = x
        
        if output is None:
            output = self.ml_model(x)
        
        return output, physics_loss
    
    def hybrid_prediction(
        self,
        x_dict: Dict[str, float],
        x_tensor: torch.Tensor
    ) -> Dict[str, any]:
        """
        Make hybrid prediction combining physics and ML.
        
        This is the core innovation of SGML: combining first-principles with data-driven learning.
        
        Args:
            x_dict: Dictionary for physics model
            x_tensor: Tensor for ML model
        
        Returns:
            Dictionary with hybrid predictions
        """
        # Physics prediction
        physics_pred = self.physics_prediction(x_dict)
        
        # ML prediction
        with torch.no_grad():
            ml_output, physics_loss = self.ml_prediction(x_tensor)
            ml_pred = ml_output.cpu().numpy()
        
        # Hybrid combination (weighted average)
        hybrid_conversion = (
            self.config.physics_weight * physics_pred['equilibrium_conversion'] +
            self.config.ml_weight * float(ml_pred[0, 0] if ml_pred.ndim > 1 else ml_pred[0])
        )
        
        return {
            'physics_prediction': physics_pred,
            'ml_prediction': float(ml_pred[0, 0] if ml_pred.ndim > 1 else ml_pred[0]),
            'hybrid_prediction': hybrid_conversion,
            'physics_loss': float(physics_loss),
            'confidence': 1.0 - abs(
                physics_pred['equilibrium_conversion'] - float(ml_pred[0, 0] if ml_pred.ndim > 1 else ml_pred[0])
            )
        }
    
    def train_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            x_batch: Input batch
            y_batch: Target batch
        
        Returns:
            Dictionary with loss values
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions, physics_loss = self.ml_prediction(x_batch)
        
        # Data loss (MSE)
        data_loss = nn.functional.mse_loss(predictions, y_batch)
        
        # Total loss (data + physics)
        if self.config.use_physics_loss:
            total_loss = data_loss + self.config.physics_loss_weight * physics_loss
        else:
            total_loss = data_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': float(total_loss),
            'data_loss': float(data_loss),
            'physics_loss': float(physics_loss)
        }
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, List[float]]:
        """
        Train SGML model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        
        Returns:
            Training history
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        
        history = {
            'train_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'val_loss': []
        }
        
        logger.info(f"Training SGML model for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.ml_model.train()
            epoch_losses = []
            
            # Mini-batch training
            n_samples = X_train_tensor.shape[0]
            indices = torch.randperm(n_samples)
            
            for i in range(0, n_samples, self.config.batch_size):
                batch_indices = indices[i:i+self.config.batch_size]
                x_batch = X_train_tensor[batch_indices]
                y_batch = y_train_tensor[batch_indices]
                
                losses = self.train_step(x_batch, y_batch)
                epoch_losses.append(losses)
            
            # Average losses
            avg_losses = {
                key: np.mean([l[key] for l in epoch_losses])
                for key in epoch_losses[0].keys()
            }
            
            history['train_loss'].append(avg_losses['total_loss'])
            history['data_loss'].append(avg_losses['data_loss'])
            history['physics_loss'].append(avg_losses['physics_loss'])
            
            # Validation
            if X_val is not None and y_val is not None:
                self.ml_model.eval()
                X_val_tensor = torch.FloatTensor(X_val)
                y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
                
                with torch.no_grad():
                    val_pred, val_physics_loss = self.ml_prediction(X_val_tensor)
                    val_loss = nn.functional.mse_loss(val_pred, y_val_tensor)
                    history['val_loss'].append(float(val_loss))
            
            # Logging
            if (epoch + 1) % 10 == 0:
                log_msg = f"Epoch {epoch+1}/{self.config.num_epochs} - "
                log_msg += f"Loss: {avg_losses['total_loss']:.4f} "
                log_msg += f"(Data: {avg_losses['data_loss']:.4f}, "
                log_msg += f"Physics: {avg_losses['physics_loss']:.4f})"
                
                if history['val_loss']:
                    log_msg += f", Val: {history['val_loss'][-1]:.4f}"
                
                logger.info(log_msg)
        
        logger.info("Training complete")
        return history
    
    def predict(
        self,
        X: np.ndarray,
        use_hybrid: bool = True
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            use_hybrid: Use hybrid prediction (physics + ML)
        
        Returns:
            Predictions
        """
        self.ml_model.eval()
        X_tensor = torch.FloatTensor(X)
        
        with torch.no_grad():
            predictions, _ = self.ml_prediction(X_tensor)
        
        return predictions.numpy()


# Convenience function
def create_sgml_model(
    input_dim: int,
    output_dim: int = 1,
    use_physics_constraints: bool = True
) -> SGMLFramework:
    """
    Create SGML model with default configuration.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension
        use_physics_constraints: Use physics-informed layers
    
    Returns:
        Configured SGML framework
    """
    config = SGMLConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        use_thermodynamic_constraints=use_physics_constraints,
        use_mass_balance_constraints=use_physics_constraints
    )
    
    return SGMLFramework(config)
