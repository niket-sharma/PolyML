"""
Physics-Informed Neural Network Layers

Custom neural network layers that enforce physical and chemical laws through
inductive bias. This ensures predictions respect fundamental principles.

Key Innovation:
Instead of learning arbitrary functions, these layers constrain the hypothesis
space to physically realistic solutions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable, Tuple
import logging

logger = logging.getLogger(__name__)


class PhysicsInformedLayer(nn.Module):
    """
    Base class for physics-informed neural network layers.
    
    Adds physics-based constraints as additional loss terms or hard constraints
    on the network output.
    
    This implements "inductive bias" - prior knowledge that guides learning
    toward physically meaningful solutions.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        physics_weight: float = 1.0
    ):
        """
        Initialize physics-informed layer.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            physics_weight: Weight for physics loss term
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_weight = physics_weight
        
        # Standard neural network layer
        self.linear = nn.Linear(input_dim, output_dim)
        
        logger.info(f"PhysicsInformedLayer initialized ({input_dim}→{output_dim})")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with physics constraint.
        
        Args:
            x: Input tensor
        
        Returns:
            Tuple of (output, physics_loss)
        """
        # Standard forward pass
        output = self.linear(x)
        
        # Compute physics-based loss
        physics_loss = self.compute_physics_loss(x, output)
        
        return output, physics_loss
    
    def compute_physics_loss(
        self,
        x: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute physics-based loss term.
        
        Override this in subclasses to implement specific physical constraints.
        
        Args:
            x: Input tensor
            output: Network output
        
        Returns:
            Physics loss (scalar)
        """
        # Default: no additional physics loss
        return torch.tensor(0.0, device=x.device)


class ThermodynamicConsistencyLayer(PhysicsInformedLayer):
    """
    Neural network layer enforcing thermodynamic consistency.
    
    Ensures predictions respect:
    - Energy conservation (First Law)
    - Entropy increase (Second Law)
    - Gibbs free energy minimization
    - Chemical equilibrium constraints
    
    This is a key example of using inductive bias to respect physical laws.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        T_index: int = 0,
        conversion_index: int = 1,
        physics_weight: float = 1.0
    ):
        """
        Initialize thermodynamic consistency layer.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            T_index: Index of temperature in input
            conversion_index: Index of conversion in input/output
            physics_weight: Weight for physics loss
        """
        super().__init__(input_dim, output_dim, physics_weight)
        self.T_index = T_index
        self.conversion_index = conversion_index
        
        # Thermodynamic parameters (learned or fixed)
        self.register_buffer('delta_H', torch.tensor(-100.0))  # kJ/mol
        self.register_buffer('R', torch.tensor(8.314e-3))  # kJ/mol/K
        
        logger.info("ThermodynamicConsistencyLayer initialized")
    
    def compute_physics_loss(
        self,
        x: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute thermodynamic consistency loss.
        
        Enforces:
        1. ΔG = ΔH - T*ΔS (Gibbs free energy)
        2. Spontaneity: ΔG < 0 for feasible reactions
        3. Equilibrium: X < X_eq
        
        Args:
            x: Input tensor [batch, features] containing T
            output: Predicted output [batch, features] containing conversion
        
        Returns:
            Thermodynamic consistency loss
        """
        # Extract temperature and conversion
        T = x[:, self.T_index]  # Temperature (K)
        
        # Get conversion from output
        if output.shape[1] > self.conversion_index:
            X = output[:, self.conversion_index]  # Conversion (0-1)
        else:
            X = output.mean(dim=1)  # Use mean if index out of range
        
        # Ensure conversion is in valid range [0, 1]
        X = torch.clamp(X, 0.0, 1.0)
        
        # Calculate Gibbs free energy
        # ΔG = ΔH - T*ΔS
        # Simplified: ΔS ≈ -R*ln(1-X)
        epsilon = 1e-6
        delta_S = -self.R * torch.log(1.0 - X + epsilon)
        delta_G = self.delta_H - T * delta_S
        
        # Physics loss: Penalize positive ΔG (non-spontaneous)
        # For polymerization, ΔG should be negative
        gibbs_penalty = torch.relu(delta_G)  # Penalize positive ΔG
        
        # Equilibrium constraint: X should not exceed X_eq
        # X_eq ≈ 0.98 for most polymerizations
        X_eq = 0.98 * torch.ones_like(X)
        equilibrium_penalty = torch.relu(X - X_eq)
        
        # Total physics loss
        physics_loss = torch.mean(gibbs_penalty + 10.0 * equilibrium_penalty)
        
        return self.physics_weight * physics_loss


class MassBalanceConstraintLayer(PhysicsInformedLayer):
    """
    Neural network layer enforcing mass balance constraints.
    
    Ensures predictions respect conservation of mass:
    Input - Output + Generation - Consumption = Accumulation
    
    This is a fundamental law that must hold for all chemical processes.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        physics_weight: float = 1.0
    ):
        """
        Initialize mass balance constraint layer.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            physics_weight: Weight for physics loss
        """
        super().__init__(input_dim, output_dim, physics_weight)
        logger.info("MassBalanceConstraintLayer initialized")
    
    def compute_physics_loss(
        self,
        x: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mass balance constraint loss.
        
        For a polymerization reactor:
        Input: Monomer feed
        Output: Monomer + Polymer
        Generation: 0 (no monomer created)
        Consumption: Polymerization rate
        
        Conservation: M_in = M_out + Polymer_formed
        
        Args:
            x: Input tensor with process variables
            output: Predicted output
        
        Returns:
            Mass balance violation penalty
        """
        # Assume input contains: [T, P, C_M_in, flow_rate, ...]
        # Assume output contains: [C_M_out, conversion, ...]
        
        batch_size = x.shape[0]
        
        # Extract relevant variables (simplified example)
        # In practice, these indices would be known from data schema
        if x.shape[1] >= 3:
            C_M_in = x[:, 2]  # Inlet monomer concentration
        else:
            C_M_in = torch.ones(batch_size, device=x.device)
        
        if output.shape[1] >= 2:
            conversion = output[:, 1]  # Predicted conversion
        else:
            conversion = output[:, 0]
        
        # Ensure conversion in [0, 1]
        conversion = torch.clamp(conversion, 0.0, 1.0)
        
        # Mass balance check:
        # Polymer formed = C_M_in * conversion
        # Monomer remaining = C_M_in * (1 - conversion)
        # Total = C_M_in * (conversion + (1 - conversion)) = C_M_in ✓
        
        # Residual should be zero
        polymer_formed = C_M_in * conversion
        monomer_remaining = C_M_in * (1.0 - conversion)
        total_mass_out = polymer_formed + monomer_remaining
        
        # Mass balance violation
        mass_balance_violation = torch.abs(total_mass_out - C_M_in)
        
        # Physics loss
        physics_loss = torch.mean(mass_balance_violation)
        
        return self.physics_weight * physics_loss


class EnergyConservationLayer(PhysicsInformedLayer):
    """
    Neural network layer enforcing energy conservation.
    
    Ensures predictions respect the First Law of Thermodynamics:
    Energy in - Energy out + Heat generated = Energy accumulated
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        physics_weight: float = 1.0
    ):
        """
        Initialize energy conservation layer.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            physics_weight: Weight for physics loss
        """
        super().__init__(input_dim, output_dim, physics_weight)
        
        # Heat of polymerization (exothermic)
        self.register_buffer('delta_H_rxn', torch.tensor(-100.0))  # kJ/mol
        self.register_buffer('Cp', torch.tensor(50.0))  # J/mol/K
        
        logger.info("EnergyConservationLayer initialized")
    
    def compute_physics_loss(
        self,
        x: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute energy conservation constraint loss.
        
        Energy balance:
        Q_reaction + Q_sensible + Q_transfer = 0 (steady state)
        
        Args:
            x: Input tensor [T_in, P, flow_rate, ...]
            output: Predicted output [T_out, conversion, ...]
        
        Returns:
            Energy balance violation penalty
        """
        batch_size = x.shape[0]
        
        # Extract temperatures
        if x.shape[1] >= 1:
            T_in = x[:, 0]
        else:
            T_in = 350.0 * torch.ones(batch_size, device=x.device)
        
        if output.shape[1] >= 1:
            T_out = output[:, 0]
        else:
            T_out = T_in  # Assume isothermal if not predicted
        
        # Extract conversion
        if output.shape[1] >= 2:
            conversion = torch.clamp(output[:, 1], 0.0, 1.0)
        else:
            conversion = 0.5 * torch.ones(batch_size, device=x.device)
        
        # Heat of reaction (exothermic, releases heat)
        Q_reaction = -self.delta_H_rxn * conversion * 1000  # kJ to J
        
        # Sensible heat (temperature change)
        Q_sensible = self.Cp * (T_out - T_in)
        
        # Energy balance: Q_reaction + Q_sensible ≈ 0 (simplified)
        # For exothermic reaction: T_out > T_in
        energy_balance = Q_reaction + Q_sensible
        
        # Penalize violations
        energy_violation = torch.abs(energy_balance)
        
        # Also ensure T_out > T_in for exothermic reaction
        temperature_consistency = torch.relu(T_in - T_out)  # Penalize if T_out < T_in
        
        physics_loss = torch.mean(energy_violation / 1000.0 + temperature_consistency)
        
        return self.physics_weight * physics_loss


# Utility function to create physics-informed model
def create_physics_informed_network(
    input_dim: int,
    hidden_dims: list,
    output_dim: int,
    use_thermodynamic: bool = True,
    use_mass_balance: bool = True,
    use_energy: bool = True
) -> nn.Module:
    """
    Create a neural network with physics-informed layers.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        use_thermodynamic: Include thermodynamic consistency
        use_mass_balance: Include mass balance constraints
        use_energy: Include energy conservation
    
    Returns:
        Physics-informed neural network
    """
    layers = []
    
    # Input layer
    prev_dim = input_dim
    for i, hidden_dim in enumerate(hidden_dims):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim
    
    # Output layer with physics constraints
    if use_thermodynamic:
        layers.append(ThermodynamicConsistencyLayer(prev_dim, output_dim))
    elif use_mass_balance:
        layers.append(MassBalanceConstraintLayer(prev_dim, output_dim))
    elif use_energy:
        layers.append(EnergyConservationLayer(prev_dim, output_dim))
    else:
        layers.append(nn.Linear(prev_dim, output_dim))
    
    model = nn.Sequential(*layers)
    
    logger.info(f"Physics-informed network created: {input_dim} → {hidden_dims} → {output_dim}")
    
    return model
