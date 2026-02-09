"""
Inductive Bias Layers for Neural Networks

These custom neural network layers enforce physical and chemical laws,
demonstrating how "inductive bias" ensures predictions respect fundamental
principles of chemistry and physics.

Key Innovation: Unlike standard neural networks that can violate physical laws,
these layers constrain predictions to be physically consistent.
"""

from .physics_informed_layers import (
    PhysicsInformedLayer,
    ThermodynamicConsistencyLayer,
    MassBalanceConstraintLayer
)
from .monotonic_layers import MonotonicityConstraint
from .conservation_layers import EnergyConservationLayer, MomentumConservationLayer

__all__ = [
    "PhysicsInformedLayer",
    "ThermodynamicConsistencyLayer",
    "MassBalanceConstraintLayer",
    "MonotonicityConstraint",
    "EnergyConservationLayer",
    "MomentumConservationLayer",
]
