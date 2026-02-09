"""
Science-Guided Machine Learning (SGML) Framework

A novel framework that merges first-principle modeling with machine learning
to optimize polyolefin processes.

Key Components:
- First-Principle Models: Polymerization kinetics, thermodynamics, reactor models
- Inductive Bias Layers: Physics-informed neural networks ensuring law compliance
- SGML Framework: Integration of physics and ML for enhanced predictions
- Process Optimization: Multi-objective optimization with physical constraints

This framework demonstrates how inductive bias in neural networks can ensure
predictions respect fundamental laws of chemistry and physics.
"""

from .core import SGMLFramework, SGMLConfig
from .physics import (
    PolymerizationKinetics,
    ThermodynamicConstraints,
    MassBalanceModel,
    ReactorModel
)
from .inductive_bias import (
    PhysicsInformedLayer,
    ThermodynamicConsistencyLayer,
    MassBalanceConstraintLayer,
    MonotonicityConstraint
)
from .optimization import SGMLOptimizer

__version__ = "2.0.0"

__all__ = [
    "SGMLFramework",
    "SGMLConfig",
    "PolymerizationKinetics",
    "ThermodynamicConstraints",
    "MassBalanceModel",
    "ReactorModel",
    "PhysicsInformedLayer",
    "ThermodynamicConsistencyLayer",
    "MassBalanceConstraintLayer",
    "MonotonicityConstraint",
    "SGMLOptimizer",
]
