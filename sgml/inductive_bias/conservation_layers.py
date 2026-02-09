"""
Conservation Law Layers

Re-exports energy conservation from physics_informed_layers
"""

from .physics_informed_layers import EnergyConservationLayer

# Placeholder for momentum conservation
class MomentumConservationLayer(EnergyConservationLayer):
    """Momentum conservation layer (simplified, inherits from energy)."""
    pass

__all__ = ["EnergyConservationLayer", "MomentumConservationLayer"]
