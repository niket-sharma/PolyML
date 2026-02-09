"""
Mass Balance Models for Polymerization Reactors

First-principle mass balance equations based on conservation of mass.
These are fundamental laws that must be satisfied in any chemical process.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MassBalanceModel:
    """
    Mass balance model for polymerization systems.
    
    Fundamental law: Mass in - Mass out + Generation - Consumption = Accumulation
    
    This is a core principle of chemical engineering that must be respected
    by any predictive model.
    """
    
    def __init__(self):
        """Initialize mass balance model."""
        logger.info("Mass balance model initialized")
    
    def component_balance(
        self,
        F_in: float,
        F_out: float,
        generation_rate: float,
        consumption_rate: float,
        accumulation_rate: float
    ) -> float:
        """
        Generic component mass balance equation.
        
        F_in - F_out + Gen - Cons = Acc
        
        Args:
            F_in: Inlet flow rate (mol/s or kg/s)
            F_out: Outlet flow rate
            generation_rate: Generation rate
            consumption_rate: Consumption rate
            accumulation_rate: Accumulation rate
        
        Returns:
            Mass balance residual (should be ~0)
        """
        balance = F_in - F_out + generation_rate - consumption_rate - accumulation_rate
        return abs(balance)
    
    def monomer_balance(
        self,
        F_M_in: float,
        F_M_out: float,
        R_p: float,
        V: float,
        dC_M_dt: float
    ) -> float:
        """
        Monomer mass balance.
        
        For continuous reactor:
        F_M_in - F_M_out - R_p*V = V * dC_M/dt
        
        Args:
            F_M_in: Monomer inlet molar flow (mol/s)
            F_M_out: Monomer outlet molar flow (mol/s)
            R_p: Polymerization rate (mol/L/s)
            V: Reactor volume (L)
            dC_M_dt: Monomer concentration change rate (mol/L/s)
        
        Returns:
            Balance residual
        """
        consumption = R_p * V
        accumulation = V * dC_M_dt
        
        balance = F_M_in - F_M_out - consumption - accumulation
        return abs(balance)
    
    def polymer_balance(
        self,
        F_P_in: float,
        F_P_out: float,
        R_p: float,
        V: float,
        dC_P_dt: float
    ) -> float:
        """
        Polymer mass balance.
        
        F_P_in - F_P_out + R_p*V = V * dC_P/dt
        
        Args:
            F_P_in: Polymer inlet flow (mol/s)
            F_P_out: Polymer outlet flow (mol/s)
            R_p: Polymerization rate (mol/L/s)
            V: Reactor volume (L)
            dC_P_dt: Polymer concentration change rate (mol/L/s)
        
        Returns:
            Balance residual
        """
        generation = R_p * V
        accumulation = V * dC_P_dt
        
        balance = F_P_in - F_P_out + generation - accumulation
        return abs(balance)
    
    def total_mass_balance(
        self,
        mass_in: float,
        mass_out: float,
        accumulation: float
    ) -> float:
        """
        Overall mass balance (conservation of mass).
        
        Σ(mass_in) - Σ(mass_out) = Accumulation
        
        Args:
            mass_in: Total mass in (kg/s)
            mass_out: Total mass out (kg/s)
            accumulation: Rate of mass accumulation (kg/s)
        
        Returns:
            Balance residual
        """
        balance = mass_in - mass_out - accumulation
        return abs(balance)


class CSTRModel(MassBalanceModel):
    """
    Continuous Stirred Tank Reactor (CSTR) model.
    
    Implements mass and energy balances for a well-mixed reactor.
    Fundamental model in chemical engineering.
    """
    
    def __init__(self, volume: float = 100.0):
        """
        Initialize CSTR model.
        
        Args:
            volume: Reactor volume (L)
        """
        super().__init__()
        self.volume = volume
        logger.info(f"CSTR model initialized (V = {volume} L)")
    
    def steady_state_conversion(
        self,
        F_M_in: float,
        k_p: float,
        C_P_avg: float,
        tau: Optional[float] = None
    ) -> float:
        """
        Calculate steady-state monomer conversion in CSTR.
        
        At steady state:
        X = k_p * C_P * τ / (1 + k_p * C_P * τ)
        
        where τ = V/F is residence time
        
        Args:
            F_M_in: Monomer inlet flow rate (L/s)
            k_p: Propagation rate constant (L/mol/s)
            C_P_avg: Average active site concentration (mol/L)
            tau: Residence time (s), calculated if None
        
        Returns:
            Steady-state conversion (0-1)
        """
        if tau is None:
            tau = self.volume / F_M_in  # Residence time
        
        # Damköhler number (dimensionless reaction rate)
        Da = k_p * C_P_avg * tau
        
        # Conversion for first-order kinetics in CSTR
        conversion = Da / (1.0 + Da)
        
        return np.clip(conversion, 0.0, 1.0)
    
    def required_volume(
        self,
        F_M_in: float,
        k_p: float,
        C_P_avg: float,
        target_conversion: float
    ) -> float:
        """
        Calculate required reactor volume for target conversion.
        
        V = F * τ = F * X / (k_p * C_P * (1 - X))
        
        Args:
            F_M_in: Monomer inlet flow rate (L/s)
            k_p: Propagation rate constant
            C_P_avg: Active site concentration
            target_conversion: Desired conversion (0-1)
        
        Returns:
            Required volume (L)
        """
        X = target_conversion
        
        # Avoid division by zero
        if X >= 0.99:
            X = 0.99
        
        tau_required = X / (k_p * C_P_avg * (1.0 - X))
        V_required = F_M_in * tau_required
        
        return V_required
    
    def simulate_dynamics(
        self,
        F_M_in: float,
        C_M_in: float,
        k_p: float,
        C_P: float,
        time_hours: float,
        n_steps: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Simulate CSTR dynamics.
        
        dC_M/dt = (F/V) * (C_M_in - C_M) - k_p * C_P * C_M
        
        Args:
            F_M_in: Volumetric flow rate (L/s)
            C_M_in: Inlet monomer concentration (mol/L)
            k_p: Propagation rate constant
            C_P: Active site concentration (mol/L)
            time_hours: Simulation time (hours)
            n_steps: Number of time steps
        
        Returns:
            Dictionary with time series
        """
        dt = (time_hours * 3600) / n_steps
        
        # Initialize
        time = np.linspace(0, time_hours, n_steps)
        C_M = np.zeros(n_steps)
        conversion = np.zeros(n_steps)
        
        # Initial condition (empty reactor)
        C_M[0] = 0.0
        
        # Euler integration
        for i in range(1, n_steps):
            # CSTR balance
            flow_term = (F_M_in / self.volume) * (C_M_in - C_M[i-1])
            reaction_term = k_p * C_P * C_M[i-1]
            
            dC_M_dt = flow_term - reaction_term
            C_M[i] = C_M[i-1] + dC_M_dt * dt
            
            # Ensure non-negative
            C_M[i] = max(C_M[i], 0.0)
            
            # Conversion
            conversion[i] = (C_M_in - C_M[i]) / C_M_in * 100 if C_M_in > 0 else 0
        
        return {
            'time': time,
            'monomer_concentration': C_M,
            'conversion': conversion,
            'residence_time': self.volume / F_M_in
        }


class ReactorModel:
    """Base class for reactor models (for inheritance)."""
    pass


class PlugFlowReactor(ReactorModel):
    """
    Plug Flow Reactor (PFR) model.
    
    Ideal tubular reactor with no axial mixing.
    """
    
    def __init__(self, length: float = 10.0, diameter: float = 0.5):
        """
        Initialize PFR model.
        
        Args:
            length: Reactor length (m)
            diameter: Reactor diameter (m)
        """
        self.length = length
        self.diameter = diameter
        self.volume = np.pi * (diameter/2)**2 * length * 1000  # m³ to L
        logger.info(f"PFR model initialized (L={length}m, D={diameter}m, V={self.volume:.1f}L)")
    
    def steady_state_profile(
        self,
        C_M_in: float,
        k_p: float,
        C_P: float,
        n_points: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Calculate steady-state concentration profile along reactor.
        
        For PFR: dC_M/dz = -(k_p * C_P * C_M) / u
        
        where u is velocity and z is axial position
        
        Args:
            C_M_in: Inlet monomer concentration (mol/L)
            k_p: Propagation rate constant
            C_P: Active site concentration (mol/L)
            n_points: Number of spatial points
        
        Returns:
            Dictionary with profiles
        """
        # Axial positions
        z = np.linspace(0, self.length, n_points)
        dz = self.length / n_points
        
        # Assume constant velocity
        u = 0.1  # m/s (typical)
        
        # Initialize
        C_M = np.zeros(n_points)
        C_M[0] = C_M_in
        
        # Integrate along reactor
        for i in range(1, n_points):
            dC_M_dz = -(k_p * C_P * C_M[i-1]) / u
            C_M[i] = C_M[i-1] + dC_M_dz * dz
            C_M[i] = max(C_M[i], 0.0)
        
        # Conversion
        conversion = (C_M_in - C_M) / C_M_in * 100
        
        return {
            'position': z,
            'monomer_concentration': C_M,
            'conversion': conversion,
            'outlet_conversion': conversion[-1]
        }
