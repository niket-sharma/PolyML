"""
Polymerization Kinetics Models

First-principle models for polyolefin polymerization kinetics including:
- Ziegler-Natta catalyst kinetics
- Chain propagation, termination, and transfer
- Molecular weight distribution calculations
- Reaction rate equations

These models capture fundamental chemical laws that govern polymerization.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class KineticParameters:
    """Kinetic parameters for polymerization reactions."""
    
    # Activation energies (kJ/mol)
    Ea_propagation: float = 40.0
    Ea_termination: float = 10.0
    Ea_initiation: float = 50.0
    Ea_chain_transfer: float = 30.0
    
    # Pre-exponential factors (various units)
    A_propagation: float = 1.0e7
    A_termination: float = 1.0e8
    A_initiation: float = 1.0e6
    A_chain_transfer: float = 1.0e5
    
    # Universal gas constant (kJ/mol/K)
    R: float = 8.314e-3


class PolymerizationKinetics:
    """
    Base class for polymerization kinetics models.
    
    Implements fundamental chemical kinetics for polymer chain growth:
    - Initiation: Catalyst + Monomer → Active site
    - Propagation: Active site + Monomer → Longer chain
    - Termination: Active site → Dead polymer
    - Chain transfer: Active site + Transfer agent → Dead polymer + New active site
    
    These are first-principle models based on chemical reaction engineering.
    """
    
    def __init__(self, parameters: Optional[KineticParameters] = None):
        """
        Initialize polymerization kinetics model.
        
        Args:
            parameters: Kinetic parameters, uses defaults if None
        """
        self.params = parameters or KineticParameters()
        logger.info("Polymerization kinetics model initialized")
    
    def rate_constant(self, A: float, Ea: float, T: float) -> float:
        """
        Calculate reaction rate constant using Arrhenius equation.
        
        k = A * exp(-Ea / RT)
        
        This is a fundamental law of chemical kinetics.
        
        Args:
            A: Pre-exponential factor
            Ea: Activation energy (kJ/mol)
            T: Temperature (K)
        
        Returns:
            Rate constant k
        """
        return A * np.exp(-Ea / (self.params.R * T))
    
    def propagation_rate(self, T: float, C_M: float, C_P: float) -> float:
        """
        Calculate propagation rate: Active site + Monomer → Longer chain.
        
        Rate = k_p * [M] * [P*]
        
        Args:
            T: Temperature (K)
            C_M: Monomer concentration (mol/L)
            C_P: Active site concentration (mol/L)
        
        Returns:
            Propagation rate (mol/L/s)
        """
        k_p = self.rate_constant(
            self.params.A_propagation,
            self.params.Ea_propagation,
            T
        )
        return k_p * C_M * C_P
    
    def termination_rate(self, T: float, C_P: float) -> float:
        """
        Calculate termination rate: Active site → Dead polymer.
        
        Rate = k_t * [P*]^2  (bimolecular termination)
        
        Args:
            T: Temperature (K)
            C_P: Active site concentration (mol/L)
        
        Returns:
            Termination rate (mol/L/s)
        """
        k_t = self.rate_constant(
            self.params.A_termination,
            self.params.Ea_termination,
            T
        )
        return k_t * (C_P ** 2)
    
    def initiation_rate(self, T: float, C_cat: float, C_M: float) -> float:
        """
        Calculate initiation rate: Catalyst + Monomer → Active site.
        
        Rate = k_i * [Cat] * [M]
        
        Args:
            T: Temperature (K)
            C_cat: Catalyst concentration (mol/L)
            C_M: Monomer concentration (mol/L)
        
        Returns:
            Initiation rate (mol/L/s)
        """
        k_i = self.rate_constant(
            self.params.A_initiation,
            self.params.Ea_initiation,
            T
        )
        return k_i * C_cat * C_M
    
    def chain_transfer_rate(self, T: float, C_P: float, C_A: float) -> float:
        """
        Calculate chain transfer rate.
        
        Rate = k_tr * [P*] * [A]
        
        Args:
            T: Temperature (K)
            C_P: Active site concentration (mol/L)
            C_A: Chain transfer agent concentration (mol/L)
        
        Returns:
            Chain transfer rate (mol/L/s)
        """
        k_tr = self.rate_constant(
            self.params.A_chain_transfer,
            self.params.Ea_chain_transfer,
            T
        )
        return k_tr * C_P * C_A
    
    def calculate_molecular_weight(
        self,
        T: float,
        C_M: float,
        C_P: float,
        C_A: float = 0.0
    ) -> Tuple[float, float]:
        """
        Calculate number-average and weight-average molecular weight.
        
        Based on kinetic chain length and moment analysis.
        
        Args:
            T: Temperature (K)
            C_M: Monomer concentration (mol/L)
            C_P: Active site concentration (mol/L)
            C_A: Chain transfer agent concentration (mol/L)
        
        Returns:
            Tuple of (Mn, Mw) - number and weight average MW (g/mol)
        """
        # Monomer molecular weight (ethylene = 28 g/mol for polyethylene)
        M_monomer = 28.0
        
        # Calculate rates
        R_p = self.propagation_rate(T, C_M, C_P)
        R_t = self.termination_rate(T, C_P)
        R_tr = self.chain_transfer_rate(T, C_P, C_A) if C_A > 0 else 0.0
        
        # Kinetic chain length (average number of monomer units per chain)
        # ν = R_p / (R_t + R_tr)
        denominator = R_t + R_tr
        if denominator < 1e-10:
            # Avoid division by zero
            kinetic_chain_length = 1000.0
        else:
            kinetic_chain_length = R_p / denominator
        
        # Number-average molecular weight
        Mn = kinetic_chain_length * M_monomer
        
        # Weight-average molecular weight (simplified)
        # For living polymerization: Mw/Mn ≈ 1 + 1/ν
        # For free radical: Mw/Mn ≈ 1.5 to 2.0
        polydispersity = 1.0 + 1.0 / max(kinetic_chain_length, 1.0)
        Mw = Mn * polydispersity
        
        return Mn, Mw
    
    def simulate_batch_reactor(
        self,
        T: float,
        C_M_0: float,
        C_cat_0: float,
        time_hours: float,
        n_steps: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Simulate batch polymerization reactor using kinetic model.
        
        Solves ODEs:
        d[M]/dt = -R_p
        d[P*]/dt = R_i - R_t - R_tr
        
        Args:
            T: Temperature (K)
            C_M_0: Initial monomer concentration (mol/L)
            C_cat_0: Initial catalyst concentration (mol/L)
            time_hours: Simulation time (hours)
            n_steps: Number of time steps
        
        Returns:
            Dictionary with time series of concentrations
        """
        dt = (time_hours * 3600) / n_steps  # Convert to seconds
        
        # Initialize arrays
        time = np.linspace(0, time_hours, n_steps)
        C_M = np.zeros(n_steps)
        C_P = np.zeros(n_steps)
        conversion = np.zeros(n_steps)
        Mn_profile = np.zeros(n_steps)
        
        # Initial conditions
        C_M[0] = C_M_0
        C_P[0] = C_cat_0 * 0.1  # Assume 10% catalyst activation
        
        # Simple Euler integration
        for i in range(1, n_steps):
            # Calculate rates
            R_p = self.propagation_rate(T, C_M[i-1], C_P[i-1])
            R_t = self.termination_rate(T, C_P[i-1])
            R_i = self.initiation_rate(T, C_cat_0, C_M[i-1])
            
            # Update concentrations
            C_M[i] = C_M[i-1] - R_p * dt
            C_P[i] = C_P[i-1] + (R_i - R_t) * dt
            
            # Ensure non-negative
            C_M[i] = max(C_M[i], 0.0)
            C_P[i] = max(C_P[i], 0.0)
            
            # Calculate conversion
            conversion[i] = (C_M_0 - C_M[i]) / C_M_0 * 100
            
            # Calculate molecular weight
            Mn, _ = self.calculate_molecular_weight(T, C_M[i], C_P[i])
            Mn_profile[i] = Mn
        
        return {
            'time': time,
            'monomer_concentration': C_M,
            'active_sites': C_P,
            'conversion': conversion,
            'number_average_MW': Mn_profile
        }


class ZieglerNattaKinetics(PolymerizationKinetics):
    """
    Specialized Ziegler-Natta catalyst kinetics for polyolefin production.
    
    Extends base polymerization kinetics with specific behavior for
    Ziegler-Natta catalysts used in industrial polyethylene/polypropylene production.
    """
    
    def __init__(self):
        """Initialize Ziegler-Natta kinetics with specific parameters."""
        # Ziegler-Natta specific parameters
        params = KineticParameters(
            Ea_propagation=35.0,  # Lower activation energy
            Ea_termination=15.0,
            Ea_initiation=45.0,
            A_propagation=2.0e7,  # Higher pre-exponential factor
            A_termination=5.0e7,
            A_initiation=1.5e6
        )
        super().__init__(parameters=params)
        logger.info("Ziegler-Natta kinetics model initialized")
    
    def stereoregularity_factor(self, T: float) -> float:
        """
        Calculate stereoregularity (isotacticity/syndiotacticity) factor.
        
        Ziegler-Natta catalysts produce stereoregular polymers.
        This is a simplified model of chain tacticity.
        
        Args:
            T: Temperature (K)
        
        Returns:
            Stereoregularity factor (0-1)
        """
        # Higher temperature → lower stereoregularity
        T_ref = 350.0  # Reference temperature (K)
        return 0.95 * np.exp(-0.001 * (T - T_ref))
    
    def calculate_properties(
        self,
        T: float,
        C_M: float,
        C_P: float
    ) -> Dict[str, float]:
        """
        Calculate comprehensive polymer properties.
        
        Args:
            T: Temperature (K)
            C_M: Monomer concentration (mol/L)
            C_P: Active site concentration (mol/L)
        
        Returns:
            Dictionary of polymer properties
        """
        Mn, Mw = self.calculate_molecular_weight(T, C_M, C_P)
        stereo = self.stereoregularity_factor(T)
        
        return {
            'number_average_MW': Mn,
            'weight_average_MW': Mw,
            'polydispersity': Mw / Mn if Mn > 0 else 1.0,
            'stereoregularity': stereo,
            'isotacticity': stereo,  # For polypropylene
            'propagation_rate': self.propagation_rate(T, C_M, C_P)
        }


# Convenience functions for quick calculations
def calculate_polymerization_rate(T: float, C_M: float, C_P: float) -> float:
    """
    Quick calculation of polymerization rate.
    
    Args:
        T: Temperature (K)
        C_M: Monomer concentration (mol/L)
        C_P: Active site concentration (mol/L)
    
    Returns:
        Polymerization rate (mol/L/s)
    """
    model = PolymerizationKinetics()
    return model.propagation_rate(T, C_M, C_P)


def predict_molecular_weight(T: float, C_M: float, C_P: float) -> Tuple[float, float]:
    """
    Quick prediction of molecular weight distribution.
    
    Args:
        T: Temperature (K)
        C_M: Monomer concentration (mol/L)
        C_P: Active site concentration (mol/L)
    
    Returns:
        Tuple of (Mn, Mw)
    """
    model = PolymerizationKinetics()
    return model.calculate_molecular_weight(T, C_M, C_P)
