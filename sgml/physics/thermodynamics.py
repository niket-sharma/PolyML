"""
Thermodynamic Models for Polymer Systems

First-principle thermodynamic models including:
- Flory-Huggins theory for polymer solutions
- Energy balance equations
- Phase equilibria
- Enthalpy and entropy calculations

These models ensure predictions respect thermodynamic laws.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ThermodynamicParameters:
    """Thermodynamic parameters for polymer systems."""
    
    # Flory-Huggins interaction parameter
    chi: float = 0.5
    
    # Heat capacities (J/mol/K)
    Cp_polymer: float = 50.0
    Cp_monomer: float = 35.0
    Cp_solvent: float = 75.0
    
    # Enthalpies (kJ/mol)
    delta_H_polymerization: float = -100.0  # Exothermic
    delta_H_mixing: float = 5.0
    
    # Universal gas constant
    R: float = 8.314  # J/mol/K


class ThermodynamicConstraints:
    """
    Thermodynamic constraint models for polymer processes.
    
    Implements fundamental thermodynamic laws:
    - First law: Energy conservation
    - Second law: Entropy increase
    - Gibbs free energy minimization
    - Chemical potential equilibrium
    """
    
    def __init__(self, parameters: Optional[ThermodynamicParameters] = None):
        """
        Initialize thermodynamic constraints.
        
        Args:
            parameters: Thermodynamic parameters
        """
        self.params = parameters or ThermodynamicParameters()
        logger.info("Thermodynamic constraints initialized")
    
    def energy_balance(
        self,
        T_in: float,
        T_out: float,
        flow_rate: float,
        reaction_rate: float,
        heat_transfer: float = 0.0
    ) -> float:
        """
        Calculate energy balance for polymerization reactor.
        
        First Law of Thermodynamics:
        Q - W = ΔH = m*Cp*ΔT + ΔH_rxn
        
        Args:
            T_in: Inlet temperature (K)
            T_out: Outlet temperature (K)
            flow_rate: Mass flow rate (kg/s)
            reaction_rate: Reaction rate (mol/s)
            heat_transfer: External heat transfer (W)
        
        Returns:
            Energy balance residual (should be ~0 for consistency)
        """
        # Sensible heat change
        delta_T = T_out - T_in
        sensible_heat = flow_rate * self.params.Cp_polymer * delta_T
        
        # Reaction heat
        reaction_heat = reaction_rate * self.params.delta_H_polymerization * 1000  # kJ to J
        
        # Energy balance
        energy_in = sensible_heat + heat_transfer
        energy_out = reaction_heat
        
        residual = abs(energy_in - energy_out)
        return residual
    
    def gibbs_free_energy(
        self,
        T: float,
        conversion: float
    ) -> float:
        """
        Calculate Gibbs free energy change.
        
        ΔG = ΔH - T*ΔS
        
        Negative ΔG indicates spontaneous polymerization.
        
        Args:
            T: Temperature (K)
            conversion: Monomer conversion (0-1)
        
        Returns:
            Gibbs free energy change (kJ/mol)
        """
        # Enthalpy change (exothermic for polymerization)
        delta_H = self.params.delta_H_polymerization * conversion
        
        # Entropy change (decrease due to ordering)
        # Simplified: ΔS ≈ -R * ln(1 - conversion)
        delta_S = -self.params.R * np.log(max(1 - conversion, 0.01)) / 1000  # J to kJ
        
        # Gibbs free energy
        delta_G = delta_H - T * delta_S
        
        return delta_G
    
    def check_thermodynamic_feasibility(
        self,
        T: float,
        P: float,
        conversion: float
    ) -> Tuple[bool, str]:
        """
        Check if operating conditions are thermodynamically feasible.
        
        Args:
            T: Temperature (K)
            P: Pressure (bar)
            conversion: Target conversion (0-1)
        
        Returns:
            Tuple of (is_feasible, reason)
        """
        # Check Gibbs free energy
        delta_G = self.gibbs_free_energy(T, conversion)
        
        if delta_G > 0:
            return False, f"Positive ΔG = {delta_G:.2f} kJ/mol, reaction not spontaneous"
        
        # Check temperature range (typical for polyolefins: 50-150°C)
        if T < 323 or T > 423:
            return False, f"Temperature {T:.1f} K outside typical range (323-423 K)"
        
        # Check pressure (typical: 1-50 bar for gas-phase)
        if P < 1 or P > 100:
            return False, f"Pressure {P:.1f} bar outside reasonable range (1-100 bar)"
        
        return True, "Thermodynamically feasible"
    
    def equilibrium_conversion(self, T: float, P: float) -> float:
        """
        Calculate thermodynamic equilibrium conversion.
        
        At equilibrium: ΔG = 0
        
        Args:
            T: Temperature (K)
            P: Pressure (bar)
        
        Returns:
            Equilibrium conversion (0-1)
        """
        # For polymerization, typically goes to near-completion
        # Simplified model based on Gibbs free energy
        
        # At equilibrium: ΔH = T*ΔS
        # Solve for conversion
        
        # For most polymerizations, X_eq ≈ 0.95-0.99
        # Temperature effect: higher T → slightly lower X_eq
        
        X_eq = 0.98 - 0.0005 * (T - 350.0)
        
        # Pressure effect (higher P → higher X_eq for gas phase)
        X_eq += 0.001 * (P - 1.0)
        
        return np.clip(X_eq, 0.0, 1.0)


class FloryHugginsModel:
    """
    Flory-Huggins theory for polymer solution thermodynamics.
    
    Describes mixing behavior of polymers and solvents, accounting for:
    - Combinatorial entropy (size difference)
    - Enthalpic interactions (chi parameter)
    - Chemical potential
    - Phase separation
    """
    
    def __init__(self, chi: float = 0.5):
        """
        Initialize Flory-Huggins model.
        
        Args:
            chi: Flory-Huggins interaction parameter
        """
        self.chi = chi
        logger.info(f"Flory-Huggins model initialized (χ = {chi})")
    
    def mixing_free_energy(
        self,
        phi_polymer: float,
        N: float = 1000.0
    ) -> float:
        """
        Calculate Gibbs free energy of mixing per site.
        
        ΔG_mix/RT = (1-φ)*ln(1-φ)/N + φ*ln(φ) + χ*φ*(1-φ)
        
        This is the classic Flory-Huggins equation.
        
        Args:
            phi_polymer: Polymer volume fraction (0-1)
            N: Degree of polymerization (chain length)
        
        Returns:
            ΔG_mix/(RT) - dimensionless free energy
        """
        phi_s = 1.0 - phi_polymer  # Solvent fraction
        
        # Avoid log(0)
        phi_p_safe = np.clip(phi_polymer, 1e-10, 1.0 - 1e-10)
        phi_s_safe = np.clip(phi_s, 1e-10, 1.0 - 1e-10)
        
        # Combinatorial entropy term
        entropy_term = (phi_s_safe * np.log(phi_s_safe) / N + 
                        phi_p_safe * np.log(phi_p_safe))
        
        # Enthalpic interaction term
        enthalpy_term = self.chi * phi_polymer * phi_s
        
        delta_G_mix = entropy_term + enthalpy_term
        
        return delta_G_mix
    
    def chemical_potential_polymer(
        self,
        phi_polymer: float,
        N: float = 1000.0,
        T: float = 350.0
    ) -> float:
        """
        Calculate polymer chemical potential.
        
        μ_polymer/RT = ln(φ) + (1-1/N)*(1-φ) + χ*(1-φ)^2
        
        Args:
            phi_polymer: Polymer volume fraction
            N: Degree of polymerization
            T: Temperature (K)
        
        Returns:
            Chemical potential (J/mol)
        """
        phi_s = 1.0 - phi_polymer
        phi_p_safe = np.clip(phi_polymer, 1e-10, 1.0 - 1e-10)
        
        mu_polymer_RT = (np.log(phi_p_safe) + 
                         (1.0 - 1.0/N) * phi_s + 
                         self.chi * (phi_s ** 2))
        
        R = 8.314  # J/mol/K
        mu_polymer = mu_polymer_RT * R * T
        
        return mu_polymer
    
    def spinodal_curve(self, N: float = 1000.0) -> np.ndarray:
        """
        Calculate spinodal decomposition curve.
        
        Spinodal condition: d²ΔG/dφ² = 0
        
        Args:
            N: Degree of polymerization
        
        Returns:
            Array of polymer fractions at spinodal
        """
        # Spinodal condition: 1/φ + 1/((1-φ)*N) - 2*χ = 0
        # For symmetric case, critical χ = (1 + 1/√N)² / 2
        
        chi_critical = 0.5 * (1.0 + 1.0/np.sqrt(N)) ** 2
        
        if self.chi < chi_critical:
            # Homogeneous solution, no phase separation
            return np.array([])
        
        # Calculate spinodal curve (simplified)
        phi_range = np.linspace(0.01, 0.99, 100)
        spinodal_points = []
        
        for phi in phi_range:
            # Second derivative of ΔG
            d2G = 1.0/phi + 1.0/((1.0-phi)*N) - 2.0*self.chi
            
            if abs(d2G) < 0.1:  # Near spinodal
                spinodal_points.append(phi)
        
        return np.array(spinodal_points)
    
    def predict_phase_behavior(
        self,
        phi_polymer: float,
        N: float = 1000.0
    ) -> Dict[str, any]:
        """
        Predict phase behavior (single phase vs. phase separation).
        
        Args:
            phi_polymer: Polymer volume fraction
            N: Degree of polymerization
        
        Returns:
            Dictionary with phase behavior info
        """
        # Critical chi for phase separation
        chi_critical = 0.5 * (1.0 + 1.0/np.sqrt(N)) ** 2
        
        # Free energy of mixing
        delta_G = self.mixing_free_energy(phi_polymer, N)
        
        # Chemical potential
        mu = self.chemical_potential_polymer(phi_polymer, N)
        
        # Determine phase state
        if self.chi < chi_critical:
            phase_state = "single_phase"
            stable = True
        else:
            spinodal = self.spinodal_curve(N)
            if len(spinodal) > 0:
                if spinodal.min() < phi_polymer < spinodal.max():
                    phase_state = "two_phase_unstable"
                    stable = False
                else:
                    phase_state = "two_phase_metastable"
                    stable = True
            else:
                phase_state = "single_phase"
                stable = True
        
        return {
            'phase_state': phase_state,
            'stable': stable,
            'chi': self.chi,
            'chi_critical': chi_critical,
            'delta_G_mix': delta_G,
            'chemical_potential': mu,
            'polymer_fraction': phi_polymer
        }


# Convenience functions
def check_energy_balance(T_in: float, T_out: float, flow_rate: float, reaction_rate: float) -> float:
    """
    Quick energy balance check.
    
    Args:
        T_in: Inlet temperature (K)
        T_out: Outlet temperature (K)
        flow_rate: Flow rate (kg/s)
        reaction_rate: Reaction rate (mol/s)
    
    Returns:
        Energy balance residual
    """
    model = ThermodynamicConstraints()
    return model.energy_balance(T_in, T_out, flow_rate, reaction_rate)


def calculate_gibbs_energy(T: float, conversion: float) -> float:
    """
    Quick Gibbs free energy calculation.
    
    Args:
        T: Temperature (K)
        conversion: Conversion (0-1)
    
    Returns:
        ΔG (kJ/mol)
    """
    model = ThermodynamicConstraints()
    return model.gibbs_free_energy(T, conversion)
