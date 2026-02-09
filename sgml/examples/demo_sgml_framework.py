#!/usr/bin/env python3
"""
Demo: SGML Framework for Polyolefin Process Optimization

Demonstrates:
1. First-principle physics predictions
2. ML with inductive bias
3. Hybrid predictions (physics + ML)
4. Process optimization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sgml import SGMLFramework, SGMLConfig
from sgml.physics import PolymerizationKinetics, ThermodynamicConstraints
from sgml.optimization import SGMLOptimizer


def demo_physics_models():
    """Demonstrate first-principle physics models."""
    print("\n" + "="*70)
    print("DEMO 1: First-Principle Physics Models")
    print("="*70)
    
    # Polymerization kinetics
    kinetics = PolymerizationKinetics()
    
    print("\n🔬 Polymerization Kinetics:")
    T = 350.0  # K
    C_M = 1.0  # mol/L
    C_P = 0.01  # mol/L
    
    R_p = kinetics.propagation_rate(T, C_M, C_P)
    Mn, Mw = kinetics.calculate_molecular_weight(T, C_M, C_P)
    
    print(f"   Temperature: {T} K")
    print(f"   Propagation Rate: {R_p:.4e} mol/L/s")
    print(f"   Number-average MW: {Mn:.0f} g/mol")
    print(f"   Weight-average MW: {Mw:.0f} g/mol")
    print(f"   Polydispersity: {Mw/Mn:.2f}")
    
    # Thermodynamics
    thermo = ThermodynamicConstraints()
    
    print("\n🌡️  Thermodynamics:")
    conversion = 0.85
    delta_G = thermo.gibbs_free_energy(T, conversion)
    X_eq = thermo.equilibrium_conversion(T, P=10.0)
    
    print(f"   Conversion: {conversion:.1%}")
    print(f"   ΔG: {delta_G:.2f} kJ/mol")
    print(f"   Equilibrium Conversion: {X_eq:.1%}")
    print(f"   Feasible: {'✅ Yes' if delta_G < 0 else '❌ No'}")


def demo_sgml_framework():
    """Demonstrate SGML framework with hybrid predictions."""
    print("\n" + "="*70)
    print("DEMO 2: SGML Framework (Physics + ML + Inductive Bias)")
    print("="*70)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    
    # Features: [T, P, C_M, C_cat, flow_rate]
    X = np.random.randn(n_samples, 5)
    X[:, 0] = X[:, 0] * 20 + 350  # T ~ N(350, 20)
    X[:, 1] = X[:, 1] * 5 + 10  # P ~ N(10, 5)
    X[:, 2] = np.abs(X[:, 2]) + 0.5  # C_M > 0
    
    # Target: conversion (simplified function of T and C_M)
    y = 0.8 + 0.001 * (X[:, 0] - 350) + 0.1 * X[:, 2] + 0.05 * np.random.randn(n_samples)
    y = np.clip(y, 0, 1)
    
    # Split data
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"\n📊 Dataset:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Validation samples: {X_val.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    
    # Create SGML model
    config = SGMLConfig(
        input_dim=5,
        output_dim=1,
        hidden_dims=[32, 16],
        use_thermodynamic_constraints=True,
        physics_loss_weight=0.5,
        num_epochs=50
    )
    
    sgml = SGMLFramework(config)
    
    print("\n🧠 Training SGML Model (Physics + ML)...")
    history = sgml.train(X_train, y_train, X_val, y_val)
    
    print(f"\n✅ Training Complete:")
    print(f"   Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final Data Loss: {history['data_loss'][-1]:.4f}")
    print(f"   Final Physics Loss: {history['physics_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"   Final Validation Loss: {history['val_loss'][-1]:.4f}")
    
    # Test hybrid prediction
    print("\n🔄 Hybrid Prediction (Physics + ML):")
    test_input_dict = {
        'temperature': 360.0,
        'pressure': 12.0,
        'monomer_concentration': 1.2,
        'catalyst_concentration': 0.05,
        'active_sites': 0.01
    }
    test_input_tensor = torch.FloatTensor([[360.0, 12.0, 1.2, 0.05, 0.1]])
    
    result = sgml.hybrid_prediction(test_input_dict, test_input_tensor)
    
    print(f"   Physics Prediction: {result['physics_prediction']['equilibrium_conversion']:.3f}")
    print(f"   ML Prediction: {result['ml_prediction']:.3f}")
    print(f"   Hybrid Prediction: {result['hybrid_prediction']:.3f}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Physics Loss: {result['physics_loss']:.4f}")


def demo_process_optimization():
    """Demonstrate process optimization using SGML."""
    print("\n" + "="*70)
    print("DEMO 3: Process Optimization")
    print("="*70)
    
    # Create simple SGML model
    config = SGMLConfig(input_dim=3, output_dim=1)
    sgml = SGMLFramework(config)
    
    # Create optimizer
    optimizer = SGMLOptimizer(sgml)
    
    print("\n🎯 Optimizing for maximum conversion...")
    
    optimal = optimizer.optimize_conditions(
        objective="maximize_conversion",
        bounds={
            'temperature': (323, 423),
            'pressure': (1, 50),
            'monomer_concentration': (0.1, 5.0)
        }
    )
    
    print(f"\n✅ Optimal Conditions Found:")
    print(f"   Temperature: {optimal['temperature']:.1f} K ({optimal['temperature']-273:.1f}°C)")
    print(f"   Pressure: {optimal['pressure']:.1f} bar")
    print(f"   Monomer Concentration: {optimal['monomer_concentration']:.2f} mol/L")
    print(f"   Predicted Conversion: {optimal['objective_value']:.1%}")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("🚀 SGML Framework Demonstration")
    print("Science-Guided Machine Learning for Polyolefin Optimization")
    print("="*70)
    
    try:
        demo_physics_models()
        demo_sgml_framework()
        demo_process_optimization()
        
        print("\n" + "="*70)
        print("✅ All demos completed successfully!")
        print("="*70)
        print("\nKey Takeaways:")
        print("  1. ✅ Physics models capture fundamental laws (kinetics, thermodynamics)")
        print("  2. ✅ Inductive bias ensures ML respects physical constraints")
        print("  3. ✅ Hybrid predictions combine accuracy + physical consistency")
        print("  4. ✅ Optimization finds optimal conditions using SGML")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
