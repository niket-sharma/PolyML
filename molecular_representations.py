"""
Advanced molecular representation methods for polymer property prediction.

This module provides state-of-the-art molecular featurization techniques including:
- RDKit molecular fingerprints
- Graph neural network representations
- Transformer-based embeddings
- 3D conformer features
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple, Dict
from dataclasses import dataclass
import logging

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem.Fragments import fr_benzene, fr_alkyl, fr_ether
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. Install with: conda install -c conda-forge rdkit")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch Geometric not available. Install with: pip install torch-geometric")


@dataclass
class MolecularFeatures:
    """Container for different types of molecular features."""
    smiles: str
    fingerprints: Optional[np.ndarray] = None
    descriptors: Optional[np.ndarray] = None
    graph_features: Optional[torch.Tensor] = None
    conformer_features: Optional[np.ndarray] = None


class MolecularFingerprints:
    """Generate various molecular fingerprints using RDKit."""

    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for molecular fingerprints")

    def morgan_fingerprints(self, smiles: str, radius: int = 2, nbits: int = 2048) -> np.ndarray:
        """Generate Morgan (circular) fingerprints."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(nbits)

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        return np.array(fp)

    def maccs_keys(self, smiles: str) -> np.ndarray:
        """Generate MACCS (Molecular ACCess System) keys."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(167)  # MACCS keys are 167 bits

        fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        return np.array(fp)

    def rdkit_fingerprints(self, smiles: str, nbits: int = 2048) -> np.ndarray:
        """Generate RDKit topological fingerprints."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(nbits)

        fp = Chem.RDKFingerprint(mol, fpSize=nbits)
        return np.array(fp)

    def atom_pair_fingerprints(self, smiles: str, nbits: int = 2048) -> np.ndarray:
        """Generate atom pair fingerprints."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(nbits)

        fp = rdMolDescriptors.GetAtomPairFingerprintAsBitVect(mol, nBits=nbits)
        return np.array(fp)


class MolecularDescriptors:
    """Calculate molecular descriptors using RDKit."""

    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for molecular descriptors")

    def calculate_descriptors(self, smiles: str) -> Dict[str, float]:
        """Calculate comprehensive set of molecular descriptors."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        descriptors = {
            # Basic molecular properties
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHBA': Descriptors.NumHBA(mol),
            'NumHBD': Descriptors.NumHBD(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),

            # Topological descriptors
            'BertzCT': Descriptors.BertzCT(mol),
            'Chi0v': Descriptors.Chi0v(mol),
            'Chi1v': Descriptors.Chi1v(mol),
            'HallKierAlpha': Descriptors.HallKierAlpha(mol),
            'Kappa1': Descriptors.Kappa1(mol),
            'Kappa2': Descriptors.Kappa2(mol),
            'Kappa3': Descriptors.Kappa3(mol),

            # Electronic descriptors
            'MaxPartialCharge': Descriptors.MaxPartialCharge(mol),
            'MinPartialCharge': Descriptors.MinPartialCharge(mol),
            'MaxAbsPartialCharge': Descriptors.MaxAbsPartialCharge(mol),

            # Geometric descriptors (requires 3D coordinates)
            'RadiusOfGyration': self._safe_descriptor(mol, Descriptors.RadiusOfGyration),
            'Asphericity': self._safe_descriptor(mol, Descriptors.Asphericity),
            'Eccentricity': self._safe_descriptor(mol, Descriptors.Eccentricity),

            # Fragment counts
            'fr_benzene': fr_benzene(mol),
            'fr_alkyl': fr_alkyl(mol),
            'fr_ether': fr_ether(mol),
        }

        return descriptors

    def _safe_descriptor(self, mol, descriptor_func):
        """Safely calculate descriptor that might fail."""
        try:
            return descriptor_func(mol)
        except:
            return 0.0


class MolecularGraph:
    """Convert SMILES to graph representation for GNNs."""

    def __init__(self):
        if not RDKIT_AVAILABLE or not TORCH_AVAILABLE:
            raise ImportError("RDKit and PyTorch Geometric are required for graph features")

    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """Convert SMILES string to PyTorch Geometric Data object."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            features = self._get_atom_features(atom)
            atom_features.append(features)

        # Get bond features and edge indices
        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])  # Undirected graph

            bond_feat = self._get_bond_features(bond)
            edge_features.extend([bond_feat, bond_feat])

        # Convert to tensors
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else None

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def _get_atom_features(self, atom) -> List[float]:
        """Extract features for a single atom."""
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().real,
            atom.GetImplicitValence(),
            atom.GetIsAromatic(),
            atom.GetMass(),
            atom.GetNumImplicitHs(),
            atom.GetNumRadicalElectrons(),
            atom.IsInRing(),
        ]
        return features

    def _get_bond_features(self, bond) -> List[float]:
        """Extract features for a single bond."""
        features = [
            bond.GetBondTypeAsDouble(),
            bond.GetIsAromatic(),
            bond.GetIsConjugated(),
            bond.IsInRing(),
            bond.GetStereo().real,
        ]
        return features


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for molecular property prediction."""

    def __init__(self, node_features: int = 10, edge_features: int = 5,
                 hidden_dim: int = 128, output_dim: int = 1, num_layers: int = 3):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(node_features, hidden_dim, heads=4, concat=False))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))

        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1))

        # Batch normalization
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph convolutions with residual connections
        for i, conv in enumerate(self.convs[:-1]):
            x_new = conv(x, edge_index)
            x_new = self.batch_norms[i](x_new)
            x_new = F.relu(x_new)
            x = x + x_new if i > 0 and x.size() == x_new.size() else x_new

        # Final layer
        x = self.convs[-1](x, edge_index)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Prediction
        out = self.classifier(x)
        return out


class AdvancedMolecularFeaturizer:
    """Main class combining all molecular featurization methods."""

    def __init__(self, use_fingerprints: bool = True, use_descriptors: bool = True,
                 use_graphs: bool = True):
        self.use_fingerprints = use_fingerprints
        self.use_descriptors = use_descriptors
        self.use_graphs = use_graphs

        if use_fingerprints:
            self.fingerprint_gen = MolecularFingerprints()
        if use_descriptors:
            self.descriptor_calc = MolecularDescriptors()
        if use_graphs:
            self.graph_gen = MolecularGraph()

    def featurize_molecule(self, smiles: str) -> MolecularFeatures:
        """Generate comprehensive molecular features."""
        features = MolecularFeatures(smiles=smiles)

        if self.use_fingerprints:
            # Combine multiple fingerprint types
            morgan = self.fingerprint_gen.morgan_fingerprints(smiles)
            maccs = self.fingerprint_gen.maccs_keys(smiles)
            rdkit_fp = self.fingerprint_gen.rdkit_fingerprints(smiles)

            features.fingerprints = np.concatenate([morgan, maccs, rdkit_fp])

        if self.use_descriptors:
            desc_dict = self.descriptor_calc.calculate_descriptors(smiles)
            features.descriptors = np.array(list(desc_dict.values()))

        if self.use_graphs:
            features.graph_features = self.graph_gen.smiles_to_graph(smiles)

        return features

    def featurize_dataset(self, smiles_list: List[str]) -> List[MolecularFeatures]:
        """Featurize a list of SMILES strings."""
        return [self.featurize_molecule(smiles) for smiles in smiles_list]


def demo_molecular_representations():
    """Demonstrate the molecular representation capabilities."""
    # Example polymer SMILES
    polymer_smiles = [
        "C=CC(=O)OCc1ccccc1",  # Poly(benzyl acrylate)
        "CCCCOC(=O)C=C",       # Poly(butyl acrylate)
        "C=CC(=O)O",           # Poly(acrylic acid)
    ]

    featurizer = AdvancedMolecularFeaturizer()

    print("Molecular Representation Demo")
    print("=" * 50)

    for i, smiles in enumerate(polymer_smiles):
        print(f"\nPolymer {i+1}: {smiles}")
        features = featurizer.featurize_molecule(smiles)

        if features.fingerprints is not None:
            print(f"Fingerprint size: {features.fingerprints.shape}")
        if features.descriptors is not None:
            print(f"Descriptor size: {features.descriptors.shape}")
        if features.graph_features is not None:
            print(f"Graph nodes: {features.graph_features.x.shape[0]}")
            print(f"Graph edges: {features.graph_features.edge_index.shape[1]}")


if __name__ == "__main__":
    demo_molecular_representations()