"""
Transformer-based models for molecular and time-series prediction.

This module implements state-of-the-art transformer architectures for:
- Molecular property prediction using SMILES
- Time-series forecasting for process optimization
- Multi-modal fusion of molecular and process data
"""

import math
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Union
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoConfig,
        BertModel, BertConfig, BertTokenizer,
        TrainingArguments, Trainer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. Install with: pip install transformers")


class SMILESTokenizer:
    """Advanced SMILES tokenizer using chemical vocabulary."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = self._build_chemical_vocab()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

    def _build_chemical_vocab(self) -> List[str]:
        """Build vocabulary of chemical tokens."""
        # Common SMILES tokens
        single_char = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H']
        double_char = ['Si', 'Mg', 'Al', 'Ca', 'Fe', 'Zn', 'Na', 'K']
        special_chars = ['(', ')', '[', ']', '=', '#', '\\', '/', '+', '-', '@']
        numbers = [str(i) for i in range(10)]

        # Ring indicators
        rings = ['%10', '%11', '%12', '%13', '%14', '%15']

        # Aromatic atoms
        aromatic = ['c', 'n', 'o', 's', 'p']

        # Special tokens
        special = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

        vocab = special + single_char + double_char + aromatic + special_chars + numbers + rings

        # Pad to vocab_size
        while len(vocab) < self.vocab_size:
            vocab.append(f'[UNUSED_{len(vocab)}]')

        return vocab[:self.vocab_size]

    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize SMILES string into chemical tokens."""
        tokens = []
        i = 0
        while i < len(smiles):
            # Try double character tokens first
            if i < len(smiles) - 1:
                two_char = smiles[i:i+2]
                if two_char in self.token_to_id:
                    tokens.append(two_char)
                    i += 2
                    continue

            # Single character tokens
            char = smiles[i]
            if char in self.token_to_id:
                tokens.append(char)
            else:
                tokens.append('[UNK]')
            i += 1

        return tokens

    def encode(self, smiles: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Encode SMILES to token IDs."""
        tokens = ['[CLS]'] + self.tokenize(smiles) + ['[SEP]']

        # Truncate or pad
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + ['[SEP]']
        else:
            tokens += ['[PAD]'] * (max_length - len(tokens))

        input_ids = [self.token_to_id.get(token, self.token_to_id['[UNK]']) for token in tokens]
        attention_mask = [1 if token != '[PAD]' else 0 for token in tokens]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MolecularTransformer(nn.Module):
    """Transformer model for molecular property prediction."""

    def __init__(self, vocab_size: int = 1000, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 output_dim: int = 1, max_length: int = 512):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_length)

        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Embedding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Create attention mask for transformer
        src_key_padding_mask = ~attention_mask.bool()

        # Transformer encoding
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Global average pooling (excluding padding)
        mask_expanded = attention_mask.unsqueeze(-1).expand(encoded.size()).float()
        sum_embeddings = torch.sum(encoded * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        # Prediction
        output = self.predictor(pooled)
        return output


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for time-series forecasting."""

    def __init__(self, input_dim: int, output_dim: int = 1, hidden_dim: int = 256,
                 num_heads: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Variable selection networks
        self.variable_selection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )

        # Gated residual network
        self.grn = GatedResidualNetwork(hidden_dim, hidden_dim, dropout)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # Position-wise feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_projection(x)

        # Variable selection
        variable_weights = self.variable_selection(x)
        x = x * variable_weights

        # Gated residual network
        x = self.grn(x)

        # Multi-head attention
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)

        # Position-wise feed forward
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)

        # Output projection (use last timestep)
        output = self.output_projection(x[:, -1, :])
        return output


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for temporal fusion transformer."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Skip connection
        if input_dim != hidden_dim:
            self.skip_connection = nn.Linear(input_dim, hidden_dim)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main path
        h = F.elu(self.linear1(x))
        h = self.dropout(h)
        h = self.linear2(h)

        # Gating
        g = torch.sigmoid(self.gate(h))
        h = h * g

        # Skip connection
        skip = self.skip_connection(x)

        # Add and normalize
        output = self.layer_norm(h + skip)
        return output


class MultiModalTransformer(nn.Module):
    """Multi-modal transformer combining molecular and process data."""

    def __init__(self, molecular_vocab_size: int = 1000, process_dim: int = 9,
                 d_model: int = 512, nhead: int = 8, num_layers: int = 6,
                 output_dim: int = 1):
        super().__init__()

        # Molecular encoder
        self.molecular_encoder = MolecularTransformer(
            vocab_size=molecular_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            output_dim=d_model
        )

        # Process data encoder
        self.process_encoder = nn.Sequential(
            nn.Linear(process_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )

        # Cross-attention fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )

        # Final predictor
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, molecular_input: Dict[str, torch.Tensor],
                process_input: torch.Tensor) -> torch.Tensor:

        # Encode molecular data
        mol_features = self.molecular_encoder(
            molecular_input['input_ids'],
            molecular_input['attention_mask']
        )

        # Encode process data
        proc_features = self.process_encoder(process_input)

        # Add sequence dimension for cross-attention
        mol_features = mol_features.unsqueeze(1)  # [batch, 1, d_model]
        proc_features = proc_features.unsqueeze(1)  # [batch, 1, d_model]

        # Cross-attention
        mol_attended, _ = self.cross_attention(mol_features, proc_features, proc_features)
        proc_attended, _ = self.cross_attention(proc_features, mol_features, mol_features)

        # Concatenate and predict
        combined = torch.cat([
            mol_attended.squeeze(1),
            proc_attended.squeeze(1)
        ], dim=-1)

        output = self.predictor(combined)
        return output


class MolecularDataset(Dataset):
    """Dataset for molecular transformer training."""

    def __init__(self, smiles_list: List[str], targets: List[float],
                 tokenizer: SMILESTokenizer, max_length: int = 512):
        self.smiles_list = smiles_list
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        target = self.targets[idx]

        # Tokenize SMILES
        encoded = self.tokenizer.encode(smiles, self.max_length)

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'target': target
        }


def create_transformer_pipeline(model_type: str = "molecular") -> nn.Module:
    """Factory function to create different transformer models."""

    if model_type == "molecular":
        return MolecularTransformer(
            vocab_size=1000,
            d_model=512,
            nhead=8,
            num_layers=6,
            output_dim=1
        )

    elif model_type == "temporal":
        return TemporalFusionTransformer(
            input_dim=9,  # Process variables
            output_dim=1,
            hidden_dim=256,
            num_heads=8,
            num_layers=4
        )

    elif model_type == "multimodal":
        return MultiModalTransformer(
            molecular_vocab_size=1000,
            process_dim=9,
            d_model=512,
            nhead=8,
            num_layers=6,
            output_dim=1
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def demo_transformer_models():
    """Demonstrate transformer model capabilities."""
    print("Transformer Models Demo")
    print("=" * 50)

    # Create tokenizer
    tokenizer = SMILESTokenizer(vocab_size=1000)

    # Example SMILES
    smiles = "C=CC(=O)OCc1ccccc1"
    print(f"SMILES: {smiles}")

    # Tokenize
    tokens = tokenizer.tokenize(smiles)
    print(f"Tokens: {tokens}")

    # Encode
    encoded = tokenizer.encode(smiles, max_length=64)
    print(f"Encoded shape: {encoded['input_ids'].shape}")

    # Create models
    models = {
        "Molecular Transformer": create_transformer_pipeline("molecular"),
        "Temporal Fusion Transformer": create_transformer_pipeline("temporal"),
        "Multi-Modal Transformer": create_transformer_pipeline("multimodal")
    }

    for name, model in models.items():
        print(f"\n{name}:")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        if "Molecular" in name:
            # Test with molecular input
            with torch.no_grad():
                output = model(encoded['input_ids'].unsqueeze(0),
                             encoded['attention_mask'].unsqueeze(0))
                print(f"Output shape: {output.shape}")

        elif "Temporal" in name:
            # Test with time series input
            with torch.no_grad():
                x = torch.randn(1, 10, 9)  # batch, seq_len, features
                output = model(x)
                print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    demo_transformer_models()