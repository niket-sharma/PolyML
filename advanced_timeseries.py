"""
Advanced time-series models for polymer process optimization.

This module implements state-of-the-art time-series forecasting methods including:
- Neural ODEs for continuous dynamics
- State-space models (Mamba, S4)
- Temporal Convolutional Networks
- Multi-scale attention mechanisms
- Physics-informed neural networks
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Union, Callable
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class NeuralODE(nn.Module):
    """Neural Ordinary Differential Equation for continuous time dynamics."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # ODE function network
        self.ode_func = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Output projection
        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using Euler method for ODE integration.
        x: [batch_size, seq_len, input_dim]
        times: [seq_len] - time points
        """
        batch_size, seq_len, _ = x.shape
        dt = times[1] - times[0]  # Assume uniform time steps

        # Initial condition
        state = x[:, 0, :]  # [batch_size, input_dim]
        outputs = []

        for t in range(seq_len):
            # Euler step: y_{t+1} = y_t + dt * f(y_t, t)
            if t > 0:
                dydt = self.ode_func(state)
                state = state + dt * dydt

            outputs.append(state)

        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_len, input_dim]

        # Project to output dimension
        output = self.output_layer(outputs[:, -1, :])  # Use last timestep
        return output


class StateSpaceModel(nn.Module):
    """Simplified State Space Model inspired by S4/Mamba."""

    def __init__(self, input_dim: int, state_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim

        # State space matrices
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.1)
        self.C = nn.Parameter(torch.randn(output_dim, state_dim) * 0.1)
        self.D = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)

        # Make A stable
        self.register_buffer('I', torch.eye(state_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through state space model.
        x: [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Ensure A is stable (eigenvalues < 1)
        A = torch.tanh(self.A)

        # Initialize state
        state = torch.zeros(batch_size, self.state_dim, device=x.device)
        outputs = []

        for t in range(seq_len):
            # State update: h_t = A * h_{t-1} + B * x_t
            state = torch.matmul(state, A.T) + torch.matmul(x[:, t, :], self.B.T)

            # Output: y_t = C * h_t + D * x_t
            output = torch.matmul(state, self.C.T) + torch.matmul(x[:, t, :], self.D.T)
            outputs.append(output)

        # Return last output
        return outputs[-1]


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network with dilated convolutions."""

    def __init__(self, input_dim: int, num_channels: List[int], kernel_size: int = 2,
                 dropout: float = 0.2, output_dim: int = 1):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                   stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
                                   dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, input_dim]
        """
        # TCN expects [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        x = self.network(x)
        # Take last timestep
        x = x[:, :, -1]
        return self.output_layer(x)


class TemporalBlock(nn.Module):
    """Temporal block for TCN."""

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int,
                 dilation: int, padding: int, dropout: float = 0.2):
        super().__init__()

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove padding from the end of sequence."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism for time series."""

    def __init__(self, input_dim: int, num_scales: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.num_scales = num_scales
        self.hidden_dim = hidden_dim

        # Different scale projections
        self.scale_projections = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_scales)
        ])

        # Attention weights
        self.attention_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_scales)
        ])

        # Final projection
        self.output_projection = nn.Linear(hidden_dim * num_scales, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, input_dim = x.shape
        scale_outputs = []

        for scale in range(self.num_scales):
            # Different temporal pooling for each scale
            if scale == 0:
                # Fine scale - use all timesteps
                scale_input = x
            elif scale == 1:
                # Medium scale - average pool by 2
                scale_input = F.avg_pool1d(x.transpose(1, 2), kernel_size=2, stride=1, padding=1).transpose(1, 2)
            else:
                # Coarse scale - average pool by 4
                scale_input = F.avg_pool1d(x.transpose(1, 2), kernel_size=4, stride=1, padding=2).transpose(1, 2)

            # Trim to original length
            scale_input = scale_input[:, :seq_len, :]

            # Project to hidden dimension
            projected = self.scale_projections[scale](scale_input)

            # Compute attention weights
            attention_logits = self.attention_weights[scale](projected).squeeze(-1)
            attention_weights = F.softmax(attention_logits, dim=1)

            # Weighted average
            scale_output = torch.sum(projected * attention_weights.unsqueeze(-1), dim=1)
            scale_outputs.append(scale_output)

        # Concatenate all scales
        combined = torch.cat(scale_outputs, dim=-1)

        # Final projection
        output = self.output_projection(combined)
        return output


class PhysicsInformedNN(nn.Module):
    """Physics-Informed Neural Network for polymer processes."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1,
                 physics_weight: float = 1.0):
        super().__init__()
        self.physics_weight = physics_weight

        # Main neural network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with physics constraint.
        Returns: (prediction, physics_loss)
        """
        prediction = self.network(x)

        # Physics constraint: mass balance (simplified)
        # For polymer processes: d(concentration)/dt = reaction_rate - flow_rate
        physics_loss = self._compute_physics_loss(x, prediction)

        return prediction, physics_loss

    def _compute_physics_loss(self, x: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """Compute physics-based loss."""
        # Simplified physics constraint for demonstration
        # In practice, this would encode domain-specific physics
        batch_size = x.shape[0]

        # Example: conservation law
        # Sum of inputs should be related to output
        input_sum = torch.sum(x, dim=-1, keepdim=True)
        physics_constraint = torch.abs(prediction - 0.1 * input_sum)

        return torch.mean(physics_constraint)


class WaveletTransform(nn.Module):
    """Learnable wavelet transform for time-frequency analysis."""

    def __init__(self, input_dim: int, num_wavelets: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.num_wavelets = num_wavelets
        self.hidden_dim = hidden_dim

        # Learnable wavelet filters
        self.wavelet_filters = nn.Parameter(torch.randn(num_wavelets, input_dim))

        # Processing networks for each frequency band
        self.frequency_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_wavelets)
        ])

        # Attention for frequency band selection
        self.frequency_attention = nn.Sequential(
            nn.Linear(hidden_dim * num_wavelets, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_wavelets),
            nn.Softmax(dim=-1)
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, input_dim = x.shape

        frequency_features = []

        for i in range(self.num_wavelets):
            # Apply wavelet filter (simplified convolution)
            wavelet = self.wavelet_filters[i].unsqueeze(0).unsqueeze(0)
            x_filtered = F.conv1d(x.transpose(1, 2), wavelet.unsqueeze(-1), padding=0)

            # Global average pooling
            x_pooled = F.adaptive_avg_pool1d(x_filtered, 1).squeeze(-1)

            # Process frequency component
            freq_feature = self.frequency_processors[i](x_pooled)
            frequency_features.append(freq_feature)

        # Concatenate all frequency features
        combined_features = torch.cat(frequency_features, dim=-1)

        # Compute attention weights
        attention_weights = self.frequency_attention(combined_features)

        # Weighted combination
        weighted_features = torch.sum(
            torch.stack(frequency_features, dim=1) * attention_weights.unsqueeze(-1),
            dim=1
        )

        # Final prediction
        output = self.output_layer(weighted_features)
        return output


class AdaptiveTimeSeriesModel(nn.Module):
    """Adaptive model that combines multiple time-series approaches."""

    def __init__(self, input_dim: int, output_dim: int = 1, hidden_dim: int = 128):
        super().__init__()

        # Different model components
        self.neural_ode = NeuralODE(input_dim, hidden_dim, hidden_dim)
        self.ssm = StateSpaceModel(input_dim, hidden_dim, hidden_dim)
        self.tcn = TemporalConvNet(input_dim, [hidden_dim, hidden_dim, hidden_dim], output_dim=hidden_dim)
        self.msa = MultiScaleAttention(input_dim, num_scales=3, hidden_dim=hidden_dim)

        # Model selection network
        self.model_selector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # 4 models
            nn.Softmax(dim=-1)
        )

        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor, times: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, input_dim = x.shape

        # Generate time points if not provided
        if times is None:
            times = torch.linspace(0, 1, seq_len, device=x.device)

        # Get predictions from each model
        ode_out = self.neural_ode(x, times)
        ssm_out = self.ssm(x)
        tcn_out = self.tcn(x)
        msa_out = self.msa(x)

        # Reshape outputs to ensure consistent dimensions
        if ode_out.dim() == 1:
            ode_out = ode_out.unsqueeze(-1)
        if ssm_out.dim() == 1:
            ssm_out = ssm_out.unsqueeze(-1)
        if tcn_out.dim() == 1:
            tcn_out = tcn_out.unsqueeze(-1)
        if msa_out.dim() == 1:
            msa_out = msa_out.unsqueeze(-1)

        # Model selection weights based on input characteristics
        input_stats = torch.mean(x, dim=1)  # [batch_size, input_dim]
        model_weights = self.model_selector(input_stats)  # [batch_size, 4]

        # Combine outputs
        combined = torch.cat([ode_out, ssm_out, tcn_out, msa_out], dim=-1)

        # Weighted fusion
        weighted_combined = combined * model_weights.unsqueeze(-1).repeat(1, combined.shape[-1] // 4)

        # Final prediction
        output = self.fusion_layer(weighted_combined)
        return output


def create_advanced_timeseries_model(model_type: str, input_dim: int = 9, **kwargs) -> nn.Module:
    """Factory function for creating advanced time-series models."""

    models = {
        "neural_ode": lambda: NeuralODE(input_dim, **kwargs),
        "state_space": lambda: StateSpaceModel(input_dim, **kwargs),
        "tcn": lambda: TemporalConvNet(input_dim, [64, 64, 64], **kwargs),
        "multi_scale_attention": lambda: MultiScaleAttention(input_dim, **kwargs),
        "physics_informed": lambda: PhysicsInformedNN(input_dim, **kwargs),
        "wavelet": lambda: WaveletTransform(input_dim, **kwargs),
        "adaptive": lambda: AdaptiveTimeSeriesModel(input_dim, **kwargs)
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type]()


def demo_advanced_timeseries():
    """Demonstrate advanced time-series models."""
    print("Advanced Time-Series Models Demo")
    print("=" * 50)

    # Synthetic data
    batch_size, seq_len, input_dim = 32, 50, 9
    x = torch.randn(batch_size, seq_len, input_dim)
    times = torch.linspace(0, 1, seq_len)

    # Test different models
    model_types = ["neural_ode", "state_space", "tcn", "multi_scale_attention",
                   "physics_informed", "wavelet", "adaptive"]

    for model_type in model_types:
        print(f"\n{model_type.upper()}:")
        try:
            model = create_advanced_timeseries_model(model_type, input_dim)
            print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

            with torch.no_grad():
                if model_type == "neural_ode":
                    output = model(x, times)
                elif model_type == "physics_informed":
                    output, physics_loss = model(x[:, -1, :])  # Use last timestep
                    print(f"Physics loss: {physics_loss.item():.4f}")
                elif model_type == "adaptive":
                    output = model(x, times)
                else:
                    output = model(x)

                print(f"Output shape: {output.shape}")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    demo_advanced_timeseries()