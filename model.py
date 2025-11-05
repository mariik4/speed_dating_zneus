from enum import Enum
import torch
import torch.nn as nn


class ActivationFunctions(Enum):
    RELU = nn.ReLU
    GELU = nn.GELU
    TANH = nn.Tanh
    SIGM = nn.Sigmoid


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        output_dim: int = 1,
        activations: tuple[ActivationFunctions, ...] = (
            ActivationFunctions.RELU,),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not activations:
            raise ValueError("At least one activation must be provided")
        for activation in activations:
            if not isinstance(activation, ActivationFunctions):
                raise ValueError(f"Unsupported activation '{activation}'")
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activations[i % len(activations)].value())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
