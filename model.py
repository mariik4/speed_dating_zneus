from enum import Enum
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim

class ActivationFunctions(Enum):
    RELU = nn.ReLU
    GELU = nn.GELU
    TANH = nn.Tanh
    SIGM = nn.Sigmoid
    SOFTMAX = nn.Softmax


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dims: tuple[int, ...] = (128, 64),
            activations: tuple[ActivationFunctions, ...] = (
                ActivationFunctions.RELU,),
            dropout: float = 0.0,
    ) -> None:

        output_dim = 1 # FIXED

        super().__init__()

        if not activations:
            raise ValueError("No activations")

        # for activation in activations:
        #     if not isinstance(activation, ActivationFunctions):
        #         raise ValueError(f"Unsupported activation '{activation}'")

        if activations and len(activations) != len(hidden_dims):
            raise ValueError("The number of activations must match the number of hidden layers")
        
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activations[i].value())

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def train_epoch(self, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: str):
        self.train()
        total_loss = 0.0

        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = self(features)
            loss = criterion(logits, targets)
            loss.backward()

            optimizer.step()
            total_loss += loss.item() * targets.size(0)

        return total_loss / len(dataloader.dataset) # type: ignore

    def evaluate(self, dataloader: DataLoader, criterion: nn.Module, device: str):
        self.eval()
        total_loss = 0.0

        correct = 0
        total = 0
        with torch.no_grad():
            for features, targets in dataloader:
                features, targets = features.to(device), targets.to(device)
                
                loss = criterion(self(features), targets)
                total_loss += loss.item() * features.size(0)  
                
                preds = torch.sigmoid(self(features))

                preds = preds.view(-1)
                targets = targets.view(-1)
                
                predicted_labels = (preds >= 0.8).float()
                correct += (predicted_labels == targets).sum().item()
                total += targets.size(0)
        
        return total_loss / len(dataloader.dataset), correct / total    

    # def evaluate(self, dataloader: DataLoader, criterion: nn.Module, device: str):
    #     self.eval()
    #     total_loss = 0.0
    #     all_preds = []
    #     all_targets = []
        
    #     with torch.no_grad():
    #         for features, targets in dataloader:
    #             features, targets = features.to(device), targets.to(device)
    #             logits = self(features)
    #             loss = criterion(logits, targets)
    #             total_loss += loss.item() * features.size(0)
                
    #             preds = torch.sigmoid(logits)
                
    #             all_preds.append(preds.cpu())
    #             all_targets.append(targets.cpu())
        
    #     all_preds = torch.cat(all_preds).view(-1)
    #     all_targets = torch.cat(all_targets).view(-1)
        
    #     predicted_labels = (all_preds >= 0.7).float()
    #     correct = (predicted_labels == all_targets).sum().item()
    #     total = all_targets.size(0)
        
    #     return total_loss / len(dataloader.dataset), correct / total

    