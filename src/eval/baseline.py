import torch
import torch.nn as nn

class BlackBoxBaseline(nn.Module):
    """
    A standard Multi-Layer Perceptron (MLP) baseline that takes Layer 14 continuous activations 
    and predicts the downstream tasks directly, bypassing the Ternary Concept Bottleneck.
    This is used to compute the Concept Utilization Efficiency (CUE) metric.
    """
    def __init__(self, emb_dim: int, n_tasks: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_tasks),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)
