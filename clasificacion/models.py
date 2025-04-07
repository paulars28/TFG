import torch.nn as nn
from kan import KANLayer


class DiabetesMLP(nn.Module):
    def __init__(self):
        super(DiabetesMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class DiabetesKAN(nn.Module):
    def __init__(self):
        super(DiabetesKAN, self).__init__()
        self.kan1 = KANLayer(in_dim=8, out_dim=32)
        self.kan2 = KANLayer(in_dim=32, out_dim=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x, *_ = self.kan1(x)  # Solo cogemos el primer output
        x, *_ = self.kan2(x)
        x = self.activation(x)
        return x