import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels : int, k : int):
        super().__init__()
        
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, k, kernel_size=1, bias=False), #Do 1x1 convolution to condense information
            
            nn.BatchNorm2d(k),
            nn.ReLU(inplace=True),
            nn.Conv2d(k, k, kernel_size=3, padding=1, bias=False) #Do 3x3 covolution to get correlations
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        input_eval = self.dense_layer(x)
        return torch.cat([x, input_eval], dim=1)