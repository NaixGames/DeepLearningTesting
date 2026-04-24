import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TransitionBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, average_stride : int = 1):
        super().__init__()
        
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=average_stride)
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.layer(x)