import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import DenseLayer

class DenseModule(nn.Module):
  def __init__(self, input_channels : int, layers : int, k : int = 32):
    super(DenseModule, self).__init__()
    layers = []

    for i in range(0, layers):
      dense_layer = DenseLayer.DenseLayer(input_channels + i*k, k)
      layers.append(dense_layer)

    self.dense_block = nn.Sequential(*layers)


  def forward(self, x : torch.Tensor) -> torch.Tensor:
    return self.dense_block(x)