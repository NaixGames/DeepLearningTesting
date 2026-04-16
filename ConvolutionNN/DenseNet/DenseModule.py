import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DenseModule(nn.Module):
  def __init__(self, 
               in_channels : int = 3, 
               ch_3x3_reduce : int = 96, 
               ch_5x5_reduce : int = 16,
               ch_3x3 : int = 128,
               ch_5x5 : int = 32,
               ch_pool_proj : int = 32,
               ch_1x1 : int = 64
    ):
    super(DenseModule, self).__init__()
    


  def forward(self, x : torch.Tensor) -> torch.Tensor:
    pass