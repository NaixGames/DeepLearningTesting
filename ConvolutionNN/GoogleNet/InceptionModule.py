import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class InceptionModule(nn.Module):
  def __init__(self, 
               in_channels : int = 3, 
               ch_3x3_reduce : int = 96, 
               ch_5x5_reduce : int = 16,
               ch_3x3 : int = 128,
               ch_5x5 : int = 32,
               ch_pool_proj : int = 32,
               ch_1x1 : int = 64
    ):
    super(InceptionModule, self).__init__()
    self.conv_1x1_c1 = nn.Conv2d(in_channels, ch_3x3_reduce, (1,1), stride = 1, padding = 0)
    self.conv_1x1_c2 = nn.Conv2d(in_channels, ch_5x5_reduce, (1,1), stride = 1, padding = 0)

    self.conv_3x3_c1 = nn.Conv2d(ch_3x3_reduce, ch_3x3, (3,3), stride = 1, padding = 1)
    self.conv_5x5_c2 = nn.Conv2d(ch_5x5_reduce, ch_5x5, (5,5), stride = 1, padding = 2)

    self.pool = nn.MaxPool2d((3,3), stride = 1, padding = 1)
    self.conv_1x1_pool = nn.Conv2d(in_channels, ch_pool_proj, (1,1), stride = 1, padding = 0)

    self.conv_1x1 = nn.Conv2d(in_channels, ch_1x1, (1,1), stride = 1, padding = 0)

  def forward(self, x : torch.Tensor) -> torch.Tensor:
    x_c1 = self.conv_1x1_c1(x)
    x_c2 = self.conv_1x1_c2(x)
    
    x_c1 = self.conv_3x3_c1(x_c1)
    x_c2 = self.conv_5x5_c2(x_c2)
    
    x_pool = self.pool(x)
    x_pool = self.conv_1x1_pool(x_pool)

    x_1x1 = self.conv_1x1(x)

    return F.relu(torch.cat([x_c1, x_c2, x_pool, x_1x1], dim = 1))