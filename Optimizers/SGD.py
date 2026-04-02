import FFNN
import torch
import sys
from Optimizer import Optimizer;

sys.path.append("../Networks")
from FFNN import FFNN


class SGD(Optimizer):
  def __init__(self, network : FFNN, rate : float = 1e-3, decay = 0, momentum = 0):
    self.network = network
    self.rate = rate
    self.decay = decay
    self.momentum = momentum

    #Initialize momentum parameters
    self.speeds = [torch.zeros_like(p) for p in network.parameters()]

  def step(self) -> None:
    param_list = list(self.network.parameters())
    for i in range(len(param_list)):
      param = param_list[i]
      self.speeds[i] = self.momentum*self.speeds[i] - param.grad*self.rate
      param.data = (1-self.decay)*param.data + self.speeds[i]
