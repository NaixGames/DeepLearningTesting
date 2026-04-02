import FFNN
import torch
import sys
from Optimizer import Optimizer;


sys.path.append("../Networks")
from FFNN import FFNN

class RSMProp(Optimizer):
  def __init__(self, network : FFNN, rate : float = 1e-3, decay = 0, beta=0.9, epsilon=1e-8):
    self.network = network
    self.rate = rate
    self.decay = decay
    self.beta = beta
    self.epsilon = epsilon

    #Initialize momentum parameters
    self.speeds = [torch.zeros_like(p) for p in network.parameters()]

  def step(self) -> None:
    param_list = list(self.network.parameters())
    for i in range(len(param_list)):
      param = param_list[i]
      self.speeds[i] = self.beta*self.speeds[i] +(1 - self.beta)*param.grad*param.grad
      param.data = (1-self.decay)*param.data - self.rate * param.grad / (torch.sqrt(self.speeds[i])  + self.epsilon)