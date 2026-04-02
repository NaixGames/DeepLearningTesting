import torch
import sys
from Optimizer import Optimizer;

sys.path.append("../Networks")
from FFNN import FFNN



class Adam(Optimizer):
  def __init__(self, network : FFNN, rate : float = 1e-3, decay = 0, beta1=0.9, beta2 = 0.9, epsilon=1e-8):
    self.network = network
    self.rate = rate
    self.decay = decay
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.iteration = 0

    #Initialize momentum parameters
    self.momentums =  [torch.zeros_like(p) for p in network.parameters()]
    self.speeds = [torch.zeros_like(p) for p in network.parameters()]

  def step(self) -> None:
    self.iteration += 1

    param_list = list(self.network.parameters())
    for i in range(len(param_list)):
      param = param_list[i]
      self.momentums[i] = self.beta1*self.momentums[i] +(1 - self.beta1)*param.grad
      self.speeds[i] = self.beta2*self.speeds[i] +(1 - self.beta2)*param.grad*param.grad

      normalized_momentum = self.momentums[i] / (1 - self.beta1**self.iteration)
      normalized_speed = self.speeds[i] / (1- self.beta2**self.iteration)
      param.data = (1-self.decay)*param.data - self.rate * normalized_momentum / (torch.sqrt(normalized_speed)  + self.epsilon)