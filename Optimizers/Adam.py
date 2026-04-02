import FFNN
import Functions
import CrossEntropy 
import torch
import sys
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as functional_parse

sys.path.append("../Networks")
from FFNN import FFNN
sys.path.append("../Utils")
import Functions
import RandomDataset


class Adam():
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
      self.momentums[i] = self.beta1*self.speeds[i] +(1 - self.beta1)*param.grad
      self.speeds[i] = self.beta2*self.speeds[i] +(1 - self.beta2)*param.grad*param.grad

      normalized_momentum = self.momentums[i] / (1 - self.beta1**self.iteration)
      normalized_speed = self.speeds[i] / (1- self.beta2**self.iteration)
      param.data = (1-self.decay)*param.data - self.rate * normalized_momentum / (normalized_speed  + self.epsilon)

  def train_FFNN(self, net : FFNN, dataset : Dataset, epochs : int =1, batch_size : int =1, print_frequency : int = 10) -> None:
    for i in range(epochs):
        dataloader = DataLoader(dataset, batch_size)
        for x,y in dataloader:
            net.clear_grad()
            y_pred = net.forward(x)
            loss = CrossEntropy.CELoss(y_pred, y)
            net.backward(x, y, y_pred)
            self.step()
        
        if (i % print_frequency == 0):
            print("Current loss")
            print(loss)

  def train_FFNN_MNIST(self, net : FFNN, dataset : Dataset, epochs : int =1, batch_size : int =1 , print_frequency : int = 10, data_classes : int = 10):
    for i in range(epochs):
        dataloader = DataLoader(dataset, batch_size)
        for x,y in dataloader:
            x_parse = x.view(x.size(0), -1)
            x_parse = x_parse.to('cuda')
            y_parse = functional_parse.one_hot(y, data_classes)
            y_parse = y_parse.to('cuda')
            net.clear_grad()
            y_pred = net.forward(x_parse)
            loss = CrossEntropy.CELoss(y_pred, y_parse)
            net.backward(x_parse, y_parse, y_pred)
            self.step()
        
        if (i % print_frequency == 0):
            print("Current loss")
            print(loss)