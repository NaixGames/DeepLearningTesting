import FFNN
import Functions
import CrossEntropy 
import torch
import RandomDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as functional_parse

class SGD():
  def __init__(self, network : FFNN, rate : float = 1e-3, decay = 0):
    self.network = network
    self.rate = rate
    self.decay = decay
  
  def step(self):
    for param in self.network.parameters():
      param.data = (1-self.decay)*param.data - param.grad*self.rate

  def train_FFNN(self, net : FFNN, dataset : Dataset, epochs : int =1, batch_size : int =1, print_frequency : int = 10):
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


if __name__ == "__main__":
  # --- Pamereters definitions ---
  device = 'cuda'

  sample_size = 100
  possible_classes = 5
  features_size = 5
  inner_layers = [20, 20, 20, 20, 20, 20, 20, 20]
  inner_layers = [20, 20, 20]
  function_array = [Functions.relu, Functions.relu, Functions.relu]

  keep_prob = [0.9, 0.75, 0.75, 0.75]
  init_type = [2, 2, 2, 2, 2]

  #Param for optimizer
  step = 0.01
  frequency_loss_print = 1000
  epochs = 10000
  batch_size = 100

  assert(batch_size <= sample_size)

  #Optional parameter for some functions
  beta_swish = 0.5
  alpha_celu = 0.75

  assert(len(inner_layers) == len(function_array))


  # --- network setup ---
  #We process the functions
  derivatives_array = []
  for i in range(0,len(function_array)):
    derivatives_array.append(Functions.differentiate(function_array[i]))

  for i in range(0, len(function_array)):
    func = function_array[i]
    #if we need an extra parameter, we "project" the function and the derivative
    #Note we do this after we got the derivatives, mainly because, if not, checking the lambda would give an incorrect check,
    #since the lambda results for "different definitions" is not the same (smth smth value vs reference smth smth)
    if (func == Functions.swish):
      function_array[i] = lambda x : Functions.swish(x, beta_swish)
      derivatives_array[i] = lambda x : Functions.swish_grad(x, beta_swish)
    if (func == Functions.celu):
      function_array[i] = lambda x : Functions.celu(x, alpha_celu)
      derivatives_array[i] = lambda x : Functions.celu_grad(x, alpha_celu)

  net = FFNN.FFNN(features_size, inner_layers, function_array, derivatives_array, possible_classes, keep_prop, init_type)
  # --- Set device to GPU ---
  net.to(device)

  # --- train ---

  dataset = RandomDataset.RandomDataset(sample_size, features_size, possible_classes, device)

  print("Input:")
  print(dataset.x)
  print("Forward result:")
  y_pred = net.forward(dataset.x)
  print(y_pred)
  print("real classes")
  print(dataset.y)

  optimizer = SGD(net, step, 0.0001)
  optimizer.train_FFNN(net, dataset, epochs, batch_size, frequency_loss_print)

  print("Dataset")
  input = dataset.x
  print(input)

  print("Predicted for dataset")
  print(net.predict(input))
  print("Real answer")
  print(dataset.y)


  print("Net")
  print(net.summarize())


