import torch
import sys 

sys.path.append("../Networks")
import FFNN
from NNSerializer import NNSerializer
sys.path.append("../Utils")
import Functions
import RandomDataset
sys.path.append("../Optimizers")
from Trainer import Trainer
from SGD import SGD
from RSMProp import RSMProp
from Adam import Adam

if __name__ == "__main__":
  # --- Pamereters definitions ---
  device = 'cpu'

  sample_size = 100
  possible_classes = 5
  features_size = 5
  inner_layers = [20, 20, 20]
  function_array = [Functions.relu, Functions.relu, Functions.relu]

  keep_prob = [0.9, 0.75, 0.75, 0.75]
  init_type = [2, 2, 2, 2, 2]

  #Param for optimizer
  step = 0.01
  frequency_loss_print = 1
  epochs = 1
  batch_size = 100
  momentum = 0.1
  decay = 0.000001

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

  net = FFNN.FFNN(features_size, inner_layers, function_array, derivatives_array, possible_classes, keep_prob, init_type)
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

#  optimizer = SGD(net, step, decay, momentum)
#  optimizer = RSMProp(net, step, decay)
  optimizer = Adam(net, step, 0)

  trainer = Trainer()
  trainer.train_FFNN(net, optimizer, dataset, epochs, batch_size, frequency_loss_print)

  print("Dataset")
  input = dataset.x
  print(input)

  print("Predicted for dataset")
  print(net.predict(input))
  print("Real answer")
  print(dataset.y)


  print("Net")
  print(net.summarize())