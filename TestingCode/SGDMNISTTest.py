import sys 
import torch
import random

sys.path.append("../Networks")
import FFNN

sys.path.append("../Optimizers")
import SGD
import Adam
import Trainer

sys.path.append("../Utils")
import Functions
import CrossEntropy 
import RandomDataset

from torch.utils.data import Dataset, DataLoader



if __name__ == "__main__":
  # --- Pamereters definitions ---
  device = 'cuda'

  sample_size = 10000
  possible_classes = 10
  features_size = 784
  inner_layers = [32, 16]
  function_array = [Functions.relu, Functions.relu]

  keep_prob = [0.9, 0.75, 0.75]
  init_type = [2, 2, 2, 2]
  batch_norm = [True, True]

  #Param for optimizer
  step = 0.01
  frequency_loss_print = 50
  epochs = 10000
  batch_size = 100
  momentum = 0.1
  decay = 0.000001

  assert(batch_size <= sample_size)

  #Optional parameter for some functions
  beta_swish = 0.5
  alpha_celu = 0.75

  assert(len(inner_layers) == len(function_array))

  #Parameter for debug
  plot_data_sample = False


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

  net = FFNN.FFNN(features_size, inner_layers, function_array, derivatives_array, possible_classes, keep_prob, 
                  init_type, batch_norm, device)


  # --- Download MNIST dataset, load it and plot it ---

  from torchvision.datasets import MNIST
  from torchvision.transforms import ToTensor
  # This is for plotting
  from matplotlib.pyplot import subplots
  from random import randint

  # Download the MNIST dataset
  dataset = MNIST('mnist', train=False, transform=ToTensor(), download=True)
  print('Cantidad total de datos:',len(dataset))

  # Show some examples
  if plot_data_sample:
    n_ejemplos = 3
    fig, axs = subplots(nrows=n_ejemplos, figsize=(2,n_ejemplos*3))

    for i in range(n_ejemplos):  
        idx = random.randint(0,len(dataset))
        T, l = dataset[idx]
        img = T.view(28,28).numpy()
        axs[i].set_title("clase: "+ str(l))
        axs[i].imshow(img)
    fig.show()

  dataloader = DataLoader(dataset, batch_size)
  #optimizer = SGD.SGD(net, step)
  optimizer = Adam.Adam(net, step, decay)

  trainer = Trainer.Trainer()
  trainer.train_FFNN_MNIST(net, optimizer, dataset, epochs, batch_size, frequency_loss_print)



