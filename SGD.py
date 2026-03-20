import FFNN
import Functions
import CrossEntropy 
import torch

class SGD():
  def __init__(self, network : FFNN, rate : float = 1e-3):
    self.network = network
    self.rate = rate
  
  def step(self):
    for param in self.network.parameters():
      param.data -= param.grad*self.rate


if __name__ == "__main__":
  # --- Pamereters definitions ---
  device = 'cuda'

  sample_size = 100
  possible_classes = 5
  features_size = 5
  inner_layers = [10, 20]
  function_array = [Functions.sigmoid, Functions.tanh]

  #Param for optimizer
  step = 0.1
  frequency_loss_print = 1000

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
    #since the lambda results for "different definitions" is not the same
    if (func == Functions.swish):
      function_array[i] = lambda x : Functions.swish(x, beta_swish)
      derivatives_array[i] = lambda x : Functions.swish_grad(x, beta_swish)
    if (func == Functions.celu):
      function_array[i] = lambda x : Functions.celu(x, alpha_celu)
      derivatives_array[i] = lambda x : Functions.celu_grad(x, alpha_celu)

  red = FFNN.FFNN(features_size,[10, 20], function_array, derivatives_array, possible_classes)
  # --- Set device to GPU ---
  red.to(device)

  # --- train ---

  #Eventually erase this toy example
  x_input = torch.rand(sample_size, features_size) 
  x_input = x_input.to(device)
  
  print("Input:")
  print(x_input)
  print("Forward result:")
  y_pred = red.forward(x_input)
  print(y_pred)

  #Note it is REALLY important this is a probability measure, if not the grad computations are all wrong and the grad check gives garbage.
  #Really annoying to debug that one ... talking from experience here :)
  y = torch.randn(sample_size, possible_classes)
  y = Functions.softmax(y, 1) 
  y = y.to(device)
  print("real classes")
  print(y)
  print("u results")
  print(red.u_results)
  print("h results")
  print(red.h_results)

  red.clear_grad()
  red.backward(x_input, y, y_pred)
  print("BACK PROPAGATION FINISHED")
  print("Backpropagation finished. Doing grad check.")
  #Note this only make sense in the case the derivatives are smooth. When using relu you will get gradients "close to 1" in areas where the derivative jumps
  #Regardless of how good this computation is. Use sigmoids for more stable checks
  print(red.numeric_grad_check(x_input, y)) 


  optimizer = SGD(red, step)
  iterations = 10000
  for i in range(iterations):
    red.clear_grad()
    y_pred = red.forward(x_input)
    loss = CrossEntropy.CELoss(y_pred, y)
    red.backward(x_input, y, y_pred)
    optimizer.step()
    if (i % frequency_loss_print == 0):
      print("Current loss")
      print(loss)


""" This should work for loaded data eventually
  optimizer = SGD(red)
  for x,y in datos: #Need to define this
    y_pred = red.forward(x)
    loss = CrossEntropy.CELoss(y_pred, y)
    print("Current loss")
    print(loss)
    red.backward(x, y, y_pred)
    optimizer.step()
"""


