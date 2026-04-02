import sys
import torch

sys.path.append("../Networks")
import FFNN
sys.path.append("../Utils")
import Functions


if __name__ == "__main__":
  sample_size = 100
  possible_classes = 5
  features_size = 5
  inner_layers = [100, 200]
  function_array = [Functions.sigmoid, Functions.tanh]

  #Optional parameter for some functions
  beta_swish = 0.5
  alpha_celu = 0.75

  assert(len(inner_layers) == len(function_array))

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

  net = FFNN.FFNN(features_size, inner_layers, function_array, derivatives_array, possible_classes)
  print(net.summarize())

  x_input = torch.rand(sample_size, features_size) 
  
  print("Input:")
  print(x_input)
  print("Forward result:")
  y_pred = net.forward(x_input)
  print(y_pred)

  #Note it is REALLY important this is a probability measure, if not the grad computations are all wrong and the grad check gives garbage.
  #Really annoying to debug that one ... talking from experience here :)
  y = torch.randn(sample_size, possible_classes)
  y = Functions.softmax(y, 1) 
  print("real classes")
  print(y)
  print("u results")
  print(net.u_results)
  print("h results")
  print(net.h_results)

  net.backward(x_input, y, y_pred)
  print("BACK PROPAGATION FINISHED")
  print("Backpropagation finished. Doing grad check.")
  #Note this only make sense in the case the derivatives are smooth. When using relu you will get gradients "close to 1" in areas where the derivative jumps
  #Regardless of how good this computation is. Use sigmoids for more stable checks
  print(net.numeric_grad_check(x_input, y)) 
