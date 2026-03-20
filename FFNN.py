import torch
import Functions
import CrossEntropy

class FFNN(torch.nn.Module):
  def __init__(self, features_size : int, inner_layers_sizes : list[int], activation_functions : list[callable], derivative_functions : list[callable], categories_size : int):
    super(FFNN, self).__init__()
    
    assert(len(activation_functions) == len(derivative_functions))
    assert(len(inner_layers_sizes) == len(activation_functions))

    self.inner_layers = len(inner_layers_sizes)
    
    layers_w = []
    layers_b = []
    
    layers_w.append(torch.randn(features_size, inner_layers_sizes[0]))
    layers_b.append(torch.randn(1, inner_layers_sizes[0]))
    
    for i in range(0, len(inner_layers_sizes)-1):
      layers_w.append(torch.randn(inner_layers_sizes[i], inner_layers_sizes[i+1]))
      layers_b.append(torch.randn(1, inner_layers_sizes[i+1]))
      
    layers_w.append(torch.randn(inner_layers_sizes[-1], categories_size))
    layers_b.append(torch.randn(1, categories_size))

    self.layers_weights = torch.nn.ParameterList([torch.nn.Parameter(layers_w[i]) for i in range(len(layers_w))])
    self.layers_bias = torch.nn.ParameterList([torch.nn.Parameter(layers_b[i]) for i in range(len(layers_b))])
    self.activation_functions = activation_functions
    self.derivative_functions = derivative_functions

  def summarize(self) -> str:
    result = "Summary: " + "\n"
    for name, param in self.named_parameters():
      result += name + " - dimension " + str(param.size()) + ": + " + str(param)
      result += "\n"
      
    return result
  
  def forward(self, x : torch.Tensor) -> torch.Tensor:
    result = x
    u_results = []
    h_results = []
  

    L = self.inner_layers

    for i in range(0, L):
      result = result @ self.layers_weights[i] + self.layers_bias[i]
      u_results.append(result)
      result = self.activation_functions[i](result)
      h_results.append(result)
    
    #last layer is different! It doesnt have an activation function, but we softmax
    result = result @ self.layers_weights[L] + self.layers_bias[L]
    u_results.append(result)

    #store the reults of each later
    self.u_results = u_results
    self.h_results = h_results

    return Functions.softmax(result, 1)
  
  def clear_grad(self):
    for param in self.parameters():
      if (param.grad is not None):
        param.grad.zero_()


  def backward(self, x, y, y_pred):
  
    # Once again, last layer is different
    L = self.inner_layers
    BatchSize = x.size()[0]
    
    #Note in this iteration we accumulate grad to allow flexibility, like multiple loss functions, in the future

    dL_dlastu = (y_pred - y)/BatchSize #Assume cross entropy loss.)
    self.layers_weights[L].grad = self.h_results[L-1].t() @ dL_dlastu
    self.layers_bias[L].grad = dL_dlastu.sum(dim = 0, keepdim=True)
    dL_dlasth = dL_dlastu @ self.layers_weights[L].t()


    # Propagate backwards; note first layer is different too!

    for i in range(L-1, 0, -1):
      dL_dlastu = dL_dlasth * self.derivative_functions[i](self.u_results[i])
      self.layers_weights[i].grad = self.h_results[i-1].t() @ dL_dlastu
      self.layers_bias[i].grad = dL_dlastu.sum(dim=0, keepdim=True)
      dL_dlasth = dL_dlastu @ self.layers_weights[i].t()

    # Propagate to first layer in which the h is the input
    dL_dlastu = dL_dlasth * self.derivative_functions[0](self.u_results[0])
    self.layers_weights[0].grad = x.t() @ dL_dlastu
    self.layers_bias[0].grad = dL_dlastu.sum(dim=0, keepdim=True)

  def numeric_grad_check(self, x: torch.Tensor, y : torch.Tensor, step : float = 1e-3) -> float:
    with torch.no_grad():
      #should do the approximation to check that our computed grad is correctly computed
      #We do L^inf check, although that will give really big values when we dont expect it
      result = 0

      #We iterate through every parameter, we compute the numeric gradient at the point, and we store the max value
      for i in range(0, len(self.layers_weights)):
        layer_w = self.layers_weights[i]
        for j in range(0, len(layer_w)):
          for k in range(0, len(layer_w[j])):
            computed_grad = layer_w.grad[j][k].item()
            original_val = layer_w[j][k].item()
            self.layers_weights[i][j][k] = original_val + step
            for_val = CrossEntropy.CELoss(self.forward(x), y)
            self.layers_weights[i][j][k] = original_val - step
            back_val = CrossEntropy.CELoss(self.forward(x), y)
            num_grad = (for_val-back_val) / (2*step)
            result = max(result, abs(num_grad - computed_grad))
            self.layers_weights[i][j][k] = original_val

      return result


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

  red = FFNN(features_size, inner_layers, function_array, derivatives_array, possible_classes)
  print(red.summarize())

  x_input = torch.rand(sample_size, features_size) 
  
  print("Input:")
  print(x_input)
  print("Forward result:")
  y_pred = red.forward(x_input)
  print(y_pred)

  #Note it is REALLY important this is a probability measure, if not the grad computations are all wrong and the grad check gives garbage.
  #Really annoying to debug that one ... talking from experience here :)
  y = torch.randn(sample_size, possible_classes)
  y = Functions.softmax(y, 1) 
  print("real classes")
  print(y)
  print("u results")
  print(red.u_results)
  print("h results")
  print(red.h_results)

  red.backward(x_input, y, y_pred)
  print("BACK PROPAGATION FINISHED")
  print("Backpropagation finished. Doing grad check.")
  #Note this only make sense in the case the derivatives are smooth. When using relu you will get gradients "close to 1" in areas where the derivative jumps
  #Regardless of how good this computation is. Use sigmoids for more stable checks
  print(red.numeric_grad_check(x_input, y)) 

