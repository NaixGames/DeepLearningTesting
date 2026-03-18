import torch
import Functions

class FFNN(torch.nn.Module):
  def __init__(self, features_size : int, inner_layers_sizes : list[int], activation_functions : list[callable], derivative_functions : list[callable], categories_size : int):
    super(FFNN, self).__init__()
    
    assert(len(activation_functions) == len(derivative_functions))
    assert(len(inner_layers_sizes) == len(activation_functions))

    self.inner_layers = len(inner_layers_sizes)
    
    layers_w = []
    layers_b = []
    
    layers_w.append(torch.rand(features_size, inner_layers_sizes[0]))
    layers_b.append(torch.zeros(1, inner_layers_sizes[0]))
    
    for i in range(0, len(inner_layers_sizes)-1):
      layers_w.append(torch.rand(inner_layers_sizes[i], inner_layers_sizes[i+1]))
      layers_b.append(torch.zeros(1, inner_layers_sizes[i+1]))
      
    layers_w.append(torch.rand(inner_layers_sizes[-1], categories_size))
    layers_b.append(torch.zeros(1, categories_size))

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
  
  def backward(self, x, y, y_pred):
    # Once again, last layer is different
    L = self.inner_layers
    BatchSize = x.size()[0]

    dL_dlastuk = (y_pred - y)/BatchSize
    self.layers_weights[L].grad = self.h_results[L-1].t() @ dL_dlastuk
    self.layers_bias[L].grad = dL_dlastuk.sum(dim = 0, keepdim=True)
    dL_dlasthL = dL_dlastuk @ self.layers_weights[L].t()

    # Propagate backwards; note first layer is different too!
    for i in range(L-1, 0, -1):
      dL_dlastuk = dL_dlasthL * self.derivative_functions[i](self.u_results[i])
      self.layers_weights[i].grad = self.h_results[i-1].t() @ dL_dlastuk
      self.layers_bias[i].grad = dL_dlastuk.sum(dim=0, keepdim=True)
      dL_dlasthL = dL_dlastuk @ self.layers_weights[i].t()

    # Propagate to first layer in which the h is the input
    dL_dlastuk = dL_dlasthL * self.derivative_functions[0](self.u_results[0])
    self.layers_weights[0].grad = x.t() @ dL_dlastuk
    self.layers_bias[0].grad = dL_dlastuk.sum(dim=0, keepdim=True)

  def numeric_grad_check(self) -> float:
    #should do the approximation to check that our computed grad is not too bad
    return 0


if __name__ == "__main__":
  
  red = FFNN(20,[50,30],[Functions.relu, lambda x : Functions.swish(x, 5)], [Functions.relu_grad, lambda x : Functions.swish_grad(x, 5)],10)
  print(red.summarize())

  x_input = torch.rand(300, 20) #300 data points with 20 different possible features for 10 possible different categories
  print("Input:")
  print(x_input)
  print("Forward result:")
  y_pred = red.forward(x_input)
  print(y_pred)

  y = torch.rand(300, 1)
  red.backward(x_input, y, y_pred)
  print("Backpropagation finished.")
