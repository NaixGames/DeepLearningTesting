import torch
import Functions

class FFNN(torch.nn.Module):
  def __init__(self, input_size : int, inner_layers_sizes, activation_functions, output_size):
    super(FFNN, self).__init__()
    
    self.inner_layers = len(inner_layers_sizes)
    
    layers_w = []
    layers_b = []
    
    layers_w.append(torch.rand(input_size, inner_layers_sizes[0]))
    layers_b.append(torch.zeros(1, inner_layers_sizes[0]))
    
    for i in range(0, len(inner_layers_sizes)-1):
      layers_w.append(torch.rand(inner_layers_sizes[i], inner_layers_sizes[i+1]))
      layers_b.append(torch.zeros(1, inner_layers_sizes[i+1]))
      
    layers_w.append(torch.rand(inner_layers_sizes[-1], output_size))
    layers_b.append(torch.zeros(1, output_size))

    self.layers_weights = torch.nn.ParameterList([layers_w[i] for i in range(len(layers_w))])
    self.layers_bias = torch.nn.ParameterList([layers_b[i] for i in range(len(layers_b))])
    self.activation_functions = torch.nn.ParameterList([activation_functions[i] for i in range(len(activation_functions))])

  def summarize(self):
    result = "Summary: " + "\n"
    for name, param in self.named_parameters():
      result += name + " - dimension " + str(param.size()) + ": + " + str(param)
      result += "\n"
      
    return result
  
  def forward(self, x):
    result = x
    
    for i in range(0, self.inner_layers+1):
      result = torch.mm(result, self.layers_weights[i]) + self.layers_bias[i]
      
    return Functions.softmax(result, 1)


if __name__ == "__main__":
  red = FFNN(300,[50,30],[Functions.relu,Functions.swish],10)
  print(red.summarize())

  x_input = torch.rand(20, 300)
  print("Forward result:")
  print(red.forward(x_input))