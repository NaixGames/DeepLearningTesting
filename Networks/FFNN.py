import torch
import math
import sys
sys.path.append("../Utils")
import Functions
import CrossEntropy


#TODO: Allow for this NN to work with general tensors as input data, instead of waiting for two dimensional
#This assumption is seen when defining the first W tensor (in which I assume it is two dimensional)
#and also when doing matrix multiplication in the forward and backward methods

class FFNN(torch.nn.Module):
  def __init__(self, features_size : int, 
               inner_layers_sizes : list[int], 
               activation_functions : list[callable], 
               derivative_functions : list[callable], 
               categories_size : int, 
               keep_prop = None, 
               init_type : list[int] = None,
               batch_norm : list[bool] = None,
               device = 'cuda'):
    super(FFNN, self).__init__()
    
    assert(len(activation_functions) == len(derivative_functions))
    assert(len(inner_layers_sizes) == len(activation_functions))
    assert(keep_prop == None or len(keep_prop) == len(inner_layers_sizes) + 1)

    self.inner_layers = len(inner_layers_sizes)

    if batch_norm is None:
      batch_norm = [False] * len(inner_layers_sizes)
    assert len(batch_norm) == len(inner_layers_sizes)

    self.batch_norm = batch_norm

    if keep_prop is not None:
      for prob in keep_prop:
          assert(0 < prob <= 1)
    
    layers_w = []
    layers_b = []
    
    layers_w.append(self.give_weight_layer_init(features_size, inner_layers_sizes[0], 0, init_type))
    layers_b.append(self.give_bias_layer_init(inner_layers_sizes[0]))

    for i in range(0, len(inner_layers_sizes)-1):
      layers_w.append(self.give_weight_layer_init(inner_layers_sizes[i], inner_layers_sizes[i+1], i+1, init_type))
      layers_b.append(self.give_bias_layer_init(inner_layers_sizes[i+1]))
      
    layers_w.append(self.give_weight_layer_init(inner_layers_sizes[-1], categories_size, len(inner_layers_sizes) + 1, init_type))
    layers_b.append(self.give_bias_layer_init(categories_size))

    self.layers_weights = torch.nn.ParameterList([torch.nn.Parameter(layers_w[i]) for i in range(len(layers_w))])
    self.layers_bias = torch.nn.ParameterList([torch.nn.Parameter(layers_b[i]) for i in range(len(layers_b))])
    self.activation_functions = activation_functions
    self.derivative_functions = derivative_functions
    self.keep_prop = keep_prop
    self.bn_gamma = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(1, inner_layers_sizes[i])) for i in range(len(inner_layers_sizes))])
    self.bn_beta = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1, inner_layers_sizes[i])) for i in range(len(inner_layers_sizes))])
    self.bn_running_mean = [torch.zeros(1, size).to(device) for size in inner_layers_sizes]
    self.bn_running_var = [torch.ones(1, size).to(device) for size in inner_layers_sizes]
    #This values are for doing an exponential moving average for computting the BN values. 
    #They COULD be separate params, but I got so many at this point that I am happy keeping everything a bit cleaner with 
    #Less control. Since most likely I will always use the values below.
    self.bn_momentum = 0.9
    self.bn_eps = 1e-5

    self.device = device
    self.to(device)


  def summarize(self) -> str:
    result = "Summary: " + "\n"
    for name, param in self.named_parameters():
      result += name + " - dimension " + str(param.size()) + ": + " + str(param)
      result += "\n"
      
    return result
  
  def give_weight_layer_init(self, dim1 : int, dim2 : int, layer_depth : int, init_type : list[int]) -> torch.Tensor:
    #init 0 : N(0,1)
    #init 1 : Xavier
    #init 2: He
    if (init_type == None or layer_depth >= len(init_type) or init_type[layer_depth] == 0):
      return torch.randn(dim1, dim2)
    
    if (init_type[layer_depth] == 1):
      return torch.randn(dim1, dim2)*(math.sqrt(1/dim1))
    
    if (init_type[layer_depth] == 2):
      return torch.randn(dim1, dim2)*(math.sqrt(2/dim1))

  def give_bias_layer_init(self, dim : int) -> torch.Tensor:
    return torch.torch.randn(1, dim)*0.01

  def forward(self, x : torch.Tensor) -> torch.Tensor:
    result = x
    u_results = []
    h_results = []

    self.bn_cache = []

    L = self.inner_layers

    dropout_mask = []

    #Apply droupout on first layer
    if self.keep_prop is not None:
        prob = self.keep_prop[0]
        mask = (torch.rand_like(result) <= prob).float()
        result = (result*mask)/prob
        dropout_mask.append(mask)

    for i in range(0, L):
      result = result @ self.layers_weights[i] + self.layers_bias[i]
      u_results.append(result)

      #Apply Batch normalization
      if self.batch_norm[i]:
        result, cache = self.batch_norm_forward(result, i)
        self.bn_cache.append(cache)

      else:
        self.bn_cache.append(None)

      result = self.activation_functions[i](result)

      #Apply droupout
      if self.keep_prop is not None:
        prob = self.keep_prop[i + 1]
        mask = (torch.rand_like(result) <= prob).float()
        result = (result*mask)/prob
        dropout_mask.append(mask)

      h_results.append(result)
    
    #last layer is different! It doesnt have an activation function, but we softmax. We also do not do dropout here
    result = result @ self.layers_weights[L] + self.layers_bias[L]
    u_results.append(result)

    #store the reults of each layer
    self.u_results = u_results
    self.h_results = h_results
    self.dropout_mask = dropout_mask

    return Functions.softmax(result, 1)
  

  def predict(self, x : torch.Tensor) -> torch.Tensor:
    #same function as forward, but we don't cache the evaluations on the node to backpropagate or use the dropout mask

    result = x  
    L = self.inner_layers

    for i in range(0, L):
      result = result @ self.layers_weights[i] + self.layers_bias[i]

      if self.batch_norm[i]:
        result = self.batch_norm_inference(result, i)

      result = self.activation_functions[i](result)
 
    #last layer is different! It doesnt have an activation function, but we softmax
    result = result @ self.layers_weights[L] + self.layers_bias[L]

    return Functions.softmax(result, 1)
  

  def backward(self, x : torch.Tensor, y  : torch.Tensor, y_pred : torch.Tensor) -> None:
  
    # Once again, last layer is different
    L = self.inner_layers
    BatchSize = x.size()[0]
    
    #Note in this iteration we accumulate grad to allow flexibility, like multiple loss functions, in the future

    dL_dlastu = (y_pred - y)/BatchSize #Assume cross entropy loss.

    self.layers_weights[L].grad = self.h_results[L-1].t() @ dL_dlastu
    self.layers_bias[L].grad = dL_dlastu.sum(dim = 0, keepdim=True)
    dL_dlasth = dL_dlastu @ self.layers_weights[L].t()

    # Propagate backwards; note first layer is different too!

    for i in range(L-1, -1, -1):
      dL_dlasth = self.give_drop_grad(dL_dlasth, i + 1)

      dL_dlastu = dL_dlasth * self.derivative_functions[i](self.u_results[i])

      if self.batch_norm[i]:
        dLnorm_dlastu, dgamma, dbeta = self.batch_norm_backward(dL_dlastu, self.bn_cache[i], i)
        self.bn_gamma[i].grad = dgamma
        self.bn_beta[i].grad = dbeta
      else:
        dLnorm_dlastu = dL_dlastu

      if (i > 0):
        self.layers_weights[i].grad = self.h_results[i-1].t() @ dLnorm_dlastu
      
      else:
        x_masked = x
        x_masked = self.give_drop_grad(x_masked, 0)
        self.layers_weights[0].grad = x_masked.t() @ dLnorm_dlastu
      
      self.layers_bias[i].grad = dLnorm_dlastu.sum(dim=0, keepdim=True)
      dL_dlasth = dLnorm_dlastu @ self.layers_weights[i].t()


  def batch_norm_forward(self, x  : torch.Tensor, layer_index : int):
    mean = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, keepdim=True, unbiased=False)

    x_hat = (x - mean) / torch.sqrt(var + self.bn_eps)

    self.bn_running_mean[layer_index] = self.bn_momentum * self.bn_running_mean[layer_index] + (1 - self.bn_momentum) * mean
    self.bn_running_var[layer_index] = self.bn_momentum * self.bn_running_var[layer_index] + (1 - self.bn_momentum) * var

    out = self.bn_gamma[layer_index] * x_hat + self.bn_beta[layer_index]
    return out, (x, x_hat, mean, var)


  def batch_norm_inference(self, x : torch.Tensor, layer_index : int) -> torch.Tensor:
    mean = self.bn_running_mean[layer_index]
    var = self.bn_running_var[layer_index]

    x_hat = (x - mean) / torch.sqrt(var + self.bn_eps)
    return self.bn_gamma[layer_index] * x_hat + self.bn_beta[layer_index]
  

  def batch_norm_backward(self, dLast_u, cache : torch.Tensor , layer_index : int):
        x, x_hat, mean, var = cache
        N = x.shape[0]
        gamma = self.bn_gamma[layer_index]

        std_inv = 1.0 / torch.sqrt(var + self.bn_eps)

        dx_hat = dLast_u * gamma

        dvar = torch.sum(dx_hat * (x - mean) * -0.5 * std_inv**3, dim=0, keepdim=True)
        dmean = torch.sum(dx_hat * -std_inv, dim=0, keepdim=True) + dvar * torch.mean(-2 * (x - mean), dim=0, keepdim=True)

        dx = dx_hat * std_inv + dvar * 2 * (x - mean) / N + dmean / N

        dgamma = torch.sum(dLast_u * x_hat, dim=0, keepdim=True)
        dbeta = torch.sum(dLast_u, dim=0, keepdim=True)

        return dx, dgamma, dbeta


  def give_drop_grad(self, grad : torch.Tensor, index : int):
    if self.keep_prop is not None:
        prob = self.keep_prop[index]
        mask = self.dropout_mask[index]
        return grad * mask/prob
    
    return grad

  def clear_grad(self):
    for param in self.parameters():
      if (param.grad is not None):
        param.grad.zero_()


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
    



