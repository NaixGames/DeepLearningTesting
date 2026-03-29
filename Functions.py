import torch

def differentiate(func : callable) -> callable:
    if (func == sigmoid):
        return sigmoid_der
    if (func == tanh):
        return tanh_der
    if (func == relu):
        return relu_grad
    if (func == swish):
        return swish_grad
    if (func == celu):
        return celu_grad
    print("Function derivative not defined")
    return None

def sigmoid(T : torch.Tensor) -> torch.Tensor:
    return torch.reciprocal(1 + torch.exp(-1*T))

def tanh(T : torch.Tensor) -> torch.Tensor:
    exp = torch.exp(T)
    negexp = torch.exp(-1*T)
    return (exp-negexp)*torch.reciprocal(exp + negexp)

def sech(T : torch.Tensor) -> torch.Tensor:
    return 2*torch.reciprocal(torch.exp(T) + torch.exp(-1*T))

def relu(T : torch.Tensor) -> torch.Tensor:
  return torch.relu(T)

def swish(T : torch.Tensor, beta : float) -> torch.Tensor:
  return T*torch.sigmoid(T*beta)

def celu(T : torch.Tensor, alpha : float) -> torch.Tensor:
  return relu(T) - relu(-alpha*(torch.exp(T/alpha)-1))

def softmax(T : torch.Tensor, dim : int, stabilize : bool =True):
  targetTensor = T
  if stabilize:
    x_max = T.max(dim=dim, keepdim=True).values
    targetTensor = T-x_max
    
  exp_x = torch.exp(targetTensor)
  return exp_x / exp_x.sum(dim = dim, keepdim = True)

# Doing the derivatives
def sigmoid_der(T : torch.Tensor) -> torch.Tensor:
	return sigmoid(T)*(1-sigmoid(T))

def tanh_der(T : torch.Tensor) -> torch.Tensor:
	s = sech(T)
	return s*s

def relu_grad(T : torch.Tensor) -> torch.Tensor:
  return (T > 0).float()

def swish_grad(T : torch.Tensor, beta : float) -> torch.Tensor:
	sig = torch.sigmoid(T*beta)
	return sig * (1 + T *beta* (1 - sig))

def celu_grad(T : torch.Tensor, alpha : float) -> torch.Tensor:
  return (T>=0).float() + (T<0).float()*torch.exp(T/alpha)

  
if __name__ == "__main__":
  #This is mainly so have an easy check that it is all working fine
	x = torch.randn(3, 4)
	print("Printing the tensor")
	print(x)
	print("Printing the relu")
	print(relu(x))
	print("printing swish")
	print(swish(x,2))
	print("printing celu")
	print(celu(x,0.5))
	print("Printing softmax")
	print(softmax(x, 1))
	print(softmax(x, 0))
	print("Printing relu grad")
	print(relu_grad(x))
	print("Printing swish grad")
	print(swish_grad(x, 2))
	print("Printing celu grad")
	print(celu_grad(x, 0.5))


