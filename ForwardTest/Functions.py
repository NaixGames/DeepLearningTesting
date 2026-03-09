import torch


def relu(T):
  return torch.relu(T)

def swish(T, beta):
  return T*torch.sigmoid(T*beta)

def celu(T, alpha):
  return relu(T) - relu(-alpha*(torch.exp(T/alpha)-1))

def softmax(T, dim, stabilize=True):
  targetTensor = T
  if stabilize:
    x_max = T.max(dim=dim, keepdim=True).values
    targetTensor = T-x_max
    
  exp_x = torch.exp(targetTensor)
  return exp_x / exp_x.sum(dim = dim, keepdim = True)

  
if __name__ == "__main__":
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


