import torch

# This is all done for two dimensions, idk if it is worth to generalize
def CELoss(Q : torch.Tensor, P : torch.Tensor, estable : bool = True, epsilon : float =1e-8) -> float:
  entropy_tensor = LogEntropy(P, Q, estable, epsilon)  
  partial_sum_tensor = torch.sum(entropy_tensor, dim = 1)
  partial_sum_tensor = partial_sum_tensor/partial_sum_tensor.size(dim = 0)
  return partial_sum_tensor.sum()

def LogEntropy(P : torch.Tensor, Q : torch.Tensor, estable : bool = True, epsilon : float = 1e-8) -> torch.Tensor:
  Q_log = StableTensorLog(Q, estable, epsilon)
  return -1*P*Q_log

def StableTensorLog(Q: torch.Tensor, estable : bool = True, epsilon : float = 1e-8) -> torch.Tensor:
  Q_normalized = Q
  if (estable):
    Q_normalized = torch.relu(Q - epsilon) + epsilon
  return torch.log(Q_normalized)

# Tu código acá
def CategoricalCELoss(Q : torch.Tensor, Target : torch.Tensor, estable : bool =True, epsilon : float =1e-8) -> float:
  projected_tensor = torch.Tensor([Q[i, Target[i]] for i in range(0, Q.size()[0])])
  projected_log = StableTensorLog(projected_tensor, estable, epsilon)
  projected_log = projected_log/projected_log.size(dim = 0)
  return -1*projected_log.sum()



if __name__ == "__main__":
  #For each one this should be probabilities that different data fit on different classes
  #the first dimension "moves" in direction of different data points and the second dimension moves on possible probability classes

  P_input = torch.rand(300, 20)
  Q_input = torch.rand(300, 20)
  print("P input:")
  print(P_input)
  print("Q input:")
  print(Q_input)
  print("Entropy tensor")
  print(LogEntropy(P_input, Q_input))
  print("Cross entropy:")
  print(CELoss(Q_input, P_input))

  print("Categorical loss")
  Q_input = torch.rand(6, 4)
  print(Q_input)
  indexes = torch.tensor([1,0,3,2,2,1])
  print(CategoricalCELoss(Q_input, indexes))