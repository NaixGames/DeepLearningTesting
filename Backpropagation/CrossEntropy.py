import torch

# This is all done for two dimensions, idk if it is worth to generalize
def CELoss(Q : torch.Tensor, P : torch.Tensor, estable : bool = True, epsilon : float =1e-8) -> float:
  entropy_tensor = LogEntropy(P, Q, estable, epsilon)  
  partial_sum_tensor = torch.sum(entropy_tensor, dim = 0)
  partial_sum_tensor = partial_sum_tensor/partial_sum_tensor.size(dim = 0)
  return partial_sum_tensor.sum()

def LogEntropy(P : torch.Tensor, Q : torch.Tensor, estable : bool = True, epsilon : float = 1e-8) -> torch.Tensor:
  Q_normalized = StableTensorLog(Q, estable, epsilon)
  return -1*P*torch.log(Q_normalized)

def StableTensorLog(Q: torch.Tensor, estable : bool = True, epsilon : float = 1e-8) -> torch.Tensor:
  Q_normalized = Q
  if (estable):
    Q_normalized = torch.relu(Q - epsilon) + epsilon
  return torch.log(Q_normalized)

# Tu código acá
def CategoricalCELoss(Q : torch.Tensor, Target : torch.Tensor, estable : bool =True, epsilon : float =1e-8) -> float:
  Q_normalized = StableTensorLog(Q, estable, epsilon)
  projected_tensor = Q_normalized[Target]
  print("DEBUG")
  print(Q_normalized)
  print(projected_tensor)
  projected_tensor = projected_tensor/projected_tensor.size(dim = 0)
  return projected_tensor.sum()



if __name__ == "__main__":
  #TOdo check that indexes are correct. I am getting dizzy in which one is the category and which one is the feature

  P_input = torch.rand(20, 300)
  Q_input = torch.rand(20, 300)
  print("P input:")
  print(P_input)
  print("Q input:")
  print(Q_input)
  print("Entropy tensor")
  print(LogEntropy(P_input, Q_input))
  print("Cross entropy:")
  print(CELoss(Q_input, P_input))

  print("Categorical loss")
  Q_input = torch.rand(20, 3)
  print(Q_input)
  indexes = torch.tensor([6,0,3])
  print(CategoricalCELoss(Q_input, indexes))