from torch.utils.data import Dataset, DataLoader
import torch
import Functions


class RandomDataset(Dataset):
  def __init__(self, data_size : int, features : int, possible_classes : int, device: str = "cuda"):
    self.data_size = data_size
    x_input = torch.randn(data_size, features) 
    self.x = x_input.to(device)

    y = torch.randn(data_size, possible_classes)
    y = Functions.softmax(y, 1) 
    self.y = y.to(device)
    
  
  def __len__(self):
    return self.data_size
  
  def __getitem__(self, i):
    return self.x[i], self.y[i]