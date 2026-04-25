import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import DenseModule as DenseModule
import TransitionBlock as TransitionBlock

class DenseNet(nn.Module):
  def __init__(self, k : int = 32, num_classes  : int = 10):
    super(DenseNet, self).__init__()
    
    #This is meant to recieve images of size 32 x 32 with 3 RGB channels
    number_channels = 2*k

    # Initial Block
    self.init_conv = nn.Conv2d(3, number_channels, kernel_size=3, stride=1, padding=0, bias=False) #this makes the image to be 30x30
    self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) #this brings the image to 30x30
    
    #First dense block + transition block
    self.dense_block_1 = DenseModule.DenseModule(input_channels= 2*k, layers = 6, k = k)
    number_channels += 6*k
    self.transition_1 = TransitionBlock.TransitionBlock(number_channels, number_channels // 2, 1) #The image here is 29 x 29
    number_channels = number_channels // 2
    
    #Second dense block + transition
    self.dense_block_2 = DenseModule.DenseModule(input_channels=number_channels, layers=12, k=k)
    number_channels += 12 * k
    self.transition_2 = TransitionBlock.TransitionBlock(number_channels, number_channels // 2, 2) #The image here is 14 x 14
    number_channels = number_channels // 2

    # Third dense block + transition
    self.dense_block_3 = DenseModule.DenseModule(input_channels=number_channels, layers=24, k=k)
    number_channels += 24 * k
    self.transition_3 = TransitionBlock.TransitionBlock(number_channels, number_channels // 2, 2) #The image here is 7 x 7
    number_channels = number_channels // 2
    
    # Fourth dense block (my brother only one more dense block, I swear this is the last one :P)
    self.dense_block_4 = DenseModule.DenseModule(input_channels=number_channels, layers=16, k=k)
    number_channels += 16 * k

    # Average pooling condesing everything on 1x1
    self.bn = nn.BatchNorm2d(number_channels)
    self.relu = nn.ReLU(inplace=True)
    self.last_average = nn.AvgPool2d(kernel_size=7)
    
	  #A fully connected neural net to finish :)
    self.fc = nn.Linear(number_channels, num_classes)

  def forward(self, x : torch.Tensor) -> torch.Tensor:
    x = self.pool(self.init_conv(x))

    x = self.transition_1(self.dense_block_1(x))
    x = self.transition_2(self.dense_block_2(x))    
    x = self.transition_3(self.dense_block_3(x))

    x = self.dense_block_4(x)

    x = self.relu(self.bn(x)) 
    x = self.last_average(x)

    hidden = torch.flatten(x, 1)
    logits = self.fc(hidden)

    return {'hidden': hidden, 'logits': logits, 'aux_logits': []}
  
  def train_routine(self, epochs : int) -> None:
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    from scipy.spatial import distance

    import torchvision
    import torchvision.transforms as transforms
    from torch.optim.lr_scheduler  import StepLR
    import sys
    sys.path.append("../Common")
    from TrainUtils import train_for_classification, plot_results

    transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Definamos algunos hiper-parámetros
    BATCH_SIZE = 128
    LR = 0.1
    EPOCHS = epochs
    REPORTS_EVERY = 1

    optimizer = optim.Adam(self.parameters())
    criterion = nn.CrossEntropyLoss() 
    scheduler = StepLR(optimizer, step_size=10, gamma=LR) 

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=4*BATCH_SIZE,
                            shuffle=False, num_workers=2)

    print("starting training routine")

    train_loss, acc = train_for_classification(self, train_loader, 
                                              test_loader, optimizer, 
                                              criterion, lr_scheduler=scheduler, 
                                              epochs=EPOCHS, reports_every=REPORTS_EVERY)

    plot_results(train_loss, acc)

    #Test
    x, y = list(test_loader)[0]
    self.cpu()
    self.eval()
    y_pred = self(x)['logits'].max(dim=1)[1]

    # Veamos como se comporta el modelo
    print("Correct Test!" if (y==y_pred).sum()/len(x) >= .75 else "Failed Test! [acc]")

  

if __name__ == "__main__":
  k = 16
  num_classes = 10

  epochs = 5

  net = DenseNet(k, num_classes)
  print("training net")
  net.train_routine(epochs)