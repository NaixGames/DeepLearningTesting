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
  
  def load_params(self) -> None:
    self.load_state_dict(torch.load("DenseNet_weights.pth"))

  def save_params(self) -> None:
    torch.save(self.state_dict(), "DenseNet_weights.pth")

  def perform_evaluation_example(self) -> None:
    #We use the net on some random example
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import Compose, ToTensor, Normalize
    import matplotlib.pyplot as plt
    from random import randint
    import torch.nn.functional as TF

    tr = Compose([
        ToTensor(), 
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = CIFAR10('/.', train=False, download=False, transform=tr)
    clases = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 
            'Dog', 'Frog', 'Horse', 'Boat', 'Truck')

    w, h = 6, 3
    fig, axs = plt.subplots(h, w, figsize=(2*w,2*h))
    for i in range(h):
        for j in range(w):
            idx = randint(0,len(test_dataset))
            T, real = test_dataset[idx]
            
            # Convertimos la imagen en un batch
            X = T.view(1,3,32,32).to('cpu')
            self.eval()
            Y = TF.softmax(self(X)['logits'], dim=1)
            prob, pred = torch.max(Y, dim=1)
            prob = prob.item()
            pred = pred.item()
            
            T = T / 2 + 0.5
            T = T.permute(1,2,0) 
            
            img = T.numpy()
            title = clases[pred] + ' / ' + clases[real] 
            axs[i,j].set_title(title)
            axs[i,j].set_xticklabels([])
            axs[i,j].set_yticklabels([])
            axs[i,j].imshow(img)
    plt.show(block=True)
  
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
    REPORTS_EVERY = 4

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

    # We test the model and see if we pass 75%
    print("Correct Test!" if (y==y_pred).sum()/len(x) >= .75 else "Failed Test! [acc]")

  

if __name__ == "__main__":
  k = 16
  num_classes = 10

  load_params = False
  perform_training = True
  save_params = True
  perform_evaluation_example = True

  epochs = 10

  #Load params if possible
  import sys
  inputs = sys.argv[1:]
  for i in range(0, len(inputs)):
    if (i == 0):
      load_params = inputs[i] == "True"
    if (i == 1):
      perform_training = inputs[i] == "True"
    if (i == 2):
      save_params = inputs[i] == "True"
    if (i == 3):
      perform_evaluation_example = inputs[i] == "True"
    if (i == 4):
      epochs = int(inputs[i])

  net = DenseNet(k, num_classes)
  
  if (load_params):
    net.load_params()

  if (perform_training):
    net.train_routine(epochs)

  if (net.save_params()):
    net.save_params()

  if (perform_evaluation_example):
    net.perform_evaluation_example()