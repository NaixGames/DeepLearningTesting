import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import InceptionModule as InceptionModule


#Module used for the "predition" parts, including the ones used for the intermediate loss functions
class AuxClassifier(nn.Module):
  def __init__(self, in_channels : int, n_classes : int):
    super(AuxClassifier, self).__init__()
    self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
    self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
    self.fc1 = nn.Linear(128 * 4 * 4, 1024)
    self.fc2 = nn.Linear(1024, n_classes)
    self.dropout = nn.Dropout(0.7)

  def forward(self, x):
    x = self.avgpool(x)
    x = F.relu(self.conv(x))
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x


class GoogLeNet(nn.Module):
  def __init__(self, n_classes : int, use_aux_logits=True):
    super(GoogLeNet, self).__init__()
    
    self.use_aux_logits = use_aux_logits

    #Note this is for images of size 32 x 32 with 3 RBG channels

    # Initial layers
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0) #The padding decreases the image to be of size 30x30
    self.maxpool1 = nn.MaxPool2d(3, stride=1, ceil_mode=True)

    self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride = 1, padding = 0)
    self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1) #The padding decreases the image to be of size 28x28
    self.maxpool2 = nn.MaxPool2d(3, stride=1, ceil_mode=True) #stride 1 leaves this as 28x28. Note it has 192 channels, as pooling keeps channels!

    # Inception blocks. Recall inputs are: N_channels, 3x3reduce, 5x5reduce, 3x3size, 5x5size, maxpoolsize, 1x1size 
    # Note the resulting number of channels is (3x3size + 5x5size + maxpoolsize + 1x1size)
    self.inception3a = InceptionModule.InceptionModule(192, 96, 16, 128, 32, 32, 64)
    self.inception3b = InceptionModule.InceptionModule(256, 128, 32, 192, 96,64, 128)
    self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

    self.inception4a = InceptionModule.InceptionModule(480, 96, 16, 208, 48, 64, 192)
    self.inception4b = InceptionModule.InceptionModule(512, 112, 24, 224, 64, 64, 160)
    self.inception4c = InceptionModule.InceptionModule(512, 128, 24, 256, 64, 64, 128)
    self.inception4d = InceptionModule.InceptionModule(512, 144, 32, 288, 64, 64, 112)
    self.inception4e = InceptionModule.InceptionModule(528, 160, 32, 320, 128, 128, 256)
    self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

    self.inception5a = InceptionModule.InceptionModule(832, 160, 32, 320, 128, 128, 256)
    self.inception5b = InceptionModule.InceptionModule(832, 192, 48, 384, 128, 128, 384)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.dropout = nn.Dropout(0.4)

    # Auxiliary classifiers
    if self.use_aux_logits:
      self.aux1 = AuxClassifier(512, n_classes)
      self.aux2 = AuxClassifier(528, n_classes)

    # Output layer
    self.fc_out = nn.Linear(1024, n_classes)

  def forward(self, x):
    if self.use_aux_logits and self.training:
      aux_logits = []
    else:
      aux_logits = None


    x = F.relu(self.conv1(x))
    x = self.maxpool1(x)

    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = self.maxpool2(x)

    # Inception 3
    x = self.inception3a(x)
    x = self.inception3b(x)
    x = self.maxpool3(x)

    # Inception 4a
    x = self.inception4a(x)

    if self.use_aux_logits and self.training:
      aux_logit_1 = self.aux1(x)
      aux_logits.append(aux_logit_1)

    x = self.inception4b(x)
    x = self.inception4c(x)
    x = self.inception4d(x)

    if self.use_aux_logits and self.training:
      aux_logit_2 = self.aux2(x)
      aux_logits.append(aux_logit_2)

    x = self.inception4e(x)
    x = self.maxpool4(x)

    # Inception 5
    x = self.inception5a(x)
    x = self.inception5b(x)

    # Final layers
    hidden = self.avgpool(x)
    hidden = torch.flatten(hidden, 1)
    hidden = self.dropout(hidden)

    logits = self.fc_out(hidden)

    return {'hidden': hidden, 'logits': logits, 'aux_logits': aux_logits}
  
  def load_params(self) -> None:
    pass

  def save_params(self) -> None:
    pass

  def train_routine(self, epoch : int) -> None:
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
    EPOCHS = epoch
    REPORTS_EVERY = 1

    optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss() 
    scheduler = StepLR(optimizer, step_size=10, gamma=LR) 

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=4*BATCH_SIZE,
                            shuffle=False, num_workers=2)

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
  #Default Params
  number_classes = 10
  use_aux_logit = True

  load_params = False
  perform_traing = True
  save_params = True
  
  epochs = 5

  net = GoogLeNet(number_classes, use_aux_logit)

  if (load_params):
    net.load_params()

  if (perform_traing):
    net.train_routine(epochs)

  if (net.save_params()):
    net.save_params()