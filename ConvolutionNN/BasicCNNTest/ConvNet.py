import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        #This will expect input of size B x 3 x 32 x 32, so imagines of 32 x 32 pixels
        #We define two layers, one of 20 filters for 3x3 convolution and another of 40 filters for 5x5 convolution
        self.conv_3x3 = nn.Conv2d(3, 20, (3,3), stride = 1, padding = 1)
        self.conv_5x5 = nn.Conv2d(3, 40, (5,5), stride = 1, padding = 2)

        #The next layer will use a 1x1 convolution to decrease the number channels from (20+40)*60 to 30, by considering the
        #last layer as a 60 channel input
        self.conv_1x1 = nn.Conv2d(60, 30, (1,1), stride = 1, padding = 0)

        #We do pooling to decrease the resolution to 16x16
        self.pool = nn.MaxPool2d((2,2), stride = 2)
        #we do two fully connected layers. Note we got  30 channels and images of size 16*16
        self.fc_1 = nn.Linear(30*16*16, 200)
        self.fc_out = nn.Linear(200, 10)

        #We define dropout layers to apply after the 1x1 convolution and between the fully connected layers.
        #We also do batch normalization
        self.dropout_1 = nn.Dropout2d(0.3)
        self.dropout_2 = nn.Dropout(0.2)

        self.BN_1 = nn.BatchNorm2d(30)
        self.BN_2 = nn.BatchNorm1d(200)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        #We apply the convolution, do concatenation and apply relu
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x = F.relu(torch.cat([x_3x3, x_5x5], dim = 1))

        #Apply the 1x1 convolution, dropout, BN and pool
        x = F.relu(self.conv_1x1(x))
        x = self.dropout_1(self.BN_1(x))
        x = self.pool(x)

        #Flat the input
        x = x.view(-1, 30*16*16)

        #Apply fully connected layers. Again, do dropout and BN
        x = F.relu(self.fc_1(x))
        x = self.dropout_2(self.BN_2(x))

        y_pred = self.fc_out(x)

        return y_pred
    
if __name__ == "__main__":
    #Params
    show_example_images = False
    epoch_to_train = 10

    #We test on CIFAR10
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import Compose, ToTensor, Normalize
    import matplotlib.pyplot as plt
    from random import randint

    tr = Compose([
        ToTensor(), 
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CIFAR10('/.', train=True, download=True, transform=tr)
    test_dataset = CIFAR10('/.', train=False, download=False, transform=tr)

    clases = ('avión', 'auto', 'ave', 'gato', 'venado', 
            'perro', 'rana', 'caballo', 'barco', 'camión')
    

    if show_example_images:
        w, h = 6, 3
        fig, axs = plt.subplots(h, w, figsize=(2*w,2*h))
        for i in range(h):
            for j in range(w):
                idx = randint(0,len(test_dataset))
                T, l = test_dataset[idx]
                
                # We turn back the pixel transformation for plotting
                T = T / 2 + 0.5
                T = T.permute(1,2,0) 
                
                img = T.numpy()
                axs[i,j].set_title(clases[l])
                axs[i,j].set_xticklabels([])
                axs[i,j].set_yticklabels([])
                axs[i,j].imshow(img)

        fig.show()

    from torch.utils.data import DataLoader

    batch_size = 128

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=4*batch_size, shuffle=True)

    total_train = len(train_dataset)
    total_test = len(test_dataset)

    net = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    # To be able to report
    import sys


    net = net.to('cuda')
    for epoch in range(epoch_to_train):  
        
        net.train()
        
        running_loss, running_acc = 0.0, 0.0
        
        for i, data in enumerate(train_data):
            
            X, Y = data
            X, Y = X.to('cuda'), Y.to('cuda')
        
            optimizer.zero_grad()
            Y_pred = net(X)
            loss = criterion(Y_pred, Y)
            loss.backward()
            optimizer.step()

            # Compute the error criteria
            items = (i+1) * batch_size
            running_loss += loss.item()
            Y_pred = F.softmax(Y_pred, dim=1)
            max_prob, max_idx = torch.max(Y_pred, dim=1)
            running_acc += torch.sum(max_idx == Y).item()
            info = f'\rEpoch:{epoch+1}({items}/{total_train}), '
            info += f'Loss:{running_loss/(i+1):02.5f}, '
            info += f'Train Acc:{running_acc/items*100:02.1f}%'
            sys.stdout.write(info)
            

        #Compute test accuracy
        net.eval()
        running_acc = 0.0
        for i, data in enumerate(test_data):
            X, Y = data
            X, Y = X.to('cuda'), Y.to('cuda')
            Y_pred = net(X)
            Y_pred = F.softmax(Y_pred, dim=1)
            max_prob, max_idx = torch.max(Y_pred, dim=1)
            running_acc += torch.sum(max_idx == Y).item()
        info = f', Test Acc:{running_acc/total_test*100:02.2f}%.\n'
        sys.stdout.write(info)


    #We use the net on some random example
    w, h = 6, 3
    fig, axs = plt.subplots(h, w, figsize=(2*w,2*h))
    for i in range(h):
        for j in range(w):
            idx = randint(0,len(test_dataset))
            T, real = test_dataset[idx]
            
            # Convertimos la imagen en un batch
            X = T.view(1,3,32,32).to('cuda')
            Y = F.softmax(net(X), dim=1)
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
