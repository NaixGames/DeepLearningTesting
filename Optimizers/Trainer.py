import CrossEntropy 
import torch
import sys
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as functional_parse
import Optimizer;

sys.path.append("../Networks")
from FFNN import FFNN


class Trainer():
    def train_FFNN(self, net : FFNN, optimizer : Optimizer, dataset : Dataset, epochs : int =1, batch_size : int =1, 
                   print_frequency : int = 10, compute_error_rate : bool = True) -> None:
        for i in range(epochs):
            dataloader = DataLoader(dataset, batch_size)
            for x,y in dataloader:
                net.clear_grad()
                y_pred = net.forward(x)
                loss = CrossEntropy.CELoss(y_pred, y)
                net.backward(x, y, y_pred)
                optimizer.step()
            
            if (i % print_frequency == 0):
                print("Current loss")
                print(loss.item())

                if (compute_error_rate):
                    print("Current error rate")
                    print(self.compute_error_rate(net, dataset))


    def train_FFNN_MNIST(self, net : FFNN, optimizer : Optimizer, dataset : Dataset, epochs : int =1, 
                         batch_size : int =1, print_frequency : int = 10, data_classes : int = 10, 
                         compute_error_rate : bool = True) -> None:
        
        for i in range(epochs):
            dataloader = DataLoader(dataset, batch_size)
            for x,y in dataloader:
                x_parse = x.view(x.size(0), -1)
                x_parse = x_parse.to('cuda')
                y_parse = functional_parse.one_hot(y, data_classes)
                y_parse = y_parse.to('cuda')
                net.clear_grad()
                y_pred = net.forward(x_parse)
                loss = CrossEntropy.CELoss(y_pred, y_parse)
                net.backward(x_parse, y_parse, y_pred)
                optimizer.step()
            
            if (i % print_frequency == 0):
                print("Current loss")
                print(loss.item())

                if (compute_error_rate):
                    print("Current error rate")
                    print(self.compute_error_rate_MNIST(net, dataset))


    def compute_error_rate(self, net : FFNN, dataset : Dataset) -> float:
        errors = 0
        datasize = len(dataset)
        dataloader = DataLoader(dataset, datasize)

        for x,y in dataloader:
            y_pred = net.predict(x).argmax(dim=1)
            
            y_class = y.argmax(dim = 1)
            
            errors += (y_pred != y_class).sum().item()

        return errors / datasize
    

    def compute_error_rate_MNIST(self, net : FFNN, dataset : Dataset) -> float:
        errors = 0
        datasize = len(dataset)
        dataloader = DataLoader(dataset, datasize)

        for x,y in dataloader:
            x_parse = x.view(x.size(0), -1)
            x_parse = x_parse.to('cuda')
            y_pred = net.predict(x_parse).argmax(dim=1)
            y_parse = y.to('cuda')
            errors += (y_pred != y_parse).sum().item()

        return errors / datasize