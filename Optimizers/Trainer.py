import CrossEntropy 
import torch
import sys
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as functional_parse
import Optimizer;

sys.path.append("../Networks")
from FFNN import FFNN


class Trainer():
    def train_FFNN(self, net : FFNN, optimizer : Optimizer, dataset : Dataset, epochs : int =1, batch_size : int =1, print_frequency : int = 10) -> None:
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
                print(loss)

    def train_FFNN_MNIST(self,net : FFNN, optimizer : Optimizer, dataset : Dataset, epochs : int =1, batch_size : int =1 , print_frequency : int = 10, data_classes : int = 10):
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
                print(loss)