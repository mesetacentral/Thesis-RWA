import torch
from torch.nn import *
from torch.utils import *


class NNRegressor(Module):
    def __init__(self, inDim, hiddenDim, outDim):
        super().__init__()
        self.fc1 = Linear(inDim, hiddenDim)
        self.activation1 = ReLU()
        self.fc3 = Linear(hiddenDim, outDim)
        
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)
    
    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.Tensor(x)
        out = self.fc1(x)
        out = self.activation1(out)
        out = self.fc3(out)
        
        return out

    def fit(self, x, y, criterion, optimizer, epochs=100, batch_size=32):
        if type(x) != torch.Tensor:
            x = torch.Tensor(x)
            y = torch.Tensor(y)
        
        for t in range(epochs):
            for i in range(int(x.shape[0] / batch_size)):
                indices = torch.randperm(x.shape[0])[:batch_size]

                x_batch = x[indices]
                y_batch = y[indices]
                
                # activate training mode
                self.train()
                y_pred = self.forward(x_batch) # x_train is the whole batch, so we are doing batch gradient descent
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.eval()


class local_NNRegressor(Module):
    def __init__(self, inDim, hiddenDim, outDim):
        super().__init__()
        self.fc1 = Linear(inDim, hiddenDim)
        self.activation1 = ReLU()
        self.fc2 = Linear(hiddenDim, outDim)
        
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
    
    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.Tensor(x)
        out = self.fc1(x)
        out = self.activation1(out)
        out = self.fc2(out)
        
        return out

    def fit(self, x, y, criterion, optimizer, epochs=100, batch_size=32):
        if type(x) != torch.Tensor:
            x = torch.Tensor(x)
            y = torch.Tensor(y)
        
        for t in range(epochs):
            for i in range(int(x.shape[0] / batch_size)):
                indices = torch.randperm(x.shape[0])[:batch_size]

                x_batch = x[indices]
                y_batch = y[indices]
                
                # activate training mode
                self.train()
                y_pred = self.forward(x_batch) # x_train is the whole batch, so we are doing batch gradient descent
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                self.eval()