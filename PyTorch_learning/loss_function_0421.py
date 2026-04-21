import torch
import torch.nn as nn

#demo 01: cross entropy loss
def dm01():
    y_true = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    y_pred = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    criterion = nn.CrossEntropyLoss()
    loss = criterion(y_pred, y_true)
    print(f'loss: {loss}')
    
#demo 02: binary cross entropy loss
def dm02():
    y_true = torch.tensor([0, 1, 0], dtype=torch.float32)
    y_pred = torch.tensor([0.69, 0.2, 0.1])
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(y_pred, y_true)
    print(f'loss: {loss}')
    
def dm03():
    y_true = torch.tensor([2, 2, 2], dtype=torch.float32)
    y_pred = torch.tensor([1.5, 2.5, 3.0], dtype=torch.float32, requires_grad=True)
    criterion = nn.L1Loss()
    loss = criterion(y_pred, y_true)
    print(f'loss: {loss}')
    
def dm04():
    y_true = torch.tensor([2, 2, 2], dtype=torch.float32)
    y_pred = torch.tensor([1.5, 2.5, 3.0], dtype=torch.float32, requires_grad=True)
    criterion = nn.MSELoss()
    loss = criterion(y_pred, y_true)
    print(f'loss: {loss}')
    
def dm05():
    y_true = torch.tensor([2, 2, 2], dtype=torch.float32)
    y_pred = torch.tensor([1.5, 2.5, 3.0], dtype=torch.float32, requires_grad=True)
    criterion = nn.SmoothL1Loss()
    loss = criterion(y_pred, y_true)
    print(f'loss: {loss}')
    
if __name__ == '__main__':
    dm01()
    dm02()
    dm03()
    dm04()
    dm05()