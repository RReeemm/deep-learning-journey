import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary   

BATCH_SIZE = 8

#1. dataset
def create_dataset():
    train_dataset = CIFAR10(root ='D:\GitHub\deep-learning-journey\dataset', train=True, transform=ToTensor(), download=False)
    test_dataset = CIFAR10(root ='D:\GitHub\deep-learning-journey\dataset', train=False, transform=ToTensor(), download=False)
    return train_dataset, test_dataset
    
class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3,6,3,1,0)
        self.pool1 = nn.MaxPool2d(2,2,0)
        self.conv2 = nn.Conv2d(6,16,3,1,0)
        self.pool2 = nn.MaxPool2d(2,2,0)
        self.linear1 = nn.Linear(576,120)
        self.linear2 = nn.Linear(120,84)
        self.output = nn.Linear(84,10)
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        #((8,16,6,6)->(8,576)
        x = x.reshape(x.size(0),-1)
        #print(f'x.shape:{x.shape}')
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.output(x)
    
def train(train_dataset):
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ImageModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    epochs = 10
    for epoch in range(epochs):
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()
        for x, y in dataloader:
            model.train()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(torch.argmax(y_pred, dim= -1))
            # print(y)
            # print(torch.argmax(y_pred, dim=-1).sum())
            total_correct += (torch.argmax(y_pred, dim =-1) == y).sum()
            total_loss += loss.item() * len(y)
            total_samples += len(y)
        print(f'epoch:{epoch + 1}, loss:{total_loss / total_samples:.5f},acc:{total_correct / total_samples:.2f},time:{time.time()-start:.2f}s')
    torch.save(model.state_dict(),'./model/image_model.pth')
    
    
    
def evaluate(test_dataset):
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = ImageModel()
    model.load_state_dict(torch.load('./model/image_model.pth'))
    total_correct, total_samples = 0,0
    for x, y in dataloader:
        model.eval()
        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim =-1)
        total_correct += (y_pred == y).sum()
        total_samples += len(y)
    print(f'acc:{total_correct / total_samples:.2f}')
        

if __name__ == "__main__":
    
    train_dataset, test_dataset = create_dataset()
    '''
    print(f'train:{train_dataset.data.shape}')
    print(f'test:{test_dataset.data.shape}')
    plt.figure(figsize=(2,2))
    plt.imshow(train_dataset.data[7])
    plt.title(train_dataset.targets[7])
    plt.show()
    '''
    #model = ImageModel()
    #summary(model,(3,32,32), batch_size=1)
    
    train(train_dataset)
    evaluate(test_dataset)