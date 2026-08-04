import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from torchsummary import summary
from sklearn.preprocessing import StandardScaler
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# todo 1.创建数据集
def create_dataset():
    data = pd.read_csv('data/mobile_price_range_data.csv')  # Load your dataset here
    x ,y = data.iloc[:, :-1], data.iloc[:, -1]
    x = x.astype(np.float32)
    y = y.astype(np.int64)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3, stratify=y)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.tensor(y_test.values))
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))

# todo 2.def layer
class PhonePriceModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 512)
        self.linear4 = nn.Linear(512, 128)
        self.output = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        x = self.output(x)          #softmax is included in CrossEntropyLoss, so we don't need to apply it here
        return x
    
# todo 3.train
def train(train_dataset, input_dim, output_dim):
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model = PhonePriceModel(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epochs = 50
    for epoch in range(epochs):
        total_loss, batch_num = 0, 0
        start_time = time.time()
        for x, y in train_loader:
            model.train()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_num += 1
        #print(f'epoch: {epoch}, loss: {total_loss/batch_num:.4f}, time: {time.time()-start_time:.2f}s')
    #print(f'\n\n模型的参数信息：\n{model.state_dict()}\n\n')
    torch.save(model.state_dict(), './model/model.pth')
    
# todo 4.test
def evaluate(test_dataset, input_dim, output_dim):
    model = PhonePriceModel(input_dim, output_dim)
    model.load_state_dict(torch.load('./model/model.pth'))
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            model.eval()
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, dim = 1)
            correct += (y_pred == y).sum()
        print(f'Accuracy: {correct.item()}/{len(test_dataset)} = {correct.item()/len(test_dataset):.4f}')

if __name__ == "__main__":
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    model = PhonePriceModel(input_dim, output_dim)
    summary(model,input_size=(16, input_dim))
    train(train_dataset, input_dim, output_dim)
    evaluate(test_dataset, input_dim, output_dim)