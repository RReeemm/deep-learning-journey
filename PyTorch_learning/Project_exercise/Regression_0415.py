import torch
from torch.utils.data import DataLoader ,TensorDataset
from torch import nn
from torch import optim
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

#1. dataset
def create_dataset():
    x, y, coef = make_regression(n_samples=100, n_features=1, noise=10, coef=True, bias=14.5, random_state=3)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y, coef

def train(x, y, coef):
    #1. make dataset
    dataset = TensorDataset(x, y)
    #2. make dataloader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    #3. make model, criterion, optimizer
    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    #4. train
    #4.1 define epochs and loss list
    epochs ,loss_list ,total_loss, total_sample = 100, [], 0.0, 0
    #4.2 train loop
    for epoch in range(epochs):
        for train_x, train_y in dataloader:
            #4.2.1 forward
            y_pred = model(train_x)
            #4.2.2 compute loss
            loss = criterion(y_pred, train_y.reshape(-1, 1))
            total_loss += loss.item()
            total_sample += 1
            #4.2.3 backward
            optimizer.zero_grad()
            loss.backward()
            #4.2.4 update
            optimizer.step()
        #4.2.5 record loss and sample
        loss_list.append(total_loss / total_sample)
        print(f'times:{epoch + 1},mean loss:{total_loss / total_sample}')  
    #5. print final parameters and plot loss curve
    print(f'{epochs}times mean loss:{loss_list}')
    print(f'w:{model.weight}, b:{model.bias}, coef:{coef}')
    #6. plot loss curve
    plt.plot(range(1, epochs + 1), loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid()
    plt.show()
    #7. plot data points
    plt.scatter(x,y)
    #7.1 plot predicted line and true line
    y_pred = torch.tensor(data = [v * model.weight + model.bias for v in x], dtype=torch.float32)
    y_true = torch.tensor(data = [v * coef + 14.5 for v in x], dtype=torch.float32)
    #7.2 plot predicted line and true line
    plt.plot(x, y_pred, color='red', label='Predicted')
    plt.plot(x, y_true, color='blue', label='True')
    plt.legend()    
    plt.grid()
    plt.show()

if __name__ == '__main__':
    x, y, coef = create_dataset()
    #print(f'x: {x[:5]}, y: {y[:5]}, coef: {coef}')
    train(x, y, coef)
