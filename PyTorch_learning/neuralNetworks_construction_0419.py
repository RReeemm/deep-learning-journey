import torch
import torch.nn as nn
from torchsummary import summary

class ModelDemo(nn.Module):
    #1 init
    def __init__(self):
        #1.1 call super init
        super().__init__()
        #1.2 define layers
        self.linear1 = nn.Linear(3, 3)
        self.linear2 = nn.Linear(3, 2)
        self.output = nn.Linear(2, 2)
        #1.3 initialize weights and bias
        #hidden layer 1
        nn.init.xavier_normal_(self.linear1.weight, gain=1.0)
        nn.init.zeros_(self.linear1.bias)
        #hidden layer 2
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.linear2.bias)
        
    #forward
    def forward(self, x):
        #2.1 hidden layer 1
        x = torch.sigmoid(self.linear1(x))
        #2.2 hidden layer 2
        x = torch.relu(self.linear2(x))
        #2.3 output layer
        x = torch.softmax(self.output(x), dim=-1)
        return x
    
#train
def train():
    #1. create model and print summary
    model = ModelDemo()
    #2. create data
    data = torch.randn(5, 3)
    print(f'input data: {data}')
    print(f'datashape: {data.shape}')
    print(f'data_requires_grad: {data.requires_grad}')
    #3. forward pass
    output = model(data)
    print(f'output: {output}')
    print(f'output datashape: {output.shape}')
    print(f'output requires_grad: {output.requires_grad}')
    
    #4. print model summary
    print('model summary:')
    summary(model, input_size=(5, 3))
    print("==============================")
    for name, param in model.named_parameters():
        print(f'{name}: {param.data}')

if __name__ == '__main__':
    train()