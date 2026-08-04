import torch
import torch.nn as nn

def dm01():
    #1.
    t1 = torch.randint(0, 10, size=(1,4)).float()
    print(t1)
    #2.
    #2.1.create a linear layer
    linear1 = nn.Linear(4, 5)
    #2.2.
    l1 = linear1(t1)
    print("l1:", l1)
    #2.3.
    output1 = torch.relu(l1)
    print("output1:", output1)
    
    #3.
    dropout = nn.Dropout(p=0.5) #dropout layer with 50% probability
    d1 = dropout(output1)
    print("d1:", d1)
    
    
    
    
if __name__ == "__main__":
    dm01()
