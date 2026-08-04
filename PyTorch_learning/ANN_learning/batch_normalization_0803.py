import torch
import torch.nn as nn

def dm01():
    #1.
    input_2d = torch.randn(size=(1,2,3,4))
    print("input_2d:", input_2d)
    
    #2.BN
    bn2d = nn.BatchNorm2d(num_features=2, eps = 1e-5, momentum=0.1, affine=True) #num_features = number of channels
    output_2d = bn2d(input_2d)
    print("output_2d:", output_2d)
    
def dm02():
    #1.
    input_1d = torch.randn(size=(2,2))
    print("input_1d:", input_1d)
    linear1 = nn.Linear(2, 4)
    l1 = linear1(input_1d)
    print("l1:", l1)
    
    #2.BN
    bn1d = nn.BatchNorm1d(num_features=4, eps = 1e-5, momentum=0.1, affine=True) #num_features = number of channels
    output_1d = bn1d(l1)
    print("output_1d:", output_1d)

if __name__ == "__main__":
    #dm01()
    dm02()