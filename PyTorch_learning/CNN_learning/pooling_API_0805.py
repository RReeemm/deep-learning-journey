import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def dm01():
    inputs = torch.tensor([
        [
            [0,1,2],
            [3,4,5],
            [6,7,8]
        ]
    ])
    print(f'inputs: {inputs}, shape: {inputs.shape}')   #(1, 3, 3)
    #create pooling layer
    pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    outputs1 = pool1(inputs)
    print(f'outputs1: {outputs1}, shape: {outputs1.shape}')  #(1, 2, 2)
    pool2 = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    outputs2 = pool2(inputs)
    print(f'outputs2: {outputs2}, shape: {outputs2.shape}')


def dm02():
    inputs = torch.tensor([
        [
            [0,1,2],
            [3,4,5],
            [6,7,8]
        ],
        [
            [10,20,30],
            [40,50,60],
            [70,80,90],
        ],
        [
            [11,22,33],
            [44,55,66],
            [77,88,99]
        ]
    ])
    print(f'inputs: {inputs}, shape: {inputs.shape}')   #(3, 3, 3)
    #create pooling layer
    pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    outputs1 = pool1(inputs)
    print(f'outputs1: {outputs1}, shape: {outputs1.shape}')  #(3, 2, 2)
    pool2 = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    outputs2 = pool2(inputs)
    print(f'outputs2: {outputs2}, shape: {outputs2.shape}')


if __name__ == "__main__":
    #dm01()
    dm02()