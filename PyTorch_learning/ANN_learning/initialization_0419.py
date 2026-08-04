import torch.nn as nn

#uniform distribution initialization
def dm01():
    linear = nn.Linear(3, 2)
    nn.init.uniform_(linear.weight)
    nn.init.uniform_(linear.bias)
    print(f'weight: {linear.weight.data}, bias: {linear.bias.data}')
    
#constant initialization
def dm02():
    linear = nn.Linear(3, 2)
    nn.init.constant_(linear.weight, 0.5)
    nn.init.constant_(linear.bias, 0.1)
    print(f'weight: {linear.weight.data}, bias: {linear.bias.data}')
    
#zero initialization
def dm03():
    linear = nn.Linear(3, 2)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    print(f'weight: {linear.weight.data}, bias: {linear.bias.data}')
    
#one initialization
def dm04():
    linear = nn.Linear(3, 2)
    nn.init.ones_(linear.weight)
    nn.init.ones_(linear.bias)
    print(f'weight: {linear.weight.data}, bias: {linear.bias.data}')
    
#normal distribution initialization
def dm05():
    linear = nn.Linear(3, 2)
    nn.init.normal_(linear.weight, mean=0.0, std=1.0)
    nn.init.normal_(linear.bias, mean=0.0, std=1.0)
    print(f'weight: {linear.weight.data}, bias: {linear.bias.data}')

#kaming uniform initialization
def dm06():
    linear = nn.Linear(3, 2)
    nn.init.kaiming_uniform_(linear.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(linear.bias, mode='fan_in', nonlinearity='relu')
    print(f'weight: {linear.weight.data}, bias: {linear.bias.data}')
    
#xavier normal initialization
def dm07():
    linear = nn.Linear(3, 2)
    nn.init.xavier_normal_(linear.weight, gain=1.0)
    nn.init.xavier_normal_(linear.bias, gain=1.0)
    print(f'weight: {linear.weight.data}, bias: {linear.bias.data}')
    
if __name__ == '__main__':
    #dm01()
    #dm02()
    #dm03()
    #dm04()
    #dm05()
    dm06()
    