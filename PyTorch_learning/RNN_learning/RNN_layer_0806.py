import torch
import torch.nn as nn

rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)
#1.number of words in every sentense
#2.number of sentense
#3.dim of word vector(input_size)
x = torch.randn(size=(5,32,128))

#1.dim of hidden layer(num_layers)
#2.number of sentense
#3.dim of hidden(hidden_size)
h0 = torch.randn(size=(1,32,256))
output, h1 = rnn(x, h0)
def dm01():
    pass

if __name__ == '__main__':
    dm01()
    
