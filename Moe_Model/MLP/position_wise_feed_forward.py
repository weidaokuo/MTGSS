"""
@author : weidaokuo
@when : 2023-10-18
"""
from torch import nn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.linear2(x)
        return x

class Output_layer(nn.Module):

    def __init__(self, input_dim, hidden1, output, drop_prob=0.1):
        super(Output_layer, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden1)
        self.linear2 = nn.Linear(hidden1, output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


import torch
if __name__=="__main__":
    encoder_input = torch.randn(4, 1, 1)
    ffn = PositionwiseFeedForward(1, 1)
    ffn = ffn(encoder_input)
    print(ffn.shape)
    print(ffn)