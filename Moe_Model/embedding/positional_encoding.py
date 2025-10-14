"""
@author : weidaokuo
@when : 2024-10-27
"""
import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, sqs_len, max_len):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model.
        :param max_len: max length of the whole strain stress curvle.
        :param sqs_len: length of the model input.
        """
        super(PositionalEncoding, self).__init__()
        self.sqs_len = sqs_len
        self.d_model = d_model
        
        if d_model%2==1:
            d_model = d_model + 1
        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient
        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position
        _2i = torch.arange(0, d_model, step=2).float()                
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words
        #print("self.encoding:",self.encoding.shape)

        if self.d_model%2==1:
            self.encoding = self.encoding[:,:-1]
    def forward(self, x, start_index):
        """
        x:[batch_size, max_len, d_model]
        start_index:[index1, index2, ...],the length of this list 
        """
        #print("x.shape",x.shape)
        out_list = []
        #x_dim = x.shape[2]
        self.encoding = self.encoding.to(x.device)
        for i,j in zip(x, start_index):
            encoding_  = self.encoding[j:j+self.sqs_len, :] 
            out = i + encoding_
            out_list.append(out)
        x = torch.stack(out_list,axis=0)           
        return x




if __name__=="__main__":
    encoder_input = torch.randn(8, 256, 1024)
    pos = PositionalEncoding(1024, 256, 1024)
    pos_out = pos(encoder_input, start_index=[7,8,9,1,6,7,8,13])
    print(pos_out.shape)
