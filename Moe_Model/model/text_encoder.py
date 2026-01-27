"""
@author : weidaokuo
@when : 2024-4-16
"""

import sys
sys.path.append('/home/weidaokuo/wdkhome/A800/20240316/Strain_Stress_Multimodal/models/model7')
import torch
from torch import nn
from config import ModelConfig





class Text_encoder(nn.Module):
        
    def __init__(self, d_model, ffn_hidden):
        super(Text_encoder, self).__init__()
        
        self.linear1 = nn.Linear(ModelConfig.bert_output_dim, d_model)         
        
        self.linear2 = nn.Linear(ffn_hidden, ffn_hidden)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        param x: the input x shape is [batch, 1, 768].
       
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x        



import torch
if __name__=="__main__":
    text = torch.randn((4,1,768))
    ffn = Text_encoder(128, 128)
    ffn = ffn(text)
    print(ffn.shape)
































