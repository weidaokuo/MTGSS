"""
@author : weidaokuo
@when : 2023-10-18
"""
import sys
sys.path.append('/home/weidaokuo/wdkhome/A800/20240316/Strain_Stress_Multimodal/models/model7')
import torch
from torch import nn
from blocks.GPT_layer import GPTlayer
from embedding.positional_encoding import PositionalEncoding

class GPT02(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super(GPT02, self).__init__()
        
        self.layers = nn.ModuleList([GPTlayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
                        
    def forward(self, text_encoder, trg, src_mask):
        """
        param text_encoder: the shape of text_encoder is [batch, 1, 128]
        param trg : the shape of trg is [batch, seq_length, 128]
        """
        for layer in self.layers:
            trg = layer(text_encoder, trg, src_mask)
     
        return trg
        


import torch
if __name__=="__main__":

    decoder_input = torch.randn(4, 100, 128)
    text_encoder = torch.randn(4, 1, 128)
    gpt2 = GPT02(128, 128, 8, 8, 0.1)
    out = gpt2(text_encoder, decoder_input, src_mask=None)
    print(out.shape)
    
    
"""
    print(gpt2.named_parameters())
    for i, (name, param) in enumerate(gpt2.named_parameters()):
        if 'gpt02' in name or 'linear2' in name:
            param.requires_grad = False
            print("ok")
        else:
            print("no")
    out = gpt2(decoder_input)
    print(out.shape)
"""













