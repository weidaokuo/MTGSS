import sys
import os
current_file_path = os.path.abspath(__file__)
nb_alloy_path = os.path.dirname(os.path.dirname(current_file_path))
if nb_alloy_path not in sys.path:
    sys.path.insert(0, nb_alloy_path)
from torch import nn
import torch
from Moe_Model.MLP.moe import MoE
from Moe_Model.norm.layer_norm import LayerNorm, RMSNorm
from Moe_Model.attention.multi_head_attention import MultiHeadAttention
from Args import args
from Moe_Model.embedding.positional_encoding import PositionalEncoding


class Encoder_layer(nn.Module):

    def __init__(self, args):
        super(Encoder_layer, self).__init__()
        self.attention = MultiHeadAttention(d_model=args.d_model, n_head=args.n_head, dropout=args.drop_prob)
        self.cross_attention = MultiHeadAttention(d_model=args.d_model, n_head=args.n_head, dropout=args.drop_prob)
        self.norm = RMSNorm(dim=args.d_model)
        self.dropout = nn.Dropout(p=args.drop_prob)        
        self.ffn = MoE(args)   

    def forward(self, dec, text_encoder):
        _x = dec
        x = self.attention(q=dec, k=dec, v=dec)
        x = self.norm(self.dropout(x) + _x)

        _x = x
        x = self.cross_attention(q=x, k=text_encoder, v=text_encoder)  # 交叉注意力机制中没有mask
        x = self.norm(self.dropout(x) + _x)
        
        # ffn
        _x = x
        weights, indices, x = self.ffn(x)
        x = self.norm(self.dropout(x) + _x)
        return weights, indices, x



class Strain_Stress_Encoder(nn.Module):  
    def __init__(self, args):
            super(Strain_Stress_Encoder, self).__init__()
            self.emb = PositionalEncoding(args.input_embedding_dim, args.sqs_length, args.max_len)
            self.layers = nn.ModuleList([Encoder_layer(args) for _ in range(args.n_layers)])
            self.input_layer1 = nn.Linear(args.HEA_input_dim, args.HEA_output_dim)
            self.input_layer2 = nn.Linear(args.strain_input_dim, args.strain_output_dim)
            self.output_layer = nn.Linear(args.d_model, args.output_d_model)
            self.cross_linear = nn.Linear(args.input_procross_dim, args.d_model)
            
    def forward(self, HEA_data, strain_data, text_data, start_index):
        weights_list = []
        indices_list = []
        x1 = self.input_layer1(HEA_data)
        x2 = self.input_layer2(strain_data)
        x3 = self.cross_linear(text_data)
        x = torch.cat((x1, x2), dim=2)
        x = self.emb(x, start_index)
        for layer in self.layers:
            weights, indices, x = layer(x, x3)
            weights_list.append(weights)
            indices_list.append(indices)
        output = self.output_layer(x)
        if args.output_moe_weights:
            return weights_list, indices_list, output.squeeze(dim=2)
        else:
            return output.squeeze(dim=2)
        
        

























