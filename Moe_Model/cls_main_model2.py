


import sys
import os
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取 Nb_alloy 的绝对路径
# 假设 Nb_alloy 是 model.py 的上两级目录
nb_alloy_path = os.path.dirname(os.path.dirname(current_file_path))


# 将 Nb_alloy 添加到 sys.path
if nb_alloy_path not in sys.path:
    sys.path.insert(0, nb_alloy_path)

#print(nb_alloy_path)

from torch import nn
import torch
from Moe_Model.MLP.moe import MoE
from Moe_Model.norm.layer_norm import LayerNorm, RMSNorm
from Moe_Model.attention.multi_head_attention import MultiHeadAttention
from Args import args
from Moe_Model.embedding.positional_encoding import PositionalEncoding
import torch.nn.functional as F

class Encoder_layer(nn.Module):

    def __init__(self, args):
        super(Encoder_layer, self).__init__()
        self.attention = MultiHeadAttention(d_model=args.d_model, n_head=args.n_head, dropout=args.drop_prob)
        self.cross_attention = MultiHeadAttention(d_model=args.d_model, n_head=args.n_head, dropout=args.drop_prob)
        self.norm = RMSNorm(dim=args.d_model)
        self.dropout = nn.Dropout(p=args.drop_prob)        
        #moe输入参数
        self.ffn = MoE(args)   

    def forward(self, dec, text_encoder):
        # self attention
        _x = dec
        x = self.attention(q=dec, k=dec, v=dec)
        x = self.norm(self.dropout(x) + _x)
        
        #此处添加交叉注意力机制
        _x = x
        x = self.cross_attention(q=x, k=text_encoder, v=text_encoder)  # 交叉注意力机制中没有mask
        x = self.norm(self.dropout(x) + _x)
        
        # ffn
        _x = x
        x = self.ffn(x)
        x = self.norm(self.dropout(x) + _x)
        return x







#第一个模型，前向输入为高熵合金特征+经过linear升维后的应变，二者拼接到一起，然后经过基于index的position encoding
class Strain_Stress_Encoder(nn.Module):  
    def __init__(self, args):
            super(Strain_Stress_Encoder, self).__init__()
            self.learnable_embedding = nn.Parameter(torch.randn(1, 1, args.d_model)) 
            self.emb = PositionalEncoding(args.input_embedding_dim, args.sqs_length, args.max_len)
            self.layers = nn.ModuleList([Encoder_layer(args) for _ in range(args.n_layers)])
            
            self.input_layer = nn.Linear(args.HEA_input_dim+1, args.d_model)
            
            #self.cross_linear = nn.Linear(args.input_procross_dim, args.d_model)
            # 非线性降维模块
            self.cross_linear = nn.Sequential(
                nn.Linear(args.input_procross_dim, args.d_model),  # 线性变换
                nn.ReLU(),                         # 激活函数
                nn.Linear(args.d_model, args.d_model)  # 线性变换
                )
                
            self.output_layer = nn.Sequential(
                nn.Linear(args.d_model, args.split_length),  # 线性变换
                nn.ReLU(),                         # 激活函数
                nn.Linear(args.split_length, args.split_length)  # 线性变换
                )    
           
            
    def forward(self, HEA_data, strain_data, text_data, start_index):
        batch_size = HEA_data.size(0)
         # 将可学习的 Embedding 向量扩展到 batch_size
        learnable_embedding = self.learnable_embedding.expand(batch_size, -1, -1)
        
        #print("x2.shape", x2.shape)
        x3 = self.cross_linear(text_data)
        #print("x3.shape", x3.shape)     #[8, 1, 512]
        x = torch.cat((HEA_data, strain_data), dim=-1)   
        x = self.input_layer(x)      
        #print("x.shape:",x.shape) 
        x = self.emb(x, start_index)
        #print("经过embedding之后的：",x.shape)  # [8,512,dim]
        #拼接可学习的cls_embedding
        x = torch.cat([learnable_embedding, x], dim=1)
        for layer in self.layers:
            x = layer(x, x3)
        global_embedding = x[:, 0, :].squeeze(dim=1)  
        #print("global_embedding.shape：",global_embedding.shape)  # [8,512,dim]
        output = self.output_layer(global_embedding)      
        return output
        
        
if __name__=='__main__':
    model = Strain_Stress_Encoder(args)
    HEA_data = torch.randn([8, 512, 15])
    strain_data = torch.randn([8, 512, 1])
    text_data = torch.randn([8, 1, 1536])
    #start_index传入的是tensor形式还是list形式都可
    #start_index = torch.tensor([5, 45, 46, 8, 9, 48, 52, 12])
    start_index = [5, 45, 46, 8, 9, 48, 52, 12]    
    output = model(HEA_data, strain_data, text_data, start_index)
    print(output.shape)           #[8, 512]
























