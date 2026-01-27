"""
@author : weidaokuo
@when : 2024-4-16
"""
import sys
sys.path.append('/home/weidaokuo/wdkhome/A800/20240316/Strain_Stress_Multimodal/models/model7')
import torch
from torch import nn
from embedding.positional_encoding import PositionalEncoding
from config import ModelConfig
from layers.layer_norm import LayerNorm
from model.gpt import GPT02
from model.text_encoder import Text_encoder



class StrainStressGPT(nn.Module):
    def __init__(self, ModelConfig):
        super(StrainStressGPT, self).__init__()
        
        param_config = ModelConfig()
        self.emb = PositionalEncoding(param_config.input_d_model,
                                param_config.max_len
                                )
        
        self.gpt02 = GPT02(param_config.d_model, param_config.ffn_hidden, param_config.n_head, param_config.n_layers, param_config.drop_prob)
        # text encoder
        self.text_encoder = Text_encoder(param_config.d_model, param_config.ffn_hidden)
        
        self.linear1 = nn.Linear(param_config.ffn_hidden*param_config.max_len, param_config.feature_output_dim)  # 预测下一时刻潜变量维度feature_output_dim=2
        self.linear2 = nn.Linear(param_config.ffn_hidden*param_config.max_len, param_config.predict_stress_length)  # 预测未来600个应力点predict_stress_length=600
        self.input_layer = nn.Linear(param_config.input_d_model, param_config.d_model)     # 经过一个线性层，将14维度的输入转化为128维度的输入

    def forward(self, text, trg):
        src_mask = self.make_trg_mask(trg)
        src_mask = src_mask.to(trg.device)
        #trg = self.input_layer(trg)
        trg = self.emb(trg)
        trg = self.input_layer(trg)
        text_embedding = self.text_encoder(text)
        trg = self.gpt02(text_embedding, trg, src_mask)

        # 下游任务
        batch_size = trg.shape[0]
        trg = trg.reshape(batch_size, -1)
        output_feature = self.linear1(trg)
        output_stress = self.linear2(trg)
        return output_feature, output_stress
        
    def make_trg_mask(self, trg):  #trg形状为[B，sqs_length, 512].这里假设每个单词的维度为512。交叉注意力机制中没有mask。
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor)
        return trg_sub_mask   
        
      
import torch
if __name__=="__main__":
    decoder_input = torch.randn(4, 100, 8)
    #text = "Hello, my name is Walker."
    text = torch.randn(4, 1, 768)
    gpt2 = StrainStressGPT(ModelConfig)
    print(gpt2.named_parameters())
    for i, (name, param) in enumerate(gpt2.named_parameters()):
        if 'gpt02' in name or 'linear3' in name:
            param.requires_grad = False
            print("ok")
        else:
            print("no")
    out_feature, out_stress = gpt2(text, decoder_input)
    print(out_stress.shape)
    print(out_feature.shape)
    
    
    
    
    
    