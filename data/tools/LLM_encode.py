import numpy as py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import os
import random
import pickle
import lmdb
from transformers import AutoTokenizer, AutoModel
from typing import Literal
#from data_trans_lmdb import data_to_lmdb





class Model_Select(object):

    @classmethod
    def steelbert(cls, text_input, device):
        """
        利用steelbert模型完成对工艺的编码。text_input形如：
        texts = ["A composite steel plate for marine construction was fabricated using 316L stainless steel."]       
        """
        #print("text_input:", text_input) 
        #加载本地下载好的steelbert模型
        model_path = "/home/weidaokuo/project/strain_stress/Nb_alloy/steelbert_model"    
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(device)  # Move model to GPU if available
        inputs = tokenizer(text_input, return_tensors='pt', padding=True, truncation=False).to(device)         
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        hidden_states = outputs.hidden_states       # 所有层的hidden_states
        last_hidden_state = hidden_states[-1]       # 最后一层输出的hidden_states
        cls_embeddings = last_hidden_state[:, 0, :] # 去除cls词对应的向量。shape:[1, 768]
        #print("cls_embeddings.shape:",cls_embeddings.shape)
        return cls_embeddings
        
    @classmethod
    def qwen(cls, text_input, device):
        print("the qwen model is not finished! please use other model.")
        pass



class LLM_process(Model_Select):
    """
    处理流程：
        1、self.csv_path是成分工艺和物理特征的csv文件所在地址。不需要应力应变曲线。
        2、self._read_data()完成了对成分，工艺的文字描述转换，并保存到self.data中。
    """

    def __init__(self, model_name = Literal["steelbert", "qwen"], device=True):
        super().__init__()
    
        """
        :param composition_element_list is a list like ["Mo", "Nb", "Ta"]
        :param composition_fraction_list is a list like [0.96, 0.2, 0.5].（weight fraction）
        """
        self.model_name = model_name
        if device:            
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
          
    def composition_encode(self, element_list:list, fraction_list:list):
        value_list = []
        if hasattr(self, self.model_name):
            for i, j in zip(element_list, fraction_list):
                ouput = getattr(self, self.model_name)([i], self.device)*j
                value_list.append(ouput)
        stacked_tensors = torch.stack(value_list, dim=0)
        summed_tensor = torch.sum(stacked_tensors, dim=0)
        return summed_tensor
    
    def direct_encode(self, process_list:list):
        if hasattr(self, self.model_name):
            ouput = getattr(self, self.model_name)(process_list, self.device)
        return ouput
        
    def steelbert_method(self, element_list:list, fraction_list:list, process_list:list): 
        """
        输出steelbert_output的形状为[1, 1536]
        """
        fraction_list = LLM_process.fraction_list_charge(fraction_list)
        tensor1 = self.composition_encode(element_list, fraction_list)
        tensor2 = self.direct_encode(process_list)
        steelbert_output = torch.cat((tensor1, tensor2), dim=1)
        return steelbert_output
    
    @staticmethod
    def fraction_list_charge(lst):
        """
        判断列表中的最大值：
        - 如果最大值 <= 1，不做任何变化。
        - 如果存在大于 1 的值，将所有数值除以 100。
        """
        if max(lst) > 1:
            return [x / 100 for x in lst]
        return lst    


if __name__=='__main__':
    model = LLM_process("steelbert")
    output = model.composition_encode(["Nb", "Hf", "C"], [96, 3, 1])
    output1 = model.direct_encode(["we are using rolling and melt method as it is processing method."])
    steelbert_output = model.steelbert_method(["Nb", "Hf", "C"], [96, 3, 1], ["we are using rolling and melt method as it is processing method."])
    steelbert_output = steelbert_output.cpu().numpy()
    print(steelbert_output, steelbert_output.shape)
























    
        
        
        
        
        
        
        
        