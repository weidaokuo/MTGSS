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
        model_path = " "    
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(device)  
        inputs = tokenizer(text_input, return_tensors='pt', padding=True, truncation=False).to(device)         
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        hidden_states = outputs.hidden_states       
        last_hidden_state = hidden_states[-1]       
        cls_embeddings = last_hidden_state[:, 0, :] 
        #print("cls_embeddings.shape:",cls_embeddings.shape)
        return cls_embeddings
        
    @classmethod
    def qwen(cls, text_input, device):
        print("Users can also add it themselves using a decoder model like Qwen.")
        pass



class LLM_process(Model_Select):
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
        fraction_list = LLM_process.fraction_list_charge(fraction_list)
        tensor1 = self.composition_encode(element_list, fraction_list)
        tensor2 = self.direct_encode(process_list)
        steelbert_output = torch.cat((tensor1, tensor2), dim=1)
        return steelbert_output
    
    @staticmethod
    def fraction_list_charge(lst):
        if max(lst) > 1:
            return [x / 100 for x in lst]
        return lst    


























    
        
        
        
        
        
        
        
        